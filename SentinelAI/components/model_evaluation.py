import json
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from SentinelAI.logger import logging
from SentinelAI.exception import CustomException
from SentinelAI.entity.config_entity import ModelEvaluationConfig
from SentinelAI.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)
from services.classifier import HateSpeechClassifier, WordTokenizer, clean_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
    ):
        self.config = model_evaluation_config
        self.trainer_artifacts = model_trainer_artifacts
        self.data_artifacts = data_transformation_artifacts

    def _load_model(self, model_dir: str):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        tokenizer = WordTokenizer.load(os.path.join(model_dir, "vocab.json"))

        model = HateSpeechClassifier(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            lstm_hidden=cfg["lstm_hidden"],
            attn_heads=cfg["attn_heads"],
            dropout=0.0,
        ).to(DEVICE)

        state = torch.load(
            os.path.join(model_dir, "model.pt"), map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(state)
        model.eval()
        return model, tokenizer, cfg["max_length"]

    def evaluate_model(self, model_dir: str) -> float:
        df = pd.read_csv(self.data_artifacts.transformed_data_path).dropna(subset=["tweet"])
        texts = df["tweet"].astype(str).apply(clean_text).tolist()
        labels = df["label"].astype(int).tolist()

        model, tokenizer, max_len = self._load_model(model_dir)

        all_preds = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                ids = [tokenizer.encode(t, max_len) for t in batch]
                tensor = torch.tensor(ids, dtype=torch.long).to(DEVICE)
                probs = torch.sigmoid(model(tensor)).cpu().numpy()
                all_preds.extend((probs > 0.5).astype(int))

        acc = accuracy_score(labels, all_preds)
        f1 = f1_score(labels, all_preds)
        logging.info(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")
        return f1

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("Starting model evaluation")
            current_score = self.evaluate_model(self.trainer_artifacts.trained_model_path)
            best_model_path = self.config.BEST_MODEL_DIR_PATH

            if not os.path.exists(best_model_path):
                is_model_accepted = True
            else:
                best_score = self.evaluate_model(best_model_path)
                is_model_accepted = current_score > best_score

            return ModelEvaluationArtifacts(
                is_model_accepted=is_model_accepted,
                evaluated_model_path=self.trainer_artifacts.trained_model_path,
                evaluation_score=current_score,
            )

        except Exception as e:
            raise CustomException(e, sys) from e
