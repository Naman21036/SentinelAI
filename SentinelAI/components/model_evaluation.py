import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from SentinelAI.logger import logging
from SentinelAI.exception import CustomException
from SentinelAI.entity.config_entity import ModelEvaluationConfig
from SentinelAI.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)


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

    def evaluate_model(self, model_path, tokenizer_path):

        df = pd.read_csv(self.data_artifacts.transformed_data_path)
        df = df.dropna(subset=["tweet"])

        texts = df["tweet"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        batch_size = 32  

        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(labels, all_preds)
        f1 = f1_score(labels, all_preds)

        logging.info(f"Evaluation Accuracy: {acc}")
        logging.info(f"Evaluation F1: {f1}")

        return f1
    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:

        try:
            logging.info("Starting Model Evaluation")

            current_model_path = self.trainer_artifacts.trained_model_path
            current_tokenizer_path = self.config.CURRENT_TOKENIZER_PATH

            current_score = self.evaluate_model(
                current_model_path, current_tokenizer_path
            )

            best_model_path = self.config.BEST_MODEL_DIR_PATH

            if not os.path.exists(best_model_path):
                is_model_accepted = True
            else:
                best_score = self.evaluate_model(
                    best_model_path,
                    best_model_path.replace("sentinelai_model", "sentinelai_tokenizer"),
                )

                is_model_accepted = current_score > best_score

            return ModelEvaluationArtifacts(
                is_model_accepted=is_model_accepted,
                evaluated_model_path=current_model_path,
                evaluation_score=current_score,
            )

        except Exception as e:
            raise CustomException(e, sys) from e