import os
import json
import sys

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader

from SentinelAI.logger import logging
from SentinelAI.exception import CustomException
from SentinelAI.entity.config_entity import ModelTrainerConfig
from SentinelAI.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)
from SentinelAI.constants import TWEET, LABEL
from services.classifier import (
    HateSpeechClassifier,
    WordTokenizer,
    clean_text,
    VOCAB_SIZE, EMBED_DIM, LSTM_HIDDEN, ATTN_HEADS, DROPOUT, MAX_LENGTH,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
PATIENCE = 4


class _TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.seqs = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_artifacts = data_transformation_artifacts
        self.config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("Reading transformed dataset")
            df = pd.read_csv(self.data_artifacts.transformed_data_path)
            df = df.dropna(subset=[TWEET])
            df[TWEET] = df[TWEET].astype(str)

            X = df[TWEET].to_numpy()
            y = df[LABEL].to_numpy().astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            logging.info("Building vocabulary")
            tokenizer = WordTokenizer(VOCAB_SIZE)
            tokenizer.fit(X_train)

            train_loader = DataLoader(
                _TweetDataset(X_train, y_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True
            )
            test_loader = DataLoader(
                _TweetDataset(X_test, y_test, tokenizer), batch_size=BATCH_SIZE
            )

            logging.info("Building BiLSTM + Attention model")
            model = HateSpeechClassifier(
                vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                lstm_hidden=LSTM_HIDDEN, attn_heads=ATTN_HEADS, dropout=DROPOUT,
            ).to(DEVICE)

            pos_weight = torch.tensor(
                [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
            ).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=2, factor=0.5,
            )

            best_f1, best_state, no_improve = 0.0, {}, 0

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for seqs, labels in train_loader:
                    seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    criterion(model(seqs), labels).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for seqs, labels in test_loader:
                        probs = torch.sigmoid(model(seqs.to(DEVICE))).cpu().numpy()
                        preds.extend((probs > 0.5).astype(int))
                        trues.extend(labels.numpy().astype(int))

                f1 = f1_score(trues, preds)
                scheduler.step(f1)
                logging.info(f"Epoch {epoch}/{EPOCHS}  f1={f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break

            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            model.load_state_dict(best_state)

            model_path = os.path.join(self.config.MODEL_DIR, "model.pt")
            vocab_path = os.path.join(self.config.MODEL_DIR, "vocab.json")
            config_path = os.path.join(self.config.MODEL_DIR, "config.json")

            torch.save(model.state_dict(), model_path)
            tokenizer.save(vocab_path)

            with open(config_path, "w") as f:
                json.dump({
                    "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM,
                    "lstm_hidden": LSTM_HIDDEN, "attn_heads": ATTN_HEADS,
                    "dropout": DROPOUT, "max_length": MAX_LENGTH,
                }, f, indent=2)

            logging.info(f"Model training complete. Best F1: {best_f1:.4f}")
            logging.info(classification_report(trues, preds, target_names=["Safe", "Hate/Toxic"]))

            return ModelTrainerArtifacts(trained_model_path=self.config.MODEL_DIR)

        except Exception as e:
            raise CustomException(e, sys) from e
