import os
import sys
import torch
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from SentinelAI.logger import logging
from SentinelAI.exception import CustomException
from SentinelAI.entity.config_entity import ModelTrainerConfig
from SentinelAI.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts
)
from SentinelAI.constants import TWEET, LABEL


class ModelTrainer:

    def __init__(self,
                 data_transformation_artifacts: DataTransformationArtifacts,
                 model_trainer_config: ModelTrainerConfig):

        self.data_artifacts = data_transformation_artifacts
        self.config = model_trainer_config


    def compute_metrics(self, pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )

        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


    def initiate_model_trainer(self) -> ModelTrainerArtifacts:

        try:
            logging.info("Reading transformed dataset")

            df = pd.read_csv(self.data_artifacts.transformed_data_path)

            df = df.dropna(subset=[TWEET])
            df[TWEET] = df[TWEET].astype(str)
            print(df[LABEL].unique())
            print(df[LABEL].min(), df[LABEL].max())

            train_df, val_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_dataset = Dataset.from_pandas(
                train_df[[TWEET, LABEL]].rename(columns={TWEET: "text"})
            )

            val_dataset = Dataset.from_pandas(
                val_df[[TWEET, LABEL]].rename(columns={TWEET: "text"})
            )

            logging.info("Loading tokenizer")

            tokenizer = DistilBertTokenizer.from_pretrained(
                self.config.HF_MODEL_NAME
            )

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )

            train_dataset = train_dataset.map(tokenize, batched=True)
            val_dataset = val_dataset.map(tokenize, batched=True)

            train_dataset.set_format(
                "torch",
                columns=["input_ids", "attention_mask", "label"]
            )

            val_dataset.set_format(
                "torch",
                columns=["input_ids", "attention_mask", "label"]
            )

            logging.info("Loading transformer model")

            model = DistilBertForSequenceClassification.from_pretrained(
                self.config.HF_MODEL_NAME,
                num_labels=self.config.NUM_LABELS
            )

            training_args = TrainingArguments(
                output_dir=self.config.OUTPUT_DIR,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=self.config.LEARNING_RATE,
                per_device_train_batch_size=self.config.TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=self.config.EVAL_BATCH_SIZE,
                num_train_epochs=self.config.NUM_EPOCHS,
                weight_decay=self.config.WEIGHT_DECAY,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                fp16=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )

            logging.info("Starting training")

            trainer.train()

            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            os.makedirs(self.config.TOKENIZER_DIR, exist_ok=True)

            model.save_pretrained(self.config.MODEL_DIR)
            tokenizer.save_pretrained(self.config.TOKENIZER_DIR)

            logging.info("Model training complete")

            return ModelTrainerArtifacts(
                trained_model_path=self.config.MODEL_DIR,
            )

        except Exception as e:
            raise CustomException(e, sys) from e