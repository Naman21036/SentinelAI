import os
import sys
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from SentinelAI.logger import logging
from SentinelAI.exception import CustomException


class PredictionPipeline:
    def __init__(self):

        self.model_dir = os.path.join(
            os.getcwd(),
            "artifacts",
            "ModelEvaluationArtifacts",
            "best_Model"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):

        logging.info("Running prediction")

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs).item()

            if prediction == 1:
                return "hate and abusive"
            else:
                return "no hate"

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text: str):

        logging.info("Entered run_pipeline method")

        try:
            return self.predict(text)

        except Exception as e:
            raise CustomException(e, sys) from e