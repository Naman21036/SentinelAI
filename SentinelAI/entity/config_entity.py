from dataclasses import dataclass
import os
from SentinelAI.constants import *


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.DATA_INGESTION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(os.getcwd(), DATA_DIR, ZIP_FILE_NAME)
        self.EXTRACTEED_DATA_DIR = self.DATA_INGESTION_ARTIFACTS_DIR
        self.IMBALANCED_DATA_PATH = os.path.join(self.EXTRACTEED_DATA_DIR, IMBALANCED_DATA_FILE)
        self.RAW_DATA_PATH = os.path.join(self.EXTRACTEED_DATA_DIR, RAW_DATA_FILE)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.MODEL_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, "sentinelai_model")
        self.TOKENIZER_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, "sentinelai_tokenizer")
        self.OUTPUT_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, "training_logs")


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.MODEL_EVALUATION_MODEL_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH = os.path.join(self.MODEL_EVALUATION_MODEL_DIR, BEST_MODEL_DIR)
        self.MODEL_NAME = MODEL_NAME
        self.TOKENIZER_NAME = TOKENIZER_NAME
        self.CURRENT_MODEL_PATH = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, self.MODEL_NAME)
        self.CURRENT_TOKENIZER_PATH = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, self.TOKENIZER_NAME)


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR, BEST_MODEL_DIR)
