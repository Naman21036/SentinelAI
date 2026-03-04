from dataclasses import dataclass
import os
from SentinelAI.constants import *

# Data ingestion configuration
@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.DATA_INGESTION_ARTIFACTS_DIR = os.path.join(os.getcwd(),ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(os.getcwd(),DATA_DIR, ZIP_FILE_NAME)

        #WHERE EXTRACTED FILES WILL BE SAVED
        self.EXTRACTEED_DATA_DIR = self.DATA_INGESTION_ARTIFACTS_DIR

        #FINAL CSV PATHS AFTER EXTRACTION
        self.IMBALANCED_DATA_PATH = os.path.join(self.EXTRACTEED_DATA_DIR, IMBALANCED_DATA_FILE)
        self.RAW_DATA_PATH = os.path.join(self.EXTRACTEED_DATA_DIR, RAW_DATA_FILE)

# Data transformation configuration
@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE 
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS 
        self.LABEL = LABEL
        self.TWEET = TWEET

# Model trainer configuration
@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.MODEL_DIR = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR,
            "sentinelai_model"
        )

        self.TOKENIZER_DIR = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR,
            "sentinelai_tokenizer"
        )

        self.OUTPUT_DIR = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR,
            "training_logs"
        )

        self.HF_MODEL_NAME = HF_MODEL_NAME
        self.MAX_LENGTH = MAX_LENGTH
        self.NUM_LABELS = NUM_LABELS
        self.LEARNING_RATE = LEARNING_RATE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.EVAL_BATCH_SIZE = EVAL_BATCH_SIZE
        self.WEIGHT_DECAY = WEIGHT_DECAY


#Model evaluation configuration
@dataclass
class ModelEvaluationConfig:
    def __init__(self):

        self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_EVALUATION_ARTIFACTS_DIR
        )

        self.BEST_MODEL_DIR_PATH: str = os.path.join(
            self.MODEL_EVALUATION_MODEL_DIR,
            BEST_MODEL_DIR
        )

        self.MODEL_NAME = MODEL_NAME
        self.TOKENIZER_NAME = TOKENIZER_NAME

        self.CURRENT_MODEL_PATH: str = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR,
            self.MODEL_NAME
        )

        self.CURRENT_TOKENIZER_PATH: str = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR,
            self.TOKENIZER_NAME
        )

# Model pusher configuration
@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.TRAINED_MODEL_DIR = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_TRAINER_ARTIFACTS_DIR
        )

        self.BEST_MODEL_DIR = os.path.join(
            os.getcwd(),
            ARTIFACTS_DIR,
            MODEL_EVALUATION_ARTIFACTS_DIR,
            BEST_MODEL_DIR
        )