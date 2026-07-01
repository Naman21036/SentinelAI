import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)

DATA_DIR = "data"
ZIP_FILE_NAME = "dataset.zip"
ZIP_FILE_PATH = os.path.join(DATA_DIR, ZIP_FILE_NAME)
IMBALANCED_DATA_FILE = "imbalanced_data.csv"
RAW_DATA_FILE = "raw_data.csv"

LABEL = "label"
TWEET = "tweet"

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"

DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME = "final.csv"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COLUMNS = ["Unnamed: 0", "count", "hate_speech", "offensive_language", "neither"]
CLASS = "class"

MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"

MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
BEST_MODEL_DIR = "best_Model"
MODEL_NAME = "sentinelai_model"
TOKENIZER_NAME = "sentinelai_tokenizer"
APP_HOST = "0.0.0.0"
APP_PORT = 8080
