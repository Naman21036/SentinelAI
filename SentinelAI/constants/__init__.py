import os
from datetime import datetime

# Define constants
TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)

#Dataset constants
DATA_DIR= "data"
ZIP_FILE_NAME = "dataset.zip"
ZIP_FILE_PATH = os.path.join(DATA_DIR, ZIP_FILE_NAME)
#Extracted data file names
IMBALANCED_DATA_FILE = "imbalanced_data.csv"
RAW_DATA_FILE = "raw_data.csv"

#Column names
LABEL= "label"
TWEET= "tweet"

# Data ingestion artifacts
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"

#Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TRANSFORMED_FILE_NAME = "final.csv"
DATA_DIR = "data"
ID = 'id'
AXIS = 1
INPLACE = True
DROP_COLUMNS = ['Unnamed: 0','count','hate_speech','offensive_language','neither']
CLASS = 'class'

#Model Trainer
HF_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"

#Model Evaluation
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
BEST_MODEL_DIR = "best_Model"
MODEL_NAME = "sentinelai_model"
TOKENIZER_NAME = "sentinelai_tokenizer"
APP_HOST = "0.0.0.0"
APP_PORT = 8080