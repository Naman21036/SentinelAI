import os
import sys
from zipfile import ZipFile
from SentinelAI.logger import logging
from SentinelAI.exception import CustomException
from SentinelAI.entity.config_entity import DataIngestionConfig
from SentinelAI.entity.artifact_entity import DataIngestionArtifacts
import pandas as pd
import numpy as np

class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config


    def unzip_data(self):
        logging.info("Entered unzip_data method")

        try:
            # Create artifacts directory
            os.makedirs(self.config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            # Extract zip from local path
            with ZipFile(self.config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.config.DATA_INGESTION_ARTIFACTS_DIR)

            logging.info("Zip extraction completed")

            return (
                self.config.IMBALANCED_DATA_PATH,
                self.config.RAW_DATA_PATH
            )

        except Exception as e:
            raise CustomException(e, sys) from e
    def data_validation(self):
        logging.info("Entered data_validation method")
        # Implement any data validation logic here if needed
        try:
            imb_df= pd.read_csv(self.config.IMBALANCED_DATA_PATH)
            logging.info(f'The columns in the dataset are: {imb_df.columns} and the shape of the dataset is: {imb_df.shape}')
            raw_df= pd.read_csv(self.config.RAW_DATA_PATH)
            logging.info(f'The columns in the dataset are: {raw_df.columns} and the shape of the dataset is: {raw_df.shape}')
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion")

        try:
            imbalance_data_file_path, raw_data_file_path = self.unzip_data()

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )
            self.data_validation()

            logging.info(f"Data ingestion completed: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e