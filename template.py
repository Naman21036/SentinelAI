import numpy as np
import pandas as pd
import logging
import os
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

project_name= "SentinelAI"

list_of_files= [
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configuration/gcloud_syncer.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/custom_exception.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logger.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/train_pipeline.py",
    f"{project_name}/pipeline/predict_pipeline.py",
    f"{project_name}/ml/__init__.py",
    f"{project_name}/ml/model.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/utils.py",
    "app.py",
    "demo.py",
    "requirements.txt",
    "Dockerfile",
    "setup.py",
    ".dockerignore"
]

for file_path in list_of_files:
    file_path = Path(file_path)

    filedir, filename = os.path.split(file_path)


    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for file: {filename}")


    if not file_path.exists():
        with open(file_path, "w") as f:
            pass
        logging.info(f"Created empty file: {file_path}")
    else:
        logging.info(f"{filename} already exists")
