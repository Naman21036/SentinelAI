from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str

#Data transformation artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str

#Model trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

#Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool
    evaluated_model_path: str
    evaluation_score: float

#Model pusher artifacts
@dataclass
class ModelPusherArtifacts:
    model_path:str
