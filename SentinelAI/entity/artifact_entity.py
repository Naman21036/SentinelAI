from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str


@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str


@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str


@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool
    evaluated_model_path: str
    evaluation_score: float


@dataclass
class ModelPusherArtifacts:
    model_path: str
