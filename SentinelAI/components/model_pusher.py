import os
import sys
import shutil
from SentinelAI.logger import logging
from SentinelAI.exception import CustomException
from SentinelAI.entity.config_entity import ModelPusherConfig
from SentinelAI.entity.artifact_entity import ModelPusherArtifacts
from SentinelAI.entity.artifact_entity import ModelEvaluationArtifacts


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifacts: ModelEvaluationArtifacts
    ):
        self.config = model_pusher_config
        self.evaluation_artifacts = model_evaluation_artifacts


    def initiate_model_pusher(self) -> ModelPusherArtifacts:

        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            source_dir = self.evaluation_artifacts.evaluated_model_path
            destination_dir = self.config.BEST_MODEL_DIR

            if os.path.exists(destination_dir):
                shutil.rmtree(destination_dir)

            shutil.copytree(source_dir, destination_dir)

            logging.info("Best model updated locally")

            model_pusher_artifact = ModelPusherArtifacts(
                model_path=destination_dir
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e