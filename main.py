from src.logger.custom_logging import logger
from src.pipeline.stage_01_data_ingestion_pipe import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_03_model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.pipeline.stage_04_data_evaluation import ModelEvaluationTrainingPipeline
from src.exceptions.expection import CustomException
import sys


def run_stage(stage_name, pipeline_class):
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    stages = [
        ("Data Ingestion stage", DataIngestionTrainingPipeline),
        ("Data Transformation stage", DataTransformationTrainingPipeline),
        ("Model Trainer stage", ModelTrainerTrainingPipeline),
        ('Model Evaluation stage',ModelEvaluationTrainingPipeline)

    ]

    for stage_name, pipeline_class in stages:
        run_stage(stage_name, pipeline_class)  