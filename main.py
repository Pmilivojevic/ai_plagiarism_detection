from aiusage.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from aiusage.pipelines.stage_02_data_validation import DataValidationTrainingPipeline
from aiusage.pipelines.stage_03_data_transformation import DataTransformationTrainingPipeline
from aiusage.pipelines.stage_04_model_trainer import ModelTrainerTrainingPipeline
from aiusage.pipelines.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from aiusage import logger


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e
