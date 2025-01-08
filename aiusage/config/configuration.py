from aiusage.constant import *
from aiusage.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from aiusage.utils.main_utils import create_directories, read_yaml


class ConfigurationManager:
    """
    Configuration Manager for handling and initializing configurations for various 
    pipeline stages.

    Attributes:
        config (dict): Parsed YAML configuration file containing all pipeline settings.
        params (dict): Parsed YAML file containing model hyperparameters and other parameters.
        schema (dict): Parsed YAML file containing schema definitions for validation.

    Methods:
        get_data_ingestion_config() -> DataIngestionConfig:
            Returns the configuration for the Data Ingestion stage.

        get_data_validation_config() -> DataValidationConfig:
            Returns the configuration for the Data Validation stage.

        get_data_transformation_config() -> DataTransformationConfig:
            Returns the configuration for the Data Transformation stage.

        get_model_trainer_config() -> ModelTrainerConfig:
            Returns the configuration for the Model Training stage.

        get_model_evaluation_config() -> ModelEvaluationConfig:
            Returns the configuration for the Model Evaluation stage.
    """

    def __init__(
        self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH,
        schema_file_path = SCHEMA_FILE_PATH
    ):
        """
        Initializes the ConfigurationManager by reading YAML configuration, parameter,
        and schema files.

        Args:
            config_file_path (str): Path to the configuration YAML file.
            params_file_path (str): Path to the parameters YAML file.
            schema_file_path (str): Path to the schema YAML file.
        """

        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Fetches the configuration for the Data Ingestion stage.

        Returns:
            DataIngestionConfig: Configuration object containing paths for data ingestion.
        """

        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            zip_file_path=config.zip_file_path
        )
        
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Fetches the configuration for the Data Validation stage.

        Returns:
            DataValidationConfig: Configuration object containing validation paths and schema.
        """

        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            dataset_folder=config.dataset_folder,
            dataset_xlsx=config.dataset_xlsx,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Fetches the configuration for the Data Transformation stage, including validation
        status.

        Returns:
            DataTransformationConfig: Configuration object containing paths and parameters
            for transformation.
        """

        config = self.config.data_transformation
        dataset_val_status_file = self.config.data_validation.STATUS_FILE
        
        with open(dataset_val_status_file, 'r') as f:
            status = f.read()
        
        status = bool(str.split(status)[-1])
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            dataset_folder=config.dataset_folder,
            dataset_xlsx=config.dataset_xlsx,
            train_dataset=config.train_dataset,
            test_dataset=config.test_dataset,
            params=self.params,
            dataset_val_status=status
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Fetches the configuration for the Model Training stage.

        Returns:
            ModelTrainerConfig: Configuration object containing paths and training
            parameters.
        """

        config = self.config.model_trainer
        
        create_directories([config.models, config.stats])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data=config.train_data,
            models=config.models,
            stats=config.stats,
            model_params=self.params
        )
    
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Fetches the configuration for the Model Evaluation stage.

        Returns:
            ModelEvaluationConfig: Configuration object containing paths and evaluation
            parameters.
        """

        config = self.config.model_evaluation
        params = self.params
        
        create_directories([config.eval_stats])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data=config.test_data,
            models=config.models,
            eval_stats=config.eval_stats,
            model_params=params
        )
        
        return model_evaluation_config
