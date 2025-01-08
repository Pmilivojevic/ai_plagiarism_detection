from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    A data class to hold the configuration settings for data ingestion process.

    Attributes:
        root_dir (Path): The root directory where the dataset and related files will be stored.
        zip_file_path (Path): The file path to the zip file containing the dataset.
    """

    root_dir: Path
    zip_file_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    A data class to hold the configuration settings for data validation process.

    Attributes:
        root_dir (Path): The root directory where all data validation files are located.
        dataset_folder (Path): The root directory where all data validation files are located.
        dataset_xlsx (Path): Path to the dataset in Excel format.
        STATUS_FILE (Path): Path to the file that stores the status of the data validation process.
        all_schema (list): A list representing the expected columns of dataset Exel file, used for
        validation.
    """
    
    root_dir: Path
    dataset_folder: Path
    dataset_xlsx: Path
    STATUS_FILE: Path
    all_schema: list


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    A dataclass to hold configuration settings for data transformation process.

    Attributes:
        root_dir (Path): The root directory where all data transformation files are located.
        dataset_folder (Path): The folder containing the raw dataset.
        dataset_xlsx (Path): The path to the dataset Exel file.
        train_dataset (Path): The directory where the transformed training dataset will be stored.
        test_dataset (Path): The directory where the transformed testing dataset will be stored.
        params (dict): A dictionary containing various parameters for data transformation.
        dataset_val_status (bool): A flag indicating whether the dataset has passed validation.
    """

    root_dir: Path
    dataset_folder: Path
    dataset_xlsx: Path
    train_dataset: Path
    test_dataset: Path
    params: dict
    dataset_val_status: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    A dataclass to hold configuration settings for Model Trainer process.

    Attributes:
        root_dir (Path): Root directory for storing all training-related files.
        train_data (Path): Path to the training data (e.g., a CSV or preprocessed dataset).
        models (Path): Directory to save the trained model(s).
        stats (Path): Directory to save metrics, plots, and other statistics generated during
                      training.
        model_params (dict): Dictionary of model-specific parameters, including architecture
                             details and hyperparameters like learning rate, batch size, etc.
    """

    root_dir: Path
    train_data: Path
    models: Path
    stats: Path
    model_params: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    A dataclass to hold configuration settings for model evaluation process.

    Attributes:
        root_dir (Path): The root directory for all evaluation-related files.
        test_data (Path): Path to the test dataset file.
        models (Path): Path to the directory containing the trained model files.
        eval_stats (Path): Path to save evaluation statistics (e.g., reports, plots).
        model_params (dict): Dictionary containing parameters used for evaluation, 
                             such as batch size, thresholds, and other settings.
    """

    root_dir: Path
    test_data: Path
    models: Path
    eval_stats: Path
    model_params: dict
