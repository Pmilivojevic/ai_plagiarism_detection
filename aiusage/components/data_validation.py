import os
import pandas as pd
from aiusage.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    A class to perform data validation for a dataset.

    Attributes:
        config (DataValidationConfig): The configuration settings for the data validation process.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation object.

        Args:
            config (DataValidationConfig): Configuration object containing settings for data validation.
        """

        self.config = config
    
    def validate_dataset(self)-> bool:
        """
        Validates the dataset against the expected schema and checks if all expected directories exist.

        Steps:
        1. Checks if the dataset's column names match the expected schema.
        2. Ensures that all unique names in the first column of the dataset Excel file 
           have corresponding directories in the dataset folder.
        3. Writes the validation status to a status file.

        Returns:
            bool: True if the dataset passes validation, False otherwise.

        Raises:
            Exception: If there is an unexpected error during the validation process.
        """
        
        try:
            validation_status = None

            data_df = pd.read_excel(self.config.dataset_xlsx)
            all_cols = list(data_df.columns)

            for col in self.config.all_schema:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                
                    return validation_status
                
                elif validation_status == None:
                    validation_status = True
            
            for name in data_df[data_df.columns[0]].unique():
                if name not in os.listdir(self.config.dataset_folder):
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    
                    return validation_status
            
            with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    
            return validation_status
        
        except Exception as e:
            raise e
