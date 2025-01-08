from aiusage.entity.config_entity import DataTransformationConfig
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os


class DataTransformation:
    """
    A class to handle the transformation of raw dataset files into a structured format suitable
    for training and testing.

    Attributes:
        config (DataTransformationConfig): Configuration object containing paths and parameters
        required for data transformation.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation object with the provided configuration.

        Args:
            config (DataTransformationConfig): Configuration object containing paths and parameters.
        """

        self.config = config

    def format_save_data(self):
        """
        Transforms the raw dataset into a structured format, splitting it into training and
         testing datasets.

        Process:
            1. Validates if the dataset passed validation (`dataset_val_status`).
            2. Reads data from the Excel file (`dataset_xlsx`) for sample information.
            3. Constructs textual samples by combining:
                - The question from the JSON metadata.
                - The candidate code from the corresponding file.
                - The AI-generated code from the specified file.
            4. Stores plagiarism scores as labels.
            5. Splits the data into training and testing sets based on the `test_split` parameter.
            6. Saves the training and testing sets to CSV files.

        Output:
            - CSV files for training (`train_dataset`) and testing (`test_dataset`) containing
            samples and their labels.
        """

        if self.config.dataset_val_status:
            data_df = pd.read_excel(self.config.dataset_xlsx)

            data_array = []

            for _, row in data_df.iterrows():
                text = ' Question: '

                with open(os.path.join(
                        self.config.dataset_folder,
                        row['coding_problem_id'],
                        row['coding_problem_id'] + '.json'
                    ), 'r') as f:
                
                    file = json.load(f)

                text += file['question'] + ' Candidate code: '

                for dir in os.listdir(
                            os.path.join(self.config.dataset_folder, row['coding_problem_id'])
                        ):
                    if row['coding_problem_id'] + '.' in dir and 'json' not in dir:
                        file_path = os.path.join(
                            self.config.dataset_folder,
                            row['coding_problem_id'],
                            dir
                        )

                        with open(file_path, 'r') as f:
                            script_file = f.read()

                        text += script_file + ' AI Code: '
                        break

                for dir in os.listdir(
                            os.path.join(self.config.dataset_folder, row['coding_problem_id'])
                        ):
                    if row['llm_answer_id'] in dir:
                        file_path = os.path.join(
                            self.config.dataset_folder,
                            row['coding_problem_id'],
                            dir
                        )
                
                        with open(file_path, 'r') as f:
                            script_file = f.read()

                        text += script_file
                        break
                        
                data_array.append(text)
            
            labels_array = list(data_df['plagiarism_score'])

            train_array, test_array, train_labels_array, test_labels_array = train_test_split(
                data_array,
                labels_array,
                test_size=self.config.params.test_split,
                random_state=42
            )

            train_data = pd.DataFrame(
                {
                    'sample': train_array,
                    'label': train_labels_array
                }
            )

            train_data.to_csv(self.config.train_dataset, index=False)

            test_data = pd.DataFrame(
                {
                    'sample': test_array,
                    'label': test_labels_array
                }
            )

            test_data.to_csv(self.config.test_dataset, index=False)
        else:
            print("Dataset didn't pass validation!!!")
