from aiusage.constant.costum_dataset import AICodeDataset
from aiusage.constant.costum_model import CustomBERTModel
from aiusage.entity.config_entity import ModelEvaluationConfig
from aiusage.utils.main_utils import save_json, plot_confusion_matrix, plot_roc_curve
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import pandas as pd
import os

class ModelEvaluation:
    """
    A class for evaluating trained models on a test dataset.

    Attributes:
        config (ModelEvaluationConfig): Configuration object containing paths and parameters
                                        for model evaluation.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation class with the given configuration.

        Args:
            config (ModelEvaluationConfig): Configuration object containing paths and parameters
                                            for model evaluation.
        """

        self.config = config
    
    def model_testing(self):
        """
        Tests the trained models on the test dataset, evaluates performance, and saves metrics.

        The function performs the following:
        1. Loads the test dataset and initializes the tokenizer and dataloader.
        2. Iteratively evaluates each model (for each fold) on the test dataset.
        3. Calculates evaluation metrics such as loss, accuracy, confusion matrix, and AUC score.
        4. Saves evaluation reports and visualizations like confusion matrices and ROC curves.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: ', device)

        test_df = pd.read_csv(self.config.test_data)
        test_list = list(test_df[test_df.columns[0]])
        test_labels_list = list(test_df[test_df.columns[1]])

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        test_dataset = AICodeDataset(test_list, test_labels_list, tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.model_params.batch_size,
            shuffle=False
        )
        
        criterion = nn.BCEWithLogitsLoss()

        for fold in tqdm(range(self.config.model_params.num_folds)):
            model = CustomBERTModel(dropout_prob=self.config.model_params.dropout).to(device)
            model.load_state_dict(torch.load(os.path.join(self.config.models, f'model_{fold}.pth')))
            model.eval()

            total_loss = 0
            test_preds = []
            test_labels = []

            with torch.no_grad():
                for batch in tqdm(test_loader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].unsqueeze(1).to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    preds = torch.sigmoid(outputs).cpu().detach().numpy().flatten()
                    test_preds.extend(preds)
                    test_labels.extend(labels.cpu().numpy().flatten())
            
            binary_test_preds = [
                1 if pred >= self.config.model_params.threshold else 0 for pred in test_preds
            ]

            binary_test_labels = [
                1 if label >= self.config.model_params.threshold else 0 for label in test_labels
            ]

            report = classification_report(
                binary_test_labels,
                binary_test_preds,
                zero_division=0,
                output_dict=True
            )

            save_json(
                os.path.join(self.config.eval_stats, f'report_{fold}_{self.config.model_params.threshold}'), 
                report
            )

            conf_matrix = confusion_matrix(binary_test_labels, binary_test_preds)

            plot_confusion_matrix(
                conf_matrix,
                self.config.eval_stats,
                fold,
                f'Confusion Matrix for Fold {fold} and Threshold {self.config.model_params.threshold}',
                self.config.model_params.threshold
            )

            auc_score = roc_auc_score(binary_test_labels, test_preds)
            auc_score_dict = {'auc_score': auc_score}

            save_json(
                os.path.join(self.config.eval_stats, f'auc_score_{fold}_{self.config.model_params.threshold}'), 
                auc_score_dict
            )

            plot_roc_curve(
                binary_test_labels,
                binary_test_preds,
                self.config.eval_stats,
                fold,
                self.config.model_params.threshold
            )
