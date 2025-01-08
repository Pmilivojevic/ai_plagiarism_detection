import os
import sys
import torch
from torch import nn
from transformers import AdamW
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
from aiusage.constant.costum_dataset import AICodeDataset
from aiusage.constant.costum_model import CustomBERTModel
from transformers import BertTokenizer
from aiusage.utils.main_utils import save_json, plot_metric, plot_confusion_matrix, plot_roc_curve
from aiusage.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    """
    Handles the training and validation process for a custom BERT-based model, 
    including data preparation, training across multiple folds, evaluation, 
    and metric visualization.

    Attributes:
        config (ModelTrainerConfig): Configuration object containing all paths and 
                                     parameters required for model training.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer with a given configuration.

        Args:
            config (ModelTrainerConfig): Configuration for model training, including
                                         paths and parameters.
        """

        self.config = config
    
    def train(self, epoch, model, train_loader, optimizer, criterion, device):
        """
        Train the model for a single epoch.

        Args:
            epoch (int): Current epoch number.
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            criterion (nn.Module): Loss function.
            device (str): Device to use ('cuda' or 'cpu').

        Returns:
            tuple: Average training loss and accuracy for the epoch.
        """

        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].unsqueeze(1).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.sigmoid(outputs).squeeze().cpu().detach().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            binary_preds = [
                1 if pred >= self.config.model_params.threshold else 0 for pred in train_preds
            ]

            binary_labels = [
                1 if label >= self.config.model_params.threshold else 0 for label in train_labels
            ]

            accuracy = accuracy_score(binary_labels, binary_preds)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print()
            sys.stdout.write(
                "epoch:%2d/%2d - train_loss:%.4f - train_accuracy:%.4f" %(
                    epoch,
                    self.config.model_params.num_epochs,
                    loss.item(),
                    accuracy
                )
            )
            sys.stdout.flush()
        
        return total_loss / len(train_loader), accuracy
    
    def validation(self, model, val_loader, criterion, device):
        """
        Validate the model on the validation dataset.

        Args:
            model (nn.Module): The model to validate.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.
            device (str): Device to use ('cuda' or 'cpu').

        Returns:
            tuple: Average validation loss, predictions, and ground truth labels.
        """

        model.eval()
        total_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].unsqueeze(1).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().detach().numpy().flatten()
                val_predictions.extend(preds)
                val_labels.extend(labels.cpu().numpy().flatten())
        
        return total_loss / len(val_loader), val_predictions, val_labels
    
    def train_compose(self):
        """
        Perform the full training and validation process, including:
        - Loading and tokenizing the dataset.
        - Splitting the dataset into folds.
        - Training and validating across multiple folds.
        - Saving metrics, model checkpoints, and plots.

        This method implements k-fold cross-validation and logs metrics such as 
        accuracy, loss, AUC, and confusion matrices for each fold.
        """
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: ', device)

        train_df = pd.read_csv(self.config.train_data)
        train_list = list(train_df[train_df.columns[0]])
        train_labels_list = list(train_df[train_df.columns[1]])

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        dataset = AICodeDataset(train_list, train_labels_list, tokenizer)

        kf = KFold(n_splits=self.config.model_params.num_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(dataset))):
            print(f'Fold {fold + 1}/{self.config.model_params.num_folds}')

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.model_params.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.model_params.batch_size,
                shuffle=True
            )

            model = CustomBERTModel(dropout_prob=self.config.model_params.dropout).to(device)
            optimizer = AdamW(model.parameters(), lr=self.config.model_params.lr)
            criterion = nn.BCEWithLogitsLoss()

            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            best_val_loss = float('inf')
            epochs_range = range(1, self.config.model_params.num_epochs + 1)

            for epoch in tqdm(range(self.config.model_params.num_epochs)):
                train_epoch_loss, train_epoch_acc = self.train(
                    epoch,
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device
                )

                train_losses.append(train_epoch_loss)
                train_accuracies.append(train_epoch_acc)

                val_epoch_loss, val_preds, val_labels = self.validation(
                    model,
                    val_loader,
                    criterion,
                    device
                )

                val_losses.append(val_epoch_loss)

                binary_val_preds = [
                    1 if pred >= self.config.model_params.threshold else 0 for pred in val_preds
                ]
                
                binary_val_labels = [
                    1 if label >= self.config.model_params.threshold else 0 for label in val_labels
                ]

                val_epoch_accuracy = accuracy_score(binary_val_labels, binary_val_preds)
                val_accuracies.append(val_epoch_accuracy)

                print(f'Epoch [{epoch+1}/{self.config.model_params.num_epochs}], '
                      f'Loss: {train_epoch_loss:.4f}, '
                      f'Validation Loss: {val_epoch_loss:.4f}, '
                      f'Train Accuracy: {train_epoch_acc:.2f}%, '
                      f'Validation Accuracy: {val_epoch_accuracy:.2f}%')
                
                if val_epoch_loss < best_val_loss:
                    torch.save(model.state_dict(), os.path.join(self.config.models, f'model_{fold}.pth'))
                    tokenizer.save_pretrained(self.config.models)

                    report = classification_report(
                        binary_val_labels,
                        binary_val_preds,
                        zero_division=0,
                        output_dict=True
                    )

                    save_json(
                        os.path.join(self.config.stats, f'report_{fold}_{self.config.model_params.threshold}'),
                        report
                    )

                    conf_matrix = confusion_matrix(binary_val_labels, binary_val_preds)

                    plot_confusion_matrix(
                        conf_matrix,
                        self.config.stats,
                        fold,
                        f'Confusion Matrix for Fold {fold} and Threshold {self.config.model_params.threshold}',
                        self.config.model_params.threshold,
                    )

                    auc_score = roc_auc_score(binary_val_labels, val_preds)
                    auc_score_dict = {'auc_score': auc_score}

                    save_json(
                        os.path.join(self.config.stats, f'auc_score_{fold}_{self.config.model_params.threshold}'), 
                        auc_score_dict
                    )
                    
                    plot_roc_curve(
                        binary_val_labels,
                        binary_val_preds,
                        self.config.stats,
                        fold,
                        self.config.model_params.threshold
                    )
            
            plot_metric(
                self.config.stats,
                epochs_range,
                train_losses,
                val_losses,
                'Train Loss',
                'Validation Loss',
                fold,
                self.config.model_params.threshold
            )

            plot_metric(
                self.config.stats,
                epochs_range,
                train_accuracies,
                val_accuracies,
                'Train Accuracies',
                'Validation Accuracies',
                fold,
                self.config.model_params.threshold
            )
