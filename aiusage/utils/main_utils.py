import os
from box.exceptions import BoxValueError
import yaml
from aiusage import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True):
    """
    create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created.
        Defaults to False.
    """
    
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.full_load(yaml_file)
            # logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:
        raise e

@ensure_annotations
def save_json(path: str, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    
    return f"~ {size_in_kb} KB"

def plot_metric(
        metric_path,
        range,
        train_matric,
        val_matric,
        train_label,
        val_label,
        fold,
        threshold
    ):
    """
    Plots and saves the training and validation metrics for a specific fold.

    Args:
        metric_path (str): The directory path where the plot will be saved.
        range (list): Range of X axes.
        train_metric (list): A list of training metric values (e.g., loss).
        val_metric (list): A list of validation metric values (e.g., loss).
        train_label (str): Label for the training metric (used in the legend and filename).
        val_label (str): Label for the validation metric (used in the legend and filename).
        fold (int): The current fold number for cross-validation.
        threshold (float): A threshold value included in the filename for specificity.

    Saves:
        A PNG file containing the plotted metrics in the specified directory.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(range, train_matric, label=train_label)
    plt.plot(range, val_matric, label=val_label)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Train/validation Loss for Fold {fold + 1}')
    plt.legend()
    plt.savefig(
        os.path.join(
            metric_path,
            f'Train_Val_{str.split(train_label)[-1]}_{fold + 1}_thr_{threshold}.png'
        )
    )

def plot_confusion_matrix(conf_matrix, cm_path, fold, title, threshold):
    """
    Plots and saves a confusion matrix as a heatmap.

    Args:
        conf_matrix (np.ndarray): The confusion matrix to be plotted.
        cm_path (str): The directory path where the plot will be saved.
        fold (int): The current fold number for cross-validation.
        title (str): The title of the plot.
        threshold (float): A threshold value included in the filename for specificity.

    Saves:
        A PNG file containing the confusion matrix heatmap in the specified directory.
    """

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(cm_path, f'cm_fold_{fold + 1}_thr_{threshold}.png'))

def plot_roc_curve(labels, predictions, roc_curve_path, fold, threshold):
    """
    Plots and saves the Receiver Operating Characteristic (ROC) curve.

    Args:
        labels (np.ndarray): True labels of the data (binary or multi-class).
        predictions (np.ndarray): Model predictions (probabilities or scores).
        roc_curve_path (str): Directory path to save the ROC curve plot.
        fold (int): The current fold number for cross-validation.
        threshold (float): Threshold value to include in the filename for specificity.

    Saves:
        A PNG file containing the ROC curve in the specified directory.
    """

    fpr, tpr, _ = roc_curve(labels, predictions)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(
        roc_curve_path,
        f'roc_curve_fold_{fold + 1}_thr_{threshold}.png'
    ))
