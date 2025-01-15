# AI Plagiarism Detection

An AI-powered plagiarism detection tool designed to analyze text similarity and detect potential cases of plagiarism using advanced natural language processing (NLP) techniques.

## Table of Contents
- About the Project
- Technologies Used
- Installation
- Running the project

## About the Project
The AI Plagiarism Detection project aims to provide an efficient and accurate way to check for plagiarism in text-based documents by leveraging AI and NLP algorithms. The project is using Huggingface's BertForSequenceClassification model for similarity prediction. The tool identifies textual similarities, paraphrased content, and borrowed ideas while respecting the boundaries of fair use. The project consists of 5 pipelines:
- Data Ingestion
- Data Validation
- Data Transformation
- Model Trainer
- Model Evaluation

### Data Ingestion
Data ingestion pipeline is responsible for orchestrating the data ingestion steps, including downloading the dataset and extracting it for further use in the training pipeline. Only the file extraction part is implemented, the data download process isn't implemented because the download link wasn't provided and I already had a dataset locally.

### Data Validation
Data validation pipeline is responsible for orchestrating the data validation steps, ensuring that the dataset adheres to the specified schema and integrity checks before further processing in the training pipeline.

### Data Transformation
A pipeline to orchestrate the data transformation and preparation process. It groups the coding question, candidate code, and AI-generated code as one training sample, and the provided similarity score represents the coresponding label. The dataset is then separated into training and test sets.

### Model Trainer
The pipeline is responsible for orchestrating the steps involved in training a machine learning model, including data loading, model training, evaluation, and saving the trained model.

### Model Evaluation
Model evaluation pipeline orchestrates the model evaluation process, including configuration setup and invoking the evaluation process for trained models.

## Technologies Used
- Python: Core programming language.
- NLP Libraries: Transformers (Hugging Face) for deep learning models like BERT.
- Machine Learning: Tools like scikit-learn for traditional similarity measures.
- Data Handling: pandas, numpy.
- Visualization: matplotlib or seaborn for similarity heatmaps.

## Installation

### Create a Virtual Environment
```bash
virtualenv env
source env/bin/activate
```

---

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
1. Execute the main script:

```bash
python main.py
```

2. Upon running, the project generates a structured output in the artifact folder, containing results from the pipelines.

---