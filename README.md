# Fraud Detection Project – End-to-End Machine Learning Pipeline

## Overview

This project implements an end-to-end fraud detection system using real-world transaction data.  
The goal is to identify fraudulent transactions while addressing challenges such as class imbalance, feature engineering, geolocation enrichment, and model explainability.

The project follows best practices in:

- Data analysis and preprocessing
- Machine learning modeling
- Model evaluation on imbalanced data
- Explainable AI using SHAP
- Professional project organization and documentation

---

## Business Context

Financial institutions face significant losses due to fraudulent transactions.  
Early detection of fraud helps reduce losses, improve customer trust, and optimize operational costs.

This project focuses on:

- Detecting fraudulent transactions accurately
- Understanding _why_ a transaction is flagged
- Providing actionable insights for business decision-making

---

## Project Structure

fraud-detection/
│
├── .vscode/ # Editor configuration
├── .github/workflows/ # CI pipeline
│ └── unittests.yml
│
├── data/ # Ignored from Git
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned & engineered data
│
├── notebooks/ # Exploration & experimentation
│
├── src/ # Production-ready pipeline code
│
├── models/ # Trained models
│
├── tests/ # Unit tests
│
├── scripts/ # Helper scripts
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Task 1 – Data Analysis & Preprocessing

### Datasets Used

- **Fraud_Data.csv** – E-commerce fraud data
- **IpAddress_to_Country.csv** – IP-to-country mapping
- **creditcard.csv** – Credit card transaction dataset

### Key Steps Performed

- Missing value handling and duplicate removal
- Data type correction (timestamps, numeric fields)
- Exploratory Data Analysis (EDA)
- IP address conversion and country-level enrichment
- Feature engineering:
  - `time_since_signup`
  - transaction velocity features
  - hour of day and day of week
- Scaling numerical features
- Encoding categorical variables
- Class imbalance analysis and mitigation using SMOTE

Processed outputs are stored in:

data/processed/
├── fraud_data_cleaned.csv
└── fraud_processed.csv

---

## Task 2 – Model Building & Training

### Models Implemented

- **Logistic Regression** (baseline, interpretable)
- **Random Forest Classifier** (ensemble model)

### Techniques Used

- Stratified train-test split
- Class imbalance handling during training
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix

### Model Selection

The Random Forest model was selected as the final model due to:

- Higher recall for fraud cases
- Better balance between precision and recall
- Robust performance on imbalanced data

Trained models are saved in:

models/
└── random_forest.pkl

---

## Task 3 – Model Explainability (SHAP)

To ensure transparency and trust, SHAP was used to explain model predictions.

### Explainability Outputs

- Built-in Random Forest feature importance
- SHAP summary plot (global importance)
- SHAP force plots for:
  - True Positive (correct fraud detection)
  - False Positive (legitimate flagged)
  - False Negative (missed fraud)

### Key Fraud Drivers Identified

- Transaction amount
- Time since signup
- Transaction velocity
- Device and behavioral patterns

---

## Business Recommendations

Based on SHAP insights:

1. Transactions shortly after signup should receive additional verification.
2. High-frequency transactions in short time windows should trigger alerts.
3. Country and device-based risk profiling should be integrated into fraud rules.

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
Run training:

python src/explain_model.py

Run explainability:

python src/explain_model.py
```
