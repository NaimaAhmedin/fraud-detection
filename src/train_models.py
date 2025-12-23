import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# ----------------
# LOAD DATA FIRST
# ----------------
df = pd.read_csv("data/processed/fraud_processed.csv")

print("Dataset shape:", df.shape)
print("Fraud rate:", df["class"].mean())

# ----------------
# 🔥 DROP ALL NON-NUMERIC COLUMNS
# ----------------
non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

if non_numeric_cols:
    print("Dropping non-numeric columns:", non_numeric_cols)
    df = df.drop(columns=non_numeric_cols)

# ----------------
# FEATURES & TARGET
# ----------------
X = df.drop(columns=["class"])
y = df["class"]

# ----------------
# TRAIN / TEST SPLIT
# ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------
# LOGISTIC REGRESSION PIPELINE
# ----------------
log_reg_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

log_reg_pipeline.fit(X_train, y_train)

# ----------------
# EVALUATION
# ----------------
y_pred = log_reg_pipeline.predict(X_test)
y_proba = log_reg_pipeline.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ----------------
# RANDOM FOREST PIPELINE
# ----------------
rf_pipeline = Pipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

# ----------------
# RF EVALUATION
# ----------------
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))


joblib.dump(log_reg_pipeline, "models/logistic_regression.pkl")
joblib.dump(rf_pipeline, "models/random_forest.pkl")

print("Models saved successfully.")
