import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/processed/fraud_processed.csv")

print("Dataset shape:", df.shape)
print("Fraud rate:", df["class"].mean())

# Drop non-numeric columns (VERY IMPORTANT)
non_numeric = df.select_dtypes(include=["object", "datetime"]).columns.tolist()
print("Dropping non-numeric columns:", non_numeric)

df = df.drop(columns=non_numeric)

# Split features & target
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
rf_pipeline.fit(X_train, y_train)

# Predict
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
print("\nRandom Forest Results")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# 🔑 EXPOSE MODEL FOR EXPLAINABILITY
rf_model = rf_pipeline.named_steps["rf"]
