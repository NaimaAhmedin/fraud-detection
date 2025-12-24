import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ===============================
# Load processed data
# ===============================
df = pd.read_csv("data/processed/fraud_processed.csv")

# Drop non-numeric columns
df = df.drop(columns=df.select_dtypes(include=["object", "datetime"]).columns)

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# Train Random Forest again
# (SHAP needs the trained model)
# ===============================
rf_pipeline = Pipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_model = rf_pipeline.named_steps["rf"]

# ===============================
# 1️⃣ Feature Importance
# ===============================
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top10 = feat_imp.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=top10)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance_top10.png", dpi=300, bbox_inches="tight")
plt.close()



# ===============================
# 2️⃣ SHAP Analysis
# ===============================
explainer = shap.TreeExplainer(rf_model)

X_test_sample = X_test.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Global SHAP importance# SHAP bar plot
shap.summary_plot(
    shap_values[1],
    X_test_sample,
    plot_type="bar",
    show=False
)
plt.savefig("reports/figures/shap_summary_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# SHAP beeswarm plot
shap.summary_plot(
    shap_values[1],
    X_test_sample,
    show=False
)
plt.savefig("reports/figures/shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()


# ===============================
# 3️⃣ SHAP Force Plots
# ===============================
y_pred = rf_pipeline.predict(X_test)

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = y_pred

# Select examples
tp = results[(results.actual == 1) & (results.predicted == 1)].iloc[0]
fp = results[(results.actual == 0) & (results.predicted == 1)].iloc[0]
fn = results[(results.actual == 1) & (results.predicted == 0)].iloc[0]

shap.initjs()

# True Positive
shap.save_html(
    "reports/figures/force_true_positive.html",
    shap.force_plot(
        explainer.expected_value[1],
        explainer.shap_values(tp.drop(["actual", "predicted"]))[1],
        tp.drop(["actual", "predicted"])
    )
)


# False Positive
shap.save_html(
    "reports/figures/force_true_positive.html",
    shap.force_plot(
        explainer.expected_value[1],
        explainer.shap_values(fp.drop(["actual", "predicted"]))[1],
        fp.drop(["actual", "predicted"])
    )
)


# False Negative
shap.save_html(
    "reports/figures/force_true_positive.html",
    shap.force_plot(
        explainer.expected_value[1],
        explainer.shap_values(fn.drop(["actual", "predicted"]))[1],
        fn.drop(["actual", "predicted"])
    )
)

