# ==============================================================
# PHASE 3.2 - MODEL EVALUATION & VISUALIZATION
# Bank Customer Churn Prediction
# ==============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# 2. PHASE 3.2 START
# ==============================================================

print("=" * 70)
print("PHASE 3.2 - MODEL EVALUATION & VISUALIZATION")
print("=" * 70)


# ==============================================================
# 3. LOAD TEST DATA
# ==============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)


X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze()


print(f"\nX_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n✓ Test data loaded successfully")


# ==============================================================
# 4. LOAD TRAINED MODELS
# ==============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)


logistic_model = joblib.load(
    MODELS_DIR / "logistic_regression.pkl"
)

random_forest_model = joblib.load(
    MODELS_DIR / "random_forest.pkl"
)

xgb_model = joblib.load(
    MODELS_DIR / "xgboost.pkl"
)


print("\n✓ Logistic Regression loaded")
print("✓ Random Forest loaded")
print("✓ XGBoost loaded")


models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgb_model
}


# ==============================================================
# 5. GENERATE PREDICTIONS
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING MODEL PREDICTIONS")
print("=" * 70)


predictions = {}
probabilities = {}


for model_name, model in models.items():

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    predictions[model_name] = y_pred

    probabilities[model_name] = y_prob

    print(f"✓ Predictions generated: {model_name}")


# ==============================================================
# 6. CONFUSION MATRICES
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING CONFUSION MATRICES")
print("=" * 70)


for model_name in models.keys():

    y_pred = predictions[model_name]

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print(f"\n{model_name}")
    print(cm)


    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Churned",
            "Churned"
        ]
    )


    display.plot(
        ax=ax,
        values_format="d"
    )


    ax.set_title(
        f"Confusion Matrix - {model_name}",
        fontsize=14,
        fontweight="bold"
    )


    plt.tight_layout()


    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


    save_path = (
        RESULTS_DIR /
        f"confusion_matrix_{filename}.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"✓ Saved: {save_path.name}"
    )


# ==============================================================
# 7. ROC CURVE COMPARISON
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING ROC CURVE")
print("=" * 70)


plt.figure(
    figsize=(9, 7)
)


for model_name in models.keys():

    y_prob = probabilities[model_name]


    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )


# Random classifier reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate",
    fontsize=11
)

plt.ylabel(
    "True Positive Rate",
    fontsize=11
)


plt.title(
    "ROC Curve Comparison",
    fontsize=15,
    fontweight="bold"
)


plt.legend(
    loc="lower right"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


roc_path = (
    RESULTS_DIR /
    "roc_curve_comparison.png"
)


plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"✓ ROC curve saved: {roc_path.name}"
)


# ==============================================================
# 8. MODEL PERFORMANCE COMPARISON
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING MODEL PERFORMANCE COMPARISON")
print("=" * 70)


comparison_path = (
    RESULTS_DIR /
    "model_comparison.csv"
)


comparison_df = pd.read_csv(
    comparison_path
)


print("\nModel Comparison:")
print(
    comparison_df.to_string(
        index=False
    )
)


# --------------------------------------------------------------
# Convert metrics to percentages
# --------------------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]


x = np.arange(
    len(comparison_df["Model"])
)

width = 0.18


fig, ax = plt.subplots(
    figsize=(12, 7)
)


for i, metric in enumerate(metrics):

    values = (
        comparison_df[metric] * 100
    )


    ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=metric
    )


ax.set_xlabel(
    "Model",
    fontsize=11
)

ax.set_ylabel(
    "Score (%)",
    fontsize=11
)


ax.set_title(
    "Machine Learning Model Performance Comparison",
    fontsize=15,
    fontweight="bold"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    comparison_df["Model"]
)


ax.set_ylim(
    0,
    100
)


ax.legend()


ax.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


performance_path = (
    RESULTS_DIR /
    "model_performance_comparison.png"
)


plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"✓ Performance comparison saved: "
    f"{performance_path.name}"
)


# ==============================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ==============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


feature_names_path = (
    MODELS_DIR /
    "feature_names.pkl"
)


feature_names = joblib.load(
    feature_names_path
)


rf_importance = (
    random_forest_model.feature_importances_
)


rf_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": rf_importance

})


rf_importance_df = (
    rf_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    "\nTop 10 Random Forest Features:"
)

print(
    rf_importance_df.head(10).to_string(
        index=False
    )
)


# Top 10 only for visualization

top_rf = (
    rf_importance_df
    .head(10)
    .sort_values(
        by="Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_rf["Feature"],
    top_rf["Importance"]
)


plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)


plt.title(
    "Top 10 Random Forest Feature Importance",
    fontsize=15,
    fontweight="bold"
)


plt.tight_layout()


rf_path = (
    RESULTS_DIR /
    "random_forest_feature_importance.png"
)


plt.savefig(
    rf_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\n✓ Random Forest feature importance saved: "
    f"{rf_path.name}"
)


# ==============================================================
# 10. XGBOOST FEATURE IMPORTANCE
# ==============================================================

print("\n" + "=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)


xgb_importance = (
    xgb_model.feature_importances_
)


xgb_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": xgb_importance

})


xgb_importance_df = (
    xgb_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    "\nTop 10 XGBoost Features:"
)


print(
    xgb_importance_df.head(10).to_string(
        index=False
    )
)


# Top 10 visualization

top_xgb = (
    xgb_importance_df
    .head(10)
    .sort_values(
        by="Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_xgb["Feature"],
    top_xgb["Importance"]
)


plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)


plt.title(
    "Top 10 XGBoost Feature Importance",
    fontsize=15,
    fontweight="bold"
)


plt.tight_layout()


xgb_path = (
    RESULTS_DIR /
    "xgboost_feature_importance.png"
)


plt.savefig(
    xgb_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\n✓ XGBoost feature importance saved: "
    f"{xgb_path.name}"
)


# ==============================================================
# 11. SAVE FEATURE IMPORTANCE DATA
# ==============================================================

rf_csv_path = (
    RESULTS_DIR /
    "random_forest_feature_importance.csv"
)


xgb_csv_path = (
    RESULTS_DIR /
    "xgboost_feature_importance.csv"
)


rf_importance_df.to_csv(
    rf_csv_path,
    index=False
)


xgb_importance_df.to_csv(
    xgb_csv_path,
    index=False
)


print(
    "\n✓ Random Forest feature importance CSV saved"
)

print(
    "✓ XGBoost feature importance CSV saved"
)


# ==============================================================
# 12. FINAL OUTPUT SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 3.2 MODEL EVALUATION COMPLETED")
print("=" * 70)


print("""
Generated visualizations:

results/
│
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_random_forest.png
├── confusion_matrix_xgboost.png
│
├── roc_curve_comparison.png
├── model_performance_comparison.png
│
├── random_forest_feature_importance.png
├── xgboost_feature_importance.png
│
├── random_forest_feature_importance.csv
└── xgboost_feature_importance.csv
""")


print("=" * 70)
print("NEXT STEP:")
print("Analyze the visualizations and perform")
print("threshold tuning / final model selection.")
print("=" * 70)