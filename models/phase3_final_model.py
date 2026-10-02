# ==============================================================
# PHASE 3.4 - FINAL MODEL SELECTION
# Bank Customer Churn Prediction
# ==============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
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
# 2. PHASE START
# ==============================================================

print("=" * 70)
print("PHASE 3.4 - FINAL MODEL SELECTION")
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


# ==============================================================
# 5. DEFINE FINAL THRESHOLDS
# ==============================================================

# Thresholds obtained from Phase 3.3 threshold analysis

LOGISTIC_THRESHOLD = 0.50

RANDOM_FOREST_THRESHOLD = 0.60

XGBOOST_THRESHOLD = 0.30


print("\n" + "=" * 70)
print("FINAL CLASSIFICATION THRESHOLDS")
print("=" * 70)

print(
    f"\nLogistic Regression : {LOGISTIC_THRESHOLD:.2f}"
)

print(
    f"Random Forest      : {RANDOM_FOREST_THRESHOLD:.2f}"
)

print(
    f"XGBoost             : {XGBOOST_THRESHOLD:.2f}"
)


# ==============================================================
# 6. MODEL EVALUATION FUNCTION
# ==============================================================

def evaluate_model(
    model,
    model_name,
    threshold
):

    print("\n" + "-" * 70)
    print(f"EVALUATING: {model_name}")
    print("-" * 70)


    # ----------------------------------------------------------
    # Prediction probabilities
    # ----------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # ----------------------------------------------------------
    # Apply classification threshold
    # ----------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # ----------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    # ----------------------------------------------------------
    # Display metrics
    # ----------------------------------------------------------

    print("\nPerformance:")

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not Churned",
                "Churned"
            ],
            zero_division=0
        )
    )


    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }


# ==============================================================
# 7. EVALUATE ALL MODELS
# ==============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


final_results = []


# Logistic Regression

final_results.append(
    evaluate_model(
        logistic_model,
        "Logistic Regression",
        LOGISTIC_THRESHOLD
    )
)


# Random Forest

final_results.append(
    evaluate_model(
        random_forest_model,
        "Random Forest",
        RANDOM_FOREST_THRESHOLD
    )
)


# XGBoost

final_results.append(
    evaluate_model(
        xgb_model,
        "XGBoost",
        XGBOOST_THRESHOLD
    )
)


# ==============================================================
# 8. CREATE FINAL RESULTS TABLE
# ==============================================================

final_results_df = pd.DataFrame(
    final_results
)


print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)


display_columns = [
    "Model",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]


print(
    final_results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==============================================================
# 9. SELECT FINAL MODEL
# ==============================================================

# Selection criterion:
# Highest F1-score after threshold tuning.

best_index = final_results_df[
    "F1_Score"
].idxmax()


best_model_name = final_results_df.loc[
    best_index,
    "Model"
]


best_threshold = final_results_df.loc[
    best_index,
    "Threshold"
]


best_f1 = final_results_df.loc[
    best_index,
    "F1_Score"
]


best_accuracy = final_results_df.loc[
    best_index,
    "Accuracy"
]


best_precision = final_results_df.loc[
    best_index,
    "Precision"
]


best_recall = final_results_df.loc[
    best_index,
    "Recall"
]


best_roc_auc = final_results_df.loc[
    best_index,
    "ROC_AUC"
]


# ==============================================================
# 10. DISPLAY SELECTED MODEL
# ==============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)


print(
    f"\nSelected Model : {best_model_name}"
)

print(
    f"Threshold      : {best_threshold:.2f}"
)

print(
    f"Accuracy       : {best_accuracy:.4f}"
)

print(
    f"Precision      : {best_precision:.4f}"
)

print(
    f"Recall         : {best_recall:.4f}"
)

print(
    f"F1 Score       : {best_f1:.4f}"
)

print(
    f"ROC-AUC        : {best_roc_auc:.4f}"
)


# ==============================================================
# 11. SAVE FINAL MODEL
# ==============================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)


if best_model_name == "Logistic Regression":

    final_model = logistic_model

elif best_model_name == "Random Forest":

    final_model = random_forest_model

elif best_model_name == "XGBoost":

    final_model = xgb_model

else:

    raise ValueError(
        "Unknown model selected."
    )


final_model_path = (
    MODELS_DIR /
    "final_churn_model.pkl"
)


joblib.dump(
    final_model,
    final_model_path
)


print(
    f"\n✓ Final model saved:"
    f"\n  {final_model_path}"
)


# ==============================================================
# 12. SAVE FINAL THRESHOLD
# ==============================================================

threshold_path = (
    MODELS_DIR /
    "final_threshold.pkl"
)


joblib.dump(
    best_threshold,
    threshold_path
)


print(
    f"\n✓ Final threshold saved:"
    f"\n  {threshold_path}"
)


# ==============================================================
# 13. SAVE FINAL MODEL INFORMATION
# ==============================================================

final_model_info = {

    "Model": best_model_name,

    "Threshold": best_threshold,

    "Accuracy": best_accuracy,

    "Precision": best_precision,

    "Recall": best_recall,

    "F1_Score": best_f1,

    "ROC_AUC": best_roc_auc
}


final_info_df = pd.DataFrame(
    [final_model_info]
)


final_info_path = (
    RESULTS_DIR /
    "final_model_summary.csv"
)


final_info_df.to_csv(
    final_info_path,
    index=False
)


print(
    f"\n✓ Final model summary saved:"
    f"\n  {final_info_path}"
)


# ==============================================================
# 14. SAVE COMPLETE MODEL COMPARISON
# ==============================================================

comparison_path = (
    RESULTS_DIR /
    "final_model_comparison.csv"
)


final_results_df.to_csv(
    comparison_path,
    index=False
)


print(
    f"\n✓ Complete model comparison saved:"
    f"\n  {comparison_path}"
)


# ==============================================================
# 15. FINAL CONFUSION MATRIX
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL CONFUSION MATRIX")
print("=" * 70)


final_probabilities = final_model.predict_proba(
    X_test
)[:, 1]


final_predictions = (
    final_probabilities >= best_threshold
).astype(int)


final_cm = confusion_matrix(
    y_test,
    final_predictions
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=final_cm,
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
    f"Final Model Confusion Matrix\n"
    f"{best_model_name} (Threshold = {best_threshold:.2f})",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


final_cm_path = (
    RESULTS_DIR /
    "final_model_confusion_matrix.png"
)


plt.savefig(
    final_cm_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\n✓ Final confusion matrix saved:"
    f"\n  {final_cm_path}"
)


# ==============================================================
# 16. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 3.4 FINAL MODEL SELECTION COMPLETED")
print("=" * 70)


print(
    f"""
FINAL MODEL
-----------
Model       : {best_model_name}
Threshold   : {best_threshold:.2f}

Accuracy    : {best_accuracy:.4f}
Precision   : {best_precision:.4f}
Recall      : {best_recall:.4f}
F1 Score    : {best_f1:.4f}
ROC-AUC     : {best_roc_auc:.4f}

FILES CREATED
-------------
models/
├── final_churn_model.pkl
└── final_threshold.pkl

results/
├── final_model_summary.csv
├── final_model_comparison.csv
└── final_model_confusion_matrix.png
"""
)


print("=" * 70)

print(
    "PHASE 3 MODEL DEVELOPMENT & EVALUATION COMPLETED"
)

print("=" * 70)

print(
    "\nNEXT PHASE:"
)

print(
    "Phase 4 - Prediction System / Model Deployment"
)

print("=" * 70)