# ==============================================================
# PHASE 3.3 - THRESHOLD ANALYSIS
# Bank Customer Churn Prediction
# ==============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
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
# 2. PHASE 3.3 START
# ==============================================================

print("=" * 70)
print("PHASE 3.3 - THRESHOLD ANALYSIS")
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


random_forest_model = joblib.load(
    MODELS_DIR / "random_forest.pkl"
)

xgb_model = joblib.load(
    MODELS_DIR / "xgboost.pkl"
)


print("\n✓ Random Forest loaded")
print("✓ XGBoost loaded")


# ==============================================================
# 5. GENERATE PROBABILITIES
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTION PROBABILITIES")
print("=" * 70)


rf_probabilities = random_forest_model.predict_proba(
    X_test
)[:, 1]


xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]


print("\n✓ Random Forest probabilities generated")
print("✓ XGBoost probabilities generated")


# ==============================================================
# 6. DEFINE THRESHOLDS
# ==============================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


print("\n" + "=" * 70)
print("THRESHOLDS TO BE TESTED")
print("=" * 70)

print(thresholds)


# ==============================================================
# 7. THRESHOLD ANALYSIS FUNCTION
# ==============================================================

def analyze_thresholds(
    y_true,
    probabilities,
    model_name
):

    results = []


    print("\n" + "=" * 70)
    print(f"{model_name.upper()} THRESHOLD ANALYSIS")
    print("=" * 70)


    for threshold in thresholds:

        # ------------------------------------------------------
        # Convert probability to class prediction
        # ------------------------------------------------------

        y_pred = (
            probabilities >= threshold
        ).astype(int)


        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )


        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )


        cm = confusion_matrix(
            y_true,
            y_pred
        )


        tn, fp, fn, tp = cm.ravel()


        # ------------------------------------------------------
        # Store results
        # ------------------------------------------------------

        results.append({

            "Model": model_name,

            "Threshold": threshold,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1_Score": f1,

            "True_Negatives": tn,

            "False_Positives": fp,

            "False_Negatives": fn,

            "True_Positives": tp
        })


    results_df = pd.DataFrame(
        results
    )


    # ----------------------------------------------------------
    # Display results
    # ----------------------------------------------------------

    print("\n")

    print(
        results_df[
            [
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    return results_df


# ==============================================================
# 8. RANDOM FOREST THRESHOLD ANALYSIS
# ==============================================================

rf_results = analyze_thresholds(
    y_test,
    rf_probabilities,
    "Random Forest"
)


# ==============================================================
# 9. XGBOOST THRESHOLD ANALYSIS
# ==============================================================

xgb_results = analyze_thresholds(
    y_test,
    xgb_probabilities,
    "XGBoost"
)


# ==============================================================
# 10. SAVE THRESHOLD RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("SAVING THRESHOLD RESULTS")
print("=" * 70)


rf_path = (
    RESULTS_DIR /
    "threshold_analysis_random_forest.csv"
)

xgb_path = (
    RESULTS_DIR /
    "threshold_analysis_xgboost.csv"
)


rf_results.to_csv(
    rf_path,
    index=False
)


xgb_results.to_csv(
    xgb_path,
    index=False
)


print(
    f"\n✓ Random Forest results saved:"
    f"\n  {rf_path}"
)


print(
    f"\n✓ XGBoost results saved:"
    f"\n  {xgb_path}"
)


# ==============================================================
# 11. FIND BEST F1 THRESHOLD
# ==============================================================

print("\n" + "=" * 70)
print("BEST F1-SCORE THRESHOLDS")
print("=" * 70)


best_rf = rf_results.loc[
    rf_results["F1_Score"].idxmax()
]


best_xgb = xgb_results.loc[
    xgb_results["F1_Score"].idxmax()
]


print("\nRandom Forest:")
print(
    f"Best Threshold : {best_rf['Threshold']:.2f}"
)

print(
    f"Accuracy       : {best_rf['Accuracy']:.4f}"
)

print(
    f"Precision      : {best_rf['Precision']:.4f}"
)

print(
    f"Recall         : {best_rf['Recall']:.4f}"
)

print(
    f"F1 Score       : {best_rf['F1_Score']:.4f}"
)


print("\nXGBoost:")
print(
    f"Best Threshold : {best_xgb['Threshold']:.2f}"
)

print(
    f"Accuracy       : {best_xgb['Accuracy']:.4f}"
)

print(
    f"Precision      : {best_xgb['Precision']:.4f}"
)

print(
    f"Recall         : {best_xgb['Recall']:.4f}"
)

print(
    f"F1 Score       : {best_xgb['F1_Score']:.4f}"
)


# ==============================================================
# 12. THRESHOLD PERFORMANCE VISUALIZATION
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING THRESHOLD PERFORMANCE VISUALIZATION")
print("=" * 70)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


# --------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------

ax.plot(
    rf_results["Threshold"],
    rf_results["Precision"],
    marker="o",
    label="Random Forest - Precision"
)


ax.plot(
    rf_results["Threshold"],
    rf_results["Recall"],
    marker="o",
    label="Random Forest - Recall"
)


ax.plot(
    rf_results["Threshold"],
    rf_results["F1_Score"],
    marker="o",
    label="Random Forest - F1"
)


# --------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------

ax.plot(
    xgb_results["Threshold"],
    xgb_results["Precision"],
    marker="s",
    linestyle="--",
    label="XGBoost - Precision"
)


ax.plot(
    xgb_results["Threshold"],
    xgb_results["Recall"],
    marker="s",
    linestyle="--",
    label="XGBoost - Recall"
)


ax.plot(
    xgb_results["Threshold"],
    xgb_results["F1_Score"],
    marker="s",
    linestyle="--",
    label="XGBoost - F1"
)


# --------------------------------------------------------------
# Chart formatting
# --------------------------------------------------------------

ax.set_xlabel(
    "Classification Threshold",
    fontsize=11
)

ax.set_ylabel(
    "Score",
    fontsize=11
)


ax.set_title(
    "Precision, Recall and F1-Score vs Classification Threshold",
    fontsize=14,
    fontweight="bold"
)


ax.set_ylim(
    0,
    1
)


ax.grid(
    alpha=0.3
)


ax.legend(
    loc="best"
)


plt.tight_layout()


threshold_plot_path = (
    RESULTS_DIR /
    "threshold_performance.png"
)


plt.savefig(
    threshold_plot_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\n✓ Threshold performance chart saved:"
    f"\n  {threshold_plot_path}"
)


# ==============================================================
# 13. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 3.3 THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)


print("""
Generated files:

results/
│
├── threshold_analysis_random_forest.csv
├── threshold_analysis_xgboost.csv
└── threshold_performance.png
""")


print("=" * 70)

print(
    "NEXT STEP: Analyze threshold results and "
    "determine the appropriate operating threshold."
)

print("=" * 70)