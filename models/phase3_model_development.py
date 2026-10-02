# ==============================================================
# PHASE 3 - MODEL DEVELOPMENT & COMPARATIVE EVALUATION
# Bank Customer Churn Prediction
# ==============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

# Get the project root directory
# __file__ = .../Bank_Customer_Churn_Risk/models/phase3_model_development.py
# parent.parent = .../Bank_Customer_Churn_Risk

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create required directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 2. PHASE 3 START
# ==============================================================

print("=" * 70)
print("PHASE 3 - MODEL DEVELOPMENT & COMPARATIVE EVALUATION")
print("=" * 70)


# ==============================================================
# 3. DISPLAY PROJECT PATHS
# ==============================================================

print("\n" + "=" * 70)
print("PROJECT PATHS")
print("=" * 70)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Data Folder  : {DATA_DIR}")
print(f"Models Folder: {MODELS_DIR}")
print(f"Results Folder: {RESULTS_DIR}")


# ==============================================================
# 4. CHECK REQUIRED FILES
# ==============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    DATA_DIR / "X_train.csv",
    DATA_DIR / "X_test.csv",
    DATA_DIR / "y_train.csv",
    DATA_DIR / "y_test.csv"
]

for file in required_files:

    if file.exists():
        print(f"✓ Found: {file.name}")
    else:
        print(f"✗ Missing: {file}")

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}\n\n"
            f"Please run Phase 2 preprocessing first."
        )


# ==============================================================
# 5. LOAD PROCESSED DATA
# ==============================================================

print("\n" + "=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)


X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze()


print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")


# ==============================================================
# 6. DATA VALIDATION
# ==============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)


# Missing values
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

print(f"\nMissing values in X_train: {train_missing}")
print(f"Missing values in X_test : {test_missing}")


# Target distribution
print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# Target percentages
print("\nTraining Target Percentage:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)


print("\nTesting Target Percentage:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .round(2)
)


# ==============================================================
# 7. CHECK FEATURE COUNT
# ==============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

print(f"\nNumber of training features: {X_train.shape[1]}")
print(f"Number of testing features : {X_test.shape[1]}")

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "Training and testing feature counts do not match!"
    )

print("\n✓ Training and testing feature counts match")


# ==============================================================
# 8. INITIALIZE MODELS
# ==============================================================

print("\n" + "=" * 70)
print("INITIALIZING MACHINE LEARNING MODELS")
print("=" * 70)


# --------------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# --------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,
    n_jobs=-1
)


models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgb_model
}


print("\n✓ Logistic Regression initialized")
print("✓ Random Forest initialized")
print("✓ XGBoost initialized")


# ==============================================================
# 9. TRAIN AND EVALUATE MODELS
# ==============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING & EVALUATION")
print("=" * 70)


results = []


for model_name, model in models.items():

    print("\n")
    print("-" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("-" * 70)


    # ----------------------------------------------------------
    # Train model
    # ----------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    print("✓ Model training completed")


    # ----------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # ----------------------------------------------------------
    # Evaluation Metrics
    # ----------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )


    # ----------------------------------------------------------
    # Store Results
    # ----------------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc
    })


    # ----------------------------------------------------------
    # Display Metrics
    # ----------------------------------------------------------

    print("\nMODEL PERFORMANCE")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


    # ----------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)


    # ----------------------------------------------------------
    # Classification Report
    # ----------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Churned",
                "Churned"
            ],
            zero_division=0
        )
    )


# ==============================================================
# 10. MODEL COMPARISON
# ==============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


results_df = pd.DataFrame(
    results
)


print("\n")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==============================================================
# 11. SAVE MODEL COMPARISON
# ==============================================================

comparison_path = (
    RESULTS_DIR / "model_comparison.csv"
)


results_df.to_csv(
    comparison_path,
    index=False
)


print(
    f"\n✓ Model comparison saved to:"
    f"\n  {comparison_path}"
)


# ==============================================================
# 12. SAVE TRAINED MODELS
# ==============================================================

print("\n" + "=" * 70)
print("SAVING TRAINED MODELS")
print("=" * 70)


logistic_path = (
    MODELS_DIR / "logistic_regression.pkl"
)

random_forest_path = (
    MODELS_DIR / "random_forest.pkl"
)

xgb_path = (
    MODELS_DIR / "xgboost.pkl"
)


joblib.dump(
    logistic_model,
    logistic_path
)


joblib.dump(
    random_forest_model,
    random_forest_path
)


joblib.dump(
    xgb_model,
    xgb_path
)


print("\n✓ Logistic Regression saved")
print(f"  → {logistic_path}")


print("\n✓ Random Forest saved")
print(f"  → {random_forest_path}")


print("\n✓ XGBoost saved")
print(f"  → {xgb_path}")


# ==============================================================
# 13. FINAL PHASE 3 SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 3 MODEL DEVELOPMENT COMPLETED")
print("=" * 70)


print("""
Generated outputs:

results/
    └── model_comparison.csv

models/
    ├── preprocessor.pkl
    ├── feature_names.pkl
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
""")


print("=" * 70)
print("NEXT STEP:")
print("Confusion Matrix Visualization + ROC Curve +")
print("Feature Importance + Model Comparison")
print("=" * 70)