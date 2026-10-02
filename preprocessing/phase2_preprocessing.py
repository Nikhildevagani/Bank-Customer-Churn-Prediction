import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "European_Bank.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 2 - DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Shape:", df.shape)


# ============================================================
# 2. DATA QUALITY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows before cleaning:", df.duplicated().sum())


# ============================================================
# 3. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = df.dropna(how="all")

print("\nShape after removing completely empty rows:", df.shape)


# ============================================================
# 4. REMOVE DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    print(f"\nRemoving {duplicate_count} duplicate rows...")
    df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ============================================================
# 5. HANDLE INCOMPLETE RECORDS
# ============================================================

print("\nMissing values after removing empty rows:")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])


# Since this dataset contains only a few incomplete records,
# remove records that cannot be used for supervised learning.

df = df.dropna()

print("\nShape after removing incomplete records:", df.shape)


# ============================================================
# 6. REMOVE NON-PREDICTIVE COLUMNS
# ============================================================

columns_to_remove = [
    "CustomerId",
    "Surname",
    "Year"
]

existing_columns = [
    col for col in columns_to_remove
    if col in df.columns
]

df = df.drop(columns=existing_columns)

print("\nRemoved columns:", existing_columns)

print("\nRemaining columns:")
print(df.columns.tolist())


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# Feature 1: Balance to Salary Ratio
# ------------------------------------------------------------

df["Balance_to_Salary"] = (
    df["Balance"] /
    (df["EstimatedSalary"] + 1)
)


# ------------------------------------------------------------
# Feature 2: Product Density
# ------------------------------------------------------------

df["Product_Density"] = (
    df["NumOfProducts"] /
    (df["Tenure"] + 1)
)


# ------------------------------------------------------------
# Feature 3: Engagement × Product Interaction
# ------------------------------------------------------------

df["Engagement_Product_Interaction"] = (
    df["IsActiveMember"] *
    df["NumOfProducts"]
)


# ------------------------------------------------------------
# Feature 4: Age × Tenure Interaction
# ------------------------------------------------------------

df["Age_Tenure_Interaction"] = (
    df["Age"] *
    df["Tenure"]
)


print("\nEngineered features created:")

engineered_features = [
    "Balance_to_Salary",
    "Product_Density",
    "Engagement_Product_Interaction",
    "Age_Tenure_Interaction"
]

for feature in engineered_features:
    print("✓", feature)


# ============================================================
# 8. CHECK ENGINEERED FEATURES
# ============================================================

print("\nEngineered feature summary:")

print(
    df[engineered_features]
    .describe()
    .round(3)
)


# ============================================================
# 9. DEFINE TARGET
# ============================================================

TARGET = "Exited"

X = df.drop(columns=[TARGET])

y = df[TARGET]


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

print("\nTarget percentages:")

print(
    (y.value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 10. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = [
    "Geography",
    "Gender"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 11. STRATIFIED TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "\nTraining churn rate:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Testing churn rate:",
    round(y_test.mean() * 100, 2),
    "%"
)


# ============================================================
# 12. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 13. FIT PREPROCESSING ON TRAINING DATA ONLY
# ============================================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nPreprocessing completed.")

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 14. GET PROCESSED FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print("\nNumber of final features:", len(feature_names))

print("\nFinal features:")

for feature in feature_names:
    print("✓", feature)


# ============================================================
# 15. SAVE PROCESSED DATA
# ============================================================

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


X_train_processed_df.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)

X_test_processed_df.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# ============================================================
# 16. SAVE PREPROCESSING PIPELINE
# ============================================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)


# Save feature names for future dashboard/explainability use.

joblib.dump(
    feature_names,
    MODEL_DIR / "feature_names.pkl"
)


# ============================================================
# 17. SAVE CLEAN FEATURE-ENGINEERED DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "clean_feature_engineered_data.csv",
    index=False
)


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("\nFinal cleaned dataset:", df.shape)

print(
    "Training data:",
    X_train_processed_df.shape
)

print(
    "Testing data:",
    X_test_processed_df.shape
)

print("\nMissing values in final dataset:")
print(df.isnull().sum().sum())

print("\nDuplicate rows in final dataset:")
print(df.duplicated().sum())


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Outputs created:

data/processed/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_test.csv
    └── clean_feature_engineered_data.csv

models/
    ├── preprocessor.pkl
    └── feature_names.pkl

Next Phase:
Model Development and Comparative Evaluation
""")