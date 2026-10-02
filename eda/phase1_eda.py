import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../data/European_Bank.csv")

print("=" * 60)
print("BANK CUSTOMER CHURN - PHASE 1 EDA")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# 2. BASIC DATASET INFORMATION
# ==========================================

print("\n\nColumn Names:")
print(df.columns.tolist())

print("\n\nDataset Information:")
print(df.info())

print("\n\nFirst 5 Records:")
print(df.head())

print("\n\nStatistical Summary:")
print(df.describe())


# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n\nMissing Values:")
print(df.isnull().sum())

total_missing = df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)


# ==========================================
# 4. DUPLICATE RECORDS
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)


# ==========================================
# 5. TARGET VARIABLE
# ==========================================

print("\n\nChurn Distribution:")
print(df["Exited"].value_counts())

churn_percentage = df["Exited"].value_counts(normalize=True) * 100

print("\nChurn Percentage:")
print(churn_percentage)


# ==========================================
# 6. CHURN DISTRIBUTION VISUALIZATION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Exited"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# 7. GEOGRAPHY ANALYSIS
# ==========================================

print("\n\nChurn by Geography:")

geo_churn = df.groupby("Geography")["Exited"].agg(
    ["count", "sum", "mean"]
)

geo_churn["Churn Rate (%)"] = geo_churn["mean"] * 100

print(
    geo_churn[
        ["count", "sum", "Churn Rate (%)"]
    ].round(2)
)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Geography",
    hue="Exited"
)

plt.title("Customer Churn by Geography")
plt.xlabel("Geography")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# 8. GENDER ANALYSIS
# ==========================================

print("\n\nChurn by Gender:")

gender_churn = df.groupby("Gender")["Exited"].agg(
    ["count", "sum", "mean"]
)

gender_churn["Churn Rate (%)"] = (
    gender_churn["mean"] * 100
)

print(
    gender_churn[
        ["count", "sum", "Churn Rate (%)"]
    ].round(2)
)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Gender",
    hue="Exited"
)

plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# 9. AGE ANALYSIS
# ==========================================

print("\n\nAge Statistics by Churn:")
print(
    df.groupby("Exited")["Age"].describe().round(2)
)


plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Exited",
    kde=True,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Age Distribution by Churn")
plt.xlabel("Age")
plt.ylabel("Density")

plt.tight_layout()
plt.show()


# ==========================================
# 10. CREDIT SCORE ANALYSIS
# ==========================================

print("\n\nCredit Score Statistics by Churn:")
print(
    df.groupby("Exited")["CreditScore"]
    .describe()
    .round(2)
)


# ==========================================
# 11. BALANCE ANALYSIS
# ==========================================

print("\n\nBalance Statistics by Churn:")
print(
    df.groupby("Exited")["Balance"]
    .describe()
    .round(2)
)


# ==========================================
# 12. NUMBER OF PRODUCTS
# ==========================================

print("\n\nChurn by Number of Products:")

product_churn = df.groupby("NumOfProducts")["Exited"].agg(
    ["count", "sum", "mean"]
)

product_churn["Churn Rate (%)"] = (
    product_churn["mean"] * 100
)

print(
    product_churn[
        ["count", "sum", "Churn Rate (%)"]
    ].round(2)
)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="NumOfProducts",
    hue="Exited"
)

plt.title("Customer Churn by Number of Products")
plt.xlabel("Number of Products")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# 13. ACTIVE MEMBER ANALYSIS
# ==========================================

print("\n\nChurn by Active Membership:")

active_churn = df.groupby("IsActiveMember")["Exited"].agg(
    ["count", "sum", "mean"]
)

active_churn["Churn Rate (%)"] = (
    active_churn["mean"] * 100
)

print(
    active_churn[
        ["count", "sum", "Churn Rate (%)"]
    ].round(2)
)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="IsActiveMember",
    hue="Exited"
)

plt.title("Customer Churn by Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# 14. CREDIT CARD ANALYSIS
# ==========================================

print("\n\nChurn by Credit Card Ownership:")

card_churn = df.groupby("HasCrCard")["Exited"].agg(
    ["count", "sum", "mean"]
)

card_churn["Churn Rate (%)"] = (
    card_churn["mean"] * 100
)

print(
    card_churn[
        ["count", "sum", "Churn Rate (%)"]
    ].round(2)
)


# ==========================================
# 15. TENURE ANALYSIS
# ==========================================

print("\n\nChurn by Tenure:")

tenure_churn = df.groupby("Tenure")["Exited"].mean() * 100

print(tenure_churn.round(2))


plt.figure(figsize=(10, 5))

tenure_churn.plot(
    kind="bar"
)

plt.title("Churn Rate by Tenure")
plt.xlabel("Tenure")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ==========================================
# 16. CORRELATION MATRIX
# ==========================================

numeric_columns = df.select_dtypes(
    include=np.number
)

correlation = numeric_columns.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ==========================================
# 17. CORRELATION WITH TARGET
# ==========================================

print("\n\nCorrelation with Churn (Exited):")

target_correlation = (
    correlation["Exited"]
    .sort_values(ascending=False)
)

print(target_correlation.round(3))


# ==========================================
# 18. CONSTANT COLUMN CHECK
# ==========================================

print("\n\nUnique Values per Column:")

print(df.nunique())

print("\nColumns with only one unique value:")

constant_columns = [
    col for col in df.columns
    if df[col].nunique() == 1
]

print(constant_columns)


# ==========================================
# 19. PHASE 1 CONCLUSION
# ==========================================

print("\n")
print("=" * 60)
print("PHASE 1 EDA COMPLETED")
print("=" * 60)

print("""
Key observations:

1. Dataset contains 10,000 customers.
2. No missing values were identified.
3. No duplicate records were identified.
4. Approximately 20.37% of customers have churned.
5. CustomerId and Surname are identifiers and should be removed.
6. Year is constant and should be removed.
7. Geography, age, activity status and product count show
   noticeable relationships with churn.
8. Class imbalance means accuracy alone should not be used
   for model selection.
9. Precision, Recall, F1-score and ROC-AUC will be important
   during model evaluation.

Next Phase:
Data Preprocessing and Feature Engineering.
""")