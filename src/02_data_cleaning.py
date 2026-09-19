from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "financial_fraud_detection_dataset.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("PHASE 2 - DATA CLEANING")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

print("\nColumn names after standardization:")
print(df.columns.tolist())

# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

empty_rows = df.isna().all(axis=1).sum()

print("\nCompletely empty rows:", empty_rows)

if empty_rows > 0:
    df = df.dropna(how="all").copy()

print("Shape after removing empty rows:", df.shape)

# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_rows = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_rows)

if duplicate_rows > 0:
    df = df.drop_duplicates().copy()

print("Shape after duplicate removal:", df.shape)

duplicate_transaction_ids = (
    df["Transaction_ID"].duplicated().sum()
)

print("Duplicate Transaction_ID values:", duplicate_transaction_ids)

# ============================================================
# DATE CONVERSION
# ============================================================

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

invalid_dates = df["Transaction_Date"].isna().sum()

print("\nInvalid transaction dates:", invalid_dates)

# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

missing_summary = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isna().sum().values,
    "Missing_Percentage": (
        df.isna().sum().values / len(df) * 100
    )
})

print("\nMissing Value Summary:")
print(missing_summary.to_string(index=False))

# ============================================================
# CLEAN CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Suspicious_Keyword"
]

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

print("\nCategorical columns cleaned.")

# ============================================================
# NUMERICAL VALIDATION
# ============================================================

numeric_columns = [
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days"
]

print("\nNumerical column data types:")

for column in numeric_columns:
    print(
        f"{column}: "
        f"{df[column].dtype}"
    )

# ============================================================
# NEGATIVE VALUE CHECK
# ============================================================

for column in numeric_columns:
    negative_count = (df[column] < 0).sum()

    print(
        f"{column} - negative values: "
        f"{negative_count}"
    )

# ============================================================
# TARGET VARIABLE VALIDATION
# ============================================================

print("\nFraudulent unique values:")
print(df["Fraudulent"].unique())

invalid_target_values = (
    ~df["Fraudulent"].isin([0, 1])
).sum()

print(
    "Invalid Fraudulent values:",
    invalid_target_values
)

if invalid_target_values > 0:
    raise ValueError(
        "Fraudulent contains values other than 0 and 1."
    )

# ============================================================
# BINARY VARIABLE VALIDATION
# ============================================================

print("\nIs_International unique values:")
print(df["Is_International"].unique())

invalid_international = (
    ~df["Is_International"].isin([0, 1])
).sum()

print(
    "Invalid Is_International values:",
    invalid_international
)

if invalid_international > 0:
    raise ValueError(
        "Is_International contains values other than 0 and 1."
    )

# ============================================================
# LEAKAGE ASSESSMENT
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE ASSESSMENT")
print("=" * 70)

print("\nSuspicious_Keyword distribution:")
print(
    pd.crosstab(
        df["Suspicious_Keyword"],
        df["Fraudulent"],
        margins=True
    )
)

keyword_fraud_rate = (
    df.groupby("Suspicious_Keyword")["Fraudulent"]
    .agg(
        Transaction_Count="count",
        Fraud_Count="sum",
        Fraud_Rate="mean"
    )
    .reset_index()
)

keyword_fraud_rate["Fraud_Rate"] *= 100

print("\nFraud Rate by Suspicious_Keyword:")
print(
    keyword_fraud_rate.to_string(index=False)
)

international_fraud_rate = (
    df.groupby("Is_International")["Fraudulent"]
    .agg(
        Transaction_Count="count",
        Fraud_Count="sum",
        Fraud_Rate="mean"
    )
    .reset_index()
)

international_fraud_rate["Fraud_Rate"] *= 100

print("\nFraud Rate by International Transaction:")
print(
    international_fraud_rate.to_string(index=False)
)

# ============================================================
# FRAUD RATE BY CATEGORICAL FEATURES
# ============================================================

categorical_analysis_columns = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location"
]

for column in categorical_analysis_columns:

    summary = (
        df.groupby(column)["Fraudulent"]
        .agg(
            Transaction_Count="count",
            Fraud_Count="sum",
            Fraud_Rate="mean"
        )
        .reset_index()
    )

    summary["Fraud_Rate"] *= 100

    print(f"\nFraud Rate by {column}:")
    print(summary.to_string(index=False))

    # ============================================================
# TIME-BASED FEATURES
# ============================================================

df["Transaction_Hour"] = (
    df["Transaction_Date"].dt.hour
)

df["Transaction_Day"] = (
    df["Transaction_Date"].dt.day
)

df["Transaction_Month"] = (
    df["Transaction_Date"].dt.month
)

df["Day_of_Week"] = (
    df["Transaction_Date"].dt.day_name()
)

df["Is_Weekend"] = (
    df["Transaction_Date"].dt.dayofweek >= 5
).astype(int)

print("\nTime-based features created.")

# ============================================================
# TRANSACTION BEHAVIOR FEATURES
# ============================================================

df["Amount_vs_Average_Spend"] = (
    df["Transaction_Amount"]
    / df["Average_Spend"].replace(0, np.nan)
)

df["Amount_vs_Average_Spend"] = (
    df["Amount_vs_Average_Spend"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

df["Transactions_per_Account_Age"] = (
    df["Previous_Transactions"]
    / df["Account_Age_Days"].replace(0, np.nan)
)

df["Transactions_per_Account_Age"] = (
    df["Transactions_per_Account_Age"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

print("\nBehavioral features created.")

# ============================================================
# BEHAVIORAL FEATURE VALIDATION
# ============================================================

print("\nBehavioral Feature Statistics:")

print("\nAmount_vs_Average_Spend:")
print(
    df["Amount_vs_Average_Spend"].describe()
)

print("\nTransactions_per_Account_Age:")
print(
    df["Transactions_per_Account_Age"].describe()
)

# ============================================================
# ZERO TRANSACTION AMOUNT CHECK
# ============================================================

zero_amount_count = (
    df["Transaction_Amount"] == 0
).sum()

zero_amount_fraud = (
    df.loc[
        df["Transaction_Amount"] == 0,
        "Fraudulent"
    ]
    .value_counts()
)

print("\n" + "=" * 70)
print("ZERO TRANSACTION AMOUNT ANALYSIS")
print("=" * 70)

print(
    "\nZero transaction amount records:",
    zero_amount_count
)

print("\nFraud distribution among zero-amount transactions:")
print(zero_amount_fraud)

# ============================================================
# OUTLIER CHECK
# ============================================================

print("\nTransaction Amount Statistics:")
print(
    df["Transaction_Amount"].describe()
)

print("\nAverage Spend Statistics:")
print(
    df["Average_Spend"].describe()
)

# ============================================================
# CLEANING SUMMARY
# ============================================================

cleaning_summary = pd.DataFrame({
    "Metric": [
        "Original Rows",
        "Final Rows",
        "Original Columns",
        "Final Columns",
        "Duplicate Rows Removed",
        "Invalid Dates",
        "Invalid Fraudulent Values",
        "Invalid International Values",
        "Missing Values Remaining"
    ],
    "Value": [
        5000,
        len(df),
        14,
        len(df.columns),
        duplicate_rows,
        invalid_dates,
        invalid_target_values,
        invalid_international,
        df.isna().sum().sum()
    ]
})

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(
    cleaning_summary.to_string(index=False)
)

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

OUTPUT_PATH = (
    PROCESSED_DIR
    / "transactions_cleaned.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nCleaned dataset saved to:")
print(OUTPUT_PATH)

REPORT_PATH = (
    REPORTS_DIR
    / "cleaning_summary.csv"
)

cleaning_summary.to_csv(
    REPORT_PATH,
    index=False
)

print("\nCleaning summary saved to:")
print(REPORT_PATH)

print("\n" + "=" * 70)
print("PHASE 2 COMPLETED")
print("=" * 70)