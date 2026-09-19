from pathlib import Path

import pandas as pd
import numpy as np
import mysql.connector


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "financial_fraud_detection"
MYSQL_USER = "root"

# Keep your existing MySQL password here.
# Do NOT share the password.
MYSQL_PASSWORD = "root"


# ============================================================
# DATABASE CONNECTION
# ============================================================

print("=" * 70)
print("PHASE 5 - FEATURE ENGINEERING")
print("=" * 70)

try:

    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )

    print("\nMySQL connection: SUCCESS")

except mysql.connector.Error as error:

    print("\nMySQL connection: FAILED")
    print(error)
    raise SystemExit


# ============================================================
# 1. LOAD DATA FROM MYSQL
# ============================================================

query = """
SELECT
    Transaction_ID,
    Customer_ID,
    Transaction_Date,
    Transaction_Amount,
    Merchant_Category,
    Payment_Method,
    Device_Type,
    Location,
    Is_International,
    Previous_Transactions,
    Average_Spend,
    Account_Age_Days,
    Suspicious_Keyword,
    Fraudulent,
    Transaction_Hour,
    Transaction_Day,
    Transaction_Month,
    Day_of_Week,
    Is_Weekend,
    Amount_vs_Average_Spend,
    Transactions_per_Account_Age
FROM transactions
"""

df = pd.read_sql(query, connection)

connection.close()

print("\nData loaded from MySQL successfully.")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATA VALIDATION")
print("=" * 70)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nFraud distribution:")
print(df["Fraudulent"].value_counts())

print("\nFraud rate:")
print(f"{df['Fraudulent'].mean() * 100:.2f}%")


# ============================================================
# 3. CREATE ML FEATURE: IS_NIGHT
# ============================================================

print("\n" + "=" * 70)
print("2. CREATE IS_NIGHT")
print("=" * 70)

df["Is_Night"] = (
    df["Transaction_Hour"].between(0, 5)
).astype(int)

print(df["Is_Night"].value_counts())


# ============================================================
# 4. CREATE ML FEATURE: AMOUNT_VS_AVERAGE
# ============================================================

print("\n" + "=" * 70)
print("3. CREATE AMOUNT_VS_AVERAGE")
print("=" * 70)

df["Amount_vs_Average"] = np.where(
    df["Average_Spend"] > 0,
    df["Transaction_Amount"] / df["Average_Spend"],
    0
)

print(df["Amount_vs_Average"].describe())


# ============================================================
# 5. CREATE ML FEATURE: HIGH_AMOUNT_DEVIATION
# ============================================================

print("\n" + "=" * 70)
print("4. CREATE HIGH_AMOUNT_DEVIATION")
print("=" * 70)

df["High_Amount_Deviation"] = (
    df["Amount_vs_Average"] >= 2
).astype(int)

print(df["High_Amount_Deviation"].value_counts())


# ============================================================
# 6. CREATE ML FEATURE: CUSTOMER_TRANSACTION_COUNT
# ============================================================

print("\n" + "=" * 70)
print("5. CREATE CUSTOMER_TRANSACTION_COUNT")
print("=" * 70)

df["Customer_Transaction_Count"] = (
    df.groupby("Customer_ID")["Transaction_ID"]
    .transform("count")
)

print(df["Customer_Transaction_Count"].describe())


# ============================================================
# 7. CREATE ML FEATURE: INTERNATIONAL_NIGHT
# ============================================================

print("\n" + "=" * 70)
print("6. CREATE INTERNATIONAL_NIGHT")
print("=" * 70)

df["International_Night"] = (
    (df["Is_International"] == 1)
    & (df["Is_Night"] == 1)
).astype(int)

print(df["International_Night"].value_counts())


# ============================================================
# 8. CREATE ML FEATURE: INTERNATIONAL_KEYWORD
# ============================================================

print("\n" + "=" * 70)
print("7. CREATE INTERNATIONAL_KEYWORD")
print("=" * 70)

df["International_Keyword"] = (
    (df["Is_International"] == 1)
    & (df["Suspicious_Keyword"] == "Yes")
).astype(int)

print(df["International_Keyword"].value_counts())


# ============================================================
# 9. CREATE ML FEATURE: BEHAVIORAL_RISK_COUNT
# ============================================================

print("\n" + "=" * 70)
print("8. CREATE BEHAVIORAL_RISK_COUNT")
print("=" * 70)

df["Behavioral_Risk_Count"] = (
    df["Is_International"]
    + df["Is_Night"]
    + (df["Suspicious_Keyword"] == "Yes").astype(int)
)

print(df["Behavioral_Risk_Count"].value_counts().sort_index())


# ============================================================
# 10. FINAL FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("9. FEATURE ENGINEERING CHECK")
print("=" * 70)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
for column in df.columns:
    print("-", column)


# ============================================================
# SAVE FEATURE-ENGINEERED DATA
# ============================================================

output_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_transactions.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)

# ============================================================
# 10. FINAL ML FEATURE SELECTION CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. FINAL ML FEATURE SELECTION CHECK")
print("=" * 70)

drop_features = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Is_Weekend"
]

target = "Fraudulent"

candidate_features = [
    column
    for column in df.columns
    if column not in drop_features + [target]
]

print("\nFeatures excluded from ML:")
for feature in drop_features:
    print("-", feature)

print("\nTarget variable:")
print("-", target)

print("\nCandidate ML features:")

for feature in candidate_features:
    print("-", feature)

print("\nTotal candidate ML features:", len(candidate_features))

print("\nFeature-engineered dataset saved:")
print(output_path)

# ============================================================
# 11. ML FEATURE QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("11. ML FEATURE QUALITY CHECK")
print("=" * 70)


# ------------------------------------------------------------
# 11.1 CHECK DUPLICATE / REDUNDANT FEATURES
# ------------------------------------------------------------

print("\nChecking Amount_vs_Average vs Amount_vs_Average_Spend...")

amount_difference = (
    df["Amount_vs_Average"] - df["Amount_vs_Average_Spend"]
).abs().max()

print(
    "Maximum absolute difference:",
    amount_difference
)

if amount_difference == 0:
    print(
        "RESULT: Amount_vs_Average and "
        "Amount_vs_Average_Spend are identical."
    )
    print(
        "DECISION: Keep Amount_vs_Average_Spend "
        "and remove Amount_vs_Average from ML features."
    )
else:
    print(
        "RESULT: The two features are different."
    )
    print(
        "DECISION: Keep both for model testing."
    )


# ------------------------------------------------------------
# 11.2 CUSTOMER TRANSACTION COUNT CHECK
# ------------------------------------------------------------

print("\nChecking Customer_Transaction_Count...")

print(
    "Customer_Transaction_Count statistics:"
)

print(
    df["Customer_Transaction_Count"].describe()
)

print("\nCustomer transaction count distribution:")

print(
    df["Customer_Transaction_Count"]
    .value_counts()
    .sort_index()
)

print(
    "\nIMPORTANT:"
)

print(
    "Customer_Transaction_Count was calculated "
    "using the complete dataset."
)

print(
    "Therefore, it may introduce data leakage "
    "during train/test evaluation."
)

print(
    "DECISION: Do not use this feature in the "
    "initial production-style ML model."
)


# ------------------------------------------------------------
# 11.3 FINAL ML FEATURES AFTER QUALITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL ML FEATURE SET")
print("=" * 70)


final_drop_features = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Is_Weekend",
    "Amount_vs_Average",
    "Customer_Transaction_Count"
]

final_ml_features = [
    column
    for column in df.columns
    if column not in final_drop_features + ["Fraudulent"]
]

print("\nFeatures excluded from final ML dataset:")

for feature in final_drop_features:
    print(f"- {feature}")

print("\nTarget variable:")
print("- Fraudulent")

print("\nFinal ML features:")

for feature in final_ml_features:
    print(f"- {feature}")

print(
    f"\nTotal final ML features: "
    f"{len(final_ml_features)}"
)

print("\n" + "=" * 70)
print("PHASE 5 FEATURE ENGINEERING COMPLETED")
print("=" * 70)