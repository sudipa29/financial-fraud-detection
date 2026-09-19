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
# Do NOT share the password with anyone.
MYSQL_PASSWORD = "root"


# ============================================================
# DATABASE CONNECTION
# ============================================================

print("=" * 70)
print("PHASE 4 - EXPLORATORY DATA ANALYSIS")
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
# 2. DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", missing_values.sum())


# ============================================================
# 4. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_rows = df.duplicated().sum()
duplicate_transaction_ids = df["Transaction_ID"].duplicated().sum()

print("Duplicate rows:", duplicate_rows)
print("Duplicate Transaction_IDs:", duplicate_transaction_ids)


# ============================================================
# 5. FRAUD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. FRAUD DISTRIBUTION")
print("=" * 70)

fraud_distribution = (
    df.groupby("Fraudulent")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

fraud_distribution["Percentage"] = (
    fraud_distribution["Transaction_Count"]
    / len(df)
    * 100
)

print(fraud_distribution)

fraud_distribution.to_csv(
    REPORTS_PATH / "eda_fraud_distribution.csv",
    index=False
)


# ============================================================
# 6. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("5. NUMERICAL FEATURE SUMMARY")
print("=" * 70)

numeric_columns = [
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Is_International",
    "Is_Weekend",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
    "Fraudulent"
]

numeric_summary = df[numeric_columns].describe().T

print(numeric_summary)

numeric_summary.to_csv(
    REPORTS_PATH / "eda_numeric_summary.csv"
)


# ============================================================
# 7. INTERNATIONAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. FRAUD RATE BY INTERNATIONAL TRANSACTION")
print("=" * 70)

international_analysis = (
    df.groupby("Is_International")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

international_analysis["Fraud_Rate"] = (
    international_analysis["Fraud_Count"]
    / international_analysis["Transaction_Count"]
    * 100
)

print(international_analysis)

international_analysis.to_csv(
    REPORTS_PATH / "eda_international_analysis.csv",
    index=False
)


# ============================================================
# 8. SUSPICIOUS KEYWORD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. FRAUD RATE BY SUSPICIOUS KEYWORD")
print("=" * 70)

keyword_analysis = (
    df.groupby("Suspicious_Keyword")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

keyword_analysis["Fraud_Rate"] = (
    keyword_analysis["Fraud_Count"]
    / keyword_analysis["Transaction_Count"]
    * 100
)

print(keyword_analysis)

keyword_analysis.to_csv(
    REPORTS_PATH / "eda_suspicious_keyword_analysis.csv",
    index=False
)


# ============================================================
# 9. MERCHANT CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. FRAUD RATE BY MERCHANT CATEGORY")
print("=" * 70)

merchant_analysis = (
    df.groupby("Merchant_Category")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

merchant_analysis["Fraud_Rate"] = (
    merchant_analysis["Fraud_Count"]
    / merchant_analysis["Transaction_Count"]
    * 100
)

merchant_analysis = merchant_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(merchant_analysis)

merchant_analysis.to_csv(
    REPORTS_PATH / "eda_merchant_analysis.csv",
    index=False
)


# ============================================================
# 10. PAYMENT METHOD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("9. FRAUD RATE BY PAYMENT METHOD")
print("=" * 70)

payment_analysis = (
    df.groupby("Payment_Method")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

payment_analysis["Fraud_Rate"] = (
    payment_analysis["Fraud_Count"]
    / payment_analysis["Transaction_Count"]
    * 100
)

payment_analysis = payment_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(payment_analysis)

payment_analysis.to_csv(
    REPORTS_PATH / "eda_payment_method_analysis.csv",
    index=False
)


# ============================================================
# 11. DEVICE TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("10. FRAUD RATE BY DEVICE TYPE")
print("=" * 70)

device_analysis = (
    df.groupby("Device_Type")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

device_analysis["Fraud_Rate"] = (
    device_analysis["Fraud_Count"]
    / device_analysis["Transaction_Count"]
    * 100
)

device_analysis = device_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(device_analysis)

device_analysis.to_csv(
    REPORTS_PATH / "eda_device_analysis.csv",
    index=False
)


# ============================================================
# 12. LOCATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("11. FRAUD RATE BY LOCATION")
print("=" * 70)

location_analysis = (
    df.groupby("Location")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

location_analysis["Fraud_Rate"] = (
    location_analysis["Fraud_Count"]
    / location_analysis["Transaction_Count"]
    * 100
)

location_analysis = location_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(location_analysis)

location_analysis.to_csv(
    REPORTS_PATH / "eda_location_analysis.csv",
    index=False
)


# ============================================================
# 13. DAY OF WEEK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("12. FRAUD RATE BY DAY OF WEEK")
print("=" * 70)

day_analysis = (
    df.groupby("Day_of_Week")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

day_analysis["Fraud_Rate"] = (
    day_analysis["Fraud_Count"]
    / day_analysis["Transaction_Count"]
    * 100
)

print(day_analysis)

day_analysis.to_csv(
    REPORTS_PATH / "eda_day_analysis.csv",
    index=False
)


# ============================================================
# 14. WEEKEND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("13. FRAUD RATE BY WEEKEND")
print("=" * 70)

weekend_analysis = (
    df.groupby("Is_Weekend")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

weekend_analysis["Fraud_Rate"] = (
    weekend_analysis["Fraud_Count"]
    / weekend_analysis["Transaction_Count"]
    * 100
)

print(weekend_analysis)

weekend_analysis.to_csv(
    REPORTS_PATH / "eda_weekend_analysis.csv",
    index=False
)


# ============================================================
# 15. TRANSACTION HOUR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("14. FRAUD RATE BY TRANSACTION HOUR")
print("=" * 70)

hour_analysis = (
    df.groupby("Transaction_Hour")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

hour_analysis["Fraud_Rate"] = (
    hour_analysis["Fraud_Count"]
    / hour_analysis["Transaction_Count"]
    * 100
)

hour_analysis = hour_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(hour_analysis)

hour_analysis.to_csv(
    REPORTS_PATH / "eda_hour_analysis.csv",
    index=False
)


# ============================================================
# 16. BEHAVIORAL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("15. BEHAVIORAL FEATURES BY FRAUD STATUS")
print("=" * 70)

behavioral_features = [
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age"
]

behavioral_analysis = (
    df.groupby("Fraudulent")[behavioral_features]
    .mean()
    .reset_index()
)

print(behavioral_analysis)

behavioral_analysis.to_csv(
    REPORTS_PATH / "eda_behavioral_analysis.csv",
    index=False
)


# ============================================================
# 17. TRANSACTION AMOUNT BY FRAUD STATUS
# ============================================================

print("\n" + "=" * 70)
print("16. TRANSACTION AMOUNT BY FRAUD STATUS")
print("=" * 70)

amount_analysis = (
    df.groupby("Fraudulent")["Transaction_Amount"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Minimum="min",
        Maximum="max"
    )
    .reset_index()
)

print(amount_analysis)

amount_analysis.to_csv(
    REPORTS_PATH / "eda_amount_analysis.csv",
    index=False
)


# ============================================================
# 18. BEHAVIORAL RATIO ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("17. BEHAVIORAL RATIO ANALYSIS")
print("=" * 70)

ratio_features = [
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age"
]

ratio_analysis = []

for feature in ratio_features:

    result = (
        df.groupby("Fraudulent")[feature]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Minimum="min",
            Maximum="max"
        )
        .reset_index()
    )

    result["Feature"] = feature

    ratio_analysis.append(result)

ratio_analysis = pd.concat(
    ratio_analysis,
    ignore_index=True
)

print(ratio_analysis)

ratio_analysis.to_csv(
    REPORTS_PATH / "eda_ratio_analysis.csv",
    index=False
)


# ============================================================
# PHASE 4.2
# CORRELATION AND COMBINED PATTERN ANALYSIS
# ============================================================


# ============================================================
# 19. NUMERICAL CORRELATION WITH FRAUD
# ============================================================

print("\n" + "=" * 70)
print("18. NUMERICAL FEATURE CORRELATION WITH FRAUD")
print("=" * 70)

correlation_features = [
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Is_International",
    "Is_Weekend",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
    "Fraudulent"
]

correlation_matrix = df[correlation_features].corr()

fraud_correlation = (
    correlation_matrix["Fraudulent"]
    .drop("Fraudulent")
    .sort_values(ascending=False)
)

print("\nCorrelation with Fraudulent:")
print(fraud_correlation)

fraud_correlation.to_csv(
    REPORTS_PATH / "eda_fraud_correlation.csv",
    header=["Correlation_with_Fraud"]
)

print("\nSaved:")
print(REPORTS_PATH / "eda_fraud_correlation.csv")


# ============================================================
# 20. FRAUD RATE BY NUMERICAL FEATURE QUARTILES
# ============================================================

print("\n" + "=" * 70)
print("19. FRAUD RATE BY NUMERICAL FEATURE QUARTILES")
print("=" * 70)

quartile_features = [
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age"
]

quartile_results = []

for feature in quartile_features:

    temp = df[[feature, "Fraudulent"]].copy()

    try:

        temp["Quartile"] = pd.qcut(
            temp[feature],
            q=4,
            duplicates="drop"
        )

        result = (
            temp.groupby(
                "Quartile",
                observed=False
            )["Fraudulent"]
            .agg(
                Transaction_Count="count",
                Fraud_Count="sum",
                Fraud_Rate="mean"
            )
            .reset_index()
        )

        result["Fraud_Rate"] = (
            result["Fraud_Rate"] * 100
        )

        result["Feature"] = feature

        quartile_results.append(result)

        print(f"\n--- {feature} ---")
        print(result)

    except ValueError:

        print(
            f"\nUnable to create quartiles for {feature}"
        )


if quartile_results:

    quartile_analysis = pd.concat(
        quartile_results,
        ignore_index=True
    )

    quartile_analysis.to_csv(
        REPORTS_PATH / "eda_numeric_quartile_analysis.csv",
        index=False
    )

    print("\nSaved:")
    print(
        REPORTS_PATH
        / "eda_numeric_quartile_analysis.csv"
    )


# ============================================================
# 21. INTERNATIONAL + SUSPICIOUS KEYWORD
# ============================================================

print("\n" + "=" * 70)
print("20. INTERNATIONAL + SUSPICIOUS KEYWORD ANALYSIS")
print("=" * 70)

international_keyword = (
    df.groupby(
        [
            "Is_International",
            "Suspicious_Keyword"
        ]
    )
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

international_keyword["Fraud_Rate"] = (
    international_keyword["Fraud_Count"]
    / international_keyword["Transaction_Count"]
    * 100
)

print(international_keyword)

international_keyword.to_csv(
    REPORTS_PATH
    / "eda_international_keyword_analysis.csv",
    index=False
)


# ============================================================
# 22. TRANSACTION HOUR GROUP
# ============================================================

print("\n" + "=" * 70)
print("21. TRANSACTION HOUR GROUP ANALYSIS")
print("=" * 70)

df["Hour_Group"] = np.select(
    [
        df["Transaction_Hour"].between(0, 5),
        df["Transaction_Hour"].between(6, 11),
        df["Transaction_Hour"].between(12, 17),
        df["Transaction_Hour"].between(18, 23)
    ],
    [
        "Overnight (00-05)",
        "Morning (06-11)",
        "Afternoon (12-17)",
        "Evening (18-23)"
    ],
    default="Unknown"
)

hour_group_analysis = (
    df.groupby("Hour_Group")
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

hour_group_analysis["Fraud_Rate"] = (
    hour_group_analysis["Fraud_Count"]
    / hour_group_analysis["Transaction_Count"]
    * 100
)

print(hour_group_analysis)

hour_group_analysis.to_csv(
    REPORTS_PATH / "eda_hour_group_analysis.csv",
    index=False
)


# ============================================================
# 23. SUSPICIOUS KEYWORD + HOUR GROUP
# ============================================================

print("\n" + "=" * 70)
print("22. SUSPICIOUS KEYWORD + HOUR GROUP ANALYSIS")
print("=" * 70)

keyword_hour_analysis = (
    df.groupby(
        [
            "Suspicious_Keyword",
            "Hour_Group"
        ]
    )
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

keyword_hour_analysis["Fraud_Rate"] = (
    keyword_hour_analysis["Fraud_Count"]
    / keyword_hour_analysis["Transaction_Count"]
    * 100
)

keyword_hour_analysis = keyword_hour_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(keyword_hour_analysis)

keyword_hour_analysis.to_csv(
    REPORTS_PATH / "eda_keyword_hour_analysis.csv",
    index=False
)

# ============================================================
# 24. INTERNATIONAL + HOUR GROUP
# ============================================================

print("\n" + "=" * 70)
print("23. INTERNATIONAL + HOUR GROUP ANALYSIS")
print("=" * 70)

international_hour_analysis = (
    df.groupby(
        [
            "Is_International",
            "Hour_Group"
        ]
    )
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

international_hour_analysis["Fraud_Rate"] = (
    international_hour_analysis["Fraud_Count"]
    / international_hour_analysis["Transaction_Count"]
    * 100
)

international_hour_analysis = international_hour_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(international_hour_analysis)

international_hour_analysis.to_csv(
    REPORTS_PATH / "eda_international_hour_analysis.csv",
    index=False
)


# ============================================================
# 25. MERCHANT + INTERNATIONAL
# ============================================================

print("\n" + "=" * 70)
print("24. MERCHANT + INTERNATIONAL ANALYSIS")
print("=" * 70)

merchant_international_analysis = (
    df.groupby(
        [
            "Merchant_Category",
            "Is_International"
        ]
    )
    .agg(
        Transaction_Count=("Transaction_ID", "count"),
        Fraud_Count=("Fraudulent", "sum"),
        Average_Amount=("Transaction_Amount", "mean")
    )
    .reset_index()
)

merchant_international_analysis["Fraud_Rate"] = (
    merchant_international_analysis["Fraud_Count"]
    / merchant_international_analysis["Transaction_Count"]
    * 100
)

merchant_international_analysis = (
    merchant_international_analysis.sort_values(
        "Fraud_Rate",
        ascending=False
    )
)

print(merchant_international_analysis)

merchant_international_analysis.to_csv(
    REPORTS_PATH
    / "eda_merchant_international_analysis.csv",
    index=False
)


# ============================================================
# 26. HIGH-RISK COMBINATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("25. HIGH-RISK COMBINATION SUMMARY")
print("=" * 70)

print("\nHighest fraud-rate International + Keyword combinations:")

high_risk_keyword = international_keyword.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(
    high_risk_keyword[
        [
            "Is_International",
            "Suspicious_Keyword",
            "Transaction_Count",
            "Fraud_Count",
            "Fraud_Rate"
        ]
    ]
)


print("\nHighest fraud-rate Hour Groups:")

high_risk_hours = hour_group_analysis.sort_values(
    "Fraud_Rate",
    ascending=False
)

print(
    high_risk_hours[
        [
            "Hour_Group",
            "Transaction_Count",
            "Fraud_Count",
            "Fraud_Rate"
        ]
    ]
)


# ============================================================
# 27. FINAL PHASE 4 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("26. PHASE 4 EDA SUMMARY")
print("=" * 70)

print("""
PHASE 4 FINDINGS:

1. Fraud represents a minority class in the dataset.
2. International transactions show a substantially higher
   fraud rate than domestic transactions.
3. Suspicious keywords show a strong association with fraud.
4. Transaction hour shows a strong behavioral pattern,
   especially during overnight hours.
5. Transaction amount alone does not strongly separate
   fraudulent and legitimate transactions.
6. Weekend status shows only a weak difference in fraud rate.
7. Behavioral ratios should be retained for later modeling
   because weak individual relationships do not mean that
   the features are useless in combination.
8. Combined feature analysis is being used to identify
   interaction patterns before model development.
9. Correlation results should be interpreted as association,
   not causation.
10. The dataset and MySQL pipeline remain validated.

Phase 4 EDA completed successfully.
""")


# ============================================================
# FINAL OUTPUT FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EDA OUTPUT FILES")
print("=" * 70)

eda_files = sorted(
    REPORTS_PATH.glob("eda_*.csv")
)

for file in eda_files:
    print(file.name)

print("\nTotal EDA output files:", len(eda_files))

print("\n" + "=" * 70)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("=" * 70)