from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PROJECT PATH & 1. LOAD DATASET
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (PROJECT_ROOT / "data" / "raw" / "financial_fraud_detection_dataset.csv")

# This is the line you are missing before Step 2!
df = pd.read_csv(DATA_PATH) 

# ... (Any other code from Part G) ...

# ==========================================
# 8. DATASET OVERVIEW (This is your Step 2 code)
# ==========================================
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print("\nNumber of Transactions:", len(df))
print("Number of Features:", df.shape[1])

print("\nData Types:")
print(df.dtypes)

# ==========================================
# 9. UNIQUE VALUES BY COLUMN
# ==========================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    print(
        f"{column}: {df[column].nunique()} unique values"
    )

# ==========================================
# 10. FRAUD DISTRIBUTION
# ==========================================

fraud_counts = df["Fraudulent"].value_counts()

print("\nFraudulent Transactions:")
print(fraud_counts)

fraud_rate = (
    df["Fraudulent"].mean() * 100
)

print(
    f"\nOverall Fraud Rate: {fraud_rate:.2f}%"
)

# ==========================================
# 11. CREATE FIGURE DIRECTORY
# ==========================================

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# 12. FRAUD DISTRIBUTION CHART
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Fraudulent"
)

plt.title(
    "Fraudulent vs Non-Fraudulent Transactions"
)

plt.xlabel("Fraudulent")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "01_fraud_distribution.png",
    dpi=300
)

plt.close()

print(
    "\nSaved: 01_fraud_distribution.png"
)

# ==========================================
# 13. DATE CONVERSION
# ==========================================

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%d-%m-%Y %H:%M"
)

print("\nTransaction date converted successfully.")

print(
    "Minimum Transaction Date:",
    df["Transaction_Date"].min()
)

print(
    "Maximum Transaction Date:",
    df["Transaction_Date"].max()
)

# ==========================================
# 14. TIME FEATURES
# ==========================================

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

print("\nTime features created.")

print(
    df[
        [
            "Transaction_Date",
            "Transaction_Hour",
            "Transaction_Day",
            "Transaction_Month",
            "Day_of_Week",
            "Is_Weekend"
        ]
    ].head()
)

# ==========================================
# PHASE 3.1 — BEHAVIORAL FEATURE ENGINEERING
# ==========================================

# Night-time transaction indicator
df["Is_Night"] = (
    df["Transaction_Hour"].between(0, 5)
).astype(int)

print("\nIs_Night distribution:")
print(df["Is_Night"].value_counts())

print("\nFraud Rate by Is_Night:")
night_analysis = (
    df.groupby("Is_Night")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

night_analysis["Fraud_Rate"] *= 100

print(night_analysis)

# ==========================================
# PHASE 3.2 — TRANSACTION BEHAVIOR FEATURE
# ==========================================

# Compare current transaction amount
# with customer's average spending behavior

df["Amount_vs_Average"] = np.where(
    df["Average_Spend"] > 0,
    df["Transaction_Amount"] / df["Average_Spend"],
    0
)

print("\nAmount_vs_Average Summary:")
print(
    df["Amount_vs_Average"].describe()
)
# Step 2 — Create high amount deviation flag
df["High_Amount_Deviation"] = (
    df["Amount_vs_Average"] >= 2
).astype(int)

print("\nHigh Amount Deviation Distribution:")
print(
    df["High_Amount_Deviation"].value_counts()
)


# Step 3 — Validate the feature against fraud
print("\nFraud Rate by High Amount Deviation:")

amount_deviation_analysis = (
    df.groupby("High_Amount_Deviation")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

amount_deviation_analysis["Fraud_Rate"] *= 100

print(amount_deviation_analysis)

# ==========================================
# PHASE 3.3 — TRANSACTION HISTORY FEATURE
# ==========================================

# Indicates whether the customer has previous
# transaction history

df["Has_Transaction_History"] = (
    df["Previous_Transactions"] > 0
).astype(int)

print("\nTransaction History Distribution:")
print(
    df["Has_Transaction_History"].value_counts()
)

print("\nFraud Rate by Transaction History:")

history_analysis = (
    df.groupby("Has_Transaction_History")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

history_analysis["Fraud_Rate"] *= 100

print(history_analysis)

# ==========================================
# PHASE 3.4 — CUSTOMER TRANSACTION FREQUENCY
# ==========================================

customer_transaction_count = (
    df.groupby("Customer_ID")["Transaction_ID"]
      .transform("count")
)

df["Customer_Transaction_Count"] = customer_transaction_count

print("\nCustomer Transaction Count Summary:")
print(
    df["Customer_Transaction_Count"].describe()
)

print("\nCustomer Transaction Count Distribution:")
print(
    df["Customer_Transaction_Count"].value_counts().sort_index()
)

print("\nFraud Rate by Customer Transaction Count:")

customer_frequency_analysis = (
    df.groupby("Customer_Transaction_Count")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

customer_frequency_analysis["Fraud_Rate"] *= 100

print(customer_frequency_analysis)

# ==========================================
# PHASE 3.5 — ACCOUNT MATURITY FEATURE
# ==========================================

df["Account_Age_Group"] = pd.cut(
    df["Account_Age_Days"],
    bins=[-1, 30, 180, 365, np.inf],
    labels=[
        "New_Account",
        "Young_Account",
        "Established_Account",
        "Mature_Account"
    ]
)

print("\nAccount Age Group Distribution:")
print(
    df["Account_Age_Group"].value_counts().sort_index()
)

print("\nFraud Rate by Account Age Group:")

account_age_analysis = (
    df.groupby(
        "Account_Age_Group",
        observed=True
    )["Fraudulent"]
    .agg(
        Transactions="count",
        Fraud_Count="sum",
        Fraud_Rate="mean"
    )
)

account_age_analysis["Fraud_Rate"] *= 100

print(account_age_analysis)

# ==========================================
# PHASE 3.6 — WEEKEND BEHAVIOR ANALYSIS
# ==========================================

print("\nFraud Rate by Weekend:")

weekend_analysis = (
    df.groupby("Is_Weekend")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

weekend_analysis["Fraud_Rate"] *= 100

print(weekend_analysis)

# ==========================================
# PHASE 3.7 — COMBINED RISK BEHAVIOR
# ==========================================

df["International_Night"] = (
    (df["Is_International"] == 1) &
    (df["Is_Night"] == 1)
).astype(int)

print("\nInternational + Night Distribution:")
print(
    df["International_Night"].value_counts()
)

print("\nFraud Rate by International + Night:")

international_night_analysis = (
    df.groupby("International_Night")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

international_night_analysis["Fraud_Rate"] *= 100

print(international_night_analysis)


df["International_Keyword"] = (
    (df["Is_International"] == 1) &
    (df["Suspicious_Keyword"] == "Yes")
).astype(int)

print("\nInternational + Suspicious Keyword Distribution:")
print(
    df["International_Keyword"].value_counts()
)

print("\nFraud Rate by International + Suspicious Keyword:")

international_keyword_analysis = (
    df.groupby("International_Keyword")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

international_keyword_analysis["Fraud_Rate"] *= 100

print(international_keyword_analysis)

# ==========================================
# PHASE 3.8 — BEHAVIORAL RISK INDICATOR
# ==========================================

df["Behavioral_Risk_Count"] = (
    df["Is_International"]
    + df["Is_Night"]
    + (df["Suspicious_Keyword"] == "Yes").astype(int)
)

print("\nBehavioral Risk Count Distribution:")
print(
    df["Behavioral_Risk_Count"].value_counts().sort_index()
)

print("\nFraud Rate by Behavioral Risk Count:")

behavioral_risk_analysis = (
    df.groupby("Behavioral_Risk_Count")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

behavioral_risk_analysis["Fraud_Rate"] *= 100

print(behavioral_risk_analysis)

# ==========================================
# 15. FRAUD BY TRANSACTION HOUR
# ==========================================

fraud_by_hour = (
    df.groupby("Transaction_Hour")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

fraud_by_hour["Fraud_Rate"] *= 100

print("\nFraud by Hour:")
print(fraud_by_hour)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=fraud_by_hour.index,
    y=fraud_by_hour["Fraud_Rate"]
)

plt.title("Fraud Rate by Transaction Hour")
plt.xlabel("Transaction Hour")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "02_fraud_by_hour.png",
    dpi=300
)

plt.close()

# ==========================================
# 16. FRAUD BY MERCHANT CATEGORY
# ==========================================

fraud_by_merchant = (
    df.groupby("Merchant_Category")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
      .sort_values(
          "Fraud_Rate",
          ascending=False
      )
)

fraud_by_merchant["Fraud_Rate"] *= 100

print("\nFraud by Merchant Category:")
print(fraud_by_merchant)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=fraud_by_merchant.reset_index(),
    x="Fraud_Rate",
    y="Merchant_Category"
)

plt.title(
    "Fraud Rate by Merchant Category"
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Merchant Category")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "03_fraud_by_merchant.png",
    dpi=300
)

plt.close()

# ==========================================
# 17. FRAUD BY PAYMENT METHOD
# ==========================================

fraud_by_payment = (
    df.groupby("Payment_Method")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
      .sort_values(
          "Fraud_Rate",
          ascending=False
      )
)

fraud_by_payment["Fraud_Rate"] *= 100

print("\nFraud by Payment Method:")
print(fraud_by_payment)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=fraud_by_payment.reset_index(),
    x="Payment_Method",
    y="Fraud_Rate"
)

plt.title(
    "Fraud Rate by Payment Method"
)

plt.xlabel("Payment Method")
plt.ylabel("Fraud Rate (%)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "04_fraud_by_payment_method.png",
    dpi=300
)

plt.close()

# ==========================================
# 18. FRAUD BY DEVICE TYPE
# ==========================================

fraud_by_device = (
    df.groupby("Device_Type")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
      .sort_values(
          "Fraud_Rate",
          ascending=False
      )
)

fraud_by_device["Fraud_Rate"] *= 100

print("\nFraud by Device Type:")
print(fraud_by_device)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=fraud_by_device.reset_index(),
    x="Device_Type",
    y="Fraud_Rate"
)

plt.title(
    "Fraud Rate by Device Type"
)

plt.xlabel("Device Type")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "05_fraud_by_device.png",
    dpi=300
)

plt.close()

# ==========================================
# 19. FRAUD BY LOCATION
# ==========================================

fraud_by_location = (
    df.groupby("Location")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
      .sort_values(
          "Fraud_Rate",
          ascending=False
      )
)

fraud_by_location["Fraud_Rate"] *= 100

print("\nFraud by Location:")
print(fraud_by_location)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=fraud_by_location.reset_index(),
    x="Fraud_Rate",
    y="Location"
)

plt.title(
    "Fraud Rate by Location"
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "06_fraud_by_location.png",
    dpi=300
)

plt.close()

# ==========================================
# 20. INTERNATIONAL TRANSACTION ANALYSIS
# ==========================================

international_analysis = (
    df.groupby("Is_International")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

international_analysis["Fraud_Rate"] *= 100

print(
    "\nInternational Transaction Analysis:"
)

print(international_analysis)

# ==========================================
# 21. SUSPICIOUS KEYWORD ANALYSIS
# ==========================================

keyword_analysis = (
    df.groupby("Suspicious_Keyword")["Fraudulent"]
      .agg(
          Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate="mean"
      )
)

keyword_analysis["Fraud_Rate"] *= 100

print(
    "\nSuspicious Keyword Analysis:"
)

print(keyword_analysis)

# ==========================================
# 22. TRANSACTION AMOUNT ANALYSIS
# ==========================================

amount_analysis = (
    df.groupby("Fraudulent")[
        "Transaction_Amount"
    ]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Minimum="min",
        Maximum="max",
        Std_Dev="std"
    )
)

print("\nTransaction Amount Analysis:")
print(amount_analysis)

# ==========================================
# 23. FRAUD AMOUNT ANALYSIS
# ==========================================

fraud_amount = df.loc[
    df["Fraudulent"] == 1,
    "Transaction_Amount"
].sum()

legitimate_amount = df.loc[
    df["Fraudulent"] == 0,
    "Transaction_Amount"
].sum()

total_amount = df["Transaction_Amount"].sum()

print(
    f"\nTotal Transaction Amount: "
    f"{total_amount:,.2f}"
)

print(
    f"Fraudulent Transaction Amount: "
    f"{fraud_amount:,.2f}"
)

print(
    f"Legitimate Transaction Amount: "
    f"{legitimate_amount:,.2f}"
)

print(
    f"Fraud Amount Percentage: "
    f"{fraud_amount / total_amount * 100:.2f}%"
)

# ==========================================
# 23.1 PHASE 2 RATIO VALIDATION
# ==========================================

overall_fraud_rate = df["Fraudulent"].mean()

international_fraud_rate = (
    df.loc[df["Is_International"] == 1, "Fraudulent"].mean()
)

domestic_fraud_rate = (
    df.loc[df["Is_International"] == 0, "Fraudulent"].mean()
)

keyword_fraud_rate = (
    df.loc[df["Suspicious_Keyword"] == "Yes", "Fraudulent"].mean()
)

no_keyword_fraud_rate = (
    df.loc[df["Suspicious_Keyword"] == "No", "Fraudulent"].mean()
)

international_ratio = (
    international_fraud_rate / domestic_fraud_rate
)

keyword_ratio = (
    keyword_fraud_rate / no_keyword_fraud_rate
)

print("\nPhase 2 Ratio Validation:")

print(
    f"International vs Domestic Fraud Risk Ratio: "
    f"{international_ratio:.2f}x"
)

print(
    f"Suspicious Keyword vs No Keyword Fraud Risk Ratio: "
    f"{keyword_ratio:.2f}x"
)

print(
    f"International Fraud Rate vs Overall Rate: "
    f"{international_fraud_rate / overall_fraud_rate:.2f}x"
)

print(
    f"Suspicious Keyword Fraud Rate vs Overall Rate: "
    f"{keyword_fraud_rate / overall_fraud_rate:.2f}x"
)

# ==========================================
# 24. NUMERICAL CORRELATION
# ==========================================

numeric_columns = [
    "Transaction_Amount",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Fraudulent"
]

correlation = df[
    numeric_columns
].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Numerical Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "07_correlation_heatmap.png",
    dpi=300
)

plt.close()

# ==========================================
# 25. DATA VALIDATION
# ==========================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

print(
    "\nNegative Transaction Amounts:",
    (df["Transaction_Amount"] < 0).sum()
)

print(
    "Negative Previous Transactions:",
    (df["Previous_Transactions"] < 0).sum()
)

print(
    "Negative Average Spend:",
    (df["Average_Spend"] < 0).sum()
)

print(
    "Negative Account Age:",
    (df["Account_Age_Days"] < 0).sum()
)
# Check duplicate transaction IDs
duplicate_transaction_ids = (
    df["Transaction_ID"].duplicated().sum()
)

print(
    "\nDuplicate Transaction IDs:",
    duplicate_transaction_ids
)

# ==========================================
# 26. SAVE EDA SUMMARY
# ==========================================

eda_summary = {
    "total_transactions": len(df),
    "total_features": df.shape[1],

    "fraud_transactions": int(
        df["Fraudulent"].sum()
    ),

    "fraud_rate_percent": round(
        df["Fraudulent"].mean() * 100,
        2
    ),

    "total_transaction_amount": round(
        df["Transaction_Amount"].sum(),
        2
    ),

    "fraud_transaction_amount": round(
        fraud_amount,
        2
    ),

    "fraud_amount_percentage": round(
        fraud_amount / total_amount * 100,
        2
    ),

    "international_domestic_risk_ratio": round(
        international_ratio,
        2
    ),

    "suspicious_keyword_no_keyword_risk_ratio": round(
        keyword_ratio,
        2
    ),

    "international_overall_risk_ratio": round(
        international_fraud_rate / overall_fraud_rate,
        2
    ),

    "suspicious_keyword_overall_risk_ratio": round(
        keyword_fraud_rate / overall_fraud_rate,
        2
    ),

    "duplicate_rows": int(
        df.duplicated().sum()
    ),

    "duplicate_transaction_ids": int(
        duplicate_transaction_ids
    )
}

eda_summary_df = pd.DataFrame(
    [eda_summary]
)

eda_summary_df.to_csv(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eda_summary.csv",
    index=False
)

print(
    "\nEDA summary saved successfully."
)