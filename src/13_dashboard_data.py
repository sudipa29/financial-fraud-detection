from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
    / "fraud_predictions_explained.csv"
)

# Feature-engineered data used to bring
# original transaction category fields
FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_transactions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fraud_dashboard_data.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "dashboard_data_summary.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD EXPLAINED PREDICTIONS
# ============================================================

print("\nLoading explained fraud predictions...")

df = pd.read_csv(INPUT_PATH)

print(
    f"Input shape: {df.shape}"
)


# ============================================================
# 1A. LOAD ORIGINAL TRANSACTION CATEGORIES
# ============================================================

print(
    "\nLoading original transaction category fields..."
)

feature_df = pd.read_csv(
    FEATURE_DATA_PATH,
    usecols=[
    "Transaction_ID",
    "Fraudulent",
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
],

)

print(
    f"Category data shape: {feature_df.shape}"
)


# ============================================================
# 1B. MERGE CATEGORY FIELDS
# ============================================================

df = df.merge(
    feature_df,
    on="Transaction_ID",
    how="left",
    validate="one_to_one",
)

print(
    f"Shape after category merge: {df.shape}"
)

# ============================================================
# 2. REQUIRED INPUT COLUMNS
# ============================================================

required_columns = [
    "Transaction_ID",
    "Transaction_Date",
    "Transaction_Amount",
    "Fraud_Probability",
    "Risk_Score",
    "Prediction_Label",
    "Fraud_Prediction",
    "Risk_Level",
    "Primary_Risk_Reason",
    "Risk_Reasons",
    "Recommended_Action",
    "Is_International",
    "Is_Night",
    "Suspicious_Keyword",
    "High_Amount_Deviation",
    "International_Night",
    "International_Keyword",
    "Behavioral_Risk_Count",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print(
    f"All required columns available: "
    f"{len(required_columns)}"
)


# ============================================================
# 3. CONVERT TRANSACTION DATE
# ============================================================

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    errors="coerce"
)


if df["Transaction_Date"].isna().any():

    raise ValueError(
        "Invalid Transaction_Date values found."
    )


# ============================================================
# 4. CREATE DATE FEATURES
# ============================================================

df["Transaction_Year"] = (
    df["Transaction_Date"].dt.year
)

df["Transaction_Month_Number"] = (
    df["Transaction_Date"].dt.month
)

df["Transaction_Month_Name"] = (
    df["Transaction_Date"].dt.month_name()
)

df["Transaction_Year_Month"] = (
    df["Transaction_Date"]
    .dt.to_period("M")
    .astype(str)
)

df["Transaction_Date_Only"] = (
    df["Transaction_Date"]
    .dt.date
    .astype(str)
)


# ============================================================
# 5. CREATE FRAUD STATUS
# ============================================================

df["Fraud_Status"] = np.where(
    df["Fraud_Prediction"] == 1,
    "Predicted Fraud",
    "Predicted Legitimate"
)


# ============================================================
# 6. CREATE TRANSACTION TYPE
# ============================================================

df["Transaction_Type"] = np.where(
    df["Is_International"] == 1,
    "International",
    "Domestic"
)


# ============================================================
# 7. CREATE TIME RISK
# ============================================================

df["Time_Risk"] = np.where(
    df["Is_Night"] == 1,
    "Night",
    "Day"
)


# ============================================================
# 8. CREATE KEYWORD RISK
# ============================================================

df["Keyword_Risk"] = np.where(
    df["Suspicious_Keyword"] == "Yes",
    "Suspicious Keyword",
    "Normal"
)


# ============================================================
# 9. CREATE HIGH RISK FLAG
# ============================================================

df["High_Risk_Flag"] = (
    df["Risk_Level"] == "High"
).astype(int)


# ============================================================
# 10. CREATE FLAGGED FRAUD AMOUNT
# ============================================================

df["Flagged_Fraud_Amount"] = (
    df["Transaction_Amount"]
    * df["Fraud_Prediction"]
)


# ============================================================
# 11. CREATE RISK SCORE BAND
# ============================================================

def risk_score_band(score):

    if score < 30:
        return "Low Risk"

    if score < 69:
        return "Medium Risk"

    return "High Risk"


df["Risk_Score_Band"] = (
    df["Risk_Score"]
    .apply(risk_score_band)
)


# ============================================================
# 12. CREATE DASHBOARD-READY COLUMN ORDER
# ============================================================

dashboard_columns = [

    # --------------------------------------------------------
    # Transaction information
    # --------------------------------------------------------

    "Transaction_ID",
    "Transaction_Date",
    "Transaction_Date_Only",
    "Transaction_Year",
    "Transaction_Month_Number",
    "Transaction_Month_Name",
    "Transaction_Year_Month",
    "Transaction_Amount",
    "Flagged_Fraud_Amount",

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    "Fraudulent",
    "Fraud_Probability",
    "Risk_Score",
    "Risk_Score_Band",
    "Prediction_Label",
    "Fraud_Prediction",
    "Fraud_Status",

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    "Risk_Level",
    "High_Risk_Flag",
    "Primary_Risk_Reason",
    "Risk_Reasons",
    "Recommended_Action",

    # --------------------------------------------------------
    # Transaction categories
    # --------------------------------------------------------

    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",

    # --------------------------------------------------------
    # Behavioral indicators
    # --------------------------------------------------------

    "Transaction_Type",
    "Time_Risk",
    "Keyword_Risk",
    "Is_International",
    "Is_Night",
    "Suspicious_Keyword",
    "High_Amount_Deviation",
    "International_Night",
    "International_Keyword",
    "Behavioral_Risk_Count",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
]


dashboard_df = df[dashboard_columns].copy()


# ============================================================
# 13. VALIDATE FINAL DATASET
# ============================================================

print("\nValidating dashboard dataset...")

print(
    f"Final rows    : {len(dashboard_df)}"
)

print(
    f"Final columns : {len(dashboard_df.columns)}"
)

print(
    f"Duplicate Transaction IDs: "
    f"{dashboard_df['Transaction_ID'].duplicated().sum()}"
)

print(
    f"Missing values: "
    f"{dashboard_df.isna().sum().sum()}"
)


if dashboard_df["Transaction_ID"].duplicated().any():

    raise ValueError(
        "Duplicate Transaction_ID values found."
    )


if dashboard_df.isna().sum().sum() > 0:

    raise ValueError(
        "Missing values found in dashboard dataset."
    )


# ============================================================
# 14. SAVE DASHBOARD DATA
# ============================================================

dashboard_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDashboard dataset saved:")

print(OUTPUT_PATH)


# ============================================================
# 15. CREATE DASHBOARD SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Total Transactions",

        "Predicted Fraud",

        "Predicted Legitimate",

        "Predicted Fraud Rate (%)",

        "High Risk Transactions",

        "Medium Risk Transactions",

        "Low Risk Transactions",

        "Total Transaction Amount",

        "Flagged Fraud Amount",

        "Average Fraud Probability",

        "Average Risk Score",
    ],

    "Value": [

        len(dashboard_df),

        int(
            dashboard_df[
                "Fraud_Prediction"
            ].sum()
        ),

        int(
            (
                dashboard_df[
                    "Fraud_Prediction"
                ] == 0
            ).sum()
        ),

        round(
            dashboard_df[
                "Fraud_Prediction"
            ].mean() * 100,
            2
        ),

        int(
            (
                dashboard_df[
                    "Risk_Level"
                ] == "High"
            ).sum()
        ),

        int(
            (
                dashboard_df[
                    "Risk_Level"
                ] == "Medium"
            ).sum()
        ),

        int(
            (
                dashboard_df[
                    "Risk_Level"
                ] == "Low"
            ).sum()
        ),

        round(
            dashboard_df[
                "Transaction_Amount"
            ].sum(),
            2
        ),

        round(
            dashboard_df[
                "Flagged_Fraud_Amount"
            ].sum(),
            2
        ),

        round(
            dashboard_df[
                "Fraud_Probability"
            ].mean(),
            4
        ),

        round(
            dashboard_df[
                "Risk_Score"
            ].mean(),
            2
        ),
    ],
})


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

print("\nDashboard summary saved:")

print(SUMMARY_PATH)


# ============================================================
# 17. DISPLAY KEY DASHBOARD METRICS
# ============================================================

print("\n" + "=" * 70)

print("DASHBOARD DATA SUMMARY")

print("=" * 70)


print(
    f"Total Transactions       : "
    f"{len(dashboard_df):,}"
)


print(
    f"Predicted Fraud          : "
    f"{dashboard_df['Fraud_Prediction'].sum():,}"
)


print(
    f"Predicted Fraud Rate     : "
    f"{dashboard_df['Fraud_Prediction'].mean() * 100:.2f}%"
)


print(
    f"High Risk                : "
    f"{(
        dashboard_df['Risk_Level'] == 'High'
    ).sum():,}"
)


print(
    f"Medium Risk              : "
    f"{(
        dashboard_df['Risk_Level'] == 'Medium'
    ).sum():,}"
)


print(
    f"Low Risk                 : "
    f"{(
        dashboard_df['Risk_Level'] == 'Low'
    ).sum():,}"
)


print(
    f"Total Transaction Amount : "
    f"{dashboard_df['Transaction_Amount'].sum():,.2f}"
)


print(
    f"Flagged Fraud Amount     : "
    f"{dashboard_df['Flagged_Fraud_Amount'].sum():,.2f}"
)


# ============================================================
# 18. TOP RISK REASONS
# ============================================================

print("\nTop Primary Risk Reasons:")

print(
    dashboard_df[
        "Primary_Risk_Reason"
    ]
    .value_counts()
    .head(10)
    .to_string()
)


# ============================================================
# 19. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

print("PHASE 13 COMPLETED SUCCESSFULLY")

print("=" * 70)

