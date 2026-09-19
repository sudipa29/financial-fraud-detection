from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_transactions.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
    / "fraud_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
)

REPORT_DIR = PROJECT_ROOT / "reports"

EXPLAINED_OUTPUT_PATH = (
    OUTPUT_DIR
    / "fraud_predictions_explained.csv"
)

RISK_REASON_SUMMARY_PATH = (
    REPORT_DIR
    / "fraud_risk_reason_summary.csv"
)


# ============================================================
# 1. CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 12 - MODEL EXPLAINABILITY & FRAUD RISK REASONS")
print("=" * 70)

print("\nLoading feature-engineered dataset...")

df = pd.read_csv(FEATURE_DATA_PATH)

print(f"Feature dataset shape: {df.shape}")


print("\nLoading prediction results...")

predictions = pd.read_csv(PREDICTION_PATH)

print(f"Prediction dataset shape: {predictions.shape}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    "Transaction_ID",
    "Fraud_Probability",
    "Risk_Score",
    "Fraud_Prediction",
    "Risk_Level",
]

required_feature_columns = [
    "Transaction_ID",
    "Is_International",
    "Is_Night",
    "Suspicious_Keyword",
    "High_Amount_Deviation",
    "International_Night",
    "International_Keyword",
    "Behavioral_Risk_Count",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
    "Transaction_Amount",
    "Average_Spend",
]

missing_prediction_columns = [
    col
    for col in required_prediction_columns
    if col not in predictions.columns
]

missing_feature_columns = [
    col
    for col in required_feature_columns
    if col not in df.columns
]

if missing_prediction_columns:
    raise ValueError(
        f"Missing prediction columns: {missing_prediction_columns}"
    )

if missing_feature_columns:
    raise ValueError(
        f"Missing feature columns: {missing_feature_columns}"
    )


# ============================================================
# 4. MERGE PREDICTIONS WITH FEATURES
# ============================================================

print("\nCombining prediction and behavioral information...")

explain_df = predictions.merge(
    df[
        [
            "Transaction_ID",
            "Is_International",
            "Is_Night",
            "Suspicious_Keyword",
            "High_Amount_Deviation",
            "International_Night",
            "International_Keyword",
            "Behavioral_Risk_Count",
            "Amount_vs_Average_Spend",
            "Transactions_per_Account_Age",
            "Average_Spend",
        ]
    ],
    on="Transaction_ID",
    how="left",
    validate="one_to_one",
)

print(f"Combined dataset shape: {explain_df.shape}")


# ============================================================
# 5. CREATE INDIVIDUAL RISK FLAGS
# ============================================================

print("\nCreating explainability flags...")


# International transaction
explain_df["Reason_International"] = (
    explain_df["Is_International"] == 1
)


# Night transaction
explain_df["Reason_Night"] = (
    explain_df["Is_Night"] == 1
)


# Suspicious keyword
explain_df["Reason_Suspicious_Keyword"] = (
    explain_df["Suspicious_Keyword"] == "Yes"
)


# High amount deviation
explain_df["Reason_High_Amount_Deviation"] = (
    explain_df["High_Amount_Deviation"] == 1
)


# International + night combination
explain_df["Reason_International_Night"] = (
    explain_df["International_Night"] == 1
)


# International + suspicious keyword combination
explain_df["Reason_International_Keyword"] = (
    explain_df["International_Keyword"] == 1
)


# Multiple behavioral risk indicators
explain_df["Reason_Multiple_Behavioral_Risks"] = (
    explain_df["Behavioral_Risk_Count"] >= 2
)


# ============================================================
# 6. CREATE HUMAN-READABLE RISK REASONS
# ============================================================

def generate_risk_reasons(row):

    reasons = []

    if row["Is_International"] == 1:
        reasons.append("International transaction")

    if row["Is_Night"] == 1:
        reasons.append("Night-time transaction")

    if row["Suspicious_Keyword"] == "Yes":
        reasons.append("Suspicious keyword detected")

    if row["High_Amount_Deviation"] == 1:
        reasons.append("Transaction amount significantly above average")

    if row["International_Night"] == 1:
        reasons.append("International + night-time pattern")

    if row["International_Keyword"] == 1:
        reasons.append("International + suspicious keyword pattern")

    if row["Behavioral_Risk_Count"] >= 2:
        reasons.append("Multiple behavioral risk indicators")

    if not reasons:
        reasons.append("No major behavioral risk indicators")

    return " | ".join(reasons)


explain_df["Risk_Reasons"] = explain_df.apply(
    generate_risk_reasons,
    axis=1
)


# ============================================================
# 7. CREATE PRIMARY RISK REASON
# ============================================================

def primary_reason(row):

    if row["International_Keyword"] == 1:
        return "International + suspicious keyword"

    if row["International_Night"] == 1:
        return "International + night-time"

    if row["Behavioral_Risk_Count"] >= 2:
        return "Multiple behavioral risk indicators"

    if row["Suspicious_Keyword"] == "Yes":
        return "Suspicious keyword detected"

    if row["Is_International"] == 1:
        return "International transaction"

    if row["Is_Night"] == 1:
        return "Night-time transaction"

    if row["High_Amount_Deviation"] == 1:
        return "High transaction amount deviation"

    return "No major risk indicator"


explain_df["Primary_Risk_Reason"] = explain_df.apply(
    primary_reason,
    axis=1
)


# ============================================================
# 8. CREATE BUSINESS RISK CATEGORY
# ============================================================

def risk_category(row):

    if row["Risk_Level"] == "High":
        return "Immediate Review"

    if row["Risk_Level"] == "Medium":
        return "Monitor"

    return "Normal"


explain_df["Recommended_Action"] = explain_df.apply(
    risk_category,
    axis=1
)


# ============================================================
# 9. SELECT FINAL OUTPUT COLUMNS
# ============================================================

final_columns = [
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


explain_df = explain_df[final_columns]


# ============================================================
# 10. SAVE EXPLAINED PREDICTIONS
# ============================================================

explain_df.to_csv(
    EXPLAINED_OUTPUT_PATH,
    index=False
)

print("\nExplainable prediction file saved:")
print(EXPLAINED_OUTPUT_PATH)


# ============================================================
# 11. RISK REASON SUMMARY
# ============================================================

print("\nCreating risk reason summary...")


reason_summary = pd.DataFrame({
    "Risk_Reason": [
        "International transaction",
        "Night-time transaction",
        "Suspicious keyword",
        "High amount deviation",
        "International + night-time",
        "International + suspicious keyword",
        "Multiple behavioral risks",
    ],
    "Transaction_Count": [
        int((explain_df["Is_International"] == 1).sum()),
        int((explain_df["Is_Night"] == 1).sum()),
        int((explain_df["Suspicious_Keyword"] == "Yes").sum()),
        int((explain_df["High_Amount_Deviation"] == 1).sum()),
        int((explain_df["International_Night"] == 1).sum()),
        int((explain_df["International_Keyword"] == 1).sum()),
        int((explain_df["Behavioral_Risk_Count"] >= 2).sum()),
    ]
})


reason_summary["Percentage_of_Transactions"] = (
    reason_summary["Transaction_Count"]
    / len(explain_df)
    * 100
).round(2)


# ============================================================
# 12. SAVE SUMMARY
# ============================================================

reason_summary.to_csv(
    RISK_REASON_SUMMARY_PATH,
    index=False
)

print("\nRisk reason summary saved:")
print(RISK_REASON_SUMMARY_PATH)


# ============================================================
# 13. DISPLAY HIGH-RISK TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP HIGH-RISK TRANSACTIONS")
print("=" * 70)

high_risk = (
    explain_df[
        explain_df["Risk_Level"] == "High"
    ]
    .sort_values(
        "Fraud_Probability",
        ascending=False
    )
    .head(10)
)


print(
    high_risk[
        [
            "Transaction_ID",
            "Transaction_Amount",
            "Fraud_Probability",
            "Risk_Score",
            "Primary_Risk_Reason",
            "Recommended_Action",
        ]
    ].to_string(index=False)
)


# ============================================================
# 14. RISK LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK LEVEL SUMMARY")
print("=" * 70)

print(
    explain_df["Risk_Level"]
    .value_counts()
    .to_string()
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 12 COMPLETED SUCCESSFULLY")
print("=" * 70)