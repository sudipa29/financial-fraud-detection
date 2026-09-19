from pathlib import Path
import json
import joblib
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

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "final_xgboost_fraud_model.pkl"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "final_xgboost_threshold.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
)

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_PATH = OUTPUT_DIR / "fraud_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "fraud_prediction_summary.csv"


# ============================================================
# FINAL ML FEATURES
# ============================================================

FEATURES = [
    "Transaction_Amount",
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Suspicious_Keyword",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Day_of_Week",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
    "Is_Night",
    "High_Amount_Deviation",
    "International_Night",
    "International_Keyword",
    "Behavioral_Risk_Count",
]


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD FEATURE-ENGINEERED DATA
# ============================================================

print("=" * 70)
print("PHASE 11 - FRAUD PREDICTION & RISK SCORING")
print("=" * 70)

print("\nLoading feature-engineered dataset...")

df = pd.read_csv(FEATURE_DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. VALIDATE FEATURES
# ============================================================

print("\nValidating ML features...")

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required ML features: {missing_features}"
    )

print(f"All {len(FEATURES)} ML features are available.")


# ============================================================
# 4. LOAD FINAL MODEL
# ============================================================

print("\nLoading final XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Final XGBoost model loaded successfully.")


# ============================================================
# 5. LOAD OPTIMAL THRESHOLD
# ============================================================

print("\nLoading optimal classification threshold...")

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)


# Support the threshold key used by the Phase 10 script.
if "optimal_threshold" in threshold_data:
    threshold = float(threshold_data["optimal_threshold"])

elif "best_threshold" in threshold_data:
    threshold = float(threshold_data["best_threshold"])

elif "threshold" in threshold_data:
    threshold = float(threshold_data["threshold"])

else:
    raise KeyError(
        "Could not find the optimal threshold in the threshold JSON file."
    )

print(f"Classification threshold: {threshold:.2f}")


# ============================================================
# 6. PREPARE FEATURES
# ============================================================

print("\nPreparing prediction features...")

X = df[FEATURES].copy()


# ============================================================
# 7. GENERATE FRAUD PROBABILITIES
# ============================================================

print("\nGenerating fraud probabilities...")

fraud_probability = model.predict_proba(X)[:, 1]

print("Fraud probabilities generated successfully.")


# ============================================================
# 8. APPLY FINAL THRESHOLD
# ============================================================

fraud_prediction = (
    fraud_probability >= threshold
).astype(int)


# ============================================================
# 9. CREATE RISK SCORE
# ============================================================

risk_score = fraud_probability * 100


# ============================================================
# 10. CREATE RISK LEVEL
# ============================================================

risk_level = np.select(
    [
        fraud_probability < 0.30,
        fraud_probability < threshold,
    ],
    [
        "Low",
        "Medium",
    ],
    default="High",
)


# ============================================================
# 11. CREATE PREDICTION OUTPUT
# ============================================================

predictions = pd.DataFrame({
    "Transaction_ID": df["Transaction_ID"],
    "Transaction_Date": df["Transaction_Date"],
    "Transaction_Amount": df["Transaction_Amount"],
    "Fraud_Probability": fraud_probability.round(6),
    "Risk_Score": risk_score.round(2),
    "Fraud_Prediction": fraud_prediction,
    "Risk_Level": risk_level,
})


# ============================================================
# 12. ADD BUSINESS-FRIENDLY LABEL
# ============================================================

predictions["Prediction_Label"] = np.where(
    predictions["Fraud_Prediction"] == 1,
    "Fraud",
    "Legitimate",
)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

predictions.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nPrediction file saved successfully:")
print(OUTPUT_PATH)


# ============================================================
# 14. PREDICTION SUMMARY
# ============================================================

total_transactions = len(predictions)

predicted_fraud = int(
    predictions["Fraud_Prediction"].sum()
)

predicted_legitimate = (
    total_transactions - predicted_fraud
)

predicted_fraud_rate = (
    predicted_fraud / total_transactions * 100
)

high_risk_count = int(
    (predictions["Risk_Level"] == "High").sum()
)

medium_risk_count = int(
    (predictions["Risk_Level"] == "Medium").sum()
)

low_risk_count = int(
    (predictions["Risk_Level"] == "Low").sum()
)

total_flagged_amount = predictions.loc[
    predictions["Fraud_Prediction"] == 1,
    "Transaction_Amount"
].sum()

average_fraud_probability = (
    predictions["Fraud_Probability"].mean()
)


# ============================================================
# 15. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FRAUD PREDICTION SUMMARY")
print("=" * 70)

print(f"Total Transactions        : {total_transactions}")
print(f"Predicted Fraud           : {predicted_fraud}")
print(f"Predicted Legitimate      : {predicted_legitimate}")
print(f"Predicted Fraud Rate      : {predicted_fraud_rate:.2f}%")

print("\nRisk Distribution:")
print(f"High Risk                 : {high_risk_count}")
print(f"Medium Risk               : {medium_risk_count}")
print(f"Low Risk                  : {low_risk_count}")

print("\nBusiness Metrics:")
print(
    f"Flagged Transaction Amount: "
    f"{total_flagged_amount:,.2f}"
)

print(
    f"Average Fraud Probability : "
    f"{average_fraud_probability:.4f}"
)


# ============================================================
# 16. CREATE SUMMARY REPORT
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
        "Flagged Transaction Amount",
        "Average Fraud Probability",
        "Classification Threshold",
    ],
    "Value": [
        total_transactions,
        predicted_fraud,
        predicted_legitimate,
        round(predicted_fraud_rate, 2),
        high_risk_count,
        medium_risk_count,
        low_risk_count,
        round(total_flagged_amount, 2),
        round(average_fraud_probability, 6),
        threshold,
    ],
})

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

print("\nSummary report saved:")
print(SUMMARY_PATH)


# ============================================================
# 17. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\nSample Predictions:")
print("-" * 70)

print(
    predictions[
        [
            "Transaction_ID",
            "Transaction_Amount",
            "Fraud_Probability",
            "Risk_Score",
            "Prediction_Label",
            "Risk_Level",
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 18. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 11 COMPLETED SUCCESSFULLY")
print("=" * 70)