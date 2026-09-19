from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_transactions.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PHASE 10 — FINAL MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 10 — FINAL XGBOOST MODEL & THRESHOLD SELECTION")
print("=" * 70)


# ============================================================
# 10.1 LOAD DATA
# ============================================================

print("\n[1] Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"]
)

df = df.sort_values(
    "Transaction_Date"
).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 10.2 FEATURES
# ============================================================

TARGET = "Fraudulent"

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

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# 10.3 CHRONOLOGICAL SPLIT
# ============================================================

print("\n[2] Creating chronological train / validation / test split...")

total_rows = len(df)

train_end = int(total_rows * 0.60)
validation_end = int(total_rows * 0.80)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:validation_end].copy()
y_val = y.iloc[train_end:validation_end].copy()

X_test = X.iloc[validation_end:].copy()
y_test = y.iloc[validation_end:].copy()

print(
    f"\nTraining   : {len(X_train)} rows"
    f"\nValidation : {len(X_val)} rows"
    f"\nTest       : {len(X_test)} rows"
)

print(
    f"\nTrain dates:"
    f"\n{df['Transaction_Date'].iloc[:train_end].min()}"
    f" → "
    f"{df['Transaction_Date'].iloc[:train_end].max()}"
)

print(
    f"\nValidation dates:"
    f"\n{df['Transaction_Date'].iloc[train_end:validation_end].min()}"
    f" → "
    f"{df['Transaction_Date'].iloc[train_end:validation_end].max()}"
)

print(
    f"\nTest dates:"
    f"\n{df['Transaction_Date'].iloc[validation_end:].min()}"
    f" → "
    f"{df['Transaction_Date'].iloc[validation_end:].max()}"
)


# ============================================================
# 10.4 PREPROCESSING
# ============================================================

categorical_features = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Suspicious_Keyword",
    "Day_of_Week",
]

numerical_features = [
    feature
    for feature in FEATURES
    if feature not in categorical_features
]


def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )


# ============================================================
# 10.5 TUNED XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("PHASE 10.5 — TRAINING TUNED XGBOOST")
print("=" * 70)

fraud_ratio = (
    (y_train == 0).sum()
    / (y_train == 1).sum()
)

print(
    f"\nTraining scale_pos_weight: "
    f"{fraud_ratio:.4f}"
)

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.5,
    reg_alpha=0.01,
    reg_lambda=5,
    scale_pos_weight=fraud_ratio,
    random_state=42,
    n_jobs=-1,
)

xgb_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor(),
        ),
        (
            "model",
            xgb_model,
        ),
    ]
)

print("\nTraining XGBoost...")

xgb_pipeline.fit(
    X_train,
    y_train,
)

print("Training completed.")


# ============================================================
# 10.6 VALIDATION PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("PHASE 10.6 — VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 70)

val_prob = xgb_pipeline.predict_proba(
    X_val
)[:, 1]

print(
    f"\nValidation ROC-AUC: "
    f"{roc_auc_score(y_val, val_prob):.4f}"
)

print(
    f"Validation PR-AUC: "
    f"{average_precision_score(y_val, val_prob):.4f}"
)


# ============================================================
# 10.7 THRESHOLD SEARCH
# ============================================================

threshold_results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.01,
):

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_pred,
    ).ravel()

    threshold_results.append(
        {
            "Threshold": round(
                float(threshold),
                2,
            ),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_report_path = (
    REPORT_DIR
    / "final_xgboost_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_report_path,
    index=False,
)


# ============================================================
# 10.8 BEST THRESHOLD
# ============================================================

best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_threshold_row["Threshold"]
)

print("\nBest threshold based on validation F1:")

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Precision : "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_threshold_row['F1']:.4f}"
)

print(
    f"\nThreshold report saved to:"
    f"\n{threshold_report_path}"
)


# ============================================================
# 10.9 TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 10.9 — FINAL TEMPORAL TEST EVALUATION")
print("=" * 70)

print(
    "\nIMPORTANT:"
    "\nThe test set was not used for threshold selection."
)

test_prob = xgb_pipeline.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# Threshold 0.50
# ------------------------------------------------------------

test_pred_050 = (
    test_prob >= 0.50
).astype(int)

metrics_050 = {
    "Accuracy": accuracy_score(
        y_test,
        test_pred_050,
    ),
    "Precision": precision_score(
        y_test,
        test_pred_050,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_test,
        test_pred_050,
        zero_division=0,
    ),
    "F1": f1_score(
        y_test,
        test_pred_050,
        zero_division=0,
    ),
    "ROC_AUC": roc_auc_score(
        y_test,
        test_prob,
    ),
    "PR_AUC": average_precision_score(
        y_test,
        test_prob,
    ),
}


print("\nXGBoost — Threshold 0.50")

for metric, value in metrics_050.items():
    print(
        f"{metric:10s}: {value:.4f}"
    )

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred_050,
    )
)


# ------------------------------------------------------------
# Optimized threshold
# ------------------------------------------------------------

test_pred_final = (
    test_prob >= best_threshold
).astype(int)

metrics_final = {
    "Accuracy": accuracy_score(
        y_test,
        test_pred_final,
    ),
    "Precision": precision_score(
        y_test,
        test_pred_final,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_test,
        test_pred_final,
        zero_division=0,
    ),
    "F1": f1_score(
        y_test,
        test_pred_final,
        zero_division=0,
    ),
    "ROC_AUC": roc_auc_score(
        y_test,
        test_prob,
    ),
    "PR_AUC": average_precision_score(
        y_test,
        test_prob,
    ),
}


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred_final,
).ravel()


print(
    f"\nXGBoost — Optimized Threshold "
    f"{best_threshold:.2f}"
)

for metric, value in metrics_final.items():
    print(
        f"{metric:10s}: {value:.4f}"
    )

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred_final,
    )
)


# ============================================================
# 10.10 FRAUD DETECTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 10.10 — FRAUD DETECTION SUMMARY")
print("=" * 70)

actual_fraud = int(
    y_test.sum()
)

print(
    f"\nActual fraud cases : {actual_fraud}"
)

print(
    f"Fraud detected     : {tp}"
)

print(
    f"Fraud missed       : {fn}"
)

print(
    f"False alarms       : {fp}"
)

print(
    f"Fraud detection rate: "
    f"{(tp / actual_fraud) * 100:.2f}%"
)


# ============================================================
# 10.11 SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 10.11 — SAVING FINAL MODEL")
print("=" * 70)

final_model_path = (
    MODEL_DIR
    / "final_xgboost_fraud_model.pkl"
)

joblib.dump(
    xgb_pipeline,
    final_model_path,
)

final_threshold_path = (
    MODEL_DIR
    / "final_xgboost_threshold.json"
)

with open(
    final_threshold_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "model": "Tuned XGBoost",
            "threshold": best_threshold,
            "threshold_selection": "Validation F1",
            "validation_roc_auc": float(
                roc_auc_score(
                    y_val,
                    val_prob,
                )
            ),
            "validation_pr_auc": float(
                average_precision_score(
                    y_val,
                    val_prob,
                )
            ),
            "test_roc_auc": float(
                metrics_final["ROC_AUC"]
            ),
            "test_pr_auc": float(
                metrics_final["PR_AUC"]
            ),
            "test_precision": float(
                metrics_final["Precision"]
            ),
            "test_recall": float(
                metrics_final["Recall"]
            ),
            "test_f1": float(
                metrics_final["F1"]
            ),
        },
        file,
        indent=4,
    )


# ============================================================
# 10.12 SAVE FINAL COMPARISON
# ============================================================

final_comparison = pd.DataFrame(
    [
        {
            "Model": "Tuned XGBoost",
            "Threshold": 0.50,
            **metrics_050,
        },
        {
            "Model": "Tuned XGBoost",
            "Threshold": best_threshold,
            **metrics_final,
        },
    ]
)

comparison_path = (
    REPORT_DIR
    / "final_xgboost_comparison.csv"
)

final_comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nFinal XGBoost model saved to:")
print(final_model_path)

print("\nFinal threshold saved to:")
print(final_threshold_path)

print("\nFinal comparison saved to:")
print(comparison_path)

print("\n" + "=" * 70)
print("PHASE 10 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"""
FINAL MODEL              : Tuned XGBoost
FINAL THRESHOLD          : {best_threshold:.2f}

TEMPORAL TEST ROC-AUC    : {metrics_final['ROC_AUC']:.4f}
TEMPORAL TEST PR-AUC     : {metrics_final['PR_AUC']:.4f}
TEMPORAL TEST PRECISION  : {metrics_final['Precision']:.4f}
TEMPORAL TEST RECALL     : {metrics_final['Recall']:.4f}
TEMPORAL TEST F1         : {metrics_final['F1']:.4f}

FRAUD DETECTED           : {tp}
FRAUD MISSED             : {fn}
FALSE ALARMS             : {fp}
"""
)

print("=" * 70)