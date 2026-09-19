from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
# PHASE 9 — TEMPORAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9 — TEMPORAL VALIDATION & FINAL MODEL SELECTION")
print("=" * 70)


# ============================================================
# 9.1 LOAD DATA
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

print(
    f"Date range: "
    f"{df['Transaction_Date'].min()} "
    f"to "
    f"{df['Transaction_Date'].max()}"
)


# ============================================================
# 9.2 DEFINE FEATURES
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
# 9.3 TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[2] Creating chronological split...")

total_rows = len(df)

train_end = int(total_rows * 0.60)
validation_end = int(total_rows * 0.80)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:validation_end].copy()
y_val = y.iloc[train_end:validation_end].copy()

X_test = X.iloc[validation_end:].copy()
y_test = y.iloc[validation_end:].copy()

dates_train = df["Transaction_Date"].iloc[:train_end]
dates_val = df["Transaction_Date"].iloc[
    train_end:validation_end
]
dates_test = df["Transaction_Date"].iloc[
    validation_end:
]

print("\nChronological split:")

print(
    f"\nTraining:"
    f"\n  Rows: {len(X_train)}"
    f"\n  Dates: {dates_train.min()} → {dates_train.max()}"
)

print(
    f"\nValidation:"
    f"\n  Rows: {len(X_val)}"
    f"\n  Dates: {dates_val.min()} → {dates_val.max()}"
)

print(
    f"\nTest:"
    f"\n  Rows: {len(X_test)}"
    f"\n  Dates: {dates_test.min()} → {dates_test.max()}"
)


print("\nFraud distribution:")

print(
    f"Train      : {y_train.sum()} fraud "
    f"({y_train.mean() * 100:.2f}%)"
)

print(
    f"Validation : {y_val.sum()} fraud "
    f"({y_val.mean() * 100:.2f}%)"
)

print(
    f"Test       : {y_test.sum()} fraud "
    f"({y_test.mean() * 100:.2f}%)"
)


# ============================================================
# 9.4 PREPROCESSING
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
# 9.5 CLASS WEIGHT
# ============================================================

fraud_ratio = (
    (y_train == 0).sum()
    / (y_train == 1).sum()
)

print(
    f"\nTraining scale_pos_weight: "
    f"{fraud_ratio:.4f}"
)


# ============================================================
# 9.6 DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
    ),

    "Tuned XGBoost": XGBClassifier(
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
    ),
}


# ============================================================
# 9.7 TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.7 — TEMPORAL MODEL COMPARISON")
print("=" * 70)

results = []

trained_models = {}

for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    trained_models[model_name] = pipeline

    # Validation predictions
    val_prob = pipeline.predict_proba(
        X_val
    )[:, 1]

    val_pred = (
        val_prob >= 0.50
    ).astype(int)

    # Test predictions
    test_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_pred = (
        test_prob >= 0.50
    ).astype(int)

    validation_metrics = {
        "Model": model_name,
        "Dataset": "Validation",
        "Threshold": 0.50,
        "Accuracy": accuracy_score(
            y_val,
            val_pred,
        ),
        "Precision": precision_score(
            y_val,
            val_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_val,
            val_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_val,
            val_pred,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_val,
            val_prob,
        ),
        "PR_AUC": average_precision_score(
            y_val,
            val_prob,
        ),
    }

    test_metrics = {
        "Model": model_name,
        "Dataset": "Test",
        "Threshold": 0.50,
        "Accuracy": accuracy_score(
            y_test,
            test_pred,
        ),
        "Precision": precision_score(
            y_test,
            test_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            test_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            test_pred,
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

    results.append(validation_metrics)
    results.append(test_metrics)

    print(
        f"Validation PR-AUC: "
        f"{validation_metrics['PR_AUC']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['PR_AUC']:.4f}"
    )


comparison_df = pd.DataFrame(results)

comparison_path = (
    REPORT_DIR
    / "temporal_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)

print(
    f"\nTemporal comparison saved to:"
    f"\n{comparison_path}"
)


# ============================================================
# 9.8 DISPLAY TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL TEST RESULTS — THRESHOLD 0.50")
print("=" * 70)

test_results = comparison_df[
    comparison_df["Dataset"] == "Test"
].copy()

test_results = test_results.sort_values(
    "PR_AUC",
    ascending=False,
)

print(
    test_results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 9.9 SELECT CANDIDATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.9 — CANDIDATE MODEL SELECTION")
print("=" * 70)

# Select based on validation PR-AUC
validation_results = comparison_df[
    comparison_df["Dataset"] == "Validation"
].copy()

candidate_row = validation_results.loc[
    validation_results["PR_AUC"].idxmax()
]

candidate_model_name = candidate_row["Model"]

print(
    f"\nCandidate selected using Validation PR-AUC:"
    f"\n{candidate_model_name}"
)

print(
    f"Validation PR-AUC: "
    f"{candidate_row['PR_AUC']:.4f}"
)


# ============================================================
# 9.10 THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.10 — TEMPORAL THRESHOLD OPTIMIZATION")
print("=" * 70)

candidate_model = trained_models[
    candidate_model_name
]

candidate_val_prob = candidate_model.predict_proba(
    X_val
)[:, 1]

threshold_results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.02,
):

    predictions = (
        candidate_val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
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

threshold_path = (
    REPORT_DIR
    / "temporal_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False,
)

best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_threshold_row["Threshold"]
)

print(
    f"\nOptimal validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation Precision: "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Validation Recall: "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['F1']:.4f}"
)


# ============================================================
# 9.11 FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.11 — FINAL TEMPORAL TEST EVALUATION")
print("=" * 70)

candidate_test_prob = candidate_model.predict_proba(
    X_test
)[:, 1]

candidate_test_pred = (
    candidate_test_prob >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    candidate_test_pred,
).ravel()

final_accuracy = accuracy_score(
    y_test,
    candidate_test_pred,
)

final_precision = precision_score(
    y_test,
    candidate_test_pred,
    zero_division=0,
)

final_recall = recall_score(
    y_test,
    candidate_test_pred,
    zero_division=0,
)

final_f1 = f1_score(
    y_test,
    candidate_test_pred,
    zero_division=0,
)

final_roc_auc = roc_auc_score(
    y_test,
    candidate_test_prob,
)

final_pr_auc = average_precision_score(
    y_test,
    candidate_test_prob,
)


print(
    f"\nFinal Model: {candidate_model_name}"
)

print(
    f"Final Threshold: {best_threshold:.2f}"
)

print(
    f"\nAccuracy : {final_accuracy:.4f}"
)

print(
    f"Precision: {final_precision:.4f}"
)

print(
    f"Recall   : {final_recall:.4f}"
)

print(
    f"F1 Score : {final_f1:.4f}"
)

print(
    f"ROC-AUC  : {final_roc_auc:.4f}"
)

print(
    f"PR-AUC   : {final_pr_auc:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        candidate_test_pred,
    )
)

print("\nFraud Detection Summary:")

print(
    f"Actual fraud cases : {tp + fn}"
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


# ============================================================
# 9.12 SAVE FINAL TEMPORAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.12 — SAVING FINAL TEMPORAL MODEL")
print("=" * 70)

final_model_path = (
    MODEL_DIR
    / "final_fraud_detection_model.pkl"
)

joblib.dump(
    candidate_model,
    final_model_path,
)

final_threshold_path = (
    MODEL_DIR
    / "final_fraud_threshold.json"
)

with open(
    final_threshold_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "model": candidate_model_name,
            "threshold": best_threshold,
            "selection_metric": "Validation PR-AUC + Validation F1 threshold",
            "validation_pr_auc": float(
                candidate_row["PR_AUC"]
            ),
            "test_pr_auc": float(
                final_pr_auc
            ),
            "test_roc_auc": float(
                final_roc_auc
            ),
            "test_precision": float(
                final_precision
            ),
            "test_recall": float(
                final_recall
            ),
            "test_f1": float(
                final_f1
            ),
        },
        file,
        indent=4,
    )


# ============================================================
# 9.13 FINAL SUMMARY
# ============================================================

print("\nFinal model saved to:")
print(final_model_path)

print("\nFinal threshold saved to:")
print(final_threshold_path)

print("\n" + "=" * 70)
print("PHASE 9 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"""
Final Candidate Model : {candidate_model_name}
Optimal Threshold     : {best_threshold:.2f}

Temporal Test PR-AUC  : {final_pr_auc:.4f}
Temporal Test ROC-AUC : {final_roc_auc:.4f}
Temporal Test Recall  : {final_recall:.4f}
Temporal Test Precision: {final_precision:.4f}
Temporal Test F1      : {final_f1:.4f}
"""
)

print("=" * 70)