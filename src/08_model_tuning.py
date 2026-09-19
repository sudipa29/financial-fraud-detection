from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


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
# PHASE 8.1 — LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8 — ADVANCED MODEL TRAINING & THRESHOLD OPTIMIZATION")
print("=" * 70)

print("\n[1] Loading feature-engineered dataset...")

df = pd.read_csv(FEATURE_DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# PHASE 8.2 — DEFINE FEATURES
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

print(f"Number of ML features: {len(FEATURES)}")
print(f"Fraudulent transactions: {y.sum()}")
print(f"Legitimate transactions: {(y == 0).sum()}")
print(f"Fraud rate: {y.mean() * 100:.2f}%")


# ============================================================
# PHASE 8.3 — TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[2] Creating train / validation / test split...")

# First: 80% development data, 20% untouched test data
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Second: split development data into train and validation
# 75% train + 25% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    stratify=y_dev,
    random_state=42,
)

print(f"\nTraining set:   {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set:       {X_test.shape}")

print(
    f"\nFraud rates:"
    f"\n  Train      : {y_train.mean() * 100:.2f}%"
    f"\n  Validation : {y_val.mean() * 100:.2f}%"
    f"\n  Test       : {y_test.mean() * 100:.2f}%"
)


# ============================================================
# PHASE 8.4 — PREPROCESSING
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

preprocessor = ColumnTransformer(
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
# PHASE 8.5 — XGBOOST BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8.5 — XGBOOST BASE MODEL")
print("=" * 70)

fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

print(f"\nXGBoost scale_pos_weight: {fraud_ratio:.4f}")

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=fraud_ratio,
    random_state=42,
    n_jobs=-1,
)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_base),
    ]
)

print("\nTraining XGBoost...")

xgb_pipeline.fit(X_train, y_train)

val_prob = xgb_pipeline.predict_proba(X_val)[:, 1]

print("\nXGBoost Validation Results — threshold = 0.50")

val_pred = (val_prob >= 0.50).astype(int)

print(f"Accuracy : {accuracy_score(y_val, val_pred):.4f}")
print(f"Precision: {precision_score(y_val, val_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_val, val_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_val, val_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, val_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_val, val_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred))


# ============================================================
# PHASE 8.6 — HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8.6 — XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

print("\nRunning RandomizedSearchCV...")
print("Scoring metric: Average Precision (PR-AUC)")

xgb_tuning_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=fraud_ratio,
    random_state=42,
    n_jobs=-1,
)

tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_tuning_model),
    ]
)

param_distributions = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.02, 0.05, 0.08, 0.10],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5, 8],
    "model__gamma": [0, 0.1, 0.3, 0.5],
    "model__reg_alpha": [0, 0.01, 0.1],
    "model__reg_lambda": [1, 2, 5],
}

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

random_search.fit(X_train, y_train)

best_xgb_pipeline = random_search.best_estimator_

print("\nBest XGBoost parameters:")

for parameter, value in random_search.best_params_.items():
    print(f"{parameter}: {value}")

print(
    f"\nBest Cross-Validation PR-AUC: "
    f"{random_search.best_score_:.4f}"
)


# ============================================================
# PHASE 8.7 — VALIDATION PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8.7 — TUNED XGBOOST VALIDATION PERFORMANCE")
print("=" * 70)

val_prob_tuned = best_xgb_pipeline.predict_proba(X_val)[:, 1]

val_pred_tuned = (val_prob_tuned >= 0.50).astype(int)

tuned_accuracy = accuracy_score(y_val, val_pred_tuned)
tuned_precision = precision_score(
    y_val,
    val_pred_tuned,
    zero_division=0,
)
tuned_recall = recall_score(
    y_val,
    val_pred_tuned,
    zero_division=0,
)
tuned_f1 = f1_score(
    y_val,
    val_pred_tuned,
    zero_division=0,
)
tuned_roc_auc = roc_auc_score(
    y_val,
    val_prob_tuned,
)
tuned_pr_auc = average_precision_score(
    y_val,
    val_prob_tuned,
)

print("\nThreshold = 0.50")

print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")
print(f"ROC-AUC  : {tuned_roc_auc:.4f}")
print(f"PR-AUC   : {tuned_pr_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred_tuned))


# ============================================================
# PHASE 8.8 — THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8.8 — FRAUD THRESHOLD OPTIMIZATION")
print("=" * 70)

threshold_results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.02,
)

for threshold in thresholds:

    predictions = (
        val_prob_tuned >= threshold
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
            "Threshold": round(float(threshold), 2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
        }
    )

threshold_df = pd.DataFrame(threshold_results)

threshold_report_path = (
    REPORT_DIR
    / "xgboost_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_report_path,
    index=False,
)

best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_threshold_row["Threshold"]
)

print("\nBest threshold based on Validation F1:")
print(f"Threshold: {best_threshold:.2f}")
print(f"Precision: {best_threshold_row['Precision']:.4f}")
print(f"Recall   : {best_threshold_row['Recall']:.4f}")
print(f"F1 Score : {best_threshold_row['F1']:.4f}")

print(
    f"\nThreshold analysis saved to:"
    f"\n{threshold_report_path}"
)


# ============================================================
# PHASE 8.9 — FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8.9 — FINAL TEST EVALUATION")
print("=" * 70)

print(
    "\nThe test set has remained untouched during "
    "model and threshold selection."
)

test_prob = best_xgb_pipeline.predict_proba(X_test)[:, 1]

# Standard threshold
test_pred_050 = (
    test_prob >= 0.50
).astype(int)

# Optimized threshold
test_pred_optimized = (
    test_prob >= best_threshold
).astype(int)


def calculate_metrics(
    y_true,
    y_pred,
    probabilities,
):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "PR_AUC": average_precision_score(
            y_true,
            probabilities,
        ),
    }


metrics_050 = calculate_metrics(
    y_test,
    test_pred_050,
    test_prob,
)

metrics_optimized = calculate_metrics(
    y_test,
    test_pred_optimized,
    test_prob,
)


print("\nXGBoost Test Performance — Threshold 0.50")

for metric, value in metrics_050.items():
    print(f"{metric:10s}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred_050))


print(
    f"\nXGBoost Test Performance — "
    f"Optimized Threshold {best_threshold:.2f}"
)

for metric, value in metrics_optimized.items():
    print(f"{metric:10s}: {value:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred_optimized,
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred_optimized,
        target_names=[
            "Legitimate",
            "Fraud",
        ],
        zero_division=0,
    )
)


# ============================================================
# PHASE 8.10 — SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8.10 — SAVING MODEL")
print("=" * 70)

import joblib

final_model_path = (
    MODEL_DIR
    / "xgboost_tuned_model.pkl"
)

joblib.dump(
    best_xgb_pipeline,
    final_model_path,
)


# Save threshold separately
threshold_path = (
    MODEL_DIR
    / "xgboost_optimal_threshold.json"
)

with open(
    threshold_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "optimal_threshold": best_threshold,
            "threshold_selection_metric": "F1",
            "threshold_selection_dataset": "validation",
        },
        file,
        indent=4,
    )


# ============================================================
# PHASE 8.11 — SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "Model": "XGBoost",
            "Threshold": 0.50,
            **metrics_050,
        },
        {
            "Model": "XGBoost",
            "Threshold": best_threshold,
            **metrics_optimized,
        },
    ]
)

comparison_path = (
    REPORT_DIR
    / "xgboost_final_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)


print(f"\nModel saved to:")
print(final_model_path)

print(f"\nOptimal threshold saved to:")
print(threshold_path)

print(f"\nComparison saved to:")
print(comparison_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"""
Best XGBoost Validation PR-AUC : {tuned_pr_auc:.4f}

Optimal Validation Threshold   : {best_threshold:.2f}

Final Test PR-AUC              : {metrics_optimized['PR_AUC']:.4f}
Final Test ROC-AUC             : {metrics_optimized['ROC_AUC']:.4f}
Final Test Precision           : {metrics_optimized['Precision']:.4f}
Final Test Recall              : {metrics_optimized['Recall']:.4f}
Final Test F1                  : {metrics_optimized['F1']:.4f}
""")

print("=" * 70)