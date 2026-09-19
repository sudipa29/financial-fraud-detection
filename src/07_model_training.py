from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_transactions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PHASE 7 - MODEL TRAINING & EVALUATION
# ============================================================

print("=" * 70)
print("PHASE 7 - FRAUD DETECTION MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("1. LOAD FEATURE-ENGINEERED DATA")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("2. DEFINE ML FEATURES")
print("=" * 70)

TARGET = "Fraudulent"

FINAL_FEATURES = [
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
    "Behavioral_Risk_Count"
]

X = df[FINAL_FEATURES].copy()
y = df[TARGET].copy()

print(f"\nNumber of features: {len(FINAL_FEATURES)}")
print(f"Target: {TARGET}")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("3. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

print(
    f"Training fraud rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Testing fraud rate: "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# 4. DEFINE FEATURE TYPES
# ============================================================

print("\n" + "=" * 70)
print("4. DEFINE FEATURE TYPES")
print("=" * 70)

categorical_features = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Suspicious_Keyword",
    "Day_of_Week"
]

numerical_features = [
    feature
    for feature in FINAL_FEATURES
    if feature not in categorical_features
]

print(f"\nCategorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")


# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("5. CREATE PREPROCESSING PIPELINE")
print("=" * 70)

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("\nPreprocessing pipeline ready.")


# ============================================================
# 6. DEFINE MODELS
# ============================================================

print("\n" + "=" * 70)
print("6. DEFINE BASELINE MODELS")
print("=" * 70)

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    )
}

for model_name in models:
    print(f"- {model_name}")


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("7. MODEL TRAINING & EVALUATION")
print("=" * 70)

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = pipeline.predict(
        X_test
    )

    # Probability predictions
    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        }
    )

    trained_models[model_name] = pipeline


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("8. MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR_AUC",
    ascending=False
).reset_index(drop=True)

print("\n")
print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. IDENTIFY BEST BASELINE MODEL
# ============================================================

print("\n" + "=" * 70)
print("9. BEST BASELINE MODEL")
print("=" * 70)

best_model_name = results_df.iloc[0]["Model"]

print(
    f"\nBest baseline model based on PR-AUC: "
    f"{best_model_name}"
)


# ============================================================
# 10. SAVE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("10. SAVE MODEL RESULTS")
print("=" * 70)

results_path = (
    REPORT_DIR
    / "model_comparison_baseline.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel comparison saved:")
print(results_path)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("11. SAVE BEST BASELINE MODEL")
print("=" * 70)

import joblib

best_model_path = (
    OUTPUT_DIR
    / "best_baseline_model.pkl"
)

joblib.dump(
    trained_models[best_model_name],
    best_model_path
)

print("\nBest model saved:")
print(best_model_path)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 7 BASELINE MODELING COMPLETED")
print("=" * 70)

print(
    f"\nModels trained: {len(models)}"
)

print(
    f"Best baseline model: {best_model_name}"
)

print(
    "\nNext step: Model tuning and threshold optimization."
)