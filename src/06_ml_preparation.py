from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


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
    / "ml"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PHASE 6 - ML DATASET PREPARATION
# ============================================================

print("=" * 70)
print("PHASE 6 - ML DATASET PREPARATION")
print("=" * 70)


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

print("\n" + "=" * 70)
print("1. LOAD FEATURE-ENGINEERED DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH
)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. VALIDATE TARGET
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET VALIDATION")
print("=" * 70)

TARGET = "Fraudulent"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print("\nTarget variable:")
print(f"- {TARGET}")

print("\nTarget distribution:")
print(
    df[TARGET].value_counts()
    .sort_index()
)

print("\nTarget percentage:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ============================================================
# 3. DEFINE FINAL ML FEATURES
# ============================================================

print("\n" + "=" * 70)
print("3. FINAL ML FEATURE SELECTION")
print("=" * 70)

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

missing_features = [
    feature
    for feature in FINAL_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing ML features: {missing_features}"
    )

X = df[FINAL_FEATURES].copy()

y = df[TARGET].copy()

print(
    f"\nNumber of ML features: {len(FINAL_FEATURES)}"
)

print("\nFinal ML features:")

for feature in FINAL_FEATURES:
    print(f"- {feature}")


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. MISSING VALUE CHECK")
print("=" * 70)

missing_values = X.isnull().sum()

print("\nMissing values in ML features:")

print(
    missing_values[missing_values > 0]
)

if missing_values.sum() == 0:
    print("\nResult: No missing values found.")
else:
    raise ValueError(
        "Missing values detected in ML features."
    )


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE TYPE IDENTIFICATION")
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

print(
    f"\nCategorical features: "
    f"{len(categorical_features)}"
)

for feature in categorical_features:
    print(f"- {feature}")

print(
    f"\nNumerical features: "
    f"{len(numerical_features)}"
)

for feature in numerical_features:
    print(f"- {feature}")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("6. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\nTesting data:")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# 7. CHECK STRATIFICATION
# ============================================================

print("\n" + "=" * 70)
print("7. TRAIN / TEST FRAUD DISTRIBUTION")
print("=" * 70)

print("\nOverall fraud rate:")
print(
    f"{y.mean() * 100:.2f}%"
)

print("\nTraining fraud rate:")
print(
    f"{y_train.mean() * 100:.2f}%"
)

print("\nTesting fraud rate:")
print(
    f"{y_test.mean() * 100:.2f}%"
)

print("\nTraining target distribution:")
print(
    y_train.value_counts()
    .sort_index()
)

print("\nTesting target distribution:")
print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 8. CREATE PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("8. CREATE PREPROCESSING PIPELINE")
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

print("\nPreprocessing pipeline created.")

print("\nNumerical processing:")
print("- StandardScaler")

print("\nCategorical processing:")
print("- OneHotEncoder")
print("- handle_unknown='ignore'")


# ============================================================
# 9. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("9. FIT PREPROCESSOR")
print("=" * 70)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\nPreprocessing completed.")

print(
    f"Processed X_train shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed X_test shape: "
    f"{X_test_processed.shape}"
)


# ============================================================
# 10. SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("10. SAVE ML DATA")
print("=" * 70)

X_train_processed_df = pd.DataFrame(
    X_train_processed
)

X_test_processed_df = pd.DataFrame(
    X_test_processed
)

X_train_processed_df["Fraudulent"] = (
    y_train.reset_index(drop=True)
)

X_test_processed_df["Fraudulent"] = (
    y_test.reset_index(drop=True)
)

train_output = (
    OUTPUT_DIR
    / "train_processed.csv"
)

test_output = (
    OUTPUT_DIR
    / "test_processed.csv"
)

X_train_processed_df.to_csv(
    train_output,
    index=False
)

X_test_processed_df.to_csv(
    test_output,
    index=False
)

print("\nTraining dataset saved:")
print(train_output)

print("\nTesting dataset saved:")
print(test_output)


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("11. FINAL ML DATA VALIDATION")
print("=" * 70)

print(
    f"\nTraining rows: "
    f"{len(X_train_processed_df)}"
)

print(
    f"Testing rows: "
    f"{len(X_test_processed_df)}"
)

print(
    f"Processed feature columns: "
    f"{X_train_processed_df.shape[1] - 1}"
)

print(
    f"Training fraud cases: "
    f"{int(y_train.sum())}"
)

print(
    f"Testing fraud cases: "
    f"{int(y_test.sum())}"
)

print("\nPhase 6 ML dataset preparation completed successfully.")