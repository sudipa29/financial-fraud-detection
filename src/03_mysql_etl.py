from pathlib import Path

import pandas as pd
import mysql.connector
from mysql.connector import Error


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_cleaned.csv"
)


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "financial_fraud_detection",
}


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
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
    "Fraudulent",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Day_of_Week",
    "Is_Weekend",
    "Amount_vs_Average_Spend",
    "Transactions_per_Account_Age",
]

# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("PHASE 3 - MYSQL ETL")
print("=" * 70)

print("\nLoading cleaned dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Shape: {df.shape}")


# ============================================================
# COLUMN VALIDATION
# ============================================================

print("\nValidating columns...")

if list(df.columns) != EXPECTED_COLUMNS:
    print("\nExpected columns:")
    print(EXPECTED_COLUMNS)

    print("\nActual columns:")
    print(list(df.columns))

    raise ValueError(
        "Column validation failed. "
        "CSV columns do not match the expected structure."
    )

print("Column validation: PASSED")


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("\nRunning data quality checks...")

# Missing values
missing_values = df.isnull().sum().sum()

print(f"Missing values: {missing_values}")

if missing_values != 0:
    raise ValueError("Missing values detected in cleaned dataset.")

# Duplicate Transaction IDs
duplicate_ids = df["Transaction_ID"].duplicated().sum()

print(f"Duplicate Transaction_ID values: {duplicate_ids}")

if duplicate_ids != 0:
    raise ValueError("Duplicate Transaction_ID values detected.")

# Fraudulent values
invalid_fraud = ~df["Fraudulent"].isin([0, 1])

print(
    f"Invalid Fraudulent values: "
    f"{invalid_fraud.sum()}"
)

if invalid_fraud.any():
    raise ValueError("Invalid Fraudulent values detected.")

# International values
invalid_international = ~df["Is_International"].isin([0, 1])

print(
    f"Invalid Is_International values: "
    f"{invalid_international.sum()}"
)

if invalid_international.any():
    raise ValueError("Invalid Is_International values detected.")

# Weekend values
invalid_weekend = ~df["Is_Weekend"].isin([0, 1])

print(
    f"Invalid Is_Weekend values: "
    f"{invalid_weekend.sum()}"
)

if invalid_weekend.any():
    raise ValueError("Invalid Is_Weekend values detected.")


print("Data quality validation: PASSED")


# ============================================================
# MYSQL CONNECTION
# ============================================================

print("\nConnecting to MySQL...")

try:

    connection = mysql.connector.connect(**DB_CONFIG)

    if connection.is_connected():

        print("MySQL connection: SUCCESS")

        cursor = connection.cursor()

        # ====================================================
        # INSERT QUERY
        # ====================================================

        insert_query = """
        INSERT INTO transactions (
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
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        # ====================================================
        # PREPARE DATA
        # ====================================================

        data = []

        for _, row in df.iterrows():

            data.append(
                (
                    row["Transaction_ID"],
                    row["Customer_ID"],
                    row["Transaction_Date"],
                    row["Transaction_Amount"],
                    row["Merchant_Category"],
                    row["Payment_Method"],
                    row["Device_Type"],
                    row["Location"],
                    row["Is_International"],
                    row["Previous_Transactions"],
                    row["Average_Spend"],
                    row["Account_Age_Days"],
                    row["Suspicious_Keyword"],
                    row["Fraudulent"],
                    row["Transaction_Hour"],
                    row["Transaction_Day"],
                    row["Transaction_Month"],
                    row["Day_of_Week"],
                    row["Is_Weekend"],
                    row["Amount_vs_Average_Spend"],
                    row["Transactions_per_Account_Age"],
                )
            )

        # ====================================================
        # INSERT DATA
        # ====================================================

        print(f"\nPreparing to insert {len(data)} records...")

        cursor.executemany(insert_query, data)

        connection.commit()

        print(
            f"Records inserted successfully: "
            f"{cursor.rowcount}"
        )

        # ====================================================
        # DATABASE VALIDATION
        # ====================================================

        cursor.execute(
            "SELECT COUNT(*) FROM transactions"
        )

        mysql_row_count = cursor.fetchone()[0]

        print(
            f"MySQL transaction count: "
            f"{mysql_row_count}"
        )

        if mysql_row_count != len(df):

            raise ValueError(
                "Row count mismatch between CSV and MySQL."
            )

        print("\nRow count validation: PASSED")

        cursor.close()
        connection.close()

        print("MySQL connection closed.")

except Error as e:

    print("\nMySQL ERROR:")
    print(e)

    if "connection" in locals() and connection.is_connected():
        connection.close()

    raise


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 ETL COMPLETED")
print("=" * 70)

print(f"CSV records:   {len(df)}")
print(f"MySQL records: {mysql_row_count}")

print("\nCSV → MySQL ETL: SUCCESS")