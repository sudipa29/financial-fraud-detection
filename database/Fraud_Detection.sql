CREATE DATABASE IF NOT EXISTS financial_fraud_detection;

USE financial_fraud_detection;

SELECT DATABASE();

CREATE TABLE transactions (
    Transaction_ID VARCHAR(50) NOT NULL,
    Customer_ID VARCHAR(50) NOT NULL,
    Transaction_Date DATETIME NOT NULL,
    Transaction_Amount DECIMAL(12,2) NOT NULL,
    Merchant_Category VARCHAR(50) NOT NULL,
    Payment_Method VARCHAR(50) NOT NULL,
    Device_Type VARCHAR(50) NOT NULL,
    Location VARCHAR(50) NOT NULL,
    Is_International TINYINT NOT NULL,
    Previous_Transactions INT NOT NULL,
    Average_Spend DECIMAL(12,2) NOT NULL,
    Account_Age_Days INT NOT NULL,
    Suspicious_Keyword VARCHAR(10) NOT NULL,
    Fraudulent TINYINT NOT NULL,

    Transaction_Hour TINYINT NOT NULL,
    Transaction_Day TINYINT NOT NULL,
    Transaction_Month TINYINT NOT NULL,
    Day_of_Week VARCHAR(20) NOT NULL,
    Is_Weekend TINYINT NOT NULL,

    Amount_vs_Average_Spend DECIMAL(12,6) NOT NULL,
    Transactions_per_Account_Age DECIMAL(12,6) NOT NULL,

    PRIMARY KEY (Transaction_ID),

    CHECK (Is_International IN (0,1)),
    CHECK (Fraudulent IN (0,1)),
    CHECK (Is_Weekend IN (0,1)),
    CHECK (Transaction_Amount >= 0),
    CHECK (Previous_Transactions >= 0),
    CHECK (Average_Spend >= 0),
    CHECK (Account_Age_Days >= 0)
);

USE financial_fraud_detection;

SHOW TABLES;
DESCRIBE transactions;
SELECT COUNT(*) AS row_count
FROM transactions;

SELECT COUNT(*) AS total_transactions
FROM transactions; /* Confirm total records */

SELECT 
    Transaction_ID,
    COUNT(*) AS duplicate_count
FROM transactions
GROUP BY Transaction_ID
HAVING COUNT(*) > 1;  /* Check duplicate Transaction IDs */

SELECT
    SUM(Transaction_ID IS NULL) AS Transaction_ID_NULLS,
    SUM(Customer_ID IS NULL) AS Customer_ID_NULLS,
    SUM(Transaction_Date IS NULL) AS Transaction_Date_NULLS,
    SUM(Transaction_Amount IS NULL) AS Transaction_Amount_NULLS,
    SUM(Fraudulent IS NULL) AS Fraudulent_NULLS
FROM transactions; /* Check NULL values */

SELECT
    Fraudulent,
    COUNT(*) AS transaction_count,
    ROUND(
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions),
        2
    ) AS percentage
FROM transactions
GROUP BY Fraudulent
ORDER BY Fraudulent; /* Check fraud distribution */

SELECT
    Is_International,
    COUNT(*) AS transaction_count,
    SUM(Fraudulent) AS fraud_count,
    ROUND(
        AVG(Fraudulent) * 100,
        2
    ) AS fraud_rate_pct
FROM transactions
GROUP BY Is_International
ORDER BY Is_International; /* Check international transactions */

SELECT
    COUNT(*) AS zero_amount_transactions,
    SUM(Fraudulent) AS fraudulent_zero_amount_transactions
FROM transactions
WHERE Transaction_Amount = 0; /* Check zero-amount transactions */

SELECT
    MIN(Amount_vs_Average_Spend) AS min_amount_ratio,
    MAX(Amount_vs_Average_Spend) AS max_amount_ratio,
    MIN(Transactions_per_Account_Age) AS min_transaction_rate,
    MAX(Transactions_per_Account_Age) AS max_transaction_rate
FROM transactions; /* Check behavioral features */

SELECT
    MIN(Transaction_Date) AS earliest_transaction,
    MAX(Transaction_Date) AS latest_transaction
FROM transactions; /* Check date range */


