# Financial Fraud Detection Model

An end-to-end financial fraud detection project using **Python, MySQL, XGBoost, Scikit-learn, Plotly and Streamlit**.

The project focuses on identifying potentially fraudulent financial transactions through data cleaning, MySQL-based ETL, exploratory data analysis, feature engineering, machine learning, fraud-risk prediction, model explainability and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Financial fraud can result in significant financial losses and operational risks for businesses. Detecting suspicious transactions requires more than simply identifying historical fraudulent transactions.

This project develops a complete fraud detection workflow that:

* Processes and cleans transaction data
* Stores and retrieves data using MySQL
* Performs exploratory data analysis
* Creates analytical and behavioral features
* Trains multiple machine learning models
* Evaluates model performance
* Selects an appropriate fraud prediction threshold
* Generates transaction-level fraud risk predictions
* Provides fraud-driver and explainability analysis
* Presents the results through an interactive Streamlit dashboard

---

## 🎯 Business Problem

Financial institutions and businesses process a large number of transactions every day. Manually reviewing every transaction is inefficient and can result in delayed identification of suspicious activity.

The objective of this project is to build a data-driven fraud detection system that can help identify transactions requiring further investigation.

The model is designed as a **decision-support system** rather than an automatic confirmation of fraud.

---

## 🎯 Project Objectives

1. Understand the transaction dataset.
2. Clean and validate the transaction data.
3. Load and process the data using MySQL.
4. Perform exploratory data analysis.
5. Identify patterns associated with fraudulent transactions.
6. Engineer useful behavioral and transaction-level features.
7. Train and compare machine learning models.
8. Evaluate fraud detection performance.
9. Optimize the prediction threshold.
10. Generate transaction-level risk predictions.
11. Analyze potential fraud drivers.
12. Build an interactive Streamlit dashboard.
13. Provide business-oriented fraud monitoring insights.

---

# 📊 Dataset

The project uses a transaction dataset containing **5,000 transactions**.

### Original dataset

The original dataset contains 14 attributes:

| Column                | Description                              |
| --------------------- | ---------------------------------------- |
| Transaction_ID        | Unique transaction identifier            |
| Customer_ID           | Customer identifier                      |
| Transaction_Date      | Transaction timestamp                    |
| Transaction_Amount    | Transaction amount                       |
| Merchant_Category     | Merchant category                        |
| Payment_Method        | Payment method                           |
| Device_Type           | Device used for the transaction          |
| Location              | Transaction location                     |
| Is_International      | Whether the transaction is international |
| Previous_Transactions | Previous transaction count               |
| Average_Spend         | Customer's average spending              |
| Account_Age_Days      | Customer account age                     |
| Suspicious_Keyword    | Suspicious keyword indicator             |
| Fraudulent            | Historical fraud label                   |

The dataset contains:

* **5,000 total transactions**
* **482 fraudulent transactions**
* **4,518 legitimate transactions**
* **9.64% observed fraud rate**

The raw dataset is not included in this repository.

---

# 🛠️ Technology Stack

## Programming

* Python

## Data Analysis

* Pandas
* NumPy

## Database / ETL

* MySQL
* MySQL Workbench
* mysql-connector-python

## Machine Learning

* Scikit-learn
* XGBoost

## Visualization

* Plotly

## Dashboard

* Streamlit

## Development

* Visual Studio Code
* Jupyter Notebook

---

# 🔄 Project Workflow

```text
Raw Transaction Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
MySQL ETL
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Threshold Optimization
        ↓
Fraud Risk Prediction
        ↓
Model Explainability
        ↓
Dashboard Data Preparation
        ↓
Streamlit Dashboard
        ↓
Business Insights
```

---

# 🧹 Data Cleaning

The data preparation stage included:

* Dataset structure inspection
* Data type validation
* Missing-value checking
* Duplicate checking
* Date/time validation
* Numerical feature validation
* Transaction amount validation
* Fraud-label validation
* Behavioral feature preparation

The cleaned dataset was subsequently used for analysis and machine learning.

---

# 🗄️ MySQL ETL

MySQL was used as part of the data engineering and ETL workflow.

The workflow included:

```text
Python Dataset
      ↓
Data Cleaning
      ↓
MySQL Database
      ↓
SQL/Data Retrieval
      ↓
Python EDA & Machine Learning
```

The MySQL layer helped demonstrate practical data extraction and transformation as part of the analytics workflow.

Database credentials are not included in this repository.

---

# 📈 Exploratory Data Analysis

The EDA stage analyzed transaction behavior and fraud patterns.

## Fraud Distribution

| Category   | Transactions |
| ---------- | -----------: |
| Legitimate |        4,518 |
| Fraudulent |          482 |
| Total      |        5,000 |

Observed fraud rate:

**9.64%**

---

## 🌍 International Transactions

The observed fraud rate differed substantially between international and domestic transactions.

| Transaction Type | Observed Fraud Rate |
| ---------------- | ------------------: |
| International    |              36.96% |
| Domestic         |              ~7.00% |

This indicates that international transaction status was an important variable for further fraud-risk analysis.

---

## 🔎 Suspicious Keyword

Transactions containing the suspicious-keyword indicator showed a higher observed fraud rate.

| Suspicious Keyword | Observed Fraud Rate |
| ------------------ | ------------------: |
| Yes                |              47.31% |
| No                 |               7.57% |

---

## 💰 Transaction Amount

Average transaction amount:

| Transaction Type | Average Amount |
| ---------------- | -------------: |
| Legitimate       |          78.88 |
| Fraudulent       |          81.99 |

Transaction amount alone was not treated as sufficient evidence of fraud.

---

# ⚙️ Feature Engineering

Additional analytical features were created to capture transaction timing and behavioral patterns.

Examples include:

* Transaction Hour
* Transaction Day
* Transaction Month
* Day of Week
* Weekend Indicator
* Amount vs Average Spend
* Transactions per Account Age

These features were used to provide additional information to the machine learning models.

---

# 🤖 Machine Learning

Multiple machine learning approaches were considered for fraud classification.

The project included models such as:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

The final model used for the fraud detection workflow was **XGBoost**.

---

# 📊 Model Evaluation

The final XGBoost model achieved:

| Metric                      |     Result |
| --------------------------- | ---------: |
| ROC-AUC                     | **0.9021** |
| Temporal Recall             | **85.23%** |
| Selected Decision Threshold |   **0.69** |

The selected threshold was used to convert predicted fraud probabilities into risk decisions.

Model performance should be interpreted in the context of the dataset, validation methodology and class distribution.

---

# ⚖️ Threshold Optimization

Fraud detection involves a trade-off between:

* Detecting more potentially fraudulent transactions
* Avoiding excessive false positives
* Managing the investigation workload

A decision threshold of **0.69** was selected during the model evaluation process.

The threshold is used to determine which predicted probabilities are sufficiently high to be flagged by the model.

---

# 🚨 Fraud Risk Prediction

The model generates risk-oriented predictions for transactions.

The dashboard categorizes transactions into:

* High Risk
* Medium Risk
* Low Risk

The current dashboard results include:

* **1,284 high/model-risk transactions**
* **438 medium-risk transactions**
* **3,278 low-risk transactions**

These are **model-generated risk categories**, not confirmed fraud labels.

The dataset contains **482 historically labeled fraudulent transactions**.

Therefore:

> A model-flagged transaction should be treated as a transaction requiring investigation, not automatically as confirmed fraud.

---

# 🔍 Fraud Drivers & Explainability

The project also analyzes variables associated with fraud risk.

Important observed patterns include:

* International transaction status
* Suspicious keyword indicator
* Transaction timing
* Spending behavior
* Transaction-to-average-spend relationship

The explainability component is intended to help analysts understand why transactions may receive higher predicted risk.

---

# 📊 Streamlit Dashboard

An interactive Streamlit dashboard was developed to present the analysis and model results.

## Dashboard Pages

### 1. Executive Overview

Provides a high-level summary of:

* Total transactions
* Fraudulent transactions
* Fraud rate
* Transaction value
* Risk indicators
* Key business KPIs

### 2. Transaction & Fraud Analysis

Provides analysis of:

* Transaction trends
* Fraud distribution
* Transaction categories
* Fraud patterns
* Transaction-level metrics

### 3. Fraud Risk Analysis

Focuses on:

* Risk segmentation
* High-risk transactions
* Fraud-risk distribution
* Risk-related transaction patterns

### 4. Model Performance

Displays:

* Model evaluation metrics
* ROC-AUC
* Recall
* Prediction performance
* Threshold information

### 5. Fraud Predictions

Provides transaction-level:

* Predicted risk
* Risk category
* Model-generated fraud indicators
* Investigation-oriented information

### 6. Fraud Drivers

Provides analysis of:

* Fraud-associated variables
* Behavioral patterns
* Feature relationships
* Explainability results

### 7. Fraud Alerts

Provides a monitoring-oriented view of transactions requiring further investigation.

### 8. Business Summary

Converts the analytical and machine-learning results into business-oriented insights and decision-support information.

---

# 🖥️ Dashboard Screenshots

## Executive Overview

![Executive Overview](screenshots/executive_overview.png)

## Transaction & Fraud Analysis

![Transaction & Fraud Analysis](screenshots/transaction_fraud_analysis.png)

## Fraud Risk Analysis

![Fraud Risk Analysis](screenshots/fraud_risk_analysis.png)

## Model Performance

![Model Performance](screenshots/model_performance.png)

## Fraud Predictions

![Fraud Predictions](screenshots/fraud_predictions.png)

## Fraud Drivers

![Fraud Drivers](screenshots/fraud_drivers.png)

## Fraud Alerts

![Fraud Alerts](screenshots/fraud_alerts.png)

## Business Summary

![Business Summary](screenshots/business_summary.png)

---

# 💡 Key Business Insights

The analysis identified several patterns relevant to fraud monitoring.

### 1. International transactions

International transactions showed a substantially higher observed fraud rate than domestic transactions in the project dataset.

### 2. Suspicious keywords

Transactions associated with suspicious keywords showed a substantially higher observed fraud rate.

### 3. Transaction behavior

Behavioral and transaction-level features provide additional information beyond transaction amount alone.

### 4. Risk-based investigation

Instead of manually reviewing every transaction, the model can help prioritize transactions requiring further investigation.

### 5. Decision support

The dashboard combines transaction analytics, model predictions and business insights in one interface.

---

# 📁 Project Structure

```text
financial-fraud-detection/
│
├── Pages/
│   ├── 01_Executive_Overview.py
│   ├── 02_Transaction_Fraud_Analysis.py
│   ├── 03_Fraud_Risk_Analysis.py
│   ├── 04_Model_Performance.py
│   ├── 05_Fraud_Predictions.py
│   ├── 06_Fraud_Drivers.py
│   ├── 07_Fraud_Alerts.py
│   └── 08_Business_Summary.py
│
├── src/
│   ├── Data processing scripts
│   ├── EDA scripts
│   ├── Feature engineering scripts
│   ├── Machine learning scripts
│   ├── Prediction scripts
│   └── Dashboard data preparation
│
├── screenshots/
│   └── Dashboard screenshots
│
├── docs/
│   └── Financial_Fraud_Detection_Project_Report.pdf
│
├── data/
│   └── README.md
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/financial-fraud-detection.git
```

Navigate to the project:

```bash
cd financial-fraud-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 🗄️ Database Configuration

The project uses MySQL for the ETL and data-processing workflow.

Database credentials should be configured locally and should **not** be committed to GitHub.

Example environment variables:

```text
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database
```

---

# ▶️ Running the Dashboard

After installing the dependencies and configuring the required data/database environment:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 🔐 Data & Security

The following are intentionally excluded from the repository:

* Raw transaction dataset
* Database credentials
* `.env` files
* Virtual environment
* Large generated datasets
* Private configuration files
* Model artifacts that are not required for demonstration

This keeps the repository suitable for public portfolio use.

---

# ⚠️ Limitations

* The dataset contains only 5,000 transactions.
* Historical fraud labels may not represent all real-world fraud scenarios.
* Model performance may change on a different dataset or population.
* Model predictions should be reviewed by an appropriate investigation process.
* A model-generated risk flag does not automatically confirm fraud.
* The project is intended as a fraud-risk decision-support prototype.

---

# 🔮 Future Enhancements

Potential future improvements include:

* Larger real-world transaction datasets
* Real-time fraud detection
* Streaming transaction monitoring
* Advanced anomaly detection
* Model retraining pipelines
* Automated alert systems
* SHAP-based explainability
* Model monitoring and drift detection
* API deployment
* Cloud deployment
* Automated investigation workflows

---

# 📄 Project Report

A detailed project report covering the complete project methodology, analysis, machine learning workflow, dashboard and business insights is available in:

```text
docs/Financial_Fraud_Detection_Project_Report.pdf
```

---

# 👩‍💻 Project Type

**Data Analytics + Machine Learning + Business Intelligence**

This project demonstrates an end-to-end workflow combining:

**Python + SQL/MySQL + Machine Learning + Data Visualization + Streamlit + Business Analysis**

---

## ⭐ Key Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* SQL / MySQL
* ETL
* Feature Engineering
* Classification
* XGBoost
* Model Evaluation
* Threshold Optimization
* Fraud Risk Analysis
* Data Visualization
* Streamlit Dashboard Development
* Business Insight Generation
* Data-driven Decision Support
