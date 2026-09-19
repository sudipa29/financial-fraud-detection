from pathlib import Path

import json
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Business Summary & Decision Support",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
    / "fraud_dashboard_data.csv"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "final_xgboost_threshold.json"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📊 Business Summary & Decision Support")

st.caption(
    "Management-level summary of fraud exposure, model performance, "
    "risk drivers, and recommended business actions."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(
            df["Transaction_Date"],
            errors="coerce"
        )

    numeric_columns = [
        "Transaction_Amount",
        "Fraud_Prediction",
        "Fraud_Probability",
        "Risk_Score",
        "Flagged_Fraud_Amount",
        "Is_International",
        "Fraudulent",
        "Amount_vs_Average_Spend",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        "Unable to load the fraud dashboard dataset."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# LOAD MODEL THRESHOLD
# ============================================================

threshold = 0.69

try:

    if THRESHOLD_PATH.exists():

        with open(
            THRESHOLD_PATH,
            "r"
        ) as f:

            threshold_data = json.load(f)

        if isinstance(threshold_data, dict):

            threshold = float(
                threshold_data.get(
                    "threshold",
                    0.69
                )
            )

        else:

            threshold = float(
                threshold_data
            )

except Exception:

    threshold = 0.69


# ============================================================
# BASIC COUNTS
# ============================================================

total_transactions = len(df)

total_transaction_amount = df[
    "Transaction_Amount"
].sum()


# ============================================================
# ACTUAL FRAUD
# ============================================================

if "Fraudulent" in df.columns:

    actual_fraud_count = int(
        df["Fraudulent"].sum()
    )

    actual_fraud_amount = df.loc[
        df["Fraudulent"] == 1,
        "Transaction_Amount"
    ].sum()

else:

    actual_fraud_count = 0
    actual_fraud_amount = 0


actual_fraud_rate = (
    actual_fraud_count
    / total_transactions
    * 100
    if total_transactions > 0
    else 0
)


# ============================================================
# PREDICTED FRAUD
# ============================================================

predicted_fraud_count = int(
    df["Fraud_Prediction"].sum()
)

predicted_fraud_rate = (
    predicted_fraud_count
    / total_transactions
    * 100
    if total_transactions > 0
    else 0
)


# ============================================================
# FLAGGED AMOUNT
# ============================================================

if "Flagged_Fraud_Amount" in df.columns:

    flagged_amount = df.loc[
        df["Fraud_Prediction"] == 1,
        "Flagged_Fraud_Amount"
    ].sum()

else:

    flagged_amount = df.loc[
        df["Fraud_Prediction"] == 1,
        "Transaction_Amount"
    ].sum()


flagged_amount_percentage = (
    flagged_amount
    / total_transaction_amount
    * 100
    if total_transaction_amount > 0
    else 0
)


# ============================================================
# RISK LEVEL COUNTS
# ============================================================

if "Risk_Level" in df.columns:

    high_risk_count = len(
        df[
            df["Risk_Level"]
            .astype(str)
            .str.upper()
            == "HIGH"
        ]
    )

    medium_risk_count = len(
        df[
            df["Risk_Level"]
            .astype(str)
            .str.upper()
            == "MEDIUM"
        ]
    )

    low_risk_count = len(
        df[
            df["Risk_Level"]
            .astype(str)
            .str.upper()
            == "LOW"
        ]
    )

else:

    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0


# ============================================================
# EXECUTIVE KPI SECTION
# ============================================================

st.subheader("📌 Executive Overview")

col1, col2, col3, col4, col5, col6 = st.columns(6)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Actual Fraud",
        f"{actual_fraud_count:,}"
    )


with col3:

    st.metric(
        "Predicted Fraud",
        f"{predicted_fraud_count:,}"
    )


with col4:

    st.metric(
        "Actual Fraud Rate",
        f"{actual_fraud_rate:.2f}%"
    )


with col5:

    st.metric(
        "Flagged Amount",
        f"₹{flagged_amount:,.2f}"
    )


with col6:

    st.metric(
        "Threshold",
        f"{threshold:.2f}"
    )


# ============================================================
# BUSINESS EXPOSURE
# ============================================================

st.divider()

st.subheader("💰 Fraud Exposure")

exposure_col1, exposure_col2, exposure_col3, exposure_col4 = (
    st.columns(4)
)


with exposure_col1:

    st.metric(
        "Total Transaction Value",
        f"₹{total_transaction_amount:,.2f}"
    )


with exposure_col2:

    st.metric(
        "Actual Fraud Amount",
        f"₹{actual_fraud_amount:,.2f}"
    )


with exposure_col3:

    st.metric(
        "Flagged Transaction Value",
        f"₹{flagged_amount:,.2f}"
    )


with exposure_col4:

    st.metric(
        "Flagged Value %",
        f"{flagged_amount_percentage:.2f}%"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("🤖 Model Effectiveness")


# These are the validated temporal test metrics
accuracy = 0.8040
precision = 0.2907
recall = 0.8523
f1_score = 0.4335
roc_auc = 0.9021
pr_auc = 0.4349


model_col1, model_col2, model_col3, model_col4, model_col5, model_col6 = (
    st.columns(6)
)


with model_col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with model_col2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with model_col3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with model_col4:

    st.metric(
        "F1 Score",
        f"{f1_score * 100:.2f}%"
    )


with model_col5:

    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )


with model_col6:

    st.metric(
        "PR-AUC",
        f"{pr_auc:.4f}"
    )


st.info(
    f"The final XGBoost model uses a tuned decision threshold of "
    f"{threshold:.2f}. The temporal test recall of {recall * 100:.2f}% "
    f"indicates that the model identifies most fraudulent transactions, "
    f"while the lower precision means some legitimate transactions may "
    f"also require review."
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.divider()

risk_left, risk_right = st.columns(2)


# ------------------------------------------------------------
# Risk Level Distribution
# ------------------------------------------------------------

with risk_left:

    st.subheader("⚠️ Current Risk Distribution")

    risk_data = pd.DataFrame(
        {
            "Risk Level": [
                "High",
                "Medium",
                "Low"
            ],
            "Transactions": [
                high_risk_count,
                medium_risk_count,
                low_risk_count
            ]
        }
    )

    fig_risk = px.bar(
        risk_data,
        x="Risk Level",
        y="Transactions",
        text="Transactions",
        title="Transactions by Risk Level"
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Transactions",
        showlegend=False
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


# ------------------------------------------------------------
# Actual vs Predicted
# ------------------------------------------------------------

with risk_right:

    st.subheader("🎯 Actual vs Predicted Fraud")

    comparison_df = pd.DataFrame(
        {
            "Category": [
                "Actual Fraud",
                "Predicted Fraud",
                "Legitimate",
            ],
            "Transactions": [
                actual_fraud_count,
                predicted_fraud_count,
                total_transactions - actual_fraud_count,
            ]
        }
    )

    fig_comparison = px.bar(
        comparison_df,
        x="Category",
        y="Transactions",
        text="Transactions",
        title="Fraud Detection Summary"
    )

    fig_comparison.update_traces(
        textposition="outside"
    )

    fig_comparison.update_layout(
        xaxis_title="Category",
        yaxis_title="Transactions",
        showlegend=False
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )


# ============================================================
# KEY FRAUD DRIVERS
# ============================================================

st.divider()

st.subheader("🔎 Key Fraud Drivers")


driver_col1, driver_col2 = st.columns(2)


# ------------------------------------------------------------
# International Fraud Rate
# ------------------------------------------------------------

with driver_col1:

    st.markdown("### 🌍 International Transactions")

    if "Is_International" in df.columns:

        international_df = df[
            df["Is_International"] == 1
        ]

        domestic_df = df[
            df["Is_International"] == 0
        ]

        international_rate = (
            international_df["Fraudulent"].mean() * 100
            if (
                not international_df.empty
                and "Fraudulent" in df.columns
            )
            else 0
        )

        domestic_rate = (
            domestic_df["Fraudulent"].mean() * 100
            if (
                not domestic_df.empty
                and "Fraudulent" in df.columns
            )
            else 0
        )

        driver_df = pd.DataFrame(
            {
                "Transaction Geography": [
                    "Domestic",
                    "International"
                ],
                "Fraud Rate": [
                    domestic_rate,
                    international_rate
                ]
            }
        )

        fig_geo = px.bar(
            driver_df,
            x="Transaction Geography",
            y="Fraud Rate",
            text=driver_df["Fraud Rate"].round(2),
            title="Actual Fraud Rate by Geography"
        )

        fig_geo.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_geo.update_layout(
            yaxis_title="Fraud Rate (%)",
            showlegend=False
        )

        st.plotly_chart(
            fig_geo,
            use_container_width=True
        )

        st.caption(
            f"International fraud rate: {international_rate:.2f}% "
            f"vs domestic: {domestic_rate:.2f}%."
        )


# ------------------------------------------------------------
# Suspicious Keyword
# ------------------------------------------------------------

with driver_col2:

    st.markdown("### 🚩 Suspicious Keyword")

    if (
        "Suspicious_Keyword" in df.columns
        and "Fraudulent" in df.columns
    ):

        keyword_df = (
            df.groupby("Suspicious_Keyword")[
                "Fraudulent"
            ]
            .mean()
            .mul(100)
            .reset_index()
        )

        keyword_df.columns = [
            "Suspicious Keyword",
            "Fraud Rate"
        ]

        fig_keyword = px.bar(
            keyword_df,
            x="Suspicious Keyword",
            y="Fraud Rate",
            text=keyword_df["Fraud Rate"].round(2),
            title="Fraud Rate by Suspicious Keyword"
        )

        fig_keyword.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_keyword.update_layout(
            yaxis_title="Fraud Rate (%)",
            showlegend=False
        )

        st.plotly_chart(
            fig_keyword,
            use_container_width=True
        )

        st.caption(
            "Transactions containing suspicious keywords "
            "show materially higher observed fraud rates."
        )


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.divider()

st.subheader("💡 Business Recommendations")


recommendation_col1, recommendation_col2 = st.columns(2)


with recommendation_col1:

    st.markdown(
        """
### 🔴 1. Prioritize High-Risk Alerts

Transactions classified as **High Risk** should be sent for
immediate manual investigation.

Recommended actions:

- Verify customer identity.
- Review transaction history.
- Check device and location consistency.
- Apply additional authentication where appropriate.
"""
    )


    st.markdown(
        """
### 🌍 2. Strengthen International Transaction Controls

International transactions demonstrate a substantially higher
fraud risk in the analyzed dataset.

Recommended actions:

- Apply enhanced verification.
- Monitor unusual international activity.
- Compare transaction location with historical behavior.
"""
    )


    st.markdown(
        """
### 🚩 3. Monitor Suspicious Keywords

Suspicious keywords are a strong behavioral indicator in the
dataset.

Recommended actions:

- Create additional transaction-review rules.
- Combine keyword indicators with other risk signals.
- Avoid relying on a single indicator alone.
"""
    )


with recommendation_col2:

    st.markdown(
        """
### 🤖 4. Use the Model as Decision Support

The model achieves high recall, making it useful for identifying
potential fraudulent activity.

However, the lower precision means some legitimate transactions
will also be flagged.

Therefore:

**Model Alert → Analyst Review → Final Decision**
"""
    )


    st.markdown(
        """
### 📊 5. Continuously Monitor Model Performance

Fraud patterns can change over time.

Recommended monitoring:

- Precision
- Recall
- False positives
- False negatives
- Fraud rate
- Alert volume
- Model drift
"""
    )


    st.markdown(
        """
### 🔐 6. Combine Multiple Risk Indicators

The strongest operational strategy is not to depend on one
variable.

Combine:

- International status
- Suspicious keywords
- Transaction timing
- Spending behavior
- Device type
- Payment method
- Historical activity
- Model probability
"""
    )


# ============================================================
# DECISION FRAMEWORK
# ============================================================

st.divider()

st.subheader("🚦 Recommended Fraud Decision Framework")


decision_df = pd.DataFrame(
    {
        "Risk Level": [
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        "Recommended Action": [
            "Immediate Investigation",
            "Enhanced Review",
            "Continue Monitoring"
        ],
        "Business Priority": [
            "Critical",
            "High",
            "Normal"
        ]
    }
)


st.dataframe(
    decision_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FINAL MANAGEMENT MESSAGE
# ============================================================

st.divider()

st.subheader("📌 Management Takeaway")


st.success(
    f"""
The fraud detection system analyzed **{total_transactions:,} transactions**
and identified **{predicted_fraud_count:,} potential fraud alerts**.

The final XGBoost model uses a **{threshold:.2f} decision threshold**
and achieved **{recall * 100:.2f}% recall** on the temporal test set.

The primary business objective should be to prioritize high-risk alerts,
strengthen controls around important fraud indicators, and use model
predictions as decision-support signals for investigators.
"""
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.divider()

st.subheader("🏁 Project Summary")

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.markdown(
        """
### What This Project Delivers

- Data cleaning and preparation
- MySQL-based ETL
- Exploratory data analysis
- Behavioral feature engineering
- Multiple fraud detection algorithms
- Final XGBoost model selection
- Temporal model validation
- Threshold optimization
- Transaction-level fraud prediction
- Risk scoring and classification
- Explainable fraud analysis
- Operational fraud alerts
- Business decision support
"""
    )


with summary_col2:

    st.markdown(
        """
### Technology Stack

**Python**
- Pandas
- NumPy
- Scikit-learn
- XGBoost

**Database**
- MySQL

**Visualization**
- Plotly
- Streamlit

**Analytics**
- Classification
- Feature engineering
- Threshold optimization
- Model evaluation
- Fraud risk analysis
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Financial Fraud Detection Dashboard | "
    "Final XGBoost Model | "
    f"Tuned Decision Threshold = {threshold:.2f}"
)