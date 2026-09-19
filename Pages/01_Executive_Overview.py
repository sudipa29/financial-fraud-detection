from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
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


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"]
    )

    df["Transaction_Date_Only"] = pd.to_datetime(
        df["Transaction_Date_Only"]
    )

    return df


df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Fraud Detection")

st.sidebar.markdown(
    """
    **Financial Fraud Detection System**

    Machine Learning powered transaction
    risk monitoring and investigation.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")


# ============================================================
# DATE FILTER
# ============================================================

min_date = df["Transaction_Date"].min().date()

max_date = df["Transaction_Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Transaction Date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)


if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = pd.Timestamp(
        selected_dates[1]
    ) + pd.Timedelta(days=1)

    filtered_df = df[
        (
            df["Transaction_Date"] >= start_date
        )
        &
        (
            df["Transaction_Date"] < end_date
        )
    ].copy()

else:

    filtered_df = df.copy()


# ============================================================
# TRANSACTION TYPE FILTER
# ============================================================

transaction_types = sorted(
    df["Transaction_Type"]
    .dropna()
    .unique()
)


selected_transaction_type = st.sidebar.multiselect(
    "Transaction Type",
    options=transaction_types,
    default=transaction_types,
)


filtered_df = filtered_df[
    filtered_df["Transaction_Type"].isin(
        selected_transaction_type
    )
]


# ============================================================
# RISK LEVEL FILTER
# ============================================================

risk_levels = [
    "High",
    "Medium",
    "Low",
]


selected_risk_levels = st.sidebar.multiselect(
    "Risk Level",
    options=risk_levels,
    default=risk_levels,
)


filtered_df = filtered_df[
    filtered_df["Risk_Level"].isin(
        selected_risk_levels
    )
]


# ============================================================
# FRAUD STATUS FILTER
# ============================================================

fraud_status_options = [
    "Predicted Fraud",
    "Predicted Legitimate",
]


selected_fraud_status = st.sidebar.multiselect(
    "Prediction Status",
    options=fraud_status_options,
    default=fraud_status_options,
)


filtered_df = filtered_df[
    filtered_df["Fraud_Status"].isin(
        selected_fraud_status
    )
]


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🛡️ Financial Fraud Detection Dashboard"
)

st.markdown(
    """
    **Machine Learning Based Transaction Risk Monitoring**
    """
)

st.markdown("---")


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )

    st.stop()


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

total_transactions = len(
    filtered_df
)

# ------------------------------------------------------------
# ACTUAL FRAUD
# ------------------------------------------------------------

actual_fraud = int(
    filtered_df["Fraudulent"].sum()
)

actual_legitimate = (
    total_transactions
    - actual_fraud
)

actual_fraud_rate = (
    actual_fraud
    / total_transactions
    * 100
)


# ------------------------------------------------------------
# MODEL PREDICTIONS
# ------------------------------------------------------------

predicted_fraud = int(
    filtered_df["Fraud_Prediction"].sum()
)

predicted_legitimate = (
    total_transactions
    - predicted_fraud
)

predicted_fraud_rate = (
    predicted_fraud
    / total_transactions
    * 100
)


# ------------------------------------------------------------
# RISK LEVELS
# ------------------------------------------------------------

high_risk = int(
    (
        filtered_df["Risk_Level"]
        == "High"
    ).sum()
)

medium_risk = int(
    (
        filtered_df["Risk_Level"]
        == "Medium"
    ).sum()
)

low_risk = int(
    (
        filtered_df["Risk_Level"]
        == "Low"
    ).sum()
)


# ------------------------------------------------------------
# TRANSACTION AMOUNTS
# ------------------------------------------------------------

transaction_amount = (
    filtered_df[
        "Transaction_Amount"
    ].sum()
)

model_flagged_amount = (
    filtered_df[
        "Flagged_Fraud_Amount"
    ].sum()
)


# ============================================================
# KPI CARDS — ROW 1
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}",
    )


with col2:

    st.metric(
        "Actual Fraudulent Transactions",
        f"{actual_fraud:,}",
    )


with col3:

    st.metric(
        "Model-Flagged Transactions",
        f"{predicted_fraud:,}",
    )


with col4:

    st.metric(
        "Model-Flagged Amount",
        f"{model_flagged_amount:,.2f}",
    )


st.markdown("")


# ============================================================
# KPI CARDS — ROW 2
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Actual Fraud Rate",
        f"{actual_fraud_rate:.2f}%",
    )


with col2:

    st.metric(
        "Model Flag Rate",
        f"{predicted_fraud_rate:.2f}%",
    )


with col3:

    st.metric(
        "High Risk Transactions",
        f"{high_risk:,}",
    )


with col4:

    st.metric(
        "Total Transaction Amount",
        f"{transaction_amount:,.2f}",
    )


st.markdown("---")

# ============================================================
# RISK DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Risk Level Distribution"
    )

    risk_data = (
        filtered_df[
            "Risk_Level"
        ]
        .value_counts()
        .reindex(
            ["High", "Medium", "Low"],
            fill_value=0,
        )
        .reset_index()
    )

    risk_data.columns = [
        "Risk_Level",
        "Transactions",
    ]

    fig_risk = px.bar(
    risk_data,
    x="Risk_Level",
    y="Transactions",
    title="Model-Assigned Risk Level Distribution",
    text="Transactions",
)
    
    fig_risk.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Number of Transactions",
        showlegend=False,
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )


# ============================================================
# TRANSACTION TYPE
# ============================================================

with col2:

    st.subheader(
        "Domestic vs International"
    )

    transaction_type_data = (
        filtered_df[
            "Transaction_Type"
        ]
        .value_counts()
        .reset_index()
    )

    transaction_type_data.columns = [
        "Transaction_Type",
        "Transactions",
    ]

    fig_type = px.pie(
        transaction_type_data,
        names="Transaction_Type",
        values="Transactions",
        title="Transaction Type Distribution",
        hole=0.45,
    )

    st.plotly_chart(
        fig_type,
        use_container_width=True,
    )


# ============================================================
# MONTHLY TRANSACTION TREND
# ============================================================

st.subheader(
    "Monthly Transaction Trend"
)

monthly_data = (
    filtered_df
    .groupby(
        "Transaction_Year_Month"
    )
    .agg(
        Transactions=(
            "Transaction_ID",
            "count",
        ),
        Model_Flagged=(
            "Fraud_Prediction",
            "sum",
        ),
        Transaction_Amount=(
            "Transaction_Amount",
            "sum",
        ),
    )
    .reset_index()
)


fig_monthly = px.line(
    monthly_data,
    x="Transaction_Year_Month",
    y=[
        "Transactions",
        "Model_Flagged",
    ],
    markers=True,
    title="Monthly Transaction and Model-Flagged Transaction Trend",
)


fig_monthly.update_layout(
    xaxis_title="Month",
    yaxis_title="Number of Transactions",
    legend_title="Metric",
)


st.plotly_chart(
    fig_monthly,
    use_container_width=True,
)


# ============================================================
# TOP RISK REASONS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Top Risk Reasons"
    )

    reason_data = (
        filtered_df[
            "Primary_Risk_Reason"
        ]
        .value_counts()
        .head(8)
        .reset_index()
    )

    reason_data.columns = [
        "Risk_Reason",
        "Transactions",
    ]

    fig_reason = px.bar(
        reason_data,
        x="Transactions",
        y="Risk_Reason",
        orientation="h",
        title="Most Common Risk Indicators",
        text="Transactions",
    )

    fig_reason.update_layout(
        yaxis_title="",
        xaxis_title="Transactions",
    )

    st.plotly_chart(
        fig_reason,
        use_container_width=True,
    )


# ============================================================
# HIGH RISK TRANSACTIONS
# ============================================================

with col2:

    st.subheader(
        "Highest Risk Transactions"
    )

    high_risk_transactions = (
        filtered_df[
            filtered_df["Risk_Level"]
            == "High"
        ]
        .sort_values(
            "Fraud_Probability",
            ascending=False,
        )
        [
            [
                "Transaction_ID",
                "Transaction_Date",
                "Transaction_Amount",
                "Risk_Score",
                "Primary_Risk_Reason",
            ]
        ]
        .head(10)
    )

    st.dataframe(
        high_risk_transactions,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Financial Fraud Detection System | "
    "Tuned XGBoost Model | "
    "Threshold = 0.69"
)