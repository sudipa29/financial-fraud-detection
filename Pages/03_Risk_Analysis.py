
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Risk Analysis",
    page_icon="⚠️",
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


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"]
    )

    return df


df = load_data()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚠️ Risk Analysis")

st.markdown(
    """
    Analyze transaction risk levels, fraud probabilities,
    risk scores, and the characteristics of transactions
    requiring greater investigation attention.
    """
)

st.markdown("---")


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔎 Risk Filters")


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


if (
    isinstance(selected_dates, tuple)
    and len(selected_dates) == 2
):

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = (
        pd.Timestamp(selected_dates[1])
        + pd.Timedelta(days=1)
    )

    filtered_df = df[
        (
            df["Transaction_Date"]
            >= start_date
        )
        &
        (
            df["Transaction_Date"]
            < end_date
        )
    ].copy()

else:

    filtered_df = df.copy()


# ============================================================
# RISK LEVEL FILTER
# ============================================================

risk_levels = [
    "High",
    "Medium",
    "Low",
]

available_risk_levels = [
    level
    for level in risk_levels
    if level in df["Risk_Level"].dropna().unique()
]

selected_risk_levels = st.sidebar.multiselect(
    "Risk Level",
    options=available_risk_levels,
    default=available_risk_levels,
)


filtered_df = filtered_df[
    filtered_df["Risk_Level"].isin(
        selected_risk_levels
    )
]


# ============================================================
# TRANSACTION TYPE FILTER
# ============================================================

transaction_types = sorted(
    df["Transaction_Type"]
    .dropna()
    .unique()
)

selected_transaction_types = (
    st.sidebar.multiselect(
        "Transaction Type",
        options=transaction_types,
        default=transaction_types,
    )
)


filtered_df = filtered_df[
    filtered_df["Transaction_Type"].isin(
        selected_transaction_types
    )
]


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_transactions = len(
    filtered_df
)

high_risk_transactions = int(
    (
        filtered_df["Risk_Level"]
        == "High"
    ).sum()
)

medium_risk_transactions = int(
    (
        filtered_df["Risk_Level"]
        == "Medium"
    ).sum()
)

low_risk_transactions = int(
    (
        filtered_df["Risk_Level"]
        == "Low"
    ).sum()
)

high_risk_amount = (
    filtered_df.loc[
        filtered_df["Risk_Level"] == "High",
        "Transaction_Amount",
    ].sum()
)

average_risk_score = (
    filtered_df["Risk_Score"].mean()
)

maximum_risk_score = (
    filtered_df["Risk_Score"].max()
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "High Risk Transactions",
        f"{high_risk_transactions:,}",
    )


with col2:

    st.metric(
        "High Risk Amount",
        f"{high_risk_amount:,.2f}",
    )


with col3:

    st.metric(
        "Average Risk Score",
        f"{average_risk_score:.2f}",
    )


with col4:

    st.metric(
        "Maximum Risk Score",
        f"{maximum_risk_score:.2f}",
    )


st.markdown("")


# ============================================================
# SECOND KPI ROW
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}",
    )


with col2:

    st.metric(
        "High Risk",
        f"{high_risk_transactions:,}",
    )


with col3:

    st.metric(
        "Medium Risk",
        f"{medium_risk_transactions:,}",
    )


with col4:

    st.metric(
        "Low Risk",
        f"{low_risk_transactions:,}",
    )


st.markdown("---")


# ============================================================
# RISK LEVEL DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Risk Level Distribution"
    )

    risk_distribution = (
        filtered_df["Risk_Level"]
        .value_counts()
        .reindex(
            ["High", "Medium", "Low"],
            fill_value=0,
        )
        .reset_index()
    )

    risk_distribution.columns = [
        "Risk_Level",
        "Transactions",
    ]

    fig_risk = px.bar(
        risk_distribution,
        x="Risk_Level",
        y="Transactions",
        text="Transactions",
        title="Transactions by Risk Level",
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Transactions",
        showlegend=False,
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )


# ============================================================
# RISK SCORE DISTRIBUTION
# ============================================================

with col2:

    st.subheader(
        "Risk Score Distribution"
    )

    fig_score = px.histogram(
        filtered_df,
        x="Risk_Score",
        nbins=20,
        title="Distribution of Transaction Risk Scores",
    )

    fig_score.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Number of Transactions",
    )

    st.plotly_chart(
        fig_score,
        use_container_width=True,
    )


# ============================================================
# FRAUD PROBABILITY DISTRIBUTION
# ============================================================

st.subheader(
    "Fraud Probability Distribution"
)


fig_probability = px.histogram(
    filtered_df,
    x="Fraud_Probability",
    nbins=20,
    title="Distribution of Model Fraud Probability",
)


fig_probability.update_layout(
    xaxis_title="Fraud Probability",
    yaxis_title="Number of Transactions",
)


st.plotly_chart(
    fig_probability,
    use_container_width=True,
)


# ============================================================
# RISK BY TRANSACTION TYPE
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Risk by Transaction Type"
    )

    type_risk = (
        filtered_df
        .groupby(
            [
                "Transaction_Type",
                "Risk_Level",
            ]
        )
        .size()
        .reset_index(
            name="Transactions"
        )
    )

    fig_type_risk = px.bar(
        type_risk,
        x="Transaction_Type",
        y="Transactions",
        color="Risk_Level",
        barmode="group",
        title="Risk Distribution by Transaction Type",
    )

    fig_type_risk.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Transactions",
    )

    st.plotly_chart(
        fig_type_risk,
        use_container_width=True,
    )


# ============================================================
# RISK BY SUSPICIOUS KEYWORD
# ============================================================

with col2:

    st.subheader(
        "Risk by Suspicious Keyword"
    )

    keyword_risk = (
        filtered_df
        .groupby(
            [
                "Suspicious_Keyword",
                "Risk_Level",
            ]
        )
        .size()
        .reset_index(
            name="Transactions"
        )
    )

    fig_keyword_risk = px.bar(
        keyword_risk,
        x="Suspicious_Keyword",
        y="Transactions",
        color="Risk_Level",
        barmode="group",
        title="Risk Distribution by Suspicious Keyword",
    )

    fig_keyword_risk.update_layout(
        xaxis_title="Suspicious Keyword",
        yaxis_title="Transactions",
    )

    st.plotly_chart(
        fig_keyword_risk,
        use_container_width=True,
    )


# ============================================================
# HIGH-RISK TRANSACTION AMOUNTS
# ============================================================

st.subheader(
    "Transaction Amount by Risk Level"
)


amount_risk = (
    filtered_df
    .groupby("Risk_Level")
    .agg(
        Transactions=(
            "Transaction_ID",
            "count",
        ),
        Total_Amount=(
            "Transaction_Amount",
            "sum",
        ),
        Average_Amount=(
            "Transaction_Amount",
            "mean",
        ),
    )
    .reset_index()
)


amount_risk["Risk_Level"] = pd.Categorical(
    amount_risk["Risk_Level"],
    categories=[
        "High",
        "Medium",
        "Low",
    ],
    ordered=True,
)


amount_risk = amount_risk.sort_values(
    "Risk_Level"
)


fig_amount = px.bar(
    amount_risk,
    x="Risk_Level",
    y="Total_Amount",
    text="Total_Amount",
    title="Transaction Amount by Risk Level",
)


fig_amount.update_traces(
    texttemplate="%{text:,.2f}",
    textposition="outside",
)


fig_amount.update_layout(
    xaxis_title="Risk Level",
    yaxis_title="Transaction Amount",
    showlegend=False,
)


st.plotly_chart(
    fig_amount,
    use_container_width=True,
)


# ============================================================
# TOP RISK REASONS
# ============================================================

st.subheader(
    "Top Risk Indicators"
)


risk_reason_data = (
    filtered_df
    .groupby(
        "Primary_Risk_Reason"
    )
    .agg(
        Transactions=(
            "Transaction_ID",
            "count",
        ),
        Average_Risk_Score=(
            "Risk_Score",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "Transactions",
        ascending=False,
    )
    .head(10)
)


fig_reasons = px.bar(
    risk_reason_data,
    x="Transactions",
    y="Primary_Risk_Reason",
    orientation="h",
    text="Transactions",
    title="Most Common Risk Indicators",
)


fig_reasons.update_traces(
    textposition="outside"
)


fig_reasons.update_layout(
    xaxis_title="Transactions",
    yaxis_title="Risk Indicator",
)


st.plotly_chart(
    fig_reasons,
    use_container_width=True,
)


# ============================================================
# HIGH-RISK TRANSACTION TABLE
# ============================================================

st.subheader(
    "🚨 Highest Risk Transactions"
)


high_risk_df = (
    filtered_df[
        filtered_df["Risk_Level"]
        == "High"
    ]
    .sort_values(
        [
            "Fraud_Probability",
            "Risk_Score",
        ],
        ascending=False,
    )
    [
        [
            "Transaction_ID",
            "Transaction_Date",
            "Transaction_Amount",
            "Transaction_Type",
            "Fraud_Probability",
            "Risk_Score",
            "Primary_Risk_Reason",
        ]
    ]
    .head(20)
    .copy()
)


if high_risk_df.empty:

    st.info(
        "No high-risk transactions match the selected filters."
    )

else:

    high_risk_df["Fraud_Probability"] = (
        high_risk_df["Fraud_Probability"]
        .round(4)
    )

    high_risk_df["Risk_Score"] = (
        high_risk_df["Risk_Score"]
        .round(2)
    )

    st.dataframe(
        high_risk_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown("---")

st.subheader(
    "💡 Risk Insights"
)


# ------------------------------------------------------------
# HIGH-RISK SHARE
# ------------------------------------------------------------

high_risk_percentage = (
    high_risk_transactions
    / total_transactions
    * 100
)


# ------------------------------------------------------------
# TOP RISK REASON
# ------------------------------------------------------------

if not risk_reason_data.empty:

    top_risk_reason = (
        risk_reason_data.iloc[0]
        ["Primary_Risk_Reason"]
    )

    top_risk_reason_count = int(
        risk_reason_data.iloc[0]
        ["Transactions"]
    )

else:

    top_risk_reason = "No risk indicator available"

    top_risk_reason_count = 0


# ------------------------------------------------------------
# INSIGHTS
# ------------------------------------------------------------

insight_col1, insight_col2 = st.columns(2)


with insight_col1:

    st.info(
        f"""
        **High-Risk Exposure**

        {high_risk_transactions:,} transactions are
        classified as High Risk, representing
        {high_risk_percentage:.2f}% of the filtered
        transaction population.

        The associated transaction value is
        {high_risk_amount:,.2f}.
        """
    )


with insight_col2:

    st.info(
        f"""
        **Primary Risk Indicator**

        The most frequently observed risk indicator is
        **{top_risk_reason}**, appearing in
        {top_risk_reason_count:,} transactions within
        the selected dataset.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Risk Analysis | "
    "Fraud risk classification generated using the "
    "final tuned XGBoost model with a 0.69 decision threshold."
)
