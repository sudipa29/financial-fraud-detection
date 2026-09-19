from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transaction & Fraud Analysis",
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

st.title("📊 Transaction & Fraud Analysis")

st.markdown(
    """
    Analyze transaction patterns and identify the
    behavioral and transactional factors associated
    with AI-flagged fraud.
    """
)

st.markdown("---")


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔎 Analysis Filters")


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
# PAYMENT METHOD FILTER
# ============================================================

payment_methods = sorted(
    df["Payment_Method"]
    .dropna()
    .unique()
)

selected_payment_methods = (
    st.sidebar.multiselect(
        "Payment Method",
        options=payment_methods,
        default=payment_methods,
    )
)

filtered_df = filtered_df[
    filtered_df["Payment_Method"].isin(
        selected_payment_methods
    )
]


# ============================================================
# MERCHANT CATEGORY FILTER
# ============================================================

merchant_categories = sorted(
    df["Merchant_Category"]
    .dropna()
    .unique()
)

selected_merchant_categories = (
    st.sidebar.multiselect(
        "Merchant Category",
        options=merchant_categories,
        default=merchant_categories,
    )
)

filtered_df = filtered_df[
    filtered_df["Merchant_Category"].isin(
        selected_merchant_categories
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

flagged_transactions = int(
    filtered_df[
        "Fraud_Prediction"
    ].sum()
)

flagged_rate = (
    flagged_transactions
    / total_transactions
    * 100
)

international_flagged = int(
    filtered_df[
        (
            filtered_df["Transaction_Type"]
            == "International"
        )
        &
        (
            filtered_df["Fraud_Prediction"]
            == 1
        )
    ].shape[0]
)

keyword_flagged = int(
    filtered_df[
        (
            filtered_df["Suspicious_Keyword"]
            == "Yes"
        )
        &
        (
            filtered_df["Fraud_Prediction"]
            == 1
        )
    ].shape[0]
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Flagged Transactions",
        f"{flagged_transactions:,}",
    )


with col2:

    st.metric(
        "Flagged Rate",
        f"{flagged_rate:.2f}%",
    )


with col3:

    st.metric(
        "International Flagged",
        f"{international_flagged:,}",
    )


with col4:

    st.metric(
        "Suspicious Keyword Flagged",
        f"{keyword_flagged:,}",
    )


st.markdown("---")


# ============================================================
# FRAUD RISK BY TRANSACTION HOUR
# ============================================================

st.subheader(
    "Fraud Risk by Transaction Hour"
)


hour_data = (
    filtered_df
    .groupby(
        filtered_df["Transaction_Date"].dt.hour
    )
    .agg(
        Transactions=(
            "Transaction_ID",
            "count",
        ),
        Flagged_Fraud=(
            "Fraud_Prediction",
            "sum",
        ),
    )
    .reset_index()
)


hour_data.rename(
    columns={
        "Transaction_Date": "Hour"
    },
    inplace=True,
)


hour_data["Flagged_Rate"] = (
    hour_data["Flagged_Fraud"]
    / hour_data["Transactions"]
    * 100
)


fig_hour = px.line(
    hour_data,
    x="Hour",
    y="Flagged_Rate",
    markers=True,
    title="AI-Flagged Transaction Rate by Hour",
)


fig_hour.update_layout(
    xaxis_title="Transaction Hour",
    yaxis_title="Flagged Rate (%)",
    xaxis=dict(
        dtick=1
    ),
)


st.plotly_chart(
    fig_hour,
    use_container_width=True,
)


# ============================================================
# DOMESTIC VS INTERNATIONAL
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Domestic vs International Risk"
    )

    type_analysis = (
        filtered_df
        .groupby("Transaction_Type")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count",
            ),
            Flagged=(
                "Fraud_Prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    type_analysis["Flagged_Rate"] = (
        type_analysis["Flagged"]
        / type_analysis["Transactions"]
        * 100
    )

    fig_type = px.bar(
        type_analysis,
        x="Transaction_Type",
        y="Flagged_Rate",
        text="Flagged_Rate",
        title="Flagged Rate by Transaction Type",
    )

    fig_type.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_type.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Flagged Rate (%)",
    )

    st.plotly_chart(
        fig_type,
        use_container_width=True,
    )


# ============================================================
# PAYMENT METHOD
# ============================================================

with col2:

    st.subheader(
        "Payment Method Risk"
    )

    payment_analysis = (
        filtered_df
        .groupby("Payment_Method")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count",
            ),
            Flagged=(
                "Fraud_Prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    payment_analysis["Flagged_Rate"] = (
        payment_analysis["Flagged"]
        / payment_analysis["Transactions"]
        * 100
    )

    payment_analysis = (
        payment_analysis
        .sort_values(
            "Flagged_Rate",
            ascending=False,
        )
    )

    fig_payment = px.bar(
        payment_analysis,
        x="Payment_Method",
        y="Flagged_Rate",
        text="Flagged_Rate",
        title="Flagged Rate by Payment Method",
    )

    fig_payment.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_payment.update_layout(
        xaxis_title="Payment Method",
        yaxis_title="Flagged Rate (%)",
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True,
    )


# ============================================================
# MERCHANT CATEGORY
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Merchant Category Risk"
    )

    merchant_analysis = (
        filtered_df
        .groupby("Merchant_Category")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count",
            ),
            Flagged=(
                "Fraud_Prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    merchant_analysis["Flagged_Rate"] = (
        merchant_analysis["Flagged"]
        / merchant_analysis["Transactions"]
        * 100
    )

    merchant_analysis = (
        merchant_analysis
        .sort_values(
            "Flagged_Rate",
            ascending=False,
        )
    )

    fig_merchant = px.bar(
        merchant_analysis,
        x="Flagged_Rate",
        y="Merchant_Category",
        orientation="h",
        text="Flagged_Rate",
        title="Flagged Rate by Merchant Category",
    )

    fig_merchant.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_merchant.update_layout(
        xaxis_title="Flagged Rate (%)",
        yaxis_title="Merchant Category",
    )

    st.plotly_chart(
        fig_merchant,
        use_container_width=True,
    )


# ============================================================
# DEVICE TYPE
# ============================================================

with col2:

    st.subheader(
        "Device Type Risk"
    )

    device_analysis = (
        filtered_df
        .groupby("Device_Type")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count",
            ),
            Flagged=(
                "Fraud_Prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    device_analysis["Flagged_Rate"] = (
        device_analysis["Flagged"]
        / device_analysis["Transactions"]
        * 100
    )

    fig_device = px.bar(
        device_analysis,
        x="Device_Type",
        y="Flagged_Rate",
        text="Flagged_Rate",
        title="Flagged Rate by Device Type",
    )

    fig_device.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_device.update_layout(
        xaxis_title="Device Type",
        yaxis_title="Flagged Rate (%)",
    )

    st.plotly_chart(
        fig_device,
        use_container_width=True,
    )


# ============================================================
# SUSPICIOUS KEYWORD ANALYSIS
# ============================================================

st.subheader(
    "Suspicious Keyword Risk"
)


keyword_analysis = (
    filtered_df
    .groupby("Suspicious_Keyword")
    .agg(
        Transactions=(
            "Transaction_ID",
            "count",
        ),
        Flagged=(
            "Fraud_Prediction",
            "sum",
        ),
    )
    .reset_index()
)


keyword_analysis["Flagged_Rate"] = (
    keyword_analysis["Flagged"]
    / keyword_analysis["Transactions"]
    * 100
)


fig_keyword = px.bar(
    keyword_analysis,
    x="Suspicious_Keyword",
    y="Flagged_Rate",
    text="Flagged_Rate",
    title="Flagged Rate by Suspicious Keyword Status",
)


fig_keyword.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)


fig_keyword.update_layout(
    xaxis_title="Suspicious Keyword",
    yaxis_title="Flagged Rate (%)",
)


st.plotly_chart(
    fig_keyword,
    use_container_width=True,
)


# ============================================================
# TRANSACTION AMOUNT VS FRAUD PROBABILITY
# ============================================================

st.subheader(
    "Transaction Amount vs Fraud Probability"
)


sample_df = filtered_df.copy()


if len(sample_df) > 1500:

    sample_df = sample_df.sample(
        1500,
        random_state=42,
    )


fig_scatter = px.scatter(
    sample_df,
    x="Transaction_Amount",
    y="Fraud_Probability",
    color="Risk_Level",
    hover_data=[
        "Transaction_ID",
        "Transaction_Type",
        "Payment_Method",
        "Merchant_Category",
        "Risk_Score",
    ],
    title="Transaction Amount vs Model Fraud Probability",
)


fig_scatter.update_layout(
    xaxis_title="Transaction Amount",
    yaxis_title="Fraud Probability",
)


st.plotly_chart(
    fig_scatter,
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Fraud Analysis | "
    "Predictions generated using the final tuned XGBoost model "
    "with a 0.69 decision threshold."
)