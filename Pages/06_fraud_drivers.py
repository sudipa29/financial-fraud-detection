from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Drivers | Financial Fraud Detection",
    page_icon="🧠",
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

    if not DATA_PATH.exists():
        return None

    data = pd.read_csv(DATA_PATH)

    if "Transaction_Date" in data.columns:
        data["Transaction_Date"] = pd.to_datetime(
            data["Transaction_Date"],
            errors="coerce"
        )

    numeric_columns = [
        "Transaction_Amount",
        "Fraud_Prediction",
        "Fraud_Probability",
        "Risk_Score",
        "Flagged_Fraud_Amount",
        "Previous_Transactions",
        "Average_Spend",
        "Account_Age_Days",
        "Transaction_Hour",
        "Transaction_Day",
        "Transaction_Month",
        "Is_International",
        "Is_Weekend",
        "Amount_vs_Average_Spend",
        "Transactions_per_Account_Age",
        "Fraudulent",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


df = load_data()


# ============================================================
# DATA CHECK
# ============================================================

if df is None:

    st.error(
        "Dashboard dataset was not found."
    )

    st.code(str(DATA_PATH))

    st.stop()


if df.empty:

    st.warning(
        "Dashboard dataset contains no records."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 Fraud Drivers & Explainability"
)

st.markdown(
    """
    Understand the **behavioral and transaction characteristics
    associated with fraud predictions** and investigate the main
    risk indicators identified by the fraud detection pipeline.
    """
)

st.caption(
    "Explainability view based on the project's prepared fraud prediction dataset"
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🧠 Driver Analysis Filters")

filtered_df = df.copy()


# ------------------------------------------------------------
# RISK LEVEL
# ------------------------------------------------------------

if "Risk_Level" in df.columns:

    risk_options = [
        "High",
        "Medium",
        "Low",
    ]

    available_risks = [
        risk
        for risk in risk_options
        if risk in df["Risk_Level"].astype(str).unique()
    ]

    selected_risks = st.sidebar.multiselect(
        "Risk Level",
        options=available_risks,
        default=available_risks,
    )

    filtered_df = filtered_df[
        filtered_df["Risk_Level"]
        .astype(str)
        .isin(selected_risks)
    ]


# ------------------------------------------------------------
# PREDICTION STATUS
# ------------------------------------------------------------

if "Fraud_Status" in df.columns:

    status_options = [
        "Predicted Fraud",
        "Predicted Legitimate",
    ]

    available_statuses = [
        status
        for status in status_options
        if status in df["Fraud_Status"].astype(str).unique()
    ]

    selected_statuses = st.sidebar.multiselect(
        "Prediction Status",
        options=available_statuses,
        default=available_statuses,
    )

    filtered_df = filtered_df[
        filtered_df["Fraud_Status"]
        .astype(str)
        .isin(selected_statuses)
    ]


# ------------------------------------------------------------
# INTERNATIONAL
# ------------------------------------------------------------

if "Is_International" in df.columns:

    geography = st.sidebar.selectbox(
        "Transaction Geography",
        [
            "All",
            "Domestic",
            "International",
        ],
    )

    if geography == "Domestic":

        filtered_df = filtered_df[
            filtered_df["Is_International"] == 0
        ]

    elif geography == "International":

        filtered_df = filtered_df[
            filtered_df["Is_International"] == 1
        ]


# ------------------------------------------------------------
# TRANSACTION TYPE
# ------------------------------------------------------------

if "Transaction_Type" in df.columns:

    transaction_types = sorted(
        df["Transaction_Type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_transaction_types = st.sidebar.multiselect(
        "Transaction Type",
        options=transaction_types,
        default=transaction_types,
    )

    filtered_df = filtered_df[
        filtered_df["Transaction_Type"]
        .astype(str)
        .isin(selected_transaction_types)
    ]


if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )

    st.stop()


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Driver Analysis Overview")

total_transactions = len(filtered_df)


if "Fraud_Prediction" in filtered_df.columns:

    predicted_fraud = int(
        filtered_df["Fraud_Prediction"]
        .fillna(0)
        .sum()
    )

else:

    predicted_fraud = 0


fraud_rate = (
    predicted_fraud
    / total_transactions
    * 100
)


if "Is_International" in filtered_df.columns:

    international_transactions = int(
        filtered_df["Is_International"]
        .fillna(0)
        .sum()
    )

else:

    international_transactions = 0


if "Suspicious_Keyword" in filtered_df.columns:

    suspicious_keyword_transactions = int(
        (
            filtered_df["Suspicious_Keyword"]
            .astype(str)
            .str.lower()
            .isin(["yes", "1", "true"])
        ).sum()
    )

else:

    suspicious_keyword_transactions = 0


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Transactions Analyzed",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Predicted Fraud",
        f"{predicted_fraud:,}"
    )

with col3:

    st.metric(
        "International Transactions",
        f"{international_transactions:,}"
    )

with col4:

    st.metric(
        "Suspicious Keyword",
        f"{suspicious_keyword_transactions:,}"
    )


st.divider()


# ============================================================
# TOP RISK REASONS
# ============================================================

st.subheader("🚨 Top Risk Indicators")

if "Primary_Risk_Reason" in filtered_df.columns:

    reason_data = (
        filtered_df["Primary_Risk_Reason"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
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
        text="Transactions",
        title="Most Common Risk Indicators",
    )

    fig_reason.update_layout(
        yaxis_title="",
        xaxis_title="Transactions",
    )

    st.plotly_chart(
        fig_reason,
        use_container_width=True,
    )

else:

    st.info(
        "Primary_Risk_Reason is not available in the dataset."
    )


st.divider()


# ============================================================
# INTERNATIONAL FRAUD ANALYSIS
# ============================================================

st.subheader("🌍 International Transaction Risk")

if (
    "Is_International" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    international_analysis = (
        filtered_df
        .groupby("Is_International")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count"
            ),
            Predicted_Fraud=(
                "Fraud_Prediction",
                "sum"
            ),
        )
        .reset_index()
    )

    international_analysis["Transaction_Type"] = (
        international_analysis["Is_International"]
        .map(
            {
                0: "Domestic",
                1: "International",
            }
        )
    )

    international_analysis["Fraud_Rate"] = (
        international_analysis["Predicted_Fraud"]
        / international_analysis["Transactions"]
        * 100
    )

    fig_international = px.bar(
        international_analysis,
        x="Transaction_Type",
        y="Fraud_Rate",
        text="Fraud_Rate",
        title="Predicted Fraud Rate: Domestic vs International",
    )

    fig_international.update_traces(
        texttemplate="%{text:.2f}%"
    )

    fig_international.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Predicted Fraud Rate (%)",
    )

    st.plotly_chart(
        fig_international,
        use_container_width=True,
    )


st.divider()


# ============================================================
# SUSPICIOUS KEYWORD ANALYSIS
# ============================================================

st.subheader("🔑 Suspicious Keyword Analysis")

if (
    "Suspicious_Keyword" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    keyword_data = (
        filtered_df
        .groupby("Suspicious_Keyword")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count"
            ),
            Predicted_Fraud=(
                "Fraud_Prediction",
                "sum"
            ),
        )
        .reset_index()
    )

    keyword_data["Fraud_Rate"] = (
        keyword_data["Predicted_Fraud"]
        / keyword_data["Transactions"]
        * 100
    )

    fig_keyword = px.bar(
        keyword_data,
        x="Suspicious_Keyword",
        y="Fraud_Rate",
        text="Fraud_Rate",
        title="Predicted Fraud Rate by Suspicious Keyword",
    )

    fig_keyword.update_traces(
        texttemplate="%{text:.2f}%"
    )

    fig_keyword.update_layout(
        xaxis_title="Suspicious Keyword",
        yaxis_title="Predicted Fraud Rate (%)",
    )

    st.plotly_chart(
        fig_keyword,
        use_container_width=True,
    )


st.divider()


# ============================================================
# TRANSACTION HOUR ANALYSIS
# ============================================================

st.subheader("⏰ Fraud Risk by Transaction Hour")

if (
    "Transaction_Hour" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    hourly_data = (
        filtered_df
        .groupby("Transaction_Hour")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count"
            ),
            Predicted_Fraud=(
                "Fraud_Prediction",
                "sum"
            ),
        )
        .reset_index()
    )

    hourly_data["Fraud_Rate"] = (
        hourly_data["Predicted_Fraud"]
        / hourly_data["Transactions"]
        * 100
    )

    fig_hour = px.line(
        hourly_data,
        x="Transaction_Hour",
        y="Fraud_Rate",
        markers=True,
        title="Predicted Fraud Rate by Transaction Hour",
    )

    fig_hour.update_layout(
        xaxis_title="Transaction Hour",
        yaxis_title="Predicted Fraud Rate (%)",
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True,
    )


st.divider()


# ============================================================
# AMOUNT VS AVERAGE SPEND
# ============================================================

st.subheader("💰 Transaction Amount vs Customer Average Spend")

if (
    "Amount_vs_Average_Spend" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    amount_data = filtered_df[
        [
            "Amount_vs_Average_Spend",
            "Fraud_Prediction",
        ]
    ].dropna()

    if not amount_data.empty:

        fig_amount = px.box(
            amount_data,
            x="Fraud_Prediction",
            y="Amount_vs_Average_Spend",
            points=False,
            title="Amount Relative to Average Spend",
        )

        fig_amount.update_layout(
            xaxis_title="Prediction Class",
            yaxis_title="Amount / Average Spend",
        )

        fig_amount.update_xaxes(
            tickvals=[0, 1],
            ticktext=[
                "Predicted Legitimate",
                "Predicted Fraud",
            ],
        )

        st.plotly_chart(
            fig_amount,
            use_container_width=True,
        )


st.divider()


# ============================================================
# PAYMENT METHOD ANALYSIS
# ============================================================

st.subheader("💳 Fraud Risk by Payment Method")

if (
    "Payment_Method" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    payment_data = (
        filtered_df
        .groupby("Payment_Method")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count"
            ),
            Predicted_Fraud=(
                "Fraud_Prediction",
                "sum"
            ),
        )
        .reset_index()
    )

    payment_data["Fraud_Rate"] = (
        payment_data["Predicted_Fraud"]
        / payment_data["Transactions"]
        * 100
    )

    payment_data = payment_data.sort_values(
        "Fraud_Rate",
        ascending=False
    )

    fig_payment = px.bar(
        payment_data,
        x="Payment_Method",
        y="Fraud_Rate",
        text="Fraud_Rate",
        title="Predicted Fraud Rate by Payment Method",
    )

    fig_payment.update_traces(
        texttemplate="%{text:.2f}%"
    )

    fig_payment.update_layout(
        xaxis_title="Payment Method",
        yaxis_title="Predicted Fraud Rate (%)",
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True,
    )


st.divider()


# ============================================================
# DEVICE TYPE ANALYSIS
# ============================================================

st.subheader("📱 Fraud Risk by Device Type")

if (
    "Device_Type" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    device_data = (
        filtered_df
        .groupby("Device_Type")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count"
            ),
            Predicted_Fraud=(
                "Fraud_Prediction",
                "sum"
            ),
        )
        .reset_index()
    )

    device_data["Fraud_Rate"] = (
        device_data["Predicted_Fraud"]
        / device_data["Transactions"]
        * 100
    )

    device_data = device_data.sort_values(
        "Fraud_Rate",
        ascending=False
    )

    fig_device = px.bar(
        device_data,
        x="Device_Type",
        y="Fraud_Rate",
        text="Fraud_Rate",
        title="Predicted Fraud Rate by Device Type",
    )

    fig_device.update_traces(
        texttemplate="%{text:.2f}%"
    )

    fig_device.update_layout(
        xaxis_title="Device Type",
        yaxis_title="Predicted Fraud Rate (%)",
    )

    st.plotly_chart(
        fig_device,
        use_container_width=True,
    )


st.divider()


# ============================================================
# MERCHANT CATEGORY ANALYSIS
# ============================================================

st.subheader("🏪 Fraud Risk by Merchant Category")

if (
    "Merchant_Category" in filtered_df.columns
    and "Fraud_Prediction" in filtered_df.columns
):

    merchant_data = (
        filtered_df
        .groupby("Merchant_Category")
        .agg(
            Transactions=(
                "Transaction_ID",
                "count"
            ),
            Predicted_Fraud=(
                "Fraud_Prediction",
                "sum"
            ),
        )
        .reset_index()
    )

    merchant_data["Fraud_Rate"] = (
        merchant_data["Predicted_Fraud"]
        / merchant_data["Transactions"]
        * 100
    )

    merchant_data = merchant_data.sort_values(
        "Fraud_Rate",
        ascending=False
    ).head(10)

    fig_merchant = px.bar(
        merchant_data,
        x="Fraud_Rate",
        y="Merchant_Category",
        orientation="h",
        text="Fraud_Rate",
        title="Top Merchant Categories by Predicted Fraud Rate",
    )

    fig_merchant.update_traces(
        texttemplate="%{text:.2f}%"
    )

    fig_merchant.update_layout(
        xaxis_title="Predicted Fraud Rate (%)",
        yaxis_title="Merchant Category",
    )

    st.plotly_chart(
        fig_merchant,
        use_container_width=True,
    )


st.divider()


# ============================================================
# SELECTED TRANSACTION EXPLANATION
# ============================================================

st.subheader("🔍 Transaction-Level Explanation")

if "Transaction_ID" in filtered_df.columns:

    transaction_ids = (
        filtered_df["Transaction_ID"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_id = st.selectbox(
        "Select a transaction to investigate",
        transaction_ids,
    )

    selected_row = filtered_df[
        filtered_df["Transaction_ID"]
        .astype(str)
        == selected_id
    ].iloc[0]


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # TRANSACTION PROFILE
    # --------------------------------------------------------

    with col1:

        st.markdown("### Transaction Profile")

        profile_fields = [
            ("Transaction ID", "Transaction_ID"),
            ("Customer ID", "Customer_ID"),
            ("Amount", "Transaction_Amount"),
            ("Merchant Category", "Merchant_Category"),
            ("Payment Method", "Payment_Method"),
            ("Device Type", "Device_Type"),
            ("Location", "Location"),
            ("Transaction Type", "Transaction_Type"),
        ]

        for label, column in profile_fields:

            if column in selected_row.index:

                value = selected_row[column]

                if (
                    column == "Transaction_Amount"
                    and pd.notna(value)
                ):

                    value = f"{value:,.2f}"

                st.write(
                    f"**{label}:** {value}"
                )


    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    with col2:

        st.markdown("### Model Risk Assessment")

        explanation_fields = [
            ("Prediction", "Fraud_Status"),
            ("Fraud Probability", "Fraud_Probability"),
            ("Risk Score", "Risk_Score"),
            ("Risk Level", "Risk_Level"),
            ("Primary Risk Reason", "Primary_Risk_Reason"),
        ]

        for label, column in explanation_fields:

            if column in selected_row.index:

                value = selected_row[column]

                if (
                    column == "Fraud_Probability"
                    and pd.notna(value)
                ):

                    value = f"{value:.2%}"

                elif (
                    column == "Risk_Score"
                    and pd.notna(value)
                ):

                    value = f"{value:.2f}"

                st.write(
                    f"**{label}:** {value}"
                )


    # --------------------------------------------------------
    # DRIVER INDICATORS
    # --------------------------------------------------------

    st.markdown("### Key Behavioral Indicators")

    driver_fields = [
        ("International Transaction", "Is_International"),
        ("Suspicious Keyword", "Suspicious_Keyword"),
        ("Transaction Hour", "Transaction_Hour"),
        ("Amount vs Average Spend", "Amount_vs_Average_Spend"),
        (
            "Transactions per Account Age",
            "Transactions_per_Account_Age",
        ),
        ("Previous Transactions", "Previous_Transactions"),
        ("Account Age (Days)", "Account_Age_Days"),
    ]

    driver_data = []

    for label, column in driver_fields:

        if column in selected_row.index:

            value = selected_row[column]

            if column == "Is_International":

                value = (
                    "International"
                    if value == 1
                    else "Domestic"
                )

            driver_data.append(
                {
                    "Indicator": label,
                    "Value": value,
                }
            )

    if driver_data:

        st.dataframe(
            pd.DataFrame(driver_data),
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    prediction = str(
        selected_row.get(
            "Fraud_Status",
            ""
        )
    )

    risk_level = str(
        selected_row.get(
            "Risk_Level",
            ""
        )
    )

    reason = str(
        selected_row.get(
            "Primary_Risk_Reason",
            "Not available"
        )
    )

    if prediction == "Predicted Fraud":

        st.error(
            f"""
            **Potential Fraud**

            This transaction was classified as **Predicted Fraud**
            using the tuned XGBoost classification threshold of **0.69**.

            **Risk Level:** {risk_level}

            **Primary Risk Indicator:** {reason}

            The transaction should be considered for additional
            investigation or review.
            """
        )

    else:

        st.success(
            f"""
            **Predicted Legitimate**

            This transaction was classified as **Predicted Legitimate**.

            **Risk Level:** {risk_level}

            **Primary Risk Indicator:** {reason}

            It can remain under normal monitoring unless additional
            business rules indicate otherwise.
            """
        )


# ============================================================
# BUSINESS TAKEAWAYS
# ============================================================

st.divider()

st.subheader("💡 Business Takeaways")

st.markdown(
    """
    **1. International transactions:**  
    International activity can represent a meaningful fraud-risk
    indicator and should receive additional monitoring.

    **2. Suspicious keywords:**  
    Transactions containing suspicious keywords can provide an
    additional behavioral signal for fraud investigation.

    **3. Transaction timing:**  
    Fraud risk can vary across transaction hours, making time-based
    monitoring useful for anomaly detection.

    **4. Spending behavior:**  
    Comparing transaction amount with the customer's historical
    average spend helps identify unusual transaction behavior.

    **5. Explainable investigation:**  
    Combining model probability, risk level and behavioral indicators
    allows analysts to investigate *why* a transaction was flagged
    rather than relying only on a binary fraud prediction.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Financial Fraud Detection System | "
    "Fraud Drivers & Explainability | "
    "Final XGBoost Model | "
    "Tuned Threshold = 0.69"
)