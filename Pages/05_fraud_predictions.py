from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Predictions | Financial Fraud Detection",
    page_icon="🔎",
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

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # DATE CONVERSION
    # --------------------------------------------------------

    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(
            df["Transaction_Date"],
            errors="coerce"
        )

    if "Transaction_Date_Only" in df.columns:
        df["Transaction_Date_Only"] = pd.to_datetime(
            df["Transaction_Date_Only"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

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
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


df = load_data()


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

if df is None:

    st.error(
        "Fraud dashboard data file was not found."
    )

    st.code(
        str(DATA_PATH)
    )

    st.stop()


if df.empty:

    st.warning(
        "The fraud dashboard dataset is empty."
    )

    st.stop()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🔎 Fraud Predictions & Transaction Investigation"
)

st.markdown(
    """
    **Transaction-level investigation using the final XGBoost
    fraud predictions and risk indicators.**
    """
)

st.caption(
    "Predictions generated using the project's tuned fraud-detection pipeline | "
    "Classification threshold = 0.69"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Investigation Filters")

st.sidebar.markdown(
    """
    Use the filters below to investigate suspicious transactions,
    risk levels and individual customers.
    """
)

st.sidebar.divider()


# ============================================================
# DATE FILTER
# ============================================================

filtered_df = df.copy()

if "Transaction_Date" in df.columns:

    valid_dates = df["Transaction_Date"].dropna()

    if not valid_dates.empty:

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

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

            filtered_df = filtered_df[
                (
                    filtered_df["Transaction_Date"]
                    >= start_date
                )
                &
                (
                    filtered_df["Transaction_Date"]
                    < end_date
                )
            ].copy()


# ============================================================
# TRANSACTION TYPE FILTER
# ============================================================

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


# ============================================================
# RISK LEVEL FILTER
# ============================================================

if "Risk_Level" in df.columns:

    risk_levels = [
        "High",
        "Medium",
        "Low",
    ]

    available_risk_levels = [
        level
        for level in risk_levels
        if level in df["Risk_Level"].astype(str).unique()
    ]

    selected_risk_levels = st.sidebar.multiselect(
        "Risk Level",
        options=available_risk_levels,
        default=available_risk_levels,
    )

    filtered_df = filtered_df[
        filtered_df["Risk_Level"]
        .astype(str)
        .isin(selected_risk_levels)
    ]


# ============================================================
# PREDICTION STATUS FILTER
# ============================================================

if "Fraud_Status" in df.columns:

    fraud_status_options = [
        "Predicted Fraud",
        "Predicted Legitimate",
    ]

    available_statuses = [
        status
        for status in fraud_status_options
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


# ============================================================
# INTERNATIONAL FILTER
# ============================================================

if "Is_International" in df.columns:

    international_options = [
        "All",
        "Domestic",
        "International",
    ]

    selected_international = st.sidebar.selectbox(
        "Transaction Geography",
        options=international_options,
    )

    if selected_international == "Domestic":

        filtered_df = filtered_df[
            filtered_df["Is_International"] == 0
        ]

    elif selected_international == "International":

        filtered_df = filtered_df[
            filtered_df["Is_International"] == 1
        ]


# ============================================================
# CUSTOMER SEARCH
# ============================================================

customer_search = st.sidebar.text_input(
    "Customer ID Search",
    placeholder="Enter Customer ID..."
)

if customer_search.strip():

    if "Customer_ID" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["Customer_ID"]
            .astype(str)
            .str.contains(
                customer_search.strip(),
                case=False,
                na=False,
            )
        ]


# ============================================================
# TRANSACTION SEARCH
# ============================================================

transaction_search = st.sidebar.text_input(
    "Transaction ID Search",
    placeholder="Enter Transaction ID..."
)

if transaction_search.strip():

    if "Transaction_ID" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["Transaction_ID"]
            .astype(str)
            .str.contains(
                transaction_search.strip(),
                case=False,
                na=False,
            )
        ]


# ============================================================
# EMPTY FILTER RESULT
# ============================================================

if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_transactions = len(filtered_df)


if "Fraud_Prediction" in filtered_df.columns:

    predicted_fraud = int(
        filtered_df["Fraud_Prediction"]
        .fillna(0)
        .sum()
    )

else:

    predicted_fraud = 0


predicted_legitimate = (
    total_transactions
    - predicted_fraud
)


predicted_fraud_rate = (
    predicted_fraud
    / total_transactions
    * 100
)


if "Risk_Level" in filtered_df.columns:

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

else:

    high_risk = 0
    medium_risk = 0
    low_risk = 0


if "Transaction_Amount" in filtered_df.columns:

    total_amount = (
        filtered_df["Transaction_Amount"]
        .fillna(0)
        .sum()
    )

else:

    total_amount = 0


if "Flagged_Fraud_Amount" in filtered_df.columns:

    flagged_amount = (
        filtered_df["Flagged_Fraud_Amount"]
        .fillna(0)
        .sum()
    )

else:

    flagged_amount = 0


# ============================================================
# KPI ROW 1
# ============================================================

st.subheader("📊 Investigation Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Predicted Fraud",
        f"{predicted_fraud:,}"
    )


with col3:

    st.metric(
        "Predicted Fraud Rate",
        f"{predicted_fraud_rate:.2f}%"
    )


with col4:

    st.metric(
        "Flagged Amount",
        f"{flagged_amount:,.2f}"
    )


# ============================================================
# KPI ROW 2
# ============================================================

st.markdown("")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "High Risk",
        f"{high_risk:,}"
    )


with col2:

    st.metric(
        "Medium Risk",
        f"{medium_risk:,}"
    )


with col3:

    st.metric(
        "Low Risk",
        f"{low_risk:,}"
    )


with col4:

    st.metric(
        "Transaction Amount",
        f"{total_amount:,.2f}"
    )


st.divider()


# ============================================================
# RISK DISTRIBUTION + PREDICTION DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# RISK DISTRIBUTION
# ------------------------------------------------------------

with col1:

    st.subheader("Risk Level Distribution")

    if "Risk_Level" in filtered_df.columns:

        risk_data = (
            filtered_df["Risk_Level"]
            .value_counts()
            .reindex(
                [
                    "High",
                    "Medium",
                    "Low",
                ],
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
            text="Transactions",
            title="Transactions by Risk Level",
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


# ------------------------------------------------------------
# PREDICTION DISTRIBUTION
# ------------------------------------------------------------

with col2:

    st.subheader("Prediction Distribution")

    if "Fraud_Status" in filtered_df.columns:

        prediction_data = (
            filtered_df["Fraud_Status"]
            .value_counts()
            .reset_index()
        )

        prediction_data.columns = [
            "Fraud_Status",
            "Transactions",
        ]

        fig_prediction = px.pie(
            prediction_data,
            names="Fraud_Status",
            values="Transactions",
            hole=0.45,
            title="Fraud Prediction Status",
        )

        st.plotly_chart(
            fig_prediction,
            use_container_width=True,
        )


st.divider()


# ============================================================
# FRAUD PROBABILITY DISTRIBUTION
# ============================================================

st.subheader("🎯 Fraud Probability Distribution")

if "Fraud_Probability" in filtered_df.columns:

    probability_data = filtered_df[
        "Fraud_Probability"
    ].dropna()

    if not probability_data.empty:

        fig_probability = px.histogram(
            filtered_df,
            x="Fraud_Probability",
            nbins=20,
            title=(
                "Distribution of XGBoost Fraud Probabilities"
            ),
        )

        fig_probability.add_vline(
            x=0.69,
            line_dash="dash",
            annotation_text="Tuned Threshold = 0.69",
            annotation_position="top",
        )

        fig_probability.update_layout(
            xaxis_title="Fraud Probability",
            yaxis_title="Transactions",
        )

        st.plotly_chart(
            fig_probability,
            use_container_width=True,
        )

        st.caption(
            "Transactions at or above the tuned 0.69 threshold "
            "are classified as predicted fraud."
        )


st.divider()


# ============================================================
# HIGH-RISK TRANSACTIONS
# ============================================================

st.subheader("🚨 Highest-Risk Transactions")

if "Fraud_Probability" in filtered_df.columns:

    high_risk_transactions = (
        filtered_df
        .sort_values(
            "Fraud_Probability",
            ascending=False,
        )
        .head(15)
        .copy()
    )

else:

    high_risk_transactions = (
        filtered_df
        .sort_values(
            "Risk_Score",
            ascending=False,
        )
        .head(15)
        .copy()
        if "Risk_Score" in filtered_df.columns
        else filtered_df.head(15).copy()
    )


display_columns = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Transaction_Amount",
    "Fraud_Probability",
    "Risk_Score",
    "Risk_Level",
    "Fraud_Status",
    "Primary_Risk_Reason",
]


available_display_columns = [
    column
    for column in display_columns
    if column in high_risk_transactions.columns
]


high_risk_display = high_risk_transactions[
    available_display_columns
].copy()


if "Fraud_Probability" in high_risk_display.columns:

    high_risk_display["Fraud_Probability"] = (
        high_risk_display["Fraud_Probability"]
        .map(
            lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else ""
        )
    )


if "Transaction_Amount" in high_risk_display.columns:

    high_risk_display["Transaction_Amount"] = (
        high_risk_display["Transaction_Amount"]
        .map(
            lambda x:
            f"{x:,.2f}"
            if pd.notna(x)
            else ""
        )
    )


if "Transaction_Date" in high_risk_display.columns:

    high_risk_display["Transaction_Date"] = (
        high_risk_display["Transaction_Date"]
        .dt.strftime("%Y-%m-%d %H:%M")
    )


st.dataframe(
    high_risk_display,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ============================================================
# SELECT A TRANSACTION
# ============================================================

st.subheader("🔍 Transaction Investigation")

if "Transaction_ID" in filtered_df.columns:

    transaction_ids = (
        filtered_df["Transaction_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    transaction_ids = sorted(transaction_ids)

    selected_transaction_id = st.selectbox(
        "Select Transaction ID",
        options=transaction_ids,
    )

    selected_transaction = filtered_df[
        filtered_df["Transaction_ID"]
        .astype(str)
        == selected_transaction_id
    ].iloc[0]


    # ========================================================
    # SELECTED TRANSACTION DETAILS
    # ========================================================

    left_col, right_col = st.columns(2)


    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    with left_col:

        st.markdown("### Transaction Details")

        detail_fields = [
            ("Transaction ID", "Transaction_ID"),
            ("Customer ID", "Customer_ID"),
            ("Transaction Date", "Transaction_Date"),
            ("Transaction Amount", "Transaction_Amount"),
            ("Merchant Category", "Merchant_Category"),
            ("Payment Method", "Payment_Method"),
            ("Device Type", "Device_Type"),
            ("Location", "Location"),
            ("Transaction Type", "Transaction_Type"),
        ]

        for label, column in detail_fields:

            if column in selected_transaction.index:

                value = selected_transaction[column]

                if (
                    column == "Transaction_Amount"
                    and pd.notna(value)
                ):

                    value = f"{value:,.2f}"

                elif (
                    column == "Transaction_Date"
                    and pd.notna(value)
                ):

                    value = value.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                st.write(
                    f"**{label}:** {value}"
                )


    # --------------------------------------------------------
    # MODEL RISK DETAILS
    # --------------------------------------------------------

    with right_col:

        st.markdown("### Model Risk Assessment")

        risk_fields = [
            ("Prediction Status", "Fraud_Status"),
            ("Fraud Probability", "Fraud_Probability"),
            ("Risk Score", "Risk_Score"),
            ("Risk Level", "Risk_Level"),
            ("Primary Risk Reason", "Primary_Risk_Reason"),
            ("Flagged Fraud Amount", "Flagged_Fraud_Amount"),
        ]

        for label, column in risk_fields:

            if column in selected_transaction.index:

                value = selected_transaction[column]

                if (
                    column == "Fraud_Probability"
                    and pd.notna(value)
                ):

                    value = f"{value:.2%}"

                elif (
                    column == "Flagged_Fraud_Amount"
                    and pd.notna(value)
                ):

                    value = f"{value:,.2f}"

                elif (
                    column == "Risk_Score"
                    and pd.notna(value)
                ):

                    value = f"{value:.2f}"

                st.write(
                    f"**{label}:** {value}"
                )


    # ========================================================
    # BEHAVIORAL INDICATORS
    # ========================================================

    st.markdown("### Behavioral & Transaction Indicators")

    indicator_fields = [
        ("International Transaction", "Is_International"),
        ("Previous Transactions", "Previous_Transactions"),
        ("Average Spend", "Average_Spend"),
        ("Account Age (Days)", "Account_Age_Days"),
        ("Suspicious Keyword", "Suspicious_Keyword"),
        ("Transaction Hour", "Transaction_Hour"),
        ("Day of Week", "Day_of_Week"),
        ("Weekend", "Is_Weekend"),
        ("Amount vs Average Spend", "Amount_vs_Average_Spend"),
        (
            "Transactions per Account Age",
            "Transactions_per_Account_Age",
        ),
    ]

    indicator_data = []

    for label, column in indicator_fields:

        if column in selected_transaction.index:

            value = selected_transaction[column]

            if (
                column == "Is_International"
                and pd.notna(value)
            ):

                value = (
                    "International"
                    if int(value) == 1
                    else "Domestic"
                )

            elif (
                column == "Is_Weekend"
                and pd.notna(value)
            ):

                value = (
                    "Weekend"
                    if int(value) == 1
                    else "Weekday"
                )

            elif (
                column == "Average_Spend"
                and pd.notna(value)
            ):

                value = f"{value:,.2f}"

            elif (
                column == "Amount_vs_Average_Spend"
                and pd.notna(value)
            ):

                value = f"{value:.3f}"

            elif (
                column == "Transactions_per_Account_Age"
                and pd.notna(value)
            ):

                value = f"{value:.4f}"

            indicator_data.append(
                {
                    "Indicator": label,
                    "Value": value,
                }
            )

    if indicator_data:

        st.dataframe(
            pd.DataFrame(indicator_data),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # INVESTIGATION INTERPRETATION
    # ========================================================

    st.markdown("### 🧠 Investigation Interpretation")

    prediction_status = str(
        selected_transaction.get(
            "Fraud_Status",
            ""
        )
    )

    risk_level = str(
        selected_transaction.get(
            "Risk_Level",
            ""
        )
    )

    risk_reason = str(
        selected_transaction.get(
            "Primary_Risk_Reason",
            "Not available"
        )
    )

    probability = selected_transaction.get(
        "Fraud_Probability",
        None
    )


    if prediction_status == "Predicted Fraud":

        st.error(
            f"""
            **Potential Fraud Detected**

            This transaction was classified as **Predicted Fraud**
            using the project's tuned XGBoost threshold of **0.69**.

            **Risk Level:** {risk_level}

            **Primary Risk Reason:** {risk_reason}

            **Recommended action:** Review the transaction and
            supporting customer/context information before taking
            further action.
            """
        )

    else:

        st.success(
            f"""
            **Predicted Legitimate**

            This transaction is currently classified as
            **Predicted Legitimate** by the fraud detection model.

            **Risk Level:** {risk_level}

            The transaction can remain in normal monitoring unless
            additional business rules or investigation evidence
            indicate otherwise.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Financial Fraud Detection System | "
    "Transaction Investigation | "
    "Final XGBoost Model | "
    "Tuned Threshold = 0.69"
)