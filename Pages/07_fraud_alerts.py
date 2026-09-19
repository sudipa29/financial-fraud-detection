from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Alerts & Monitoring",
    page_icon="🚨",
    layout="wide",
)


# ============================================================
# PROJECT PATH
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
# PAGE TITLE
# ============================================================

st.title("🚨 Fraud Alerts & Monitoring")
st.caption(
    "Operational monitoring of predicted fraud alerts, risk levels, "
    "and investigation priorities."
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
        "Fraud_Probability",
        "Risk_Score",
        "Fraud_Prediction",
        "Is_International",
        "Flagged_Fraud_Amount",
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
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "Transaction_ID",
    "Transaction_Date",
    "Transaction_Amount",
    "Fraud_Prediction",
    "Fraud_Probability",
    "Risk_Level",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Alert Filters")


# Risk level

risk_options = ["All"]

if "Risk_Level" in df.columns:

    available_risks = (
        df["Risk_Level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    risk_options.extend(
        sorted(available_risks)
    )

selected_risk = st.sidebar.selectbox(
    "Risk Level",
    risk_options
)


# Alert status

alert_status_options = [
    "All",
    "Predicted Fraud",
    "Legitimate"
]

selected_status = st.sidebar.selectbox(
    "Alert Status",
    alert_status_options
)


# Geography

geography_options = [
    "All",
    "Domestic",
    "International"
]

selected_geography = st.sidebar.selectbox(
    "Transaction Geography",
    geography_options
)


# Transaction type

if "Transaction_Type" in df.columns:

    transaction_types = (
        df["Transaction_Type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_transaction_type = st.sidebar.selectbox(
        "Transaction Type",
        ["All"] + sorted(transaction_types)
    )

else:

    selected_transaction_type = "All"


# Minimum probability

min_probability = st.sidebar.slider(
    "Minimum Fraud Probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


# Risk

if selected_risk != "All":

    filtered_df = filtered_df[
        filtered_df["Risk_Level"].astype(str)
        == selected_risk
    ]


# Prediction

if selected_status == "Predicted Fraud":

    filtered_df = filtered_df[
        filtered_df["Fraud_Prediction"] == 1
    ]

elif selected_status == "Legitimate":

    filtered_df = filtered_df[
        filtered_df["Fraud_Prediction"] == 0
    ]


# Geography

if selected_geography == "International":

    filtered_df = filtered_df[
        filtered_df["Is_International"] == 1
    ]

elif selected_geography == "Domestic":

    filtered_df = filtered_df[
        filtered_df["Is_International"] == 0
    ]


# Transaction type

if (
    selected_transaction_type != "All"
    and "Transaction_Type" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["Transaction_Type"].astype(str)
        == selected_transaction_type
    ]


# Probability

filtered_df = filtered_df[
    filtered_df["Fraud_Probability"]
    >= min_probability
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
# ALERT DATA
# ============================================================

alert_df = filtered_df[
    filtered_df["Fraud_Prediction"] == 1
].copy()


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("🚨 Alert Summary")

col1, col2, col3, col4, col5 = st.columns(5)


# Total alerts

total_alerts = len(alert_df)


# Alert rate

total_transactions = len(filtered_df)

alert_rate = (
    total_alerts / total_transactions * 100
    if total_transactions > 0
    else 0
)


# Flagged amount

if "Flagged_Fraud_Amount" in alert_df.columns:

    flagged_amount = alert_df[
        "Flagged_Fraud_Amount"
    ].sum()

else:

    flagged_amount = alert_df[
        "Transaction_Amount"
    ].sum()


# High risk alerts

high_risk_alerts = len(
    alert_df[
        alert_df["Risk_Level"].astype(str).str.upper()
        == "HIGH"
    ]
)


# Average probability

average_probability = (
    alert_df["Fraud_Probability"].mean() * 100
    if not alert_df.empty
    else 0
)


with col1:

    st.metric(
        "🚨 Fraud Alerts",
        f"{total_alerts:,}"
    )


with col2:

    st.metric(
        "Alert Rate",
        f"{alert_rate:.2f}%"
    )


with col3:

    st.metric(
        "💰 Flagged Amount",
        f"₹{flagged_amount:,.2f}"
    )


with col4:

    st.metric(
        "🔴 High-Risk Alerts",
        f"{high_risk_alerts:,}"
    )


with col5:

    st.metric(
        "Avg Fraud Probability",
        f"{average_probability:.2f}%"
    )


# ============================================================
# ALERT PRIORITY
# ============================================================

st.divider()

st.subheader("⚠️ Alert Priority")


priority_col1, priority_col2, priority_col3 = st.columns(3)


high_count = len(
    alert_df[
        alert_df["Risk_Level"].astype(str).str.upper()
        == "HIGH"
    ]
)

medium_count = len(
    alert_df[
        alert_df["Risk_Level"].astype(str).str.upper()
        == "MEDIUM"
    ]
)

low_count = len(
    alert_df[
        alert_df["Risk_Level"].astype(str).str.upper()
        == "LOW"
    ]
)


with priority_col1:

    st.metric(
        "🔴 High Priority",
        f"{high_count:,}"
    )


with priority_col2:

    st.metric(
        "🟠 Medium Priority",
        f"{medium_count:,}"
    )


with priority_col3:

    st.metric(
        "🟢 Low Priority",
        f"{low_count:,}"
    )


# ============================================================
# ALERT DISTRIBUTION
# ============================================================

st.divider()

left_col, right_col = st.columns(2)


# ------------------------------------------------------------
# Risk Level Distribution
# ------------------------------------------------------------

with left_col:

    st.subheader("Alert Distribution by Risk Level")

    risk_counts = (
        alert_df["Risk_Level"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk_Level",
        "Alerts"
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk_Level",
        y="Alerts",
        text="Alerts",
        title="Predicted Fraud Alerts by Risk Level"
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Number of Alerts",
        showlegend=False
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


# ------------------------------------------------------------
# Geography
# ------------------------------------------------------------

with right_col:

    st.subheader("Alerts by Transaction Geography")

    if "Is_International" in alert_df.columns:

        geography_df = alert_df.copy()

        geography_df["Geography"] = (
            geography_df["Is_International"]
            .map({
                0: "Domestic",
                1: "International"
            })
            .fillna("Unknown")
        )

        geography_counts = (
            geography_df["Geography"]
            .value_counts()
            .reset_index()
        )

        geography_counts.columns = [
            "Geography",
            "Alerts"
        ]

        fig_geo = px.pie(
            geography_counts,
            names="Geography",
            values="Alerts",
            hole=0.45,
            title="Fraud Alerts: Domestic vs International"
        )

        st.plotly_chart(
            fig_geo,
            use_container_width=True
        )

    else:

        st.info(
            "International transaction information "
            "is not available."
        )


# ============================================================
# ALERT TREND
# ============================================================

st.divider()

st.subheader("📈 Fraud Alert Trend")

if not alert_df.empty:

    trend_df = (
        alert_df
        .dropna(subset=["Transaction_Date"])
        .assign(
            Alert_Date=lambda x:
            x["Transaction_Date"].dt.date
        )
        .groupby("Alert_Date")
        .size()
        .reset_index(
            name="Alerts"
        )
    )

    fig_trend = px.line(
        trend_df,
        x="Alert_Date",
        y="Alerts",
        markers=True,
        title="Daily Predicted Fraud Alerts"
    )

    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Alerts"
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )

else:

    st.info(
        "No fraud alerts available for the selected filters."
    )


# ============================================================
# ALERT CHARACTERISTICS
# ============================================================

st.divider()

st.subheader("🔍 Alert Characteristics")

char_col1, char_col2 = st.columns(2)


# ------------------------------------------------------------
# Payment Method
# ------------------------------------------------------------

with char_col1:

    if "Payment_Method" in alert_df.columns:

        payment_df = (
            alert_df["Payment_Method"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        payment_df.columns = [
            "Payment_Method",
            "Alerts"
        ]

        fig_payment = px.bar(
            payment_df,
            x="Payment_Method",
            y="Alerts",
            text="Alerts",
            title="Fraud Alerts by Payment Method"
        )

        fig_payment.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_payment,
            use_container_width=True
        )

    else:

        st.info(
            "Payment method data is not available."
        )


# ------------------------------------------------------------
# Device Type
# ------------------------------------------------------------

with char_col2:

    if "Device_Type" in alert_df.columns:

        device_df = (
            alert_df["Device_Type"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        device_df.columns = [
            "Device_Type",
            "Alerts"
        ]

        fig_device = px.bar(
            device_df,
            x="Device_Type",
            y="Alerts",
            text="Alerts",
            title="Fraud Alerts by Device Type"
        )

        fig_device.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_device,
            use_container_width=True
        )

    else:

        st.info(
            "Device type data is not available."
        )


# ============================================================
# TOP ALERTS FOR INVESTIGATION
# ============================================================

st.divider()

st.subheader("🔎 Priority Investigation Queue")

st.caption(
    "Transactions are ranked using fraud probability and risk level "
    "to help investigators focus on the most important alerts."
)


investigation_df = alert_df.copy()


# Sort by probability first

investigation_df = investigation_df.sort_values(
    by=[
        "Fraud_Probability",
        "Transaction_Amount"
    ],
    ascending=[
        False,
        False
    ]
)


# Keep useful columns

display_columns = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Transaction_Amount",
    "Fraud_Probability",
    "Risk_Score",
    "Risk_Level",
]


optional_columns = [
    "Transaction_Type",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Primary_Risk_Reason",
]


for col in optional_columns:

    if col in investigation_df.columns:
        display_columns.append(col)


investigation_display = investigation_df[
    [
        col
        for col in display_columns
        if col in investigation_df.columns
    ]
].head(20).copy()


# Format probability

if "Fraud_Probability" in investigation_display.columns:

    investigation_display[
        "Fraud_Probability"
    ] = (
        investigation_display[
            "Fraud_Probability"
        ] * 100
    ).round(2)


# Format amount

if "Transaction_Amount" in investigation_display.columns:

    investigation_display[
        "Transaction_Amount"
    ] = investigation_display[
        "Transaction_Amount"
    ].round(2)


# Format date

if "Transaction_Date" in investigation_display.columns:

    investigation_display[
        "Transaction_Date"
    ] = investigation_display[
        "Transaction_Date"
    ].dt.strftime(
        "%Y-%m-%d %H:%M"
    )


# Rename columns

investigation_display = investigation_display.rename(
    columns={
        "Transaction_ID": "Transaction ID",
        "Customer_ID": "Customer ID",
        "Transaction_Date": "Transaction Date",
        "Transaction_Amount": "Amount",
        "Fraud_Probability": "Fraud Probability %",
        "Risk_Score": "Risk Score",
        "Risk_Level": "Risk Level",
        "Transaction_Type": "Transaction Type",
        "Payment_Method": "Payment Method",
        "Device_Type": "Device",
        "Is_International": "International",
        "Primary_Risk_Reason": "Primary Risk Reason",
    }
)


st.dataframe(
    investigation_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SELECT ALERT FOR INVESTIGATION
# ============================================================

st.divider()

st.subheader("🕵️ Alert Investigation")


if not alert_df.empty:

    alert_ids = (
        alert_df["Transaction_ID"]
        .astype(str)
        .tolist()
    )

    selected_transaction = st.selectbox(
        "Select an alert to investigate",
        alert_ids
    )

    selected_row = alert_df[
        alert_df["Transaction_ID"].astype(str)
        == selected_transaction
    ].iloc[0]


    # --------------------------------------------------------
    # Transaction profile
    # --------------------------------------------------------

    st.markdown("### Transaction Profile")

    profile_col1, profile_col2, profile_col3 = st.columns(3)


    with profile_col1:

        st.write(
            f"**Transaction ID:** "
            f"{selected_row.get('Transaction_ID', 'N/A')}"
        )

        st.write(
            f"**Customer ID:** "
            f"{selected_row.get('Customer_ID', 'N/A')}"
        )

        st.write(
            f"**Transaction Type:** "
            f"{selected_row.get('Transaction_Type', 'N/A')}"
        )


    with profile_col2:

        st.write(
            f"**Transaction Amount:** "
            f"₹{selected_row.get('Transaction_Amount', 0):,.2f}"
        )

        st.write(
            f"**Payment Method:** "
            f"{selected_row.get('Payment_Method', 'N/A')}"
        )

        st.write(
            f"**Device:** "
            f"{selected_row.get('Device_Type', 'N/A')}"
        )


    with profile_col3:

        st.write(
            f"**Location:** "
            f"{selected_row.get('Location', 'N/A')}"
        )

        st.write(
            f"**International:** "
            f"{'Yes' if selected_row.get('Is_International', 0) == 1 else 'No'}"
        )

        st.write(
            f"**Transaction Date:** "
            f"{selected_row.get('Transaction_Date', 'N/A')}"
        )


    # --------------------------------------------------------
    # Risk assessment
    # --------------------------------------------------------

    st.markdown("### Model Risk Assessment")

    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)


    with risk_col1:

        st.metric(
            "Fraud Probability",
            f"{selected_row.get('Fraud_Probability', 0) * 100:.2f}%"
        )


    with risk_col2:

        st.metric(
            "Risk Score",
            f"{selected_row.get('Risk_Score', 0):.2f}"
        )


    with risk_col3:

        st.metric(
            "Risk Level",
            str(
                selected_row.get(
                    "Risk_Level",
                    "N/A"
                )
            )
        )


    with risk_col4:

        st.metric(
            "Fraud Status",
            str(
                selected_row.get(
                    "Fraud_Status",
                    "Predicted Fraud"
                )
            )
        )


    # --------------------------------------------------------
    # Investigation recommendation
    # --------------------------------------------------------

    st.markdown("### Recommended Action")


    risk_level = str(
        selected_row.get(
            "Risk_Level",
            ""
        )
    ).upper()


    fraud_probability = float(
        selected_row.get(
            "Fraud_Probability",
            0
        )
    )


    if risk_level == "HIGH":

        st.error(
            "🔴 HIGH PRIORITY — Immediate investigation recommended."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "🟠 MEDIUM PRIORITY — Review transaction and supporting activity."
        )

    else:

        st.info(
            "🟢 LOW PRIORITY — Continue monitoring unless additional risk indicators appear."
        )


    # --------------------------------------------------------
    # Risk reason
    # --------------------------------------------------------

    if "Primary_Risk_Reason" in selected_row.index:

        reason = selected_row.get(
            "Primary_Risk_Reason",
            ""
        )

        if pd.notna(reason) and str(reason).strip():

            st.markdown("### Primary Risk Reason")

            st.info(
                str(reason)
            )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.divider()

st.subheader("💡 Operational Insights")


if total_alerts > 0:

    highest_probability = alert_df[
        "Fraud_Probability"
    ].max() * 100

    average_amount = alert_df[
        "Transaction_Amount"
    ].mean()


    insight_col1, insight_col2 = st.columns(2)


    with insight_col1:

        st.markdown(
            f"""
**Alert Monitoring**

- **{total_alerts:,}** transactions are currently classified as predicted fraud.
- **{high_count:,}** alerts are high priority.
- Average fraud probability among alerts is **{average_probability:.2f}%**.
- Highest observed fraud probability is **{highest_probability:.2f}%**.
"""
        )


    with insight_col2:

        st.markdown(
            f"""
**Financial Exposure**

- Total flagged transaction amount: **₹{flagged_amount:,.2f}**
- Average amount per fraud alert: **₹{average_amount:,.2f}**
- Alerts should be prioritized using both **risk probability and transaction value**.
- High-risk alerts should receive the fastest investigation response.
"""
        )


else:

    st.info(
        "No predicted fraud alerts are available under the current filters."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fraud Alerts & Monitoring | Final XGBoost Model | "
    "Tuned Decision Threshold = 0.69"
)