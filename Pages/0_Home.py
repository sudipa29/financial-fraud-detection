import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Financial Fraud Detection System")

st.markdown(
    """
    ### Machine Learning Based Financial Fraud Detection

    This dashboard provides an interactive environment for
    monitoring financial transactions, identifying suspicious
    activity, analyzing fraud patterns, evaluating model
    performance, and supporting fraud investigation decisions.
    """
)

st.markdown("---")


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader("📌 Project Overview")

st.write(
    """
    The Financial Fraud Detection System uses machine learning
    techniques to identify potentially fraudulent financial
    transactions.

    The system combines transaction characteristics,
    behavioral indicators, transaction patterns, and model
    predictions to assign fraud probabilities and risk levels.

    The dashboard provides analytical and operational views
    to help users understand fraud patterns, investigate
    suspicious transactions, monitor alerts, and make
    data-driven decisions.
    """
)


# ============================================================
# DASHBOARD MODULES
# ============================================================

st.subheader("📊 Dashboard Modules")

st.caption(
    "Use the navigation menu on the left to explore all "
    "eight dashboard modules."
)


# ============================================================
# MODULE INFORMATION
# ============================================================

modules = [
    (
        "1️⃣ 📊 Executive Overview",
        "High-level fraud KPIs, transaction trends, "
        "fraud exposure, risk distribution, and important "
        "business indicators.",
    ),
    (
        "2️⃣ 🔍 Transaction & Fraud Analysis",
        "Detailed analysis of transaction behavior, "
        "fraud patterns, transaction characteristics, "
        "and fraudulent activity.",
    ),
    (
        "3️⃣ ⚠️ Fraud Risk Analysis",
        "Analysis of fraud risk levels, high-risk "
        "transactions, risk indicators, and suspicious "
        "transaction patterns.",
    ),
    (
        "4️⃣ 🤖 Model Performance & Evaluation",
        "Evaluation of the final XGBoost model using "
        "accuracy, precision, recall, F1, ROC-AUC, "
        "PR-AUC, and confusion matrix analysis.",
    ),
    (
        "5️⃣ 🔎 Fraud Predictions & Investigation",
        "Transaction-level fraud predictions, fraud "
        "probabilities, risk scores, investigation, "
        "and filtering capabilities.",
    ),
    (
        "6️⃣ 🧠 Fraud Drivers & Explainability",
        "Analysis of important fraud indicators, "
        "behavioral patterns, transaction characteristics, "
        "and risk reasons behind predicted fraud.",
    ),
    (
        "7️⃣ 🚨 Fraud Alerts & Monitoring",
        "Operational monitoring of predicted fraud alerts, "
        "risk priorities, flagged values, investigation "
        "queues, and alert trends.",
    ),
    (
        "8️⃣ 📈 Business Summary & Decision Support",
        "Management-level summary of fraud exposure, "
        "model effectiveness, key fraud drivers, "
        "recommendations, and decision guidelines.",
    ),
]


# ============================================================
# DISPLAY MODULES — 2 COLUMN GRID
# ============================================================

for i in range(0, len(modules), 2):

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT CARD
    # --------------------------------------------------------

    with col1:

        title, description = modules[i]

        with st.container(border=True):

            st.subheader(title)

            st.write(description)

    # --------------------------------------------------------
    # RIGHT CARD
    # --------------------------------------------------------

    with col2:

        title, description = modules[i + 1]

        with st.container(border=True):

            st.subheader(title)

            st.write(description)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

st.markdown("---")

st.subheader("🤖 Machine Learning Model")

st.info(
    """
    The fraud detection pipeline uses machine learning
    classification techniques to estimate the probability
    that a transaction is fraudulent.

    The final selected model is XGBoost.

    A tuned decision threshold of 0.69 is used to classify
    transactions as predicted fraud or legitimate.

    The model is designed as a decision-support system,
    allowing fraud analysts to investigate high-risk
    transactions and prioritize potential fraud alerts.
    """
)


# ============================================================
# KEY MODEL HIGHLIGHTS
# ============================================================

st.subheader("📌 Key Model Highlights")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Model",
        "XGBoost",
    )


with col2:

    st.metric(
        "Decision Threshold",
        "0.69",
    )


with col3:

    st.metric(
        "Temporal Recall",
        "85.23%",
    )


with col4:

    st.metric(
        "ROC-AUC",
        "0.9021",
    )


# ============================================================
# BUSINESS DECISION FRAMEWORK
# ============================================================

st.markdown("---")

st.subheader("🚦 Fraud Decision Framework")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# HIGH RISK
# ------------------------------------------------------------

with col1:

    st.error(
        """
        🔴 HIGH RISK

        **Immediate Investigation**

        Prioritize the transaction for
        analyst review.
        """
    )


# ------------------------------------------------------------
# MEDIUM RISK
# ------------------------------------------------------------

with col2:

    st.warning(
        """
        🟠 MEDIUM RISK

        **Enhanced Review**

        Review supporting transaction
        and customer activity.
        """
    )


# ------------------------------------------------------------
# LOW RISK
# ------------------------------------------------------------

with col3:

    st.success(
        """
        🟢 LOW RISK

        **Continue Monitoring**

        No immediate investigation unless
        additional indicators appear.
        """
    )


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.markdown("---")

st.subheader("🔄 Project Workflow")

st.write(
    """
    Raw Transaction Data
    → MySQL ETL
    → Data Cleaning
    → Exploratory Data Analysis
    → Feature Engineering
    → Machine Learning
    → Model Evaluation
    → Threshold Optimization
    → Fraud Prediction
    → Risk Classification
    → Explainability
    → Fraud Alerts
    → Business Decision Support
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Financial Fraud Detection System | "
    "Zidio Development Internship | "
    "Machine Learning & Data Analytics"
)