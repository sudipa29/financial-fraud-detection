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
# NAVIGATION
# ============================================================

Pages = {
    "Dashboard": [

        st.Page(
            "Pages/0_Home.py",
            title="Home",
            icon="🏠",
        ),

        st.Page(
            "Pages/01_Executive_Overview.py",
            title="Executive Overview",
            icon="📊",
        ),

        st.Page(
            "Pages/02_Fraud_Analysis.py",
            title="Transaction & Fraud Analysis",
            icon="🔍",
        ),

        st.Page(
            "Pages/03_Risk_Analysis.py",
            title="Fraud Risk Analysis",
            icon="⚠️",
        ),

        st.Page(
            "Pages/4_model_performance.py",
            title="Model Performance & Evaluation",
            icon="🤖",
        ),

        st.Page(
            "Pages/05_fraud_predictions.py",
            title="Fraud Predictions & Investigation",
            icon="🔎",
        ),

        st.Page(
            "Pages/06_fraud_drivers.py",
            title="Fraud Drivers & Explainability",
            icon="🧠",
        ),

        st.Page(
            "Pages/07_fraud_alerts.py",
            title="Fraud Alerts & Monitoring",
            icon="🚨",
        ),

        st.Page(
            "Pages/08_business_summary.py",
            title="Business Summary & Decision Support",
            icon="📈",
        ),
    ]
}


# ============================================================
# RUN NAVIGATION
# ============================================================

pg = st.navigation(Pages)

pg.run()