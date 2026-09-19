from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Performance | Fraud Detection",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Model Performance & Evaluation")
st.markdown(
    """
    Evaluate the final **XGBoost fraud detection model** using the
    project's actual test-set performance at the tuned classification
    threshold of **0.69**.
    """
)

st.divider()


# ============================================================
# ACTUAL MODEL RESULTS
# ============================================================

THRESHOLD = 0.69

ACCURACY = 0.8040
PRECISION = 0.2907
RECALL = 0.8523
F1_SCORE = 0.4335
ROC_AUC = 0.9021
PR_AUC = 0.4349

TN = 729
FP = 183
FN = 13
TP = 75


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("🎯 Final XGBoost Performance")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Accuracy",
        f"{ACCURACY:.2%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{PRECISION:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{RECALL:.2%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{F1_SCORE:.2%}"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{ROC_AUC:.4f}"
    )

with col6:
    st.metric(
        "Threshold",
        f"{THRESHOLD:.2f}"
    )


st.markdown("")


# ============================================================
# CONFUSION MATRIX + PERFORMANCE SUMMARY
# ============================================================

left_col, right_col = st.columns([1.1, 1])


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

with left_col:

    st.subheader("🔍 Confusion Matrix")

    confusion_matrix = [
        [TN, FP],
        [FN, TP]
    ]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 20},
            hovertemplate=(
                "Actual: %{y}<br>"
                "Prediction: %{x}<br>"
                "Transactions: %{z}<extra></extra>"
            ),
            colorscale="Blues",
            showscale=False,
        )
    )

    fig_cm.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


# ------------------------------------------------------------
# PERFORMANCE INTERPRETATION
# ------------------------------------------------------------

with right_col:

    st.subheader("📈 Performance Summary")

    st.markdown(
        f"""
        ### Model
        **XGBoost — Final Fraud Detection Model**

        ### Classification Threshold
        **{THRESHOLD:.2f}**

        The threshold was tuned to improve fraud detection performance
        rather than relying on the default 0.50 probability cutoff.

        ### Key Results

        - **Recall:** {RECALL:.2%}
        - **Precision:** {PRECISION:.2%}
        - **F1 Score:** {F1_SCORE:.2%}
        - **ROC-AUC:** {ROC_AUC:.4f}
        - **PR-AUC:** {PR_AUC:.4f}

        ### Business Interpretation

        The model identifies approximately **85% of actual fraudulent
        transactions**, which is important because missing fraudulent
        transactions can create direct financial risk.

        The lower precision means that some legitimate transactions are
        also flagged as potentially fraudulent and may require additional
        review.
        """
    )


st.divider()


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

st.subheader("📋 Classification Results")

results_df = pd.DataFrame(
    {
        "Metric": [
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC",
            "Classification Threshold",
        ],
        "Value": [
            TN,
            FP,
            FN,
            TP,
            f"{ACCURACY:.4f}",
            f"{PRECISION:.4f}",
            f"{RECALL:.4f}",
            f"{F1_SCORE:.4f}",
            f"{ROC_AUC:.4f}",
            f"{PR_AUC:.4f}",
            f"{THRESHOLD:.2f}",
        ],
    }
)

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Business Insights")

insight1, insight2, insight3 = st.columns(3)

with insight1:

    st.markdown(
        f"""
        **🛡️ Strong Fraud Detection**

        The model achieves a recall of **{RECALL:.2%}**,
        meaning it captures the majority of fraudulent transactions
        in the test set.
        """
    )


with insight2:

    st.markdown(
        f"""
        **⚠️ False-Positive Trade-off**

        The model generated **{FP} false positives**.
        These legitimate transactions may require manual review.
        """
    )


with insight3:

    st.markdown(
        f"""
        **🎯 Tuned Decision Threshold**

        A threshold of **{THRESHOLD:.2f}** was selected to balance
        fraud detection and false-positive control.
        """
    )


# ============================================================
# MODEL PERFORMANCE NOTE
# ============================================================

st.info(
    """
    **Why the 0.69 threshold matters:**  
    In fraud detection, the default 0.50 probability threshold is not
    always optimal. Increasing the threshold changes the balance between
    precision and recall. The project's tuned threshold of 0.69 was selected
    based on validation performance before evaluating the model on the
    temporal test set.
    """
)