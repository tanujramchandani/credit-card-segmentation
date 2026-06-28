import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
)

# ─── Cluster metadata ────────────────────────────────────────────────────────
CLUSTER_INFO = {
    0: {
        "label": "Inactive Users",
        "color": "#6c757d",
        "description": (
            "These customers maintain a low balance and show minimal purchase or cash advance "
            "activity. They rarely use their cards and tend to keep a dormant relationship with "
            "their credit line. Business strategy: re-engagement campaigns, incentive offers, "
            "or reduced annual fees to reactivate usage."
        ),
    },
    1: {
        "label": "Cash-Dependent Customers",
        "color": "#e63946",
        "description": (
            "These customers rely heavily on cash advances rather than purchases. They exhibit "
            "high cash advance frequency and transaction counts, with relatively low purchase "
            "activity. They often carry revolving balances and make only minimum payments. "
            "Business strategy: financial literacy outreach, targeted personal loan products, "
            "or balance transfer offers."
        ),
    },
    2: {
        "label": "Premium Spenders",
        "color": "#2a9d8f",
        "description": (
            "High-value customers with large credit limits, significant purchase volumes, and "
            "near-full payment rates. They favour one-off purchases and demonstrate strong "
            "repayment behaviour. Business strategy: premium rewards programmes, higher credit "
            "limit upgrades, and exclusive concierge or travel benefits."
        ),
    },
    3: {
        "label": "Installment Buyers",
        "color": "#e9c46a",
        "description": (
            "These customers prefer spreading payments over time through installment purchases. "
            "They show moderate balances, consistent installment purchase frequency, and "
            "manageable repayment patterns. Business strategy: EMI conversion offers, "
            "no-cost installment schemes, and loyalty rewards on installment spends."
        ),
    },
}

FEATURES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]

FEATURE_CONFIG = {
    "BALANCE":                         {"min": 0.0,  "max": 20000.0, "default": 1500.0,  "step": 50.0,   "fmt": "%.2f"},
    "BALANCE_FREQUENCY":               {"min": 0.0,  "max": 1.0,    "default": 0.8,     "step": 0.01,   "fmt": "%.2f"},
    "PURCHASES":                       {"min": 0.0,  "max": 50000.0, "default": 1000.0,  "step": 50.0,   "fmt": "%.2f"},
    "ONEOFF_PURCHASES":                {"min": 0.0,  "max": 40000.0, "default": 500.0,   "step": 50.0,   "fmt": "%.2f"},
    "INSTALLMENTS_PURCHASES":          {"min": 0.0,  "max": 22000.0, "default": 500.0,   "step": 50.0,   "fmt": "%.2f"},
    "CASH_ADVANCE":                    {"min": 0.0,  "max": 47000.0, "default": 0.0,     "step": 50.0,   "fmt": "%.2f"},
    "PURCHASES_FREQUENCY":             {"min": 0.0,  "max": 1.0,    "default": 0.5,     "step": 0.01,   "fmt": "%.2f"},
    "ONEOFF_PURCHASES_FREQUENCY":      {"min": 0.0,  "max": 1.0,    "default": 0.2,     "step": 0.01,   "fmt": "%.2f"},
    "PURCHASES_INSTALLMENTS_FREQUENCY":{"min": 0.0,  "max": 1.0,    "default": 0.3,     "step": 0.01,   "fmt": "%.2f"},
    "CASH_ADVANCE_FREQUENCY":          {"min": 0.0,  "max": 1.5,    "default": 0.0,     "step": 0.01,   "fmt": "%.2f"},
    "CASH_ADVANCE_TRX":                {"min": 0,    "max": 123,    "default": 0,       "step": 1,      "fmt": "%d"},
    "PURCHASES_TRX":                   {"min": 0,    "max": 358,    "default": 10,      "step": 1,      "fmt": "%d"},
    "CREDIT_LIMIT":                    {"min": 50.0, "max": 30000.0, "default": 5000.0,  "step": 50.0,   "fmt": "%.2f"},
    "PAYMENTS":                        {"min": 0.0,  "max": 50000.0, "default": 1500.0,  "step": 50.0,   "fmt": "%.2f"},
    "MINIMUM_PAYMENTS":                {"min": 0.0,  "max": 76000.0, "default": 500.0,   "step": 50.0,   "fmt": "%.2f"},
    "PRC_FULL_PAYMENT":                {"min": 0.0,  "max": 1.0,    "default": 0.1,     "step": 0.01,   "fmt": "%.2f"},
    "TENURE":                          {"min": 6,    "max": 12,     "default": 12,      "step": 1,      "fmt": "%d"},
}


def load_models():
    base = os.path.dirname(__file__)
    kmeans = joblib.load(os.path.join(base, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    return kmeans, scaler


def build_sidebar():
    st.sidebar.header("Customer Feature Inputs")
    st.sidebar.markdown("Adjust the sliders to match a customer's profile, then click **Classify Customer**.")
    values = {}
    for feat in FEATURES:
        cfg = FEATURE_CONFIG[feat]
        if feat in ("CASH_ADVANCE_TRX", "PURCHASES_TRX", "TENURE"):
            values[feat] = float(
                st.sidebar.number_input(
                    feat,
                    min_value=int(cfg["min"]),
                    max_value=int(cfg["max"]),
                    value=int(cfg["default"]),
                    step=int(cfg["step"]),
                    key=feat,
                )
            )
        else:
            values[feat] = st.sidebar.slider(
                feat,
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=float(cfg["step"]),
                format=cfg["fmt"],
                key=feat,
            )
    return values


def cluster_profile_table():
    data = {
        "Cluster": [
            "0 - Inactive Users",
            "1 - Cash-Dependent Customers",
            "2 - Premium Spenders",
            "3 - Installment Buyers",
        ],
        "Balance":          ["Low",      "High",    "Moderate", "Moderate"],
        "Purchases":        ["Very Low", "Low",     "Very High","Moderate"],
        "Cash Advance":     ["Minimal",  "High",    "Low",      "Low"],
        "Credit Limit":     ["Low",      "Moderate","Very High","Moderate"],
        "Payment Rate":     ["Low",      "Low",     "High",     "Moderate"],
        "Installments":     ["Minimal",  "Minimal", "Low",      "High"],
        "Typical Strategy": [
            "Re-engagement campaigns",
            "Loan products / balance transfer",
            "Premium rewards / limit upgrade",
            "EMI / no-cost installment offers",
        ],
    }
    return pd.DataFrame(data)


# ─── Main ────────────────────────────────────────────────────────────────────
st.title("Credit Card Customer Segmentation")
st.markdown(
    "This application uses a **KMeans clustering model** trained on ~9,000 credit card customers "
    "to classify a new customer into one of four behavioural segments. "
    "Fill in the customer's features in the sidebar and click **Classify Customer**."
)

st.markdown("---")

tab1, tab2 = st.tabs(["Customer Classifier", "Cluster Profiles"])

with tab1:
    values = build_sidebar()

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        classify = st.button("Classify Customer", type="primary", use_container_width=True)

    if classify:
        try:
            kmeans, scaler = load_models()
        except FileNotFoundError:
            st.error(
                "Model files not found. Please run the Jupyter notebook first to train and save "
                "`kmeans_model.pkl` and `scaler.pkl` in this directory."
            )
            st.stop()

        input_array = np.array([[values[f] for f in FEATURES]])
        scaled = scaler.transform(input_array)
        cluster_id = int(kmeans.predict(scaled)[0])
        info = CLUSTER_INFO[cluster_id]

        st.markdown("### Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(
                f"""
                <div style="
                    background-color:{info['color']}22;
                    border-left: 6px solid {info['color']};
                    padding: 20px 24px;
                    border-radius: 8px;
                ">
                    <p style="font-size:13px; color:#aaa; margin:0 0 4px 0;">Assigned Cluster</p>
                    <p style="font-size:28px; font-weight:700; color:{info['color']}; margin:0;">{info['label']}</p>
                    <p style="font-size:14px; color:#888; margin:4px 0 0 0;">Cluster {cluster_id}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.info(info["description"])

        st.markdown("#### Input Summary")
        input_df = pd.DataFrame([values]).T.reset_index()
        input_df.columns = ["Feature", "Value"]
        input_df["Value"] = input_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        st.dataframe(input_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Cluster Profiles Overview")
    st.markdown(
        "The table below summarises the behavioural characteristics and recommended business "
        "strategies for each of the four customer segments identified by the model."
    )
    st.dataframe(cluster_profile_table(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Segment Descriptions")

    for cid, info in CLUSTER_INFO.items():
        with st.expander(f"Cluster {cid} - {info['label']}", expanded=(cid == 2)):
            st.markdown(
                f"""
                <div style="border-left: 4px solid {info['color']}; padding-left: 14px;">
                    {info['description']}
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("Model: KMeans K=4  |  Preprocessing: StandardScaler  |  Dataset: ~8,950 credit card customers")
