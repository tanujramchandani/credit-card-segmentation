import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 8px 20px;
    border: 1px solid rgba(255,255,255,0.08);
    font-weight: 500;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #a5b4fc !important;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}

.cluster-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Cluster metadata ────────────────────────────────────────────────────────
CLUSTER_INFO = {
    0: {
        "label": "Inactive Users",
        "color": "#6c757d",
        "color_rgb": "108,117,125",
        "emoji": "😴",
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
        "color_rgb": "230,57,70",
        "emoji": "💸",
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
        "color_rgb": "42,157,143",
        "emoji": "💎",
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
        "color_rgb": "233,196,106",
        "emoji": "📦",
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
    "BALANCE":                          {"min": 0.0,   "max": 20000.0, "default": 1500.0, "step": 50.0,  "fmt": "%.2f"},
    "BALANCE_FREQUENCY":                {"min": 0.0,   "max": 1.0,    "default": 0.8,    "step": 0.01,  "fmt": "%.2f"},
    "PURCHASES":                        {"min": 0.0,   "max": 50000.0, "default": 1000.0, "step": 50.0,  "fmt": "%.2f"},
    "ONEOFF_PURCHASES":                 {"min": 0.0,   "max": 40000.0, "default": 500.0,  "step": 50.0,  "fmt": "%.2f"},
    "INSTALLMENTS_PURCHASES":           {"min": 0.0,   "max": 22000.0, "default": 500.0,  "step": 50.0,  "fmt": "%.2f"},
    "CASH_ADVANCE":                     {"min": 0.0,   "max": 47000.0, "default": 0.0,    "step": 50.0,  "fmt": "%.2f"},
    "PURCHASES_FREQUENCY":              {"min": 0.0,   "max": 1.0,    "default": 0.5,    "step": 0.01,  "fmt": "%.2f"},
    "ONEOFF_PURCHASES_FREQUENCY":       {"min": 0.0,   "max": 1.0,    "default": 0.2,    "step": 0.01,  "fmt": "%.2f"},
    "PURCHASES_INSTALLMENTS_FREQUENCY": {"min": 0.0,   "max": 1.0,    "default": 0.3,    "step": 0.01,  "fmt": "%.2f"},
    "CASH_ADVANCE_FREQUENCY":           {"min": 0.0,   "max": 1.5,    "default": 0.0,    "step": 0.01,  "fmt": "%.2f"},
    "CASH_ADVANCE_TRX":                 {"min": 0,     "max": 123,    "default": 0,      "step": 1,     "fmt": "%d"},
    "PURCHASES_TRX":                    {"min": 0,     "max": 358,    "default": 10,     "step": 1,     "fmt": "%d"},
    "CREDIT_LIMIT":                     {"min": 50.0,  "max": 30000.0, "default": 5000.0, "step": 50.0,  "fmt": "%.2f"},
    "PAYMENTS":                         {"min": 0.0,   "max": 50000.0, "default": 1500.0, "step": 50.0,  "fmt": "%.2f"},
    "MINIMUM_PAYMENTS":                 {"min": 0.0,   "max": 76000.0, "default": 500.0,  "step": 50.0,  "fmt": "%.2f"},
    "PRC_FULL_PAYMENT":                 {"min": 0.0,   "max": 1.0,    "default": 0.1,    "step": 0.01,  "fmt": "%.2f"},
    "TENURE":                           {"min": 6,     "max": 12,     "default": 12,     "step": 1,     "fmt": "%d"},
}

# Subset used for the radar chart
RADAR_FEATURES = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT",
    "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
]
RADAR_LABELS = [
    "Balance", "Purchases", "Cash Advance",
    "Credit Limit", "Payments", "Full Payment %",
    "Purchase Freq", "Cash Adv Freq",
]


# ─── Cached helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    kmeans = joblib.load(os.path.join(base, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    return kmeans, scaler


@st.cache_data
def load_dataset():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "credit card dataset.csv"))
    df.drop(columns=["CUST_ID"], errors="ignore", inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


@st.cache_data
def compute_elbow_silhouette(_scaler, k_max=10):
    df = load_dataset()
    X = _scaler.transform(df[FEATURES].values)
    ks, inertias, silhouettes = [], [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(X, labels, sample_size=2000, random_state=42)
        )
    return ks, inertias, silhouettes


@st.cache_data
def compute_cluster_sizes(_kmeans, _scaler):
    df = load_dataset()
    X = _scaler.transform(df[FEATURES].values)
    labels = _kmeans.predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    return unique.tolist(), counts.tolist()


@st.cache_data
def compute_cluster_centers_original(_kmeans, _scaler):
    return _scaler.inverse_transform(_kmeans.cluster_centers_)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def build_sidebar():
    st.sidebar.header("🎛️ Customer Feature Inputs")
    st.sidebar.markdown(
        "Adjust the sliders to match a customer's profile, "
        "then click **Classify Customer**."
    )
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


# ─── Profile table ────────────────────────────────────────────────────────────

def cluster_profile_table():
    return pd.DataFrame({
        "Cluster": [
            "0 — Inactive Users",
            "1 — Cash-Dependent",
            "2 — Premium Spenders",
            "3 — Installment Buyers",
        ],
        "Balance":      ["Low",      "High",    "Moderate", "Moderate"],
        "Purchases":    ["Very Low", "Low",     "Very High","Moderate"],
        "Cash Advance": ["Minimal",  "High",    "Low",      "Low"],
        "Credit Limit": ["Low",      "Moderate","Very High","Moderate"],
        "Payment Rate": ["Low",      "Low",     "High",     "Moderate"],
        "Installments": ["Minimal",  "Minimal", "Low",      "High"],
        "Strategy": [
            "Re-engagement campaigns",
            "Loan products / balance transfer",
            "Premium rewards / limit upgrade",
            "EMI / no-cost installment offers",
        ],
    })


# ─── Plot: Elbow Curve ────────────────────────────────────────────────────────

def plot_elbow(ks, inertias, chosen_k):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias,
        mode="lines+markers",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=8, color="#6366f1"),
        hovertemplate="K=%{x}<br>Inertia=%{y:,.0f}<extra></extra>",
        name="Inertia",
    ))
    # Highlight chosen K point
    chosen_idx = ks.index(chosen_k)
    fig.add_trace(go.Scatter(
        x=[chosen_k], y=[inertias[chosen_idx]],
        mode="markers",
        marker=dict(size=14, color="#f59e0b", symbol="star"),
        name=f"Chosen K={chosen_k}",
        hovertemplate=f"K={chosen_k}<br>Inertia={inertias[chosen_idx]:,.0f}<extra></extra>",
    ))
    fig.add_vline(
        x=chosen_k, line_dash="dash", line_color="#f59e0b",
        annotation_text=f" K={chosen_k} ★",
        annotation_font_color="#f59e0b",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Elbow Curve — Inertia vs Number of Clusters",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Within-Cluster SSE)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", tickmode="linear"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


# ─── Plot: Silhouette ─────────────────────────────────────────────────────────

def plot_silhouette(ks, silhouettes, chosen_k):
    colors = ["#f59e0b" if k == chosen_k else "#6366f1" for k in ks]
    fig = go.Figure(go.Bar(
        x=ks, y=silhouettes,
        marker_color=colors,
        hovertemplate="K=%{x}<br>Silhouette=%{y:.4f}<extra></extra>",
        name="Silhouette Score",
    ))
    fig.add_vline(
        x=chosen_k, line_dash="dash", line_color="#f59e0b",
        annotation_text=f" K={chosen_k} ★",
        annotation_font_color="#f59e0b",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Silhouette Score vs Number of Clusters",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score (higher = better)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(showgrid=False, tickmode="linear"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=360,
    )
    return fig


# ─── Plot: Cluster Size Pie ───────────────────────────────────────────────────

def plot_cluster_size_pie(unique_labels, counts):
    labels_str = [f"C{c} — {CLUSTER_INFO[c]['label']}" for c in unique_labels]
    colors = [CLUSTER_INFO[c]["color"] for c in unique_labels]
    fig = go.Figure(go.Pie(
        labels=labels_str,
        values=counts,
        marker=dict(colors=colors, line=dict(color="#1e1e2e", width=2)),
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}<br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Cluster Size Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(t=50, b=60, l=20, r=20),
        height=400,
    )
    return fig


# ─── Plot: Cluster Size Bar ───────────────────────────────────────────────────

def plot_cluster_size_bar(unique_labels, counts):
    labels_str = [f"C{c}" for c in unique_labels]
    colors = [CLUSTER_INFO[c]["color"] for c in unique_labels]
    hover = [f"{CLUSTER_INFO[c]['label']}<br>{cnt:,} customers" for c, cnt in zip(unique_labels, counts)]
    fig = go.Figure(go.Bar(
        x=labels_str, y=counts,
        marker_color=colors,
        text=counts, textposition="outside",
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title="Customer Count per Cluster",
        xaxis_title="Cluster",
        yaxis_title="Number of Customers",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=360,
    )
    return fig


# ─── Plot: Radar Chart ────────────────────────────────────────────────────────

def normalize_for_radar(values_dict, centers_original):
    radar_idx = [FEATURES.index(f) for f in RADAR_FEATURES]
    cluster_vals = centers_original[:, radar_idx]
    user_vals = np.array([[values_dict[f] for f in RADAR_FEATURES]])
    all_rows = np.vstack([cluster_vals, user_vals])
    mins = all_rows.min(axis=0)
    maxs = all_rows.max(axis=0)
    rng = np.where(maxs - mins == 0, 1, maxs - mins)
    norm = (all_rows - mins) / rng
    return norm[:4], norm[4]     # cluster_norm, user_norm


def plot_radar(values_dict, cluster_id, centers_original):
    cluster_norm, user_norm = normalize_for_radar(values_dict, centers_original)
    cats = RADAR_LABELS + [RADAR_LABELS[0]]     # close the polygon

    fig = go.Figure()
    for cid, cinfo in CLUSTER_INFO.items():
        row = cluster_norm[cid].tolist() + [cluster_norm[cid][0]]
        is_match = (cid == cluster_id)
        fig.add_trace(go.Scatterpolar(
            r=row, theta=cats,
            mode="lines",
            name=f"C{cid} Avg — {cinfo['label']}",
            line=dict(color=cinfo["color"], width=2.5 if is_match else 1.0),
            opacity=0.9 if is_match else 0.25,
            fill="toself" if is_match else "none",
            fillcolor=f"rgba({cinfo['color_rgb']},0.12)" if is_match else "rgba(0,0,0,0)",
        ))
    # User input overlay
    user_closed = user_norm.tolist() + [user_norm[0]]
    fig.add_trace(go.Scatterpolar(
        r=user_closed, theta=cats,
        mode="lines+markers",
        name="Your Input",
        line=dict(color="#ffffff", width=2, dash="dot"),
        marker=dict(size=7, color="#ffffff"),
    ))
    fig.update_layout(
        title="Your Profile vs All Cluster Averages",
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False,
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(color="#e2e8f0", size=11),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(t=60, b=80, l=60, r=60),
        height=460,
    )
    return fig


# ─── Plot: Confidence Bar ─────────────────────────────────────────────────────

def plot_distance_confidence(distances, cluster_id):
    conf = 1.0 / (1.0 + distances)
    conf_pct = (conf / conf.sum() * 100)
    labels = [f"C{c} — {CLUSTER_INFO[c]['label']}" for c in range(4)]
    colors_bar = [
        CLUSTER_INFO[c]["color"] if c == cluster_id else "rgba(255,255,255,0.15)"
        for c in range(4)
    ]
    fig = go.Figure(go.Bar(
        x=labels,
        y=conf_pct,
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in conf_pct],
        textposition="outside",
        hovertemplate="%{x}<br>Confidence: %{y:.1f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=cluster_id,
        y=conf_pct[cluster_id],
        text="✔ Best match",
        showarrow=True,
        arrowhead=2,
        arrowcolor=CLUSTER_INFO[cluster_id]["color"],
        font=dict(color=CLUSTER_INFO[cluster_id]["color"], size=12),
        yshift=20,
    )
    fig.update_layout(
        title="Cluster Confidence (derived from distances to cluster centers)",
        xaxis_title="Cluster",
        yaxis_title="Confidence (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            range=[0, float(conf_pct.max()) * 1.3],
        ),
        margin=dict(t=60, b=40, l=60, r=20),
        height=380,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# App Layout
# ═══════════════════════════════════════════════════════════════════════════════

st.title("💳 Credit Card Customer Segmentation")
st.markdown(
    "This application uses a **KMeans clustering model** trained on ~9,000 credit card customers "
    "to classify a new customer into one of four behavioural segments. "
    "Fill in the customer's features in the sidebar and click **Classify Customer**."
)

# Global colour legend
legend_html = " ".join([
    f'<span class="cluster-badge" '
    f'style="background:rgba({info["color_rgb"]},0.18);color:{info["color"]};'
    f'border:1px solid {info["color"]};">'
    f'{info["emoji"]} C{cid} {info["label"]}</span>'
    for cid, info in CLUSTER_INFO.items()
])
st.markdown(f'<div style="margin-bottom:8px;">{legend_html}</div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Customer Classifier", "📊 Cluster Profiles", "🧠 Model Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Customer Classifier
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    values = build_sidebar()

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        classify = st.button("🚀 Classify Customer", type="primary", use_container_width=True)

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

        # Euclidean distances from input to every cluster centre (scaled space)
        distances = np.linalg.norm(scaled - kmeans.cluster_centers_, axis=1)

        # ── Result card ──
        st.markdown("### Prediction Result")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(
                f"""
                <div style="
                    background: rgba({info['color_rgb']},0.10);
                    border-left: 6px solid {info['color']};
                    padding: 22px 26px;
                    border-radius: 12px;
                ">
                    <p style="font-size:13px;color:#aaa;margin:0 0 4px 0;">Assigned Cluster</p>
                    <p style="font-size:32px;font-weight:700;color:{info['color']};margin:0;">
                        {info['emoji']} {info['label']}
                    </p>
                    <p style="font-size:14px;color:#888;margin:6px 0 0 0;">Cluster {cluster_id}</p>
                    <p style="font-size:13px;color:#bbb;margin:10px 0 0 0;">
                        Distance to centre:
                        <strong style="color:{info['color']}">{distances[cluster_id]:.4f}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.info(info["description"])

        st.markdown("---")

        # ── Radar + Confidence ──
        st.markdown("#### 📡 Profile Comparison & Cluster Confidence")
        centers_original = compute_cluster_centers_original(kmeans, scaler)
        radar_col, conf_col = st.columns(2)
        with radar_col:
            st.plotly_chart(
                plot_radar(values, cluster_id, centers_original),
                use_container_width=True,
            )
        with conf_col:
            st.plotly_chart(
                plot_distance_confidence(distances, cluster_id),
                use_container_width=True,
            )

        # ── Raw distance table ──
        st.markdown("#### 📏 Raw Distances to Each Cluster Centre")
        dist_df = pd.DataFrame({
            "Cluster": [f"C{c} — {CLUSTER_INFO[c]['label']}" for c in range(4)],
            "Distance (scaled space)": [f"{distances[c]:.4f}" for c in range(4)],
            "Result": ["✅ Best match" if c == cluster_id else "" for c in range(4)],
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Input summary ──
        st.markdown("#### 📋 Input Feature Summary")
        input_df = pd.DataFrame([values]).T.reset_index()
        input_df.columns = ["Feature", "Value"]
        input_df["Value"] = input_df["Value"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            """
            <div style="
                background: rgba(99,102,241,0.06);
                border: 1px dashed rgba(99,102,241,0.3);
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                margin-top: 20px;
            ">
                <p style="font-size:48px;margin:0;">🎯</p>
                <p style="font-size:18px;font-weight:600;color:#a5b4fc;margin:8px 0 4px 0;">
                    Ready to classify a customer
                </p>
                <p style="color:#94a3b8;margin:0;">
                    Adjust the feature sliders in the sidebar, then click
                    <strong>Classify Customer</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cluster Profiles
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Cluster Profiles Overview")
    st.markdown(
        "The table below summarises the behavioural characteristics and recommended business "
        "strategies for each of the four customer segments identified by the model."
    )
    st.dataframe(cluster_profile_table(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🥧 Cluster Size Distribution")
    try:
        km2, sc2 = load_models()
        ul2, ct2 = compute_cluster_sizes(km2, sc2)
        p2col, b2col = st.columns(2)
        with p2col:
            st.plotly_chart(plot_cluster_size_pie(ul2, ct2), use_container_width=True)
        with b2col:
            st.plotly_chart(plot_cluster_size_bar(ul2, ct2), use_container_width=True)
    except FileNotFoundError:
        st.warning("Model files not found — cluster size distribution unavailable.")

    st.markdown("---")
    st.markdown("### Segment Descriptions")
    for cid, info in CLUSTER_INFO.items():
        with st.expander(f"{info['emoji']} Cluster {cid} — {info['label']}", expanded=(cid == 2)):
            st.markdown(
                f'<div style="border-left:4px solid {info["color"]};padding-left:14px;">'
                f'{info["description"]}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 Model Insights — Why K = 4?")
    st.markdown(
        "The plots below show the **Elbow Curve** (inertia drops) and **Silhouette Score** "
        "across K = 2 … 10. K = 4 sits at the inflection point of the elbow and delivers "
        "a good balance between cluster separation and business interpretability."
    )

    try:
        km3, sc3 = load_models()
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook to generate the `.pkl` files.")
        st.stop()

    with st.spinner("Computing elbow & silhouette scores (cached after first run) …"):
        ks, inertias, silhouettes = compute_elbow_silhouette(sc3)

    chosen_k = km3.n_clusters
    chosen_idx = ks.index(chosen_k)

    el_col, sil_col = st.columns(2)
    with el_col:
        st.plotly_chart(plot_elbow(ks, inertias, chosen_k), use_container_width=True)
    with sil_col:
        st.plotly_chart(plot_silhouette(ks, silhouettes, chosen_k), use_container_width=True)

    # Key metric cards
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<p style="color:#94a3b8;font-size:13px;margin:0 0 4px;">Chosen K</p>'
            f'<p style="font-size:36px;font-weight:700;color:#a5b4fc;margin:0;">{chosen_k}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m2:
        sil_val = silhouettes[chosen_idx]
        st.markdown(
            f'<div class="metric-card">'
            f'<p style="color:#94a3b8;font-size:13px;margin:0 0 4px;">Silhouette @ K={chosen_k}</p>'
            f'<p style="font-size:36px;font-weight:700;color:#2a9d8f;margin:0;">{sil_val:.4f}</p>'
            f'<p style="color:#64748b;font-size:12px;margin:4px 0 0;">Range −1 → 1 (higher is better)</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m3:
        inertia_val = inertias[chosen_idx]
        st.markdown(
            f'<div class="metric-card">'
            f'<p style="color:#94a3b8;font-size:13px;margin:0 0 4px;">Inertia @ K={chosen_k}</p>'
            f'<p style="font-size:28px;font-weight:700;color:#e9c46a;margin:0;">{inertia_val:,.0f}</p>'
            f'<p style="color:#64748b;font-size:12px;margin:4px 0 0;">Within-cluster sum of squares</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Cluster size in Model Insights too
    st.markdown("---")
    st.markdown("### 🎯 Cluster Size Distribution")
    try:
        ul3, ct3 = compute_cluster_sizes(km3, sc3)
        p3col, b3col = st.columns(2)
        with p3col:
            st.plotly_chart(plot_cluster_size_pie(ul3, ct3), use_container_width=True)
        with b3col:
            st.plotly_chart(plot_cluster_size_bar(ul3, ct3), use_container_width=True)
    except Exception as ex:
        st.warning(f"Could not load cluster sizes: {ex}")

    # Model summary table
    st.markdown("---")
    st.markdown("### 📐 Model Summary")
    st.dataframe(
        pd.DataFrame({
            "Parameter": ["Algorithm", "K (clusters)", "Preprocessing", "Init method", "n_init", "Dataset size"],
            "Value":     ["KMeans", str(chosen_k), "StandardScaler", "k-means++", "10", "~8,950 customers"],
        }),
        use_container_width=True,
        hide_index=True,
    )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: KMeans K=4  |  Preprocessing: StandardScaler  |  "
    "Dataset: ~8,950 credit card customers  |  Built with Streamlit + Plotly"
)
