"""
MediPredict AI — Streamlit Dashboard
Interactive visualization of predictions, resource estimates, and EDA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

    .metric-card {
        background: linear-gradient(135deg, #1e1e3f, #2d2d5f);
        border: 1px solid #6c63ff44;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.3);
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0a0c0;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #a78bfa;
        border-left: 4px solid #6c63ff;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a, #1a1a3e);
        border-right: 1px solid #6c63ff33;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #a78bfa);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.5);
        transform: translateY(-2px);
    }

    h1, h2, h3 { color: white !important; }
    p, li, label { color: #c8c8e0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "patients.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pkl")
AGE_PLOT = os.path.join(PROJECT_ROOT, "age_distribution.png")
HEATMAP_PLOT = os.path.join(PROJECT_ROOT, "correlation_heatmap.png")


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def run_pipeline_cached():
    """Run the main pipeline to generate model + predictions."""
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "main.py")],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return result.returncode == 0, result.stdout, result.stderr


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediPredict AI")
    st.markdown("*Autonomous Healthcare Analytics*")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Select View",
        ["📊 Overview", "📁 Dataset", "🤖 Predictions", "🏨 Resources", "📈 EDA Plots"],
        label_visibility="collapsed",
    )
    st.divider()

    if st.button("🔄 Re-run Pipeline"):
        with st.spinner("Running pipeline..."):
            success, stdout, stderr = run_pipeline_cached()
        if success:
            st.success("Pipeline completed!")
        else:
            st.error("Pipeline failed. Check terminal.")
            st.code(stderr[:500])

    st.markdown("---")
    st.markdown(
        "<small style='color:#666'>Built with ❤️ using Streamlit + scikit-learn</small>",
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────
df = load_dataset()
model = load_model()

# ── Header ────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; background: linear-gradient(135deg,#6c63ff,#a78bfa); "
    "-webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.8rem;'>"
    "🏥 MediPredict AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#a0a0c0; font-size:1.1rem;'>"
    "Autonomous Healthcare Analytics & Resource Prediction System</p>",
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{"✅" if df is not None else "❌"}</div>'
            f'<div class="metric-label">Dataset Loaded</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        rows = len(df) if df is not None else 0
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{rows:,}</div>'
            f'<div class="metric-label">Total Records</div></div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{"✅" if model is not None else "❌"}</div>'
            f'<div class="metric-label">Model Ready</div></div>',
            unsafe_allow_html=True,
        )

    with col4:
        n_features = len(df.columns) - 1 if df is not None else 0
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{n_features}</div>'
            f'<div class="metric-label">Features Used</div></div>',
            unsafe_allow_html=True,
        )

    if df is not None:
        st.markdown('<div class="section-header">Outcome Distribution</div>', unsafe_allow_html=True)
        counts = df["Outcome"].value_counts().reset_index()
        counts.columns = ["Outcome", "Count"]
        counts["Label"] = counts["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})

        fig = px.pie(
            counts,
            names="Label",
            values="Count",
            color_discrete_sequence=["#6c63ff", "#f97316"],
            hole=0.55,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(font=dict(color="white")),
            margin=dict(t=20, b=20),
        )
        fig.update_traces(textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: Dataset
# ══════════════════════════════════════════════════════════════════
elif page == "📁 Dataset":
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)

    if df is not None:
        st.info(f"📄 **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns | Path: `{DATA_PATH}`")
        st.dataframe(df.head(50), use_container_width=True, height=400)

        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(3), use_container_width=True)
    else:
        st.error("Dataset not found. Run the pipeline first.")

# ══════════════════════════════════════════════════════════════════
# PAGE: Predictions
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Predictions":
    st.markdown('<div class="section-header">Model & Predictions</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Model not found. Please run the pipeline first.")
    else:
        st.success(f"✅ Model loaded: **{type(model).__name__}** with {model.n_estimators} estimators")

    if df is not None and model is not None:
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        feature_cols = [c for c in df.columns if c != "Outcome"]
        X = df[feature_cols].values
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        df_preds = df.copy()
        df_preds["Predicted"] = preds
        df_preds["Risk_Probability"] = np.round(probs * 100, 1)
        df_preds["Status"] = df_preds["Predicted"].map({0: "🟢 Low Risk", 1: "🔴 At Risk"})

        at_risk = int(preds.sum())
        total = len(preds)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{total:,}</div>'
                f'<div class="metric-label">Patients Assessed</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{at_risk:,}</div>'
                f'<div class="metric-label">Predicted At-Risk</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            rate = at_risk / total * 100
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{rate:.1f}%</div>'
                f'<div class="metric-label">Risk Rate</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Risk Probability Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df_preds, x="Risk_Probability", color="Status",
            nbins=30,
            color_discrete_map={"🟢 Low Risk": "#22c55e", "🔴 At Risk": "#ef4444"},
            barmode="overlay", opacity=0.75,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.6)",
            font_color="white", xaxis_title="Risk Probability (%)", yaxis_title="Count",
            legend_title="Status",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Patient Records with Predictions</div>', unsafe_allow_html=True)
        st.dataframe(
            df_preds[["Age", "Glucose", "BMI", "Risk_Probability", "Status"]].head(100),
            use_container_width=True, height=350,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE: Resources
# ══════════════════════════════════════════════════════════════════
elif page == "🏨 Resources":
    st.markdown('<div class="section-header">Hospital Resource Estimates</div>', unsafe_allow_html=True)

    if df is not None and model is not None:
        feature_cols = [c for c in df.columns if c != "Outcome"]
        preds = model.predict(df[feature_cols].values)
        at_risk = int(preds.sum())

        beds = round(at_risk * 0.3)
        doctors = round(beds / 5)
        cost = beds * 5000
        days = beds * 2

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, f"{beds:,}", "Beds Required", "🛏️"),
            (c2, f"{doctors:,}", "Doctors Needed", "👨‍⚕️"),
            (c3, f"${cost:,}", "Est. Cost (USD)", "💰"),
            (c4, f"{days:,}", "Treatment Days", "📅"),
        ]
        for col, val, label, icon in metrics:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div style="font-size:2rem">{icon}</div>'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-header">Resource Breakdown</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=["Beds", "Doctors", "Treatment Days"],
            y=[beds, doctors, days],
            marker=dict(
                color=["#6c63ff", "#a78bfa", "#f97316"],
                line=dict(color="#ffffff22", width=1),
            ),
            text=[beds, doctors, days],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.6)",
            font_color="white", showlegend=False,
            yaxis=dict(gridcolor="#ffffff15"),
            margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please run the pipeline to generate predictions first.")

# ══════════════════════════════════════════════════════════════════
# PAGE: EDA Plots
# ══════════════════════════════════════════════════════════════════
elif page == "📈 EDA Plots":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(AGE_PLOT):
            st.image(AGE_PLOT, caption="Age Distribution", use_container_width=True)
        else:
            st.info("Run the pipeline to generate the age distribution plot.")

    with col2:
        if os.path.exists(HEATMAP_PLOT):
            st.image(HEATMAP_PLOT, caption="Feature Correlation Heatmap", use_container_width=True)
        else:
            st.info("Run the pipeline to generate the correlation heatmap.")

    if df is not None:
        st.markdown('<div class="section-header">Interactive Feature Distribution</div>', unsafe_allow_html=True)
        feature = st.selectbox("Select Feature", [c for c in df.columns if c != "Outcome"])
        fig = px.histogram(
            df, x=feature, color="Outcome",
            color_discrete_map={0: "#6c63ff", 1: "#ef4444"},
            nbins=30, barmode="overlay", opacity=0.75,
            labels={"Outcome": "Diabetic"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.6)",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)
