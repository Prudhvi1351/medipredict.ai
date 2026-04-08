import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Multiple Disease Prediction",
    layout="wide",
    page_icon="🧑‍⚕️",
)

# ── Custom CSS (dark premium theme) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

.predict-card {
    background: linear-gradient(135deg, #1e1e3f, #2d2d5f);
    border: 1px solid #6c63ff44;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 8px 32px rgba(108,99,255,0.15);
}

.result-positive {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #fca5a5;
    font-weight: 600;
    font-size: 1.1rem;
}

.result-negative {
    background: linear-gradient(135deg, #14532d, #166534);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #86efac;
    font-weight: 600;
    font-size: 1.1rem;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a, #1a1a3e);
    border-right: 1px solid #6c63ff33;
}

.stTextInput > div > div > input {
    background: #1e1e3f;
    border: 1px solid #6c63ff55;
    border-radius: 8px;
    color: white;
}

.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(108,99,255,0.5);
    transform: translateY(-2px);
}

h1, h2, h3 { color: white !important; }
p, label, .stTextInput label { color: #c8c8e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(working_dir, "saved_models", filename)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load model `{filename}`: {e}")
        return None

diabetes_model      = load_model("diabetes_model.sav")
heart_disease_model = load_model("heart_disease_model.sav")
parkinsons_model    = load_model("parkinsons_model.sav")

# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#a78bfa; text-align:center;'>🏥 MediPredict</h2>",
        unsafe_allow_html=True,
    )
    selected = option_menu(
        "Disease Prediction",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0,
        styles={
            "container": {"background-color": "#12122a"},
            "icon": {"color": "#a78bfa", "font-size": "18px"},
            "nav-link": {"color": "#c8c8e0", "font-size": "15px"},
            "nav-link-selected": {
                "background": "linear-gradient(135deg,#6c63ff,#a78bfa)",
                "color": "white",
                "border-radius": "8px",
            },
        },
    )

# ── Helper ────────────────────────────────────────────────────────
def show_result(positive: bool, pos_msg: str, neg_msg: str):
    if positive:
        st.markdown(f'<div class="result-positive">🔴 {pos_msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-negative">🟢 {neg_msg}</div>', unsafe_allow_html=True)


def safe_floats(values):
    try:
        return [float(v) for v in values], None
    except ValueError as e:
        return None, f"Invalid input: {e}. Please enter numeric values only."


# ══════════════════════════════════════════════════════════════════
# Diabetes Prediction
# ══════════════════════════════════════════════════════════════════
if selected == "Diabetes Prediction":
    st.markdown(
        "<h1>🩸 Diabetes Prediction</h1>"
        "<p style='color:#a0a0c0;'>Enter patient vitals to predict diabetes risk using ML.</p>",
        unsafe_allow_html=True,
    )

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies           = st.text_input("Number of Pregnancies", placeholder="e.g. 2")
            SkinThickness         = st.text_input("Skin Thickness (mm)", placeholder="e.g. 20")
            DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.47")
        with col2:
            Glucose               = st.text_input("Glucose Level", placeholder="e.g. 120")
            Insulin               = st.text_input("Insulin Level (mu U/ml)", placeholder="e.g. 80")
            Age                   = st.text_input("Age", placeholder="e.g. 33")
        with col3:
            BloodPressure         = st.text_input("Blood Pressure (mm Hg)", placeholder="e.g. 70")
            BMI                   = st.text_input("BMI value", placeholder="e.g. 28.1")

    if st.button("🔍 Run Diabetes Test"):
        if diabetes_model is None:
            st.error("Model not loaded.")
        else:
            inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]
            values, err = safe_floats(inputs)
            if err:
                st.warning(err)
            else:
                pred = diabetes_model.predict([values])
                show_result(
                    pred[0] == 1,
                    "The person is diabetic.",
                    "The person is not diabetic.",
                )

# ══════════════════════════════════════════════════════════════════
# Heart Disease Prediction
# ══════════════════════════════════════════════════════════════════
elif selected == "Heart Disease Prediction":
    st.markdown(
        "<h1>❤️ Heart Disease Prediction</h1>"
        "<p style='color:#a0a0c0;'>Enter patient cardiac data to predict heart disease risk.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age      = st.text_input("Age", placeholder="e.g. 55")
        trestbps = st.text_input("Resting Blood Pressure", placeholder="e.g. 130")
        restecg  = st.text_input("Resting ECG results (0/1/2)", placeholder="e.g. 0")
        oldpeak  = st.text_input("ST Depression (exercise)", placeholder="e.g. 1.4")
        thal     = st.text_input("Thal (0=normal; 1=fixed; 2=reversible)", placeholder="e.g. 2")
    with col2:
        sex      = st.text_input("Sex (1=male, 0=female)", placeholder="e.g. 1")
        chol     = st.text_input("Serum Cholesterol (mg/dl)", placeholder="e.g. 250")
        thalach  = st.text_input("Max Heart Rate Achieved", placeholder="e.g. 150")
        slope    = st.text_input("Slope of Peak Exercise ST", placeholder="e.g. 1")
    with col3:
        cp       = st.text_input("Chest Pain Type (0-3)", placeholder="e.g. 2")
        fbs      = st.text_input("Fasting Blood Sugar >120 mg/dl (1=True)", placeholder="e.g. 0")
        exang    = st.text_input("Exercise Induced Angina (1=Yes)", placeholder="e.g. 0")
        ca       = st.text_input("Major Vessels (0-3)", placeholder="e.g. 1")

    if st.button("🔍 Run Heart Disease Test"):
        if heart_disease_model is None:
            st.error("Model not loaded.")
        else:
            inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
            values, err = safe_floats(inputs)
            if err:
                st.warning(err)
            else:
                pred = heart_disease_model.predict([values])
                show_result(
                    pred[0] == 1,
                    "The person has heart disease.",
                    "The person does not have heart disease.",
                )

# ══════════════════════════════════════════════════════════════════
# Parkinson's Prediction
# ══════════════════════════════════════════════════════════════════
elif selected == "Parkinsons Prediction":
    st.markdown(
        "<h1>🧠 Parkinson's Disease Prediction</h1>"
        "<p style='color:#a0a0c0;'>Enter voice measurement data to predict Parkinson's disease.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo             = st.text_input("MDVP:Fo (Hz)", placeholder="119.99")
        RAP            = st.text_input("MDVP:RAP", placeholder="0.00370")
        APQ3           = st.text_input("Shimmer:APQ3", placeholder="0.00830")
        NHR            = st.text_input("NHR", placeholder="0.00997")
        RPDE           = st.text_input("RPDE", placeholder="0.41")
    with col2:
        fhi            = st.text_input("MDVP:Fhi (Hz)", placeholder="157.30")
        PPQ            = st.text_input("MDVP:PPQ", placeholder="0.00554")
        APQ5           = st.text_input("Shimmer:APQ5", placeholder="0.01090")
        HNR            = st.text_input("HNR", placeholder="21.03")
        DFA            = st.text_input("DFA", placeholder="0.82")
    with col3:
        flo            = st.text_input("MDVP:Flo (Hz)", placeholder="74.99")
        DDP            = st.text_input("Jitter:DDP", placeholder="0.01109")
        APQ            = st.text_input("MDVP:APQ", placeholder="0.01457")
        spread1        = st.text_input("spread1", placeholder="-4.81")
        spread2        = st.text_input("spread2", placeholder="0.26")
    with col4:
        Jitter_percent = st.text_input("MDVP:Jitter (%)", placeholder="0.00370")
        Shimmer        = st.text_input("MDVP:Shimmer", placeholder="0.02971")
        DDA            = st.text_input("Shimmer:DDA", placeholder="0.02490")
        D2             = st.text_input("D2", placeholder="2.30")
    with col5:
        Jitter_Abs     = st.text_input("MDVP:Jitter (Abs)", placeholder="0.00002")
        Shimmer_dB     = st.text_input("MDVP:Shimmer (dB)", placeholder="0.28")
        PPE            = st.text_input("PPE", placeholder="0.28")

    if st.button("🔍 Run Parkinson's Test"):
        if parkinsons_model is None:
            st.error("Model not loaded.")
        else:
            inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            values, err = safe_floats(inputs)
            if err:
                st.warning(err)
            else:
                pred = parkinsons_model.predict([values])
                show_result(
                    pred[0] == 1,
                    "The person has Parkinson's disease.",
                    "The person does not have Parkinson's disease.",
                )
