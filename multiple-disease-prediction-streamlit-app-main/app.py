import os
import streamlit as st
from streamlit_option_menu import option_menu
import sys
import pandas as pd
import io

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.data_cleaning_agent import clean_data
from agents.feature_engineering_agent import create_features
from agents.model_training_agent import load_and_evaluate
from agents.resource_estimation_agent import estimate_resources
from agents.prediction_agent import run_predictions

# Set page configuration
st.set_page_config(
    page_title="Autonomous Healthcare Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
.metric-card {
    background: linear-gradient(135deg, #1e1e3f, #2d2d5f);
    border: 1px solid #6c63ff44;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(108,99,255,0.1);
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.8rem; color: #a0a0c0; text-transform: uppercase; }
h1, h2, h3 { color: white !important; }
.stSidebar { background: #12122a !important; }
.stDataFrame { border: 1px solid #6c63ff33; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: #6c63ff;'>🏥 MediPredict AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0c0;'>Autonomous Healthcare Analytics & Resource Prediction System</p>", unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        "Disease Prediction",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction", "Upload Dataset"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person", "cloud-upload"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#6c63ff"}}
    )

# ══════════════════════════════════════════════════════════════════
# FEATURE: Upload Dataset (Batch Prediction)
# ══════════════════════════════════════════════════════════════════
if selected == "Upload Dataset":
    st.title("📁 Batch Disease Prediction")
    st.markdown("Upload a patient dataset (CSV) to run autonomous batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset uploaded: {uploaded_file.name} ({len(df_raw)} records)")
            
            # --- Disease Detection Logic ---
            cols = set(df_raw.columns)
            detected_disease = None
            if "Pregnancies" in cols and "Glucose" in cols:
                detected_disease = "diabetes"
            elif "age" in cols and "trestbps" in cols:
                detected_disease = "heart"
            elif "MDVP:Fo(Hz)" in cols or "PPE" in cols:
                detected_disease = "parkinsons"
            
            if not detected_disease:
                detected_disease = st.selectbox("Select Disease Type", ["diabetes", "heart", "parkinsons"])
            else:
                st.info(f"🔍 Auto-detected disease type: **{detected_disease.capitalize()}**")

            # --- Autonomous Pipeline Execution (MATCHING main.py) ---
            with st.spinner(f"Running autonomous pipeline for {detected_disease}..."):
                # 1. Clean data (Handle missing values/duplicates)
                df_clean = clean_data(df_raw, detected_disease)
                
                # 2. Add features
                df_features = create_features(df_clean, detected_disease)
                
                # 3. Load model
                model_info = load_and_evaluate(detected_disease)
                
                if model_info:
                    # 4. Predict
                    results = run_predictions(model_info, df_features, detected_disease)
                    
                    # 5. Build Result Tables
                    df_final = df_raw.copy()
                    df_final["Prediction"] = results["predictions"]
                    df_final["Prediction_Label"] = df_final["Prediction"].map({0: "Negative", 1: "Positive"})
                    
                    pos_df = df_final[df_final["Prediction"] == 1].drop(columns=["Prediction"])
                    neg_df = df_final[df_final["Prediction"] == 0].drop(columns=["Prediction"])
                    
                    st.divider()
                    
                    # --- Resource Estimation ---
                    st.subheader("🏨 Batch Resource Estimation")
                    res = estimate_resources(len(pos_df), detected_disease)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.markdown(f"<div class='metric-card'><div class='metric-val'>{res['beds_needed']}</div><div class='metric-label'>Total Beds</div></div>", unsafe_allow_html=True)
                    with c2: st.markdown(f"<div class='metric-card'><div class='metric-val'>{res['doctors_needed']}</div><div class='metric-label'>Staff Needed</div></div>", unsafe_allow_html=True)
                    with c3: st.markdown(f"<div class='metric-card'><div class='metric-val'>${res['estimated_cost']:,}</div><div class='metric-label'>Est. Cost</div></div>", unsafe_allow_html=True)
                    with c4: st.markdown(f"<div class='metric-card'><div class='metric-val'>{res['treatment_days']}</div><div class='metric-label'>Total Days</div></div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # --- Results Tables ---
                    col_p, col_n = st.columns(2)
                    with col_p:
                        st.subheader(f"🔴 Positive Patients ({len(pos_df)})")
                        st.dataframe(pos_df, use_container_width=True)
                        csv_pos = pos_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Positive Results", csv_pos, f"positive_{detected_disease}.csv", "text/csv")
                        
                    with col_n:
                        st.subheader(f"🟢 Negative Patients ({len(neg_df)})")
                        st.dataframe(neg_df, use_container_width=True)
                        csv_neg = neg_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Negative Results", csv_neg, f"negative_{detected_disease}.csv", "text/csv")
                        
                    st.success("✅ Autonomous processing complete. Accuracy optimized.")

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════
# Single User Prediction Pages
# ══════════════════════════════════════════════════════════════════
else:
    disease_map = {
        "Diabetes Prediction": "diabetes",
        "Heart Disease Prediction": "heart",
        "Parkinsons Prediction": "parkinsons"
    }
    disease_key = disease_map[selected]
    model_info = load_and_evaluate(disease_key)

    if selected == "Diabetes Prediction":
        st.title("🩸 Diabetes Analytics")
        col1, col2, col3 = st.columns(3)
        with col1: Pregnancies = st.text_input('Number of Pregnancies', '0')
        with col2: Glucose = st.text_input('Glucose Level', '120')
        with col3: BloodPressure = st.text_input('Blood Pressure value', '70')
        with col1: SkinThickness = st.text_input('Skin Thickness value', '20')
        with col2: Insulin = st.text_input('Insulin Level', '80')
        with col3: BMI = st.text_input('BMI value', '28.1')
        with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', '0.47')
        with col2: Age = st.text_input('Age of the Person', '33')
        inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    elif selected == "Heart Disease Prediction":
        st.title("❤️ Heart Disease Analytics")
        col1, col2, col3 = st.columns(3)
        with col1: age = st.text_input('Age', '55')
        with col2: sex = st.text_input('Sex (1=M, 0=F)', '1')
        with col3: cp = st.text_input('Chest Pain types (0-3)', '2')
        with col1: trestbps = st.text_input('Resting BP', '130')
        with col2: chol = st.text_input('Serum Cholestoral', '250')
        with col3: fbs = st.text_input('Fasting Blood Sugar > 120 (1=T)', '0')
        with col1: restecg = st.text_input('Resting ECG (0/1/2)', '0')
        with col2: thalach = st.text_input('Max Heart Rate', '150')
        with col3: exang = st.text_input('Exercise Induced Angina (1=Y)', '0')
        with col1: oldpeak = st.text_input('ST depression', '1.4')
        with col2: slope = st.text_input('Slope of ST segment', '1')
        with col3: ca = st.text_input('Major vessels (0-3)', '0')
        with col1: thal = st.text_input('thal (0-2)', '2')
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    elif selected == "Parkinsons Prediction":
        st.title("🧠 Parkinson's Analytics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: fo = st.text_input('MDVP:Fo(Hz)', '119.9')
        with col2: fhi = st.text_input('MDVP:Fhi(Hz)', '157.3')
        with col3: flo = st.text_input('MDVP:Flo(Hz)', '74.9')
        with col4: jp = st.text_input('Jitter(%)', '0.003')
        with col5: ja = st.text_input('Jitter(Abs)', '0.00002')
        with col1: rap = st.text_input('MDVP:RAP', '0.0037')
        with col2: ppq = st.text_input('MDVP:PPQ', '0.0055')
        with col3: ddp = st.text_input('Jitter:DDP', '0.011')
        with col4: shim = st.text_input('MDVP:Shimmer', '0.029')
        with col5: sdb = st.text_input('Shimmer(dB)', '0.28')
        with col1: apq3 = st.text_input('Shimmer:APQ3', '0.008')
        with col2: apq5 = st.text_input('Shimmer:APQ5', '0.010')
        with col3: apq = st.text_input('MDVP:APQ', '0.014')
        with col4: dda = st.text_input('Shimmer:DDA', '0.024')
        with col5: nhr = st.text_input('NHR', '0.009')
        with col1: hnr = st.text_input('HNR', '21.0')
        with col2: rpde = st.text_input('RPDE', '0.41')
        with col3: dfa = st.text_input('DFA', '0.82')
        with col4: s1 = st.text_input('spread1', '-4.81')
        with col5: s2 = st.text_input('spread2', '0.26')
        with col1: d2 = st.text_input('D2', '2.3')
        with col2: ppe = st.text_input('PPE', '0.28')
        inputs = [fo, fhi, flo, jp, ja, rap, ppq, ddp, shim, sdb, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, s1, s2, d2, ppe]

    if st.button(f"🔍 Run {selected}"):
        model = model_info["model"] if model_info else None
        if model is None: st.error("Model not found!")
        else:
            try:
                val_input = [float(x) for x in inputs]
                prediction = model.predict([val_input])
                st.divider()
                if prediction[0] == 1:
                    st.error(f"🔴 Result: High Risk of {disease_key.capitalize()} Detected")
                    res = estimate_resources(1, disease_key)
                    st.markdown("### 🏨 Resource Requirements")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.markdown(f"<div class='metric-card'><div class='metric-val'>{res['beds_needed']}</div><div class='metric-label'>Beds</div></div>", unsafe_allow_html=True)
                    with c2: st.markdown(f"<div class='metric-card'><div class='metric-val'>{res['doctors_needed']}</div><div class='metric-label'>Staff</div></div>", unsafe_allow_html=True)
                    with c3: st.markdown(f"<div class='metric-card'><div class='metric-val'>${res['estimated_cost']:,}</div><div class='metric-label'>Cost</div></div>", unsafe_allow_html=True)
                    with c4: st.markdown(f"<div class='metric-card'><div class='metric-val'>{res['treatment_days']}</div><div class='metric-label'>Days</div></div>", unsafe_allow_html=True)
                else:
                    st.success(f"🟢 Result: No {disease_key.capitalize()} Detected")
            except Exception as e:
                st.error(f"Error: {e}. Please check numeric inputs.")

    # EDA Visuals
    st.divider()
    st.markdown("### 📈 Automated EDA (Pipeline Output)")
    col_eda1, col_eda2 = st.columns(2)
    eda_base = f"eda_outputs/{disease_key}"
    hist_p = f"{eda_base}/{disease_key}_distribution.png"
    heat_p = f"{eda_base}/{disease_key}_correlation_heatmap.png"
    with col_eda1:
        if os.path.exists(hist_p): st.image(hist_p, caption=f"{disease_key.capitalize()} Distribution")
    with col_eda2:
        if os.path.exists(heat_p): st.image(heat_p, caption=f"{disease_key.capitalize()} Correlation")
