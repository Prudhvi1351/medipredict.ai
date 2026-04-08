# 🏥 MediPredict AI

**Autonomous Healthcare Analytics & Resource Prediction System**

A complete end-to-end ML pipeline that predicts patient diabetes risk and estimates hospital resource requirements — with a beautiful interactive dashboard.

---

## 🚀 Quick Start

```bash
# 1. Navigate to project
cd healthcare_ai_project

# 2. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (downloads data, trains model, generates insights)
python main.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Dashboard opens at **http://localhost:8501**

---

## 📁 Project Structure

```
healthcare_ai_project/
├── agents/
│   ├── data_ingestion_agent.py       # Load & validate CSV
│   ├── data_cleaning_agent.py        # Deduplicate, impute, scale
│   ├── eda_agent.py                  # Generate EDA plots
│   ├── feature_engineering_agent.py  # Create risk_score feature
│   ├── model_training_agent.py       # Train RandomForestClassifier
│   ├── prediction_agent.py           # Load model & predict
│   ├── resource_estimation_agent.py  # Estimate beds/doctors/cost
│   └── insight_generation_agent.py   # Print analytics report
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── data/
│   └── patients.csv                  # Pima Indians Diabetes dataset
├── main.py                           # Full pipeline orchestrator
├── model.pkl                         # Trained model (auto-generated)
├── age_distribution.png              # EDA plot (auto-generated)
├── correlation_heatmap.png           # EDA plot (auto-generated)
└── requirements.txt
```

---

## 🤖 Agent Pipeline

```
Data Ingest → Clean → EDA → Feature Eng → Train Model → Predict → Resources → Insights
```

Each agent is modular, independently testable, and has built-in error handling with retry logic.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Overview | System status, outcome distribution pie chart |
| 📁 Dataset | Full dataset preview + descriptive statistics |
| 🤖 Predictions | Risk distribution, per-patient predictions |
| 🏨 Resources | Beds, doctors, cost, treatment day estimates |
| 📈 EDA Plots | Age histogram, correlation heatmap, interactive feature explorer |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | RandomForestClassifier (100 trees) |
| Train/Test Split | 80% / 20% |
| Accuracy | ~74% |
| Dataset | Pima Indians Diabetes (768 records, 8 features) |

---

## 🏨 Sample Output

```
Total Patients Assessed  :      154
Predicted At-Risk        :       46  (29.9%)
Beds Required            :       14
Doctors Required         :        3
Estimated Cost (USD)     :  $70,000
Treatment Days           :       28
```

---

## 📦 Tech Stack

- **ML**: scikit-learn (RandomForestClassifier)
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Persistence**: joblib

---

## 📄 Dataset

[Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) — auto-downloaded on first run.
