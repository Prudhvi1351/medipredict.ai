"""
Autonomous Healthcare Analytics Pipeline
Orchestrates multiple agents to deliver end-to-end insights.
"""
import os
import sys

# Add current directory to path so agents can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.data_ingestion_agent import load_data
from agents.data_cleaning_agent import clean_data
from agents.eda_agent import perform_eda
from agents.feature_engineering_agent import create_features
from agents.model_training_agent import load_and_evaluate
from agents.prediction_agent import run_predictions
from agents.resource_estimation_agent import estimate_resources
from agents.insight_generation_agent import generate_report

def run_pipeline(disease_type="diabetes"):
    print(f"\n{'='*60}")
    print(f" 🚀 STARTING PIPELINE: {disease_type.upper()}")
    print(f"{'='*60}")

    # 1. Ingestion
    df_raw = load_data(disease_type)
    if df_raw is None: return

    # 2. Cleaning
    df_clean = clean_data(df_raw, disease_type)

    # 3. EDA
    perform_eda(df_clean, disease_type)

    # 4. Feature Engineering
    df_features = create_features(df_clean, disease_type)

    # 5. Model Training (Load/Validate)
    model_data = load_and_evaluate(disease_type)
    if model_data is None: return

    # 6. Prediction
    prediction_data = run_predictions(model_data, df_features, disease_type)

    # 7. Resource Estimation
    resource_data = estimate_resources(prediction_data["at_risk"], disease_type)

    # 8. Insight Generation
    report = generate_report(
        prediction_data, 
        resource_data, 
        model_data["accuracy"], 
        disease_type
    )

    # Save report to file
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        f"{disease_type}_insights_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"✅ Pipeline completed for {disease_type}. Report saved to {report_path}")

def main():
    print("🏥 MEDIPREDICT AI - AUTONOMOUS MULTI-DISEASE PIPELINE")
    diseases = ["diabetes", "heart", "parkinsons"]
    
    for disease in diseases:
        try:
            run_pipeline(disease)
        except Exception as e:
            print(f"❌ Error running pipeline for {disease}: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 ALL PIPELINES COMPLETED SUCCESSFULLY!")
    print("Run 'streamlit run app.py' to view the interactive dashboard.")

if __name__ == "__main__":
    main()
