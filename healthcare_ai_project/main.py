"""
MediPredict AI — Main Pipeline
Orchestrates the full healthcare analytics workflow.
"""

import sys
import os
import traceback
import requests

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.data_ingestion_agent import load_data
from agents.data_cleaning_agent import clean_data
from agents.eda_agent import perform_eda
from agents.feature_engineering_agent import create_features
from agents.model_training_agent import train_model
from agents.prediction_agent import predict
from agents.resource_estimation_agent import estimate_resources
from agents.insight_generation_agent import generate_insights


DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
DATA_PATH = "data/patients.csv"
COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def download_dataset(url: str = DATASET_URL, path: str = DATA_PATH) -> None:
    """Download dataset if it does not exist."""
    if os.path.exists(path):
        print(f"[Main] Dataset already exists at {path}. Skipping download.")
        return

    print(f"[Main] Downloading dataset from {url}...")
    for attempt in range(1, 3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                # Write header then data
                f.write(",".join(COLUMN_NAMES) + "\n")
                f.write(response.text)
            print(f"[Main] ✅ Dataset saved to {path}")
            return
        except Exception as e:
            print(f"[Main] ⚠️  Download attempt {attempt} failed: {e}")
            if attempt == 2:
                raise


def run_step(name: str, fn, *args, **kwargs):
    """Run a pipeline step with error handling and one retry."""
    for attempt in range(1, 3):
        try:
            print(f"\n{'='*60}")
            print(f"  STEP: {name} (attempt {attempt})")
            print(f"{'='*60}")
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            print(f"[Main] ❌ Step '{name}' failed (attempt {attempt}): {e}")
            if attempt == 2:
                print(f"[Main] ⚠️  Skipping step '{name}' after 2 failures.")
                traceback.print_exc()
                return None


def main():
    print("\n" + "🏥 " * 20)
    print("  MediPredict AI — Healthcare Analytics Pipeline Starting")
    print("🏥 " * 20 + "\n")

    # ── Step 0: Download dataset ─────────────────────────────────
    run_step("Download Dataset", download_dataset)

    # ── Step 1: Load data ────────────────────────────────────────
    df_raw = run_step("Load Data", load_data, DATA_PATH)
    if df_raw is None:
        print("[Main] 🚫 Cannot continue without data.")
        return

    # ── Step 2: Clean data ───────────────────────────────────────
    df_clean = run_step("Clean Data", clean_data, df_raw)
    if df_clean is None:
        df_clean = df_raw  # fallback to raw

    # ── Step 3: EDA ──────────────────────────────────────────────
    run_step("Perform EDA", perform_eda, df_clean, ".")

    # ── Step 4: Feature engineering ──────────────────────────────
    df_features = run_step("Create Features", create_features, df_clean)
    if df_features is None:
        df_features = df_clean

    # ── Step 5: Train model ──────────────────────────────────────
    train_result = run_step("Train Model", train_model, df_features)
    if train_result is None:
        print("[Main] 🚫 Model training failed. Exiting.")
        return

    # ── Step 6: Predict ──────────────────────────────────────────
    prediction_result = run_step("Predict", predict, train_result["X_test"])
    if prediction_result is None:
        print("[Main] 🚫 Prediction failed. Exiting.")
        return

    # ── Step 7: Estimate resources ───────────────────────────────
    resource_result = run_step(
        "Estimate Resources", estimate_resources, prediction_result["predicted_at_risk"]
    )
    if resource_result is None:
        resource_result = {}

    # ── Step 8: Generate insights ─────────────────────────────────
    run_step(
        "Generate Insights",
        generate_insights,
        prediction_result,
        resource_result,
        train_result["accuracy"],
    )

    print("\n" + "✅ " * 20)
    print("  Pipeline completed successfully!")
    print("  → Run: streamlit run dashboard/app.py  to launch the dashboard.")
    print("✅ " * 20 + "\n")


if __name__ == "__main__":
    main()
