"""
Prediction Agent
Loads trained model and generates predictions on test data.
"""

import pandas as pd
import numpy as np
import joblib
import os


MODEL_PATH = "model.pkl"


def predict(X_test: np.ndarray, model_path: str = MODEL_PATH) -> dict:
    """Load model and generate predictions."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        positive_count = int(predictions.sum())
        total = len(predictions)
        positive_rate = positive_count / total * 100

        print(f"[PredictionAgent] ✅ Predictions generated for {total} patients.")
        print(f"[PredictionAgent] At-risk patients: {positive_count} ({positive_rate:.1f}%)")

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "total_patients": total,
            "predicted_at_risk": positive_count,
            "positive_rate": positive_rate,
        }

    except Exception as e:
        print(f"[PredictionAgent] ❌ Error: {e}")
        raise


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from agents.data_ingestion_agent import load_data
    from agents.data_cleaning_agent import clean_data
    from agents.feature_engineering_agent import create_features
    from agents.model_training_agent import train_model

    df = create_features(clean_data(load_data()))
    result = train_model(df)
    preds = predict(result["X_test"])
    print(preds)
