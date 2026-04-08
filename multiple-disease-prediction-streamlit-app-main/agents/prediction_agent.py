"""
Prediction Agent
Generates predictions using loaded models.
"""
import pandas as pd
import numpy as np

def run_predictions(model_data: dict, X_test: pd.DataFrame, disease_type: str = "diabetes"):
    """Generate predictions on a dataframe or array."""
    if model_data is None or X_test is None: return None

    model = model_data["model"]
    # Ensure X_test only contains the expected number of features
    # (Simple hack: standard datasets have 'Outcome' or 'target' at the end)
    features = X_test.copy()
    if "Outcome" in features.columns: features.drop(columns=["Outcome"], inplace=True)
    if "target" in features.columns: features.drop(columns=["target"], inplace=True)
    if "status" in features.columns: features.drop(columns=["status"], inplace=True)

    # For Parkinson's, drop the 'name' column if present
    if "name" in features.columns: features.drop(columns=["name"], inplace=True)

    # Convert to numpy and predict first 100 rows for the "pipeline" report
    data_to_predict = features.head(100).values
    predictions = model.predict(data_to_predict)

    at_risk_count = int(np.sum(predictions))
    total_count = len(predictions)

    print(f"[PredictionAgent] ✅ Predictions generated for {disease_type}")
    print(f"[PredictionAgent] At-risk found: {at_risk_count}/{total_count}")

    return {
        "predictions": predictions,
        "total": total_count,
        "at_risk": at_risk_count
    }
