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
    # For Parkinson's, drop the 'name' column and any engineered features
    cols_to_drop = ["Outcome", "target", "status", "name", "Jitter_Shimmer_Ratio", "Glucose_BMI_Ratio", "BP_Level"]
    for col in cols_to_drop:
        if col in features.columns:
            features.drop(columns=[col], inplace=True)

    # Convert to numpy and predict first 100 rows
    data_to_predict = features.head(100).values
    
    # Debug: check shape
    if disease_type == "parkinsons" and data_to_predict.shape[1] != 22:
        print(f"[PredictionAgent] ⚠️ Parkinson's features: {data_to_predict.shape[1]} (Expect 22)")
    
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
