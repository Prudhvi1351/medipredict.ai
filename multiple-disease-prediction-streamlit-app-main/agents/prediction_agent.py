"""
Prediction Agent
Generates predictions using loaded models with robust feature selection.
"""
import pandas as pd
import numpy as np

# Defined schemas for each disease to ensure input consistency
EXPECTED_FEATURES = {
    "diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "heart": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ],
    "parkinsons": [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", 
        "spread1", "spread2", "D2", "PPE"
    ]
}

def run_predictions(model_data: dict, X_input: pd.DataFrame, disease_type: str = "diabetes"):
    """
    Generate predictions with strict feature alignment.
    Ensures input shape matches the pre-trained model expectations.
    """
    if model_data is None or X_input is None: return None
    
    model = model_data["model"]
    expected = EXPECTED_FEATURES.get(disease_type, [])
    
    # 1. Feature Alignment Logic
    # We only keep the columns explicitly expected by the model
    # Matches case-insensitively to be robust to CSV differences
    features = pd.DataFrame()
    for col in expected:
        # Check for case-insensitive match
        match = None
        for c in X_input.columns:
            if c.strip().lower() == col.strip().lower():
                match = c
                break
        
        if match:
            features[col] = X_input[match]
        else:
            # If missing, fill with 0 to prevent crash, but log warning
            print(f"[PredictionAgent] ⚠️ Warning: Missing column '{col}'. Filling with zeros.")
            features[col] = 0.0

    # Convert to numpy and predict
    data_to_predict = features.values
    
    print(f"[PredictionAgent] Info: Using {data_to_predict.shape[1]} clinical features for {disease_type} prediction.")
    
    predictions = model.predict(data_to_predict)
    at_risk_count = int(np.sum(predictions))
    total_count = len(predictions)

    print(f"[PredictionAgent] ✅ Predictions generated. At-risk found: {at_risk_count}/{total_count}")

    return {
        "predictions": predictions,
        "total": total_count,
        "at_risk": at_risk_count,
        "features_used": list(features.columns)
    }
