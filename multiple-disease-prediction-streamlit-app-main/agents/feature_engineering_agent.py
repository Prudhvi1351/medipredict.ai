"""
Feature Engineering Agent
Adds disease-specific features.
"""
import pandas as pd

def create_features(df: pd.DataFrame, disease_type: str = "diabetes"):
    """Add engineered features."""
    if df is None: return None

    df = df.copy()
    if disease_type == "diabetes":
        # Example: Glucose to BMI ratio
        df["Glucose_BMI_Ratio"] = df["Glucose"] / (df["BMI"] + 1)
    elif disease_type == "heart":
        # Example: Pulse Pressure
        if "trestbps" in df.columns:
             df["BP_Level"] = df["trestbps"].apply(lambda x: "High" if x > 140 else "Normal")
    elif disease_type == "parkinsons":
        # Example: Jitter to Shimmer ratio
        if "MDVP:Jitter(%)" in df.columns and "MDVP:Shimmer" in df.columns:
            df["Jitter_Shimmer_Ratio"] = df["MDVP:Jitter(%)"] / (df["MDVP:Shimmer"] + 0.0001)

    print(f"[FeatureEngineeringAgent] ✅ Features created for {disease_type}")
    return df
