"""
Data Cleaning Agent
Handles duplicates and missing values for multiple disease datasets.
"""
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, disease_type: str = "diabetes"):
    """Clean and prepare data for prediction."""
    if df is None: return None

    original_shape = df.shape
    df = df.drop_duplicates()

    # Fill missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # For diabetes, replace 0 with NaN for specific columns before median imputation
    if disease_type == "diabetes":
        cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"[DataCleaningAgent] ✅ Cleaned {disease_type} data. Shape: {original_shape} -> {df.shape}")
    return df
