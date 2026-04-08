"""
Data Cleaning Agent
Responsible for cleaning, deduplication, imputation, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the input DataFrame."""
    try:
        original_shape = df.shape

        # Remove duplicates
        df = df.drop_duplicates()
        print(f"[DataCleaningAgent] ✅ Removed duplicates. Rows: {original_shape[0]} → {df.shape[0]}")

        # Columns where 0 likely means missing data
        zero_as_nan_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_as_nan_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)

        # Fill missing values with column median
        df.fillna(df.median(numeric_only=True), inplace=True)
        print(f"[DataCleaningAgent] ✅ Missing values filled with median.")

        # Scale numeric features (excluding target)
        feature_cols = [c for c in df.columns if c != "Outcome"]
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        print(f"[DataCleaningAgent] ✅ Features scaled with StandardScaler.")
        print(f"[DataCleaningAgent] Final shape: {df_scaled.shape}")

        return df_scaled

    except Exception as e:
        print(f"[DataCleaningAgent] ❌ Error during cleaning: {e}")
        raise


if __name__ == "__main__":
    from data_ingestion_agent import load_data
    df = load_data()
    df_clean = clean_data(df)
    print(df_clean.head())
