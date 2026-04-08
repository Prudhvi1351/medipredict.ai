"""
Data Ingestion Agent
Responsible for loading disease-specific datasets.
"""
import pandas as pd
import os

def load_data(disease_type: str = "diabetes"):
    """
    Load dataset for a specific disease.
    types: diabetes, heart, parkinsons
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_map = {
        "diabetes": "dataset/diabetes.csv",
        "heart": "dataset/heart.csv",
        "parkinsons": "dataset/parkinsons.csv"
    }

    filepath = os.path.join(base_dir, file_map.get(disease_type, "dataset/diabetes.csv"))

    if not os.path.exists(filepath):
        print(f"[DataIngestionAgent] ❌ Error: File not found at {filepath}")
        return None

    df = pd.read_csv(filepath)
    print(f"[DataIngestionAgent] ✅ {disease_type.capitalize()} dataset loaded successfully")
    print(f"[DataIngestionAgent] Shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data("diabetes")
    if df is not None:
        print(df.head())
