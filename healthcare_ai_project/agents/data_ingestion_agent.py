"""
Data Ingestion Agent
Responsible for loading and validating patient CSV data.
"""

import pandas as pd
import os
import sys


def load_data(filepath: str = "data/patients.csv") -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")

        df = pd.read_csv(filepath)
        print(f"[DataIngestionAgent] ✅ Dataset loaded successfully.")
        print(f"[DataIngestionAgent] Shape: {df.shape}")
        print(f"[DataIngestionAgent] Columns: {list(df.columns)}")
        return df

    except FileNotFoundError as e:
        print(f"[DataIngestionAgent] ❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[DataIngestionAgent] ❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    df = load_data()
    print(df.head())
