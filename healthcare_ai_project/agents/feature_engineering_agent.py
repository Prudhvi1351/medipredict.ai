"""
Feature Engineering Agent
Creates derived features for improved model performance.
"""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate feature-engineered columns."""
    try:
        df = df.copy()

        # Composite risk score
        df["risk_score"] = df["Glucose"] + df["BloodPressure"] + df["BMI"]

        print(f"[FeatureEngineeringAgent] ✅ Created feature: risk_score")
        print(f"[FeatureEngineeringAgent] risk_score stats:\n{df['risk_score'].describe()}")
        print(f"[FeatureEngineeringAgent] Updated shape: {df.shape}")

        return df

    except Exception as e:
        print(f"[FeatureEngineeringAgent] ❌ Error: {e}")
        raise


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from agents.data_ingestion_agent import load_data
    from agents.data_cleaning_agent import clean_data
    df = create_features(clean_data(load_data()))
    print(df[["Glucose", "BloodPressure", "BMI", "risk_score"]].head())
