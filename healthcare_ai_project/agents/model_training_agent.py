"""
Model Training Agent
Trains a RandomForestClassifier and saves the model to disk.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


MODEL_PATH = "model.pkl"


def train_model(df: pd.DataFrame, model_path: str = MODEL_PATH) -> dict:
    """Train RandomForestClassifier and save model."""
    try:
        target = "Outcome"
        feature_cols = [c for c in df.columns if c != target]

        X = df[feature_cols].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[ModelTrainingAgent] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"[ModelTrainingAgent] ✅ Model trained successfully.")
        print(f"[ModelTrainingAgent] Accuracy: {accuracy:.4f}")
        print(f"[ModelTrainingAgent] Classification Report:\n{classification_report(y_test, y_pred)}")

        # Save model
        joblib.dump(clf, model_path)
        print(f"[ModelTrainingAgent] ✅ Model saved to: {model_path}")

        return {
            "model": clf,
            "accuracy": accuracy,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_cols,
        }

    except Exception as e:
        print(f"[ModelTrainingAgent] ❌ Error: {e}")
        raise


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from agents.data_ingestion_agent import load_data
    from agents.data_cleaning_agent import clean_data
    from agents.feature_engineering_agent import create_features
    df = create_features(clean_data(load_data()))
    result = train_model(df)
    print(f"Accuracy: {result['accuracy']:.4f}")
