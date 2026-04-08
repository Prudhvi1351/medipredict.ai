"""
EDA Agent
Generates exploratory data analysis plots and saves them to disk.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Generate and save EDA plots."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # --- Age Distribution Histogram ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0f0f1a")
        fig.patch.set_facecolor("#0f0f1a")

        # Use original Age values for histogram
        age_col = "Age"
        ax.hist(df[age_col], bins=20, color="#6c63ff", edgecolor="#ffffff", alpha=0.85)
        ax.set_title("Age Distribution of Patients", color="white", fontsize=16, fontweight="bold")
        ax.set_xlabel("Age (scaled)", color="white", fontsize=12)
        ax.set_ylabel("Count", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#6c63ff")

        age_path = os.path.join(output_dir, "age_distribution.png")
        plt.tight_layout()
        plt.savefig(age_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[EDAAgent] ✅ Saved: {age_path}")

        # --- Correlation Heatmap ---
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0f0f1a")

        corr = df.corr(numeric_only=True)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            linewidths=0.5,
            linecolor="#2a2a3a",
            annot_kws={"size": 8, "color": "white"},
        )
        ax.set_title("Feature Correlation Heatmap", color="white", fontsize=16, fontweight="bold")
        ax.tick_params(colors="white")
        plt.setp(ax.get_xticklabels(), color="white")
        plt.setp(ax.get_yticklabels(), color="white")

        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[EDAAgent] ✅ Saved: {heatmap_path}")

    except Exception as e:
        print(f"[EDAAgent] ❌ Error during EDA: {e}")
        raise


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from agents.data_ingestion_agent import load_data
    from agents.data_cleaning_agent import clean_data
    df = clean_data(load_data())
    perform_eda(df)
