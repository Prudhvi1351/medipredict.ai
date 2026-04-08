"""
EDA Agent
Generates histograms and correlation heatmaps.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def perform_eda(df: pd.DataFrame, disease_type: str = "diabetes"):
    """Generate and save EDA plots."""
    if df is None: return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "eda_outputs", disease_type)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histogram (using first numerical column as example)
    plt.figure(figsize=(10, 6))
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    target_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]

    sns.histplot(df[target_col], kde=True, color='blue')
    plt.title(f"{disease_type.capitalize()} - {target_col} Distribution")
    hist_path = os.path.join(output_dir, f"{disease_type}_distribution.png")
    plt.savefig(hist_path)
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"{disease_type.capitalize()} Correlation Heatmap")
    heatmap_path = os.path.join(output_dir, f"{disease_type}_correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    print(f"[EDAAgent] ✅ EDA plots generated for {disease_type} in {output_dir}")
    return {"hist": hist_path, "heatmap": heatmap_path}
