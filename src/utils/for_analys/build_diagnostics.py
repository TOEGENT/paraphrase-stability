import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def build_diagnostics(y_data_path, pred_data_path):
    y_df = pd.read_parquet(y_data_path)
    pred_df = pd.read_parquet(pred_data_path)
    
    y_true = y_df["k_over_N"].values
    y_pred = pred_df["prediction"].values

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    indices = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(indices, y_true, label="y_true", alpha=0.7)
    ax.scatter(indices, y_pred, label="y_pred", alpha=0.7)



    ax.set(
        xlabel="Sample index",
        ylabel="Stability probability",
        title="True vs Predicted by sample",
        ylim=(0, 1)
    )

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    return fig

if __name__ == "__main__":
    build_diagnostics(Path("src\\datasets_out_split\\ag_news\\train.parquet"),
                      Path("src\\predictions\\CatBoost\\train_predictions.parquet"))