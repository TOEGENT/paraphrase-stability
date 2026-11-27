import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def build_binary_diagnostics(y_data_path, pred_data_path, threshold: float = 0.5):
    y_df = pd.read_parquet(y_data_path)
    pred_df = pd.read_parquet(pred_data_path)

    # Бинарный таргет: 1 = стабильный (k=24), 0 = нестабильный (k<24)
    y_true = (y_df["k_over_N"].values == 1.0).astype(int)

    # Вероятность класса 1 (если модель отдала только prediction — используем его)
    if "proba" in pred_df.columns:
        y_proba = pred_df["proba"].values
    else:
        y_proba = pred_df["prediction"].values.astype(float)

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length")

    y_pred = (y_proba >= threshold).astype(int)
    indices = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Истинные значения (0/1) — отображаем на двух уровнях
    ax.scatter(indices, y_true, label="y_true", alpha=0.4, s=10, color="tab:blue")

    # Предсказанные вероятности
    ax.scatter(indices, y_proba, label="y_proba", alpha=0.6, s=10, color="tab:orange")

    # Горизонтальная линия порога
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold={threshold}")

    # Подсветим ошибки (там, где y_pred != y_true)
    errors = y_pred != y_true
    if errors.any():
        ax.scatter(
            indices[errors], y_proba[errors],
            color="red", s=25, alpha=0.9, label="errors",
        )

    ax.set(
        xlabel="Sample index",
        ylabel="Stability probability",
        title="True vs Predicted probability by sample",
        ylim=(-0.05, 1.05),
    )

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    fig = build_diagnostics(
        Path("src/datasets_out_split/ag_news/train.parquet"),
        Path("src/predictions/CatBoost/train_predictions.parquet"),
    )
    fig.savefig("src/predictions/CatBoost/metadata/train/diag.png", dpi=300, bbox_inches="tight")