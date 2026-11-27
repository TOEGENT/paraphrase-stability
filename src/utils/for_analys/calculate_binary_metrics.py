import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

def calculate_binary_metrics(y_data_path, pred_data_path, threshold: float = 0.5):
    """
    y_data_path  : parquet с колонкой k_over_N (0..1)
    pred_data_path: parquet с колонкой proba (вероятность класса 1)
    threshold    : порог перевода вероятности в класс (0/1)
    """
    y_df = pd.read_parquet(y_data_path)
    pred_df = pd.read_parquet(pred_data_path)

    # 1. Бинарный таргет
    y_true = (y_df["k_over_N"].values == 1.0).astype(int)

    # 2. Вероятности от модели
    # Поддерживаем оба варианта: колонка "proba" или "prediction" (если там вероятности)
    if "proba" in pred_df.columns:
        y_proba = pred_df["proba"].values
    else:
        y_proba = pred_df["prediction"].values

    # 3. Жесткие классы по порогу
    y_pred = (y_proba >= threshold).astype(int)

    # 4. Метрики
    # pos_label=0 — считаем метрики для "нестабильного" класса (k < 24)
    # Если хотите считать для класса 1, поменяйте на pos_label=1
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_class0": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_class0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_class0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_class1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_rate_pred": float(y_pred.mean()),   # доля предсказанных 1
        "positive_rate_true": float(y_true.mean()),   # доля истинных 1
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics