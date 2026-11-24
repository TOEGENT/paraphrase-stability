import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

"""
y_dataset (оригинал)              pred_dataset (предикшены)
┌─────────────┬───────┐          ┌────────────┐
│ prompt      │ ...   │          │ prediction │
│ k           │ ...   │          ├────────────┤
│ N           │ ...   │          │ 0.123      │
│ k_over_N    │ ...   │          │ 0.456      │
│ ...         │ ...   │          │ 0.789      │
└─────────────┴───────┘          └────────────┘
     3933 строк                       3933 строк
     7 колонок                        1 колонка

calculate_metrics берет:
  y_true  = y_dataset["k_over_N"]      → shape (3933,)
  y_pred  = pred_dataset["prediction"] → shape (3933,)


"""
def calculate_metrics(y_data_path, pred_data_path):
    y_df = pd.read_parquet(y_data_path)
    pred_df = pd.read_parquet(pred_data_path)
    
    y_true = y_df["k_over_N"].values
    y_pred = pred_df["prediction"].values
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }

    return metrics