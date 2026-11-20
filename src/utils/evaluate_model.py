import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

def evaluate_model(val_path: Path, pred_path: Path, model_name: str):
    val_df = pd.read_parquet(val_path)
    pred_df = pd.read_parquet(pred_path)
    
    # Убедимся, что строки совпадают
    assert len(val_df) == len(pred_df), "Разная длина выборок!"
    
    y_true = val_df["k_over_N"].values
    y_pred = pred_df["prediction"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"MAE:  {mae:.4f} (средняя ошибка в единицах k/N)")
    print(f"RMSE: {rmse:.4f} (штрафует за большие промахи)")
    print(f"R2:   {r2:.4f} (1.0 - идеально, 0.0 - как среднее)")
    print("------------------")
    
    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    output_dir = Path("datasets/LYR/experiment_1")
    val_file = output_dir / "val.parquet"
    
    results = []
    
    # Оцениваем Dummy
    results.append(evaluate_model(val_file, output_dir / "val_dummy_pred.parquet", "Dummy"))
    
    # Оцениваем Linear Regression
    results.append(evaluate_model(val_file, output_dir / "val_linreg_pred.parquet", "Linear Reg"))
    
    # Оцениваем CatBoost
    results.append(evaluate_model(val_file, output_dir / "val_catboost_pred.parquet", "CatBoost"))
    
    # Сводим всё в таблицу
    summary_df = pd.DataFrame(results)
    print("\n🏆 Итоговая таблица:")
    print(summary_df.sort_values(by="mae"))