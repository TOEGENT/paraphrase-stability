from pathlib import Path

from utils.for_analys import analyze_preds
import predictors
import pandas as pd
def train_predict_analyze_pipe(
    train_path: Path,
    val_path: Path,
    output_dir:Path,
    predictor
):

    # Predictor directory
    predictor_dir =output_dir
    predictor_dir.mkdir(parents=True, exist_ok=True)

    # Predictions
    train_predictions,val_predictions = predictor(train_path,val_path)

    # Save predictions
    train_pred_path = predictor_dir / "train_predictions.parquet"
    val_pred_path = predictor_dir / "val_predictions.parquet"

    train_predictions.to_parquet(train_pred_path, index=False)
    val_predictions.to_parquet(val_pred_path, index=False)

    # Analyze predictions
    analyze_preds.analyze_preds(
        y_data_path=train_path,
        pred_data_path=train_pred_path,
        split = "train"
    )

    analyze_preds.analyze_preds(
        y_data_path=val_path,
        pred_data_path=val_pred_path,
        split = "val"
    )

from pathlib import Path
import importlib.util
import sys

if __name__ == "__main__":

    
    train_path = Path("src\\datasets_out_split\\ag_news\\train.parquet")
    val_path = Path("src\\datasets_out_split\\ag_news\\val.parquet")

    predictors_dir = Path("src/predictors")
    predictions_base = Path("src/predictions")

    
    predictors_dir = Path("src/predictors")
    
    for module_file in sorted(predictors_dir.glob("*.py")):
        if module_file.stem.startswith("__"): 
            continue
        output_dir = predictions_base / module_file.stem
        if output_dir.exists():
            print(f"⏭️ {module_file.stem} уже существует, пропускаем")
            continue
        # Динамический импорт модуля по пути к файлу
        spec = importlib.util.spec_from_file_location(module_file.stem, module_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_file.stem] = module
        spec.loader.exec_module(module)
        
        # Проверка наличия функции
        if not hasattr(module, "train_predict"):
            print(f"⚠️ {module_file.stem}: нет функции train_predict, пропускаем")
            continue
            

        print(f"\n🚀 Running {module_file.stem}...")
        train_predict_analyze_pipe(
            train_path, val_path, output_dir, module.train_predict
        )