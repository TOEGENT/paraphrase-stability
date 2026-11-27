from pathlib import Path
import json
import matplotlib.pyplot as plt

from utils.for_analys import calculate_binary_metrics, build_binary_diagnositcs


def analyze_preds(
    y_data_path: Path,
    pred_data_path: Path,
    split: str
):
    pred_data_path = Path(pred_data_path)
    y_data_path = Path(y_data_path)

    predictor_dir = pred_data_path.parent
    metadata_dir = predictor_dir / "metadata" / split
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    metrics = calculate_binary_metrics.calculate_binary_metrics(y_data_path, pred_data_path)

    with open(metadata_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Build diagnostics
    diag = build_binary_diagnositcs.build_binary_diagnostics(y_data_path, pred_data_path)

    diag.savefig(
        metadata_dir / "diag.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(diag)

if __name__ == "__main__":
    analyze_preds(Path("src\\datasets_out_split\\ag_news\\val.parquet"),
                    Path("src\\predictions\\CatBoost\\val_predictions.parquet"),split="val")