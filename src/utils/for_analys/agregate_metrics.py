import json
import pandas as pd
from pathlib import Path

def collect_metrics(predictions_dir: Path):
    rows = []

    for metrics_file in predictions_dir.rglob("metrics.json"):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Название модели — папка непосредственно перед metadata/train/val
        parts = metrics_file.relative_to(predictions_dir).parts
        model_name = parts[0]
        split = metrics_file.parent.name

        rows.append({
            "model": model_name,
            "split": split,
            **metrics
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("metrics.json не найдены")
        return df

    df = df.sort_values(["model", "split"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    predictions_dir = Path("src\\predictions")

    df = collect_metrics(predictions_dir)

    print(df.to_string(index=False))

    df.to_csv("metrics_comparison.csv", index=False)