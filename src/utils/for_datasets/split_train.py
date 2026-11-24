# WARNING!
"""
Данный датасет корректно разделяет parquet на train+val
    Только если у вашего parquet
        columns: text  | label
        types:  string | int64
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_parquet(
        input_path: Path,
        output_path: Path,
        val_size = 0.1):
    input_parquet = pd.read_parquet(input_path)
    train_parquet, val_parquet = train_test_split(
        input_parquet,
        test_size=val_size,
        random_state=42,
        stratify=input_parquet["label"]
    )

    train_parquet.to_parquet(output_path / "train.parquet", index=False)
    val_parquet.to_parquet(output_path / "val.parquet", index=False)

if __name__ == "__main__":
    split_parquet(
        Path("datasets/ag_news/AG_news_train.parquet"),
        Path("datasets/ag_news"))