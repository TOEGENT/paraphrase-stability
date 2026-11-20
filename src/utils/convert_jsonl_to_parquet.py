import pandas as pd

def convert_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path):
    """
    Читает JSONL и сохраняет как Parquet
    """
    df = pd.read_json(jsonl_path, lines=True)
    df.to_parquet(parquet_path, index=False)
    
    print(f"Converted {len(df)} records to {parquet_path}")