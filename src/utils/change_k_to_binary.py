import pandas as pd

df = pd.read_parquet("src/datasets_out/LYR/raw_stability.parquet")

df["k_over_N"] = (df["k_over_N"] == 1).astype(int)

df.to_parquet("src/datasets_out/LYR/raw_stability.parquet", index=False)
