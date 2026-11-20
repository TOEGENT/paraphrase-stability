import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_path: str, output_dir: str):
    df = pd.read_parquet(input_path)
    
    # 1. Создаем категории (бины) для сохранения пропорций
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [0, 1, 2, 3]
    df['stratify_col'] = pd.cut(df['k_over_N'], bins=bins, labels=labels, include_lowest=True)
    
    # 2. Делим на Train+Val и Test (80% / 20%)
    train_val, test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['stratify_col']
    )
    
    # 3. Делим Train+Val на Train и Val (64% / 16% от общего)
    train, val = train_test_split(
        train_val, 
        test_size=0.2, # 20% от 80% = 16%
        random_state=42, 
        stratify=train_val['stratify_col']
    )
    
    
    train.to_parquet(f"{output_dir}/train.parquet", index=False)
    val.to_parquet(f"{output_dir}/val.parquet", index=False)
    test.to_parquet(f"{output_dir}/test.parquet", index=False)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
