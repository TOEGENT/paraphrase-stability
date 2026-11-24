import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from pathlib import Path

def train_xgboost_embeddings(train_path: Path, val_path: Path, output_path: Path):
    # 1. Загрузка модели эмбеддингов
    print("Loading embedding model...")
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Загрузка данных
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 3. Создание эмбеддингов
    print("Creating embeddings...")
    X_train = model_emb.encode(train_df["prompt"].tolist(), show_progress_bar=True)
    y_train = train_df["k_over_N"].values
    
    X_val = model_emb.encode(val_df["prompt"].tolist(), show_progress_bar=True)
    y_val = val_df["k_over_N"].values
    
    # 4. Обучение XGBoost
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror', # Или 'reg:absoluteerror' для MAE
        eval_metric='mae',
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 5. Предсказание
    preds = model.predict(X_val)
    
    # 6. Сохранение
    val_df["prediction"] = preds
    val_df.to_parquet(output_path, index=False)
    
    print(f"XGBoost trained. Best MAE: {model.best_score}")