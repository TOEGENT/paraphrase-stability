import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sentence_transformers import SentenceTransformer
from pathlib import Path

def train_predict(train_path: Path, val_path: Path):
    # 1. Загрузка модели эмбеддингов (легкая и быстрая)
    print("Loading embedding model...")
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Загрузка данных
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 3. Превращаем тексты в векторы (эмбеддинги)
    print("Creating embeddings for Train...")
    X_train = model_emb.encode(train_df["prompt"].tolist(), show_progress_bar=True)
    y_train = train_df["k_over_N"].values
    
    print("Creating embeddings for Val...")
    X_val = model_emb.encode(val_df["prompt"].tolist(), show_progress_bar=True)
    
    # 4. Обучение MLP
    # Эмбеддинги уже нормализованы, scaler можно не использовать, но с ним иногда лучше
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128), # Нейросеть побольше, так как признаков 384
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True, # Остановится, если станет хуже
        validation_fraction=0.1
    )
    
    print("Training MLP...")
    model.fit(X_train, y_train)
    
    # 5. Предсказание
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    return pd.DataFrame({"prediction": train_preds}), pd.DataFrame({"prediction": val_preds})
