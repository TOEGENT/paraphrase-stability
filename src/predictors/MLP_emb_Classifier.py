import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from pathlib import Path

def train_predict(train_path: Path, val_path: Path):
    # 1. Загрузка модели эмбеддингов
    print("Loading embedding model...")
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Загрузка данных
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 3. Бинарный таргет (1 = стабильно k=24, 0 = нестабильно k<24)
    y_train = (train_df["k_over_N"].values == 1.0).astype(int)
    y_val = (val_df["k_over_N"].values == 1.0).astype(int)
    
    # 4. Эмбеддинги
    print("Creating embeddings for Train...")
    X_train = model_emb.encode(train_df["prompt"].tolist(), show_progress_bar=True)
    
    print("Creating embeddings for Val...")
    X_val = model_emb.encode(val_df["prompt"].tolist(), show_progress_bar=True)
    
    # 5. Масштабирование (важно для MLP!)
    # Эмбеддинги MiniLM не всегда нормализованы идеально для нейросети
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # 6. Обучение MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    
    print("Training MLP...")
    model.fit(X_train, y_train)
    
    # 7. Предсказание (классы и вероятности)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    return (
        pd.DataFrame({"prediction": train_preds, "proba": train_proba}),
        pd.DataFrame({"prediction": val_preds, "proba": val_proba})
    )