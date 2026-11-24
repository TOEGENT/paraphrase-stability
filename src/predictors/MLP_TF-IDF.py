import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def train_predict(train_path: Path, val_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Векторизация
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    y_train = train_df["k_over_N"].values
    
    X_val = vectorizer.transform(val_df["prompt"])
    
    # 3. Масштабирование (ВАЖНО для MLP!)
    # TF-IDF и так примерно в [0,1], но StandardScaler делает распределение более удобным для градиента
    scaler = StandardScaler(with_mean=False) # with_mean=False для разреженных матриц
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 4. Обучение MLP
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64), # Два слоя нейронов
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 5. Предсказание
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    return pd.DataFrame({"prediction": train_preds}), pd.DataFrame({"prediction": val_preds})
