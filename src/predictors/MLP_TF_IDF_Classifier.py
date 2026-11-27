import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
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
    X_val = vectorizer.transform(val_df["prompt"])
    
    # 3. Бинарный таргет (1 = стабильно k=24, 0 = нестабильно k<24)
    y_train = (train_df["k_over_N"].values == 1.0).astype(int)
    y_val = (val_df["k_over_N"].values == 1.0).astype(int)
    
    # 4. Масштабирование (ВАЖНО для MLP!)
    scaler = StandardScaler(with_mean=False)  # with_mean=False для разреженных матриц
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 5. Обучение MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,          # остановится, если val_loss начнет расти
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    model.fit(X_train_scaled, y_train)
    
    # 6. Предсказание (классы и вероятности)
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)
    
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    val_proba = model.predict_proba(X_val_scaled)[:, 1]

    return (
        pd.DataFrame({"prediction": train_preds, "proba": train_proba}),
        pd.DataFrame({"prediction": val_preds, "proba": val_proba})
    )