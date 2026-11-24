"""
Задача:
    Имея датасет с данными
        "prompt": prompt,
        "k": k,
        "N": N,
        "k_over_N": k / N 
    Создать и обучить dummy

"""

import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_dummy_standardized(train_path: Path, val_path: Path, output_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Векторизация текста (стандартизация входа)
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # fit_transform для train, transform для val (чтобы не было утечки данных)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    y_train = train_df["k_over_N"].values
    
    X_val = vectorizer.transform(val_df["prompt"])
    
    # 3. Обучение Dummy (он всё равно проигнорирует X, но формат будет как у "взрослых" моделей)
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)
    
    # 4. Предсказание
    preds = model.predict(X_val)
    
    # 5. Сохранение
    val_df["prediction"] = preds
    val_df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    train_dummy_standardized(Path("train.parquet"), Path("val.parquet"), Path("val_dummy_pred.parquet"))