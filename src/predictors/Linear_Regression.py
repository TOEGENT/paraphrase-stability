import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_linear_regression(train_path: Path, val_path: Path, output_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Векторизация (та же самая, что и для Dummy)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    y_train = train_df["k_over_N"].values
    
    X_val = vectorizer.transform(val_df["prompt"])
    y_val = val_df["k_over_N"].values
    
    # 3. Обучение Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Предсказание
    preds = model.predict(X_val)
    
    # 5. Сохранение
    val_df["prediction"] = preds
    val_df.to_parquet(output_path, index=False)
