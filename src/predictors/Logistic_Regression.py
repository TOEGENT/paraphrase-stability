import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_logistic_regression(train_path: Path, val_path: Path, output_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Векторизация
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    y_train = train_df["k_over_N"].values
    
    X_val = vectorizer.transform(val_df["prompt"])
    y_val = val_df["k_over_N"].values
    
    # 3. Обучение Logistic Regression
    # max_iter увеличиваем, чтобы модель точно сошлась
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 4. Предсказание (выдаст вероятности от 0 до 1)
    preds = model.predict(X_val)
    
    # 5. Сохранение
    val_df["prediction"] = preds
    val_df.to_parquet(output_path, index=False)
    
    score = model.score(X_val, y_val)
    print(f"Logistic Regression trained. R^2 on Val: {score:.4f}")