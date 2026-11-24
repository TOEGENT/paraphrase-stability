import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_catboost(train_path: Path, val_path: Path, output_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Векторизация (CatBoost умеет и в текст, но с TF-IDF он работает стабильнее на старте)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    y_train = train_df["k_over_N"].values
    
    X_val = vectorizer.transform(val_df["prompt"])
    
    # 3. Обучение CatBoost
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE', # Сразу оптимизируем под нашу метрику!
        verbose=False,
        random_seed=42
    )
    
    # CatBoost умеет работать с разреженными матрицами scipy
    model.fit(X_train, y_train, eval_set=(X_val, val_df["k_over_N"].values))
    
    # 4. Предсказание
    preds = model.predict(X_val)
    
    # 5. Сохранение
    val_df["prediction"] = preds
    val_df.to_parquet(output_path, index=False)
    
    print(f"CatBoost trained. Best MAE on Val: {model.best_score_['validation']['MAE']:.4f}")