import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_predict(train_path: Path, val_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Бинарный таргет (1 = стабильно k=24, 0 = нестабильно k<24)
    y_train = (train_df["k_over_N"].values == 1.0).astype(int)
    y_val = (val_df["k_over_N"].values == 1.0).astype(int)
    
    # 3. Векторизация
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    X_val = vectorizer.transform(val_df["prompt"])
    
    # 4. Обучение Logistic Regression
    # max_iter=1000 — чтобы сошлась на разреженных TF-IDF данных
    # class_weight='balanced' — на случай сильного дисбаланса
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # 5. Предсказание (классы 0/1 и вероятности)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    return (
        pd.DataFrame({"prediction": train_preds, "proba": train_proba}),
        pd.DataFrame({"prediction": val_preds, "proba": val_proba})
    )