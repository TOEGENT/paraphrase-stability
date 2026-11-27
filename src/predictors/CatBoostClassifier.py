import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_predict(train_path: Path, val_path: Path):
    # 1. Загрузка
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 2. Бинарный таргет (1 - k=24, 0 - k<24)
    y_train = (train_df["k_over_N"] == 1.0).astype(int).values
    y_val = (val_df["k_over_N"] == 1.0).astype(int).values
    
    # 3. Векторизация
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["prompt"])
    X_val = vectorizer.transform(val_df["prompt"])
    
    # 4. Обучение классификатора
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=False,
        random_seed=42
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # 5. Предсказание (классы 0/1 и вероятности)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    
    return (
        pd.DataFrame({"prediction": train_preds, "proba": train_proba}),
        pd.DataFrame({"prediction": val_preds, "proba": val_proba})
    )