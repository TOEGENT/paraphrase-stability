import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from pathlib import Path

def train_predict(train_path: Path, val_path: Path):
    # 1. Загрузка модели эмбеддингов
    print("Loading embedding model...")
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Загрузка данных
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 3. Создание эмбеддингов
    print("Creating embeddings...")
    X_train = model_emb.encode(train_df["prompt"].tolist(), show_progress_bar=True)
    X_val = model_emb.encode(val_df["prompt"].tolist(), show_progress_bar=True)
    
    # 4. Бинарный таргет
    y_train = (train_df["k_over_N"].values == 1.0).astype(int)
    y_val = (val_df["k_over_N"].values == 1.0).astype(int)
    
    # 5. Обучение XGBoost Classifier
    # Считаем scale_pos_weight для компенсации дисбаланса
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos if pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
        scale_pos_weight=spw
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 6. Предсказание (классы и вероятности)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    return (
        pd.DataFrame({"prediction": train_preds, "proba": train_proba}),
        pd.DataFrame({"prediction": val_preds, "proba": val_proba})
    )