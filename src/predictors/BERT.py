import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from pathlib import Path
import numpy as np


"""
Задача
    Принять на вход
        датасет train, val
    Выдать
        датасет 


"""

def train_bert_finetune(train_path: Path, val_path: Path, output_path: Path):
    # 1. Загрузка данных
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Превращаем в формат HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['prompt', 'k_over_N']])
    val_dataset = Dataset.from_pandas(val_df[['prompt', 'k_over_N']])
    
    # 2. Токенизация
    model_name = "distilbert-base-uncased" # Берем легкую версию для скорости
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
    
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # 3. Загрузка модели (для регрессии num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # 4. Настройка обучения
    training_args = TrainingArguments(
        output_dir="./bert_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    # Функция для расчета MAE во время обучения
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mae = np.mean(np.abs(predictions.flatten() - labels))
        return {"mae": mae}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # 5. Обучение
    print("🚀 Starting BERT Fine-tuning...")
    trainer.train()
    
    # 6. Предсказание на валидации
    predictions = trainer.predict(val_tokenized)
    preds = predictions.predictions.flatten()
    
    # 7. Сохранение
    val_df["prediction"] = preds
    val_df.to_parquet(output_path, index=False)
    print(f"BERT Fine-tuned. MAE: {predictions.metrics['test_mae']:.4f}")

if __name__ == "__main__":
    train_bert_finetune(Path("src\\datasets_out_split\\ag_news\\train.parquet"),
                        Path("src\\datasets_out_split\\ag_news\\val.parquet"))