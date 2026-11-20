from itertools import permutations
import pandas as pd
import hashlib
from pathlib import Path
"""
Вход: DataFrame с колонками 'text' и 'label'
Выход: Генератор пар (anchor_text, examples_list)
        examples_list — это список из 4-х кортежей (text, label)
"""
def generate_llm_inputs(parquet_path:Path, hashes_path: Path):

    df = pd.read_parquet(parquet_path)
    existing_hashes = set()
    if hashes_path.exists():

        with open(hashes_path,"r") as f:
            existing_hashes = set(line.strip() for line in f if line.strip())
            print(f"🛡️ Found {len(existing_hashes)} completed groups. Skipping them.")
    else:
        print("🚀 Starting fresh generation.")

    for i in range(0, len(df) - 4, 5):
        group = df.iloc[i:i+5]
        
        # 1. Фиксируем якорь (только текст)
        anchor_text = group.iloc[0]["text"]

        anchor_hash = hashlib.md5(anchor_text.encode('utf-8')).hexdigest()

        if anchor_hash in existing_hashes:
            continue

        # 2. Берем остальные 4 текста и их лейблы
        others = group.iloc[1:]
        others_data = list(zip(others["text"], others["label"]))
        
        # 3. Перебираем все 24 комбинации контекста
        for perm in permutations(others_data):
            # perm теперь содержит кортежи: ((text1, label1), (text2, label2), ...)
            yield anchor_text, list(perm)