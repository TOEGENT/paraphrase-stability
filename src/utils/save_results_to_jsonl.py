import json
import hashlib
from pathlib import Path
from utils import k_over_n_generator

def save_results_to_jsonl(llm_iterator, output_path: Path):
    """
    Сохраняет результаты в JSONL, а хеши — в отдельный .hashes файл.
    """
    hash_path = output_path.with_suffix('.hashes')
    
    # 1. Загружаем уже существующие хеши из метаданных
    existing_hashes = set()
    if hash_path.exists():
        with open(hash_path, 'r') as f:
            existing_hashes = set(line.strip() for line in f if line.strip())
            
    print(f"🛡️ Loaded {len(existing_hashes)} hashes from metadata.")

    # 2. Открываем JSONL для дописывания
    with open(output_path, 'a', encoding='utf-8') as f_json, \
         open(hash_path, 'a', encoding='utf-8') as f_hash:
        
        for prompt, k, N in k_over_n_generator.k_over_n_generator(llm_iterator):
            current_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            
            if current_hash in existing_hashes:
                continue
            
            record = {
                "prompt": prompt,
                "k": k,
                "N": N,
                "k_over_N": k / N if int(N) > 0 else 0.0
            }
            
            # Пишем данные
            f_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            f_json.flush()
            
            # Пишем хеш в метаданные
            f_hash.write(current_hash + '\n')
            f_hash.flush()
            
            existing_hashes.add(current_hash)