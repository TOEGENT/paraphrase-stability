"""
Задача 3.1
    Принять от s2_llm_responce_generator
        yield prompt, anchor, answer
            yield -> 
                prompt
                    'Examples: 
                        "B": 0,
                        "C": 0,
                        "D": 1,
                        "E": 1
                    Now guess
                        "A": ?',
                anchor
                    "A"
                answer
                    1
            yield -> 
                prompt
                    'Examples: 
                        "C": 0,
                        "B": 0,
                        "D": 1,
                        "E": 1
                    Now guess
                        "A": ?',
                anchor
                    "A"
                answer
                    1
            ...

    И выдать
        jsonl датасет состоящий из
            {
                prompt1: str,
                answer1: str,
                k: int,
                N: int,
                k_over_N: float
                    k/N то бишь
            },
            {
                prompt2: str,
                answer2: str,
                k: int,
                N: int,
                k_over_N: float
            },
            ...
Решение
    (1)
        Передадим s2_llm_responce_generator в s2_k_over_n_generator и получим (1)
            yield prompt_first, answer_first, k, N, k_over_N для тройки prompt,anchor,answer
                    prompt: str (тот же),
                    answer: str (тот же),
                    k: int,
                    N: int,
                    k_over_N: float
                        k/N то бишь
    (2)
        Для каждой четвёрки prompt, k, N, k_over_N (2)
            строим словарик
                record = {
                    "prompt": prompt,
                    "answer": answer,
                    "k": k,
                    "N": N,
                    "k_over_N": k / N if int(N) > 0 else 0.0
                }
    (3)
        Сохраняем датасет из словариков в нужную папку
            f_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            f_json.flush()

"""


import json
import hashlib
from pathlib import Path
from utils import s3_k_over_n_generator

def s3_save_results_to_jsonl(llm_iterator, output_path: Path):
    """
    Сохраняет результаты в JSONL, а хеши — в отдельный .hashes файл.
    """

    #hash logic start
    hash_path = output_path.with_suffix('.hashes')
    existing_hashes = set()
    if hash_path.exists():
        with open(hash_path, 'r') as f:
            existing_hashes = set(line.strip() for line in f if line.strip())
            
    print(f"🛡️ Loaded {len(existing_hashes)} hashes from metadata.")
    #hash logic end

    with open(output_path, 'a', encoding='utf-8') as f_json, \
         open(hash_path, 'a', encoding='utf-8') as f_hash:

        k_over_n = s3_k_over_n_generator.s3_k_over_n_generator(llm_iterator) #(1)
        for prompt,anchor,answer, k, N in k_over_n: #(2)

            #hash
            current_hash = hashlib.md5(anchor.encode('utf-8')).hexdigest()

            #(2)
            record = {
                "prompt": prompt,
                "answer": answer,
                "k": k,
                "N": N,
                "k_over_N": k / N if int(N) > 0 else 0.0
            }
            
            # (3)
            f_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            f_json.flush()
            
            # hash_logic
            f_hash.write(current_hash + '\n')
            f_hash.flush()
            existing_hashes.add(current_hash)