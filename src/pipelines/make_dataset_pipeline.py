"""
Задача
    Получить датасет отзывов 
    с оценкой их эмоционального окраса (pos / neg)
        text label
        "A"     1
        "B"     0
        "C"     1
            ...
    Выдать датасет для обучения регрессоров в котором хранятся
        prompt, answer, k, N, k_over_N
        
        prompt
            Промпт для модели состоящий из 
                few_shot
                    "A" : 1
                    "B" : 0
                    "C" : 1
                    "D" : 1
                И вопроса
                    "E" : ?
        answer
            Ответ модели на промпт
                1
        k
            количество совпадений ответа на вопрос с ответами
            на промпт вида 
                few_shot у которого поменяли местами примеры
                И того же вопроса
                    "E" : ?
        N
            Общее кол-во перестановок few_shot

    Решение
        Сперва Резделим весь датасет на группы по 5 (1)
                group1,group2,...
            которые будут нашими few_shot + anchor(question)
                (anchor,few_shot)1, (anchor,few_shot)2,...


        Затем для каждой группы переберём все возможные комбинации (2)
            few_shot -> few_shots = 
                (few_shot_ver1, 
                few_shot_ver2,
                    ...,
        Далее пробежимся по всем элементам few_shots (3.1)
            и создадим prompt для ллм как
                few_shot_ver
                    "A" : 1
                    "B" : 0
                    "C" : 1
                    "D" : 1
                + anchor
                    "E" : ?
            и получим от неё ответ
                1 or 0

            Пока получаем ответы сравним (3.2)
                сколько раз мы получили answer от первой версии перестановки
                от других перестановок
                и посчитаем соотношение
                    кол-во ответов у которых совпал ответ с первым
                        k
                    на
                    общее кол-во ответов (их 24)
                        N
                    
                        k_over_N

        Получив ответы и соотношение сохраним их в группы (4)
            (prompt, answer, k, N, k_over_N)1,
            (prompt, answer, k, N, k_over_N)2,

        И сохраним это всё в датасет
"""


import config
from pathlib import Path

# Импортируем все компоненты из utils
from utils import (
    s1_pair_generator,
    s3_save_results_to_jsonl,
    convert_jsonl_to_parquet,
    s2_llm_responce_generator
)


def make_dataset_pipeline(input_path: Path, output_dir: Path,label_map:dict):
    if input_path.exists():
        raise FileExistsError(
            f"Файл {input_path} уже существует! "
            "Удалите его вручную, если хотите перегенерировать датасет."
        )

    raw_jsonl_path = output_dir / f"{config.DATASET_OUTPUT_NAME}.jsonl"
    raw_parquet_path = output_dir / f"{config.DATASET_OUTPUT_NAME}.parquet"
    raw_hash_path = raw_jsonl_path.with_suffix('.hashes')
    

    input_iter = s1_pair_generator.s1_pair_generator(input_path,raw_hash_path) # (1)
    llm_iter = s2_llm_responce_generator.s2_llm_response_generator(input_iter,label_map) # (2)

    try:
        s3_save_results_to_jsonl.s3_save_results_to_jsonl(llm_iter, raw_jsonl_path) # (3.1) (3.2)
    finally:
        convert_jsonl_to_parquet.convert_jsonl_to_parquet(raw_jsonl_path,raw_parquet_path) # (4)    
    
if __name__ == "__main__":
    input_file = config.LYR_TRAIN_PATH
    output_dir = config.DATASETS_OUT_DIR / "LYR"
    label_map = config.LYR_LABEL_MAP
    make_dataset_pipeline(input_file, output_dir,label_map)
