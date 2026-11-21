from pathlib import Path
import pandas as pd
import config

"""
Задача
    Принять на вход
        src/datasets папки датасетов в каждом из которых .parquet датасеты
            ag_news/
                data1.parquet
                data2.parquet
                ...
                datan.parquet
            CARER/
                data11.parquet
                data22.parquet
                ...

            LYR/
                daTA1.parquet
                daTA2.parquet
                ...
            ...
            others/
    Выдать
        папку datasets_schemas на уровне с datasets c содержимым в виде
            datasets/
            datasets_schemas/
                dataset1/
                    schema.json
                dataset2/
                    schema.json
                dataset3/
                    schema.json
                ...
                datasetn/
                    schema.json
        json

        
    Решение
        Пройдёмся двойным циклом по папке с датасетами (1)
            for dataset_path_dir in Path.iterdir(datasets_dir):
                for dataset_path in Path.iterdir(dataset_path_dir):
        Для каждого будем извлекать его колонки (2.1)
            и конвертировать в json (2.2)
"""

def make_dataset_schemas(datasets_path_dir:Path):
    datasets_dir = datasets_path_dir
    for dataset_path_dir in Path.iterdir(datasets_dir): #(1)
        for dataset_path in Path.iterdir(dataset_path_dir): #(1)
            df = pd.read_parquet(dataset_path)
            schema = df.head().to_json
            print(schema)


if __name__ == "__main__":
    make_dataset_schemas(config.DATASETS_PATH_DIR)