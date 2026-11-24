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
        список схем если всё ок
        ошибку если не по схемам датасеты

        
    Решение
        Пройдёмся двойным циклом по папке с датасетами (1)
            for dataset_path_dir in Path.iterdir(datasets_dir):
                for dataset_path in Path.iterdir(dataset_path_dir):
        Для каждого будем извлекать его тип и конвертировать в dict (2)
            df = pd.read_parquet(dataset_path)
            schema = {df.dtypes.astype(str).to_dict()}

            чтобы добавить уникальные значения лейбла (3.1)
                labels_range = [int(item) for item in set(df["label"].unique())]
            мощность лейбла (3.2)
                labels_power = len(labels_range)

            и проверить её на ограничение в не более 10 (4)
        И выведем на экран (5)
            print(dataset_path,"ok")
            print(schema)

"""

def check_dataset_schemas(datasets_path_dir:Path,datasets_schema:dict):
    datasets_dir = datasets_path_dir
    for dataset_path_dir in Path.iterdir(datasets_dir): #(1)
        for dataset_path in Path.iterdir(dataset_path_dir): #(1)

            df = pd.read_parquet(dataset_path) #(2)
            schema = df.dtypes.astype(str).to_dict() #(2)

            if schema!=datasets_schema:
                raise TypeError(f"your dataset have {schema} schema,\n"
                                f"BUT must to have {config.DATASETS_SCHEMA} !")
            
            labels_range = [int(item) for item in set(df["label"].unique())] 
            labels_power = len(labels_range) 

            if labels_power>10: # (4)
                raise ValueError(f"dataset can have power of labels less than 10!\n"
                                 f"your have {labels_power}!")

            schema["label_power_len"] = len(labels_range) #(3.1)
            schema["label_items"] = labels_range #(3.2)

            print(dataset_path,"ok") #(5)
            print(schema) #(5)



if __name__ == "__main__":
    check_dataset_schemas(config.DATASETS_IN_DIR,config.DATASETS_SCHEMA)