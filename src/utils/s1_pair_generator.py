"""
    Задача 1
        Принять на вход 
            parquet_path
                text       label
                "A"    :    1 (pos)
                "B"    :    0 (neg)
                "C"    :    0
                "D"    :    1
                "E"    :    1
                    ...
        Выдать 
            yield якорный вопрос,якорный ответ, все комбинации few_shot вопросов
                yield ->
                        ( "A",1, (
                                (("A",1),("B",0),("C",0),("D",1),("E",1)),
                                (("B",0),("A",1),("C",0),("D",1),("E",1))
                                (("A",1),("C",0),("B",0),("D",1),("E",1)),
                                ... так 4! или 24 раза
                                ) )
                yield ->
                    ( "F",0, (аналогично только для G,H,I,J) )

    Решение
        Исходный датасет:
                text       label
                "A"    :    1 (pos)
                "B"    :    0 (neg)
                "C"    :    0
                "D"    :    1
                "E"    :    1
                    ...
        
        1) 
            Проходясь по всему датасету циклом (1.1)
                for i in range(0, len(df) - 4, 5):
            Делим весь датасет на непересекающиеся группы по 5 которые будем потом переставлять (1.2)
                group = df.iloc[i:i+5]
                    group[0]
                        text       label
                        "A"    :    1 
                        "B"    :    0 
                        "C"    :    0
                        "D"    :    1
                        "E"    :    1
                    group[1]
                        text       label
                        "F"    :    1 
                        "G"    :    1 
                        "H"    :    0
                        "I"    :    0
                        "J"    :    0
                    ...
        2) 
            Запоминаем первый отзыв как якорный (не сохраняем инфу о оценке чтобы ИИ потом сам ответил на якорь) (2.1)
                anchor_text = "A"
                anchor_text = group.iloc[0]["text"]
                anchor_answer = group.iloc[0]["label"]
            Остальные как few_shot (сохраняя инфу о оценке для примера) (2.2)
                others_data = ("B",0),("C",0),("D",1),("E",1)
                others_data = list(zip(others["text"], others["label"]))
        3)
            Для каждой группы few_shot (3.1)
                ("B",0),("C",0),("D",1),("E",1)
            перебираем все комбинации положения элементов few_shot (3.1)
                (("B",0),("C",0),("D",1),("E",1)),
                (("C",0),("B",0),("D",1),("E",1))
                (("C",0),("D",1),("B",0),("E",1)),
                ... так 4! или 24 раза 

                    for perm in permutations(others_data): 
                
            Возвращаем якорь + ответ на якорь + конкретную перестановку few_shot (3.2)
                "A",1, (("B",0),("C",0),("D",1),("E",1))
                    yield anchor_text,anchor_answer, list(perm)
"""

from itertools import permutations
import pandas as pd
import hashlib
from pathlib import Path

def s1_pair_generator(parquet_path:Path, hashes_path: Path):
    """
    input
        parquet_path
        hashes_path

    output
        yield anchor_text, anchor_answer, list(few_shot_perm)
    """

    df = pd.read_parquet(parquet_path)

    #hash logic start
    existing_hashes = set()
    if hashes_path.exists():

        with open(hashes_path,"r") as f:
            existing_hashes = set(line.strip() for line in f if line.strip())
            print(f"🛡️ Found {len(existing_hashes)} completed groups. Skipping them.")
    else:
        print("🚀 Starting fresh generation.")
    #hash logic end


    for i in range(0, len(df) - 4, 5): #(1.1)

        group = df.iloc[i:i+5] #(1.2)
        
        anchor_text = group.iloc[0]["text"] #(2.1)
        acnhor_answer = group.iloc[0]["label"]

        #hash logic start
        anchor_hash = hashlib.md5(anchor_text.encode('utf-8')).hexdigest() 

        if anchor_hash in existing_hashes:
            continue
        #hash logic end

        #(2.2)
        others = group.iloc[1:] 
        others_data = list(zip(others["text"], others["label"]))
        
        for perm in permutations(others_data): #(3.1)
            yield anchor_text,acnhor_answer, list(perm) #(3.2)