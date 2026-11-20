from itertools import permutations
import pandas as pd
"""
Вход: DataFrame с колонками 'text' и 'label'
Выход: Генератор пар (anchor_text, examples_list)
        examples_list — это список из 4-х кортежей (text, label)
"""
def generate_llm_inputs(df: pd.DataFrame):

    print("generate_llm_inputs started")
    for i in range(0, len(df) - 4, 5):
        group = df.iloc[i:i+5]
        
        # 1. Фиксируем якорь (только текст)
        anchor_text = group.iloc[0]["text"]
        
        # 2. Берем остальные 4 текста и их лейблы
        others = group.iloc[1:]
        others_data = list(zip(others["text"], others["label"]))
        
        # 3. Перебираем все 24 комбинации контекста
        for perm in permutations(others_data):
            # perm теперь содержит кортежи: ((text1, label1), (text2, label2), ...)
            yield anchor_text, list(perm)