from itertools import permutations
import pandas as pd

def generate_llm_inputs(df: pd.DataFrame):
    """
    Вход: DataFrame с колонкой 'text'
    Выход: Генератор пар (anchor_text, full_context_string)
    """
    for i in range(0, len(df) - 4, 5):
        group = df.iloc[i:i+5]
        
        # 1. Фиксируем якорь
        anchor_text = group.iloc[0]["text"]
        
        # 2. Берем остальные 4
        others_texts = group.iloc[1:]["text"].tolist()
        
        # 3. Перебираем все 24 комбинации
        for perm in permutations(others_texts):
            # 4. Собираем входную строку для LLM
            # Разделитель [SEP] можно заменить на любой другой
            full_context = f"{anchor_text} [SEP] {' [SEP] '.join(perm)}"
            
            yield anchor_text, full_context