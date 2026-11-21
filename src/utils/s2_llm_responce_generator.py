from utils import call_llm_api

"""
Задача 2
    Взять результаты s1_pair_generator
        yield якорный вопрос, все комбинации few_shot вопросов
            yield ->
                    ( "A", (
                            (("B",0),("C",0),("D",1),("E",1)),
                            (("C",0),("B",0),("D",1),("E",1))
                            (("C",0),("D",1),("B",0),("E",1)),
                            ... так 4! или 24 раза
                            ) )
            yield ->
                ( "F", (аналогично только для G,H,I,J) )
    И выдать 
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
    Решение
        Для каждого якоря и его few_shots (1)
            "A", (
                (("B",0),("C",0),("D",1),("E",1)),
                (("C",0),("B",0),("D",1),("E",1))
                (("C",0),("D",1),("B",0),("E",1)),
                ... так 4! или 24 раза
                ) 
            
                for anchor, examples in input_iterator:
                
        Формируем блок примеров (2)
            'Examples: 
                "B": 0,
                "C": 0,
                "D": 1,
                "E": 1'

                examples_block = ""
                for ex_text, ex_label in examples:
                    label_name = get_label_name(ex_label)
                    examples_block += f"Text: '{ex_text}'\nLabel: {label_name}\n\n"
        
        Формируем вопрос (3)
            'Now guess
                "A": ?'

                prompt = (
                    f"You are an emotion classifier. Analyze the text and return ONLY the label name.\n\n"
                    f"Here are some examples:\n{examples_block}\n"
                    f"Now analyze this text:\nText: '{anchor}'\nLabel:"
                )
        Передаём примеры + вопрос в ллм (4)
            answer = call_llm_api.call_llm_api(prompt)

        выбрасываем промпт, якорь, ответ ллм
            yield prompt, anchor, answer
"""
LABEL_MAP = {
    0: "Negative", 1: "Positive", 
}

def get_label_name(label_id):
    return LABEL_MAP.get(label_id, str(label_id))

def s2_llm_response_generator(input_iterator):
    """
    Вход: Генератор пар (anchor_text, examples_list)
    Выход: Генератор триплетов (prompt, anchor_text, llm_answer)
    """
    print("llm_response_generator started")

    #(1)
    for anchor, examples in input_iterator:

        # (2)
        examples_block = ""
        for ex_text, ex_label in examples:
            label_name = get_label_name(ex_label)
            examples_block += f"Text: '{ex_text}'\nLabel: {label_name}\n\n"
            
        # (3)
        prompt = (
            f"You are an emotion classifier. Analyze the text and return ONLY the label name.\n\n"
            f"Here are some examples:\n{examples_block}\n"
            f"Now analyze this text:\nText: '{anchor}'\nLabel:"
        )

        #(4)
        answer = call_llm_api.call_llm_api(prompt)
        
        #(5) 
        yield prompt, anchor, answer