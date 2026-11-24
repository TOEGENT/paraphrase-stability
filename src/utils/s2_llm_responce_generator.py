"""
Задача 2
    Взять результаты s1_pair_generator
        yield якорный вопрос,ответ на якорь, все комбинации few_shot вопросов
            yield ->
                    ( "A",1, (
                            (("B",0),("C",0),("D",1),("E",1)),
                            (("C",0),("B",0),("D",1),("E",1))
                            (("C",0),("D",1),("B",0),("E",1)),
                            ... так 4! или 24 раза
                            ) )
            yield ->
                ( "F",0, (аналогично только для G,H,I,J) )
    И выдать 
        yield prompt, anchor,anchor_answer, answer
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
                anchor_answer
                    1
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
                anchor_answer
                    1
                answer
                    1
            ...
    Решение
        Для каждого якоря, его ответа и его few_shots (1)
            "A", 1, (
                (("B",0),("C",0),("D",1),("E",1)),
                (("C",0),("B",0),("D",1),("E",1))
                (("C",0),("D",1),("B",0),("E",1)),
                ... так 4! или 24 раза
                ) 
            
                for anchor,anchor_answer, examples in input_iterator:
                
        Формируем блок примеров (2)
            'Examples: 
                "B": 0,
                "C": 0,
                "D": 1,
                "E": 1'

                examples_block = ""
                for ex_text, ex_label in examples:
                    label_name = label_map[ex_label]
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

        выбрасываем промпт, якорь, ответ на якорь, ответ ллм
            yield prompt, anchor,anchor_answer, answer
"""
from utils import call_llm_api

def s2_llm_response_generator(input_iterator,label_map):
    """
    input
        yield anchor_text, anchor_answer, list(few_shot_perm)
    
    output
        yield prompt, anchor, anchor_answer, answer    

    """
    print("llm_response_generator started")

    #(1)
    for anchor,anchor_answer, examples in input_iterator:

        # (2)
        examples_block = ""
        for ex_text,ex_label in examples:
            label_name = label_map[ex_label]
            examples_block += f"Text: '{ex_text}'\nLabel: {label_name}\n\n"
            
        # (3)
        prompt = (
            f"you're a classificator. return ONLY the label\n"
            f"Here are some examples:\n{examples_block}\n"
            f"Now analyze this text:\nText: '{anchor}'\nLabel:"
        )

        #(4)
        answer = call_llm_api.call_llm_api(prompt)
        
        #(5) 
        yield prompt, anchor,anchor_answer,answer