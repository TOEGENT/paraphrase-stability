from utils import call_llm_api

"""
Задача
    Взять результаты generate_llm_inputs
        yield anchor_text, list(perm)
    И выдать 
        yield prompt, anchor, answer

"""
LABEL_MAP = {
    0: "Negative", 1: "Positive", 
    2: "Business", 3: "Sci/Tech" 
}

def get_label_name(label_id):
    return LABEL_MAP.get(label_id, str(label_id))

def llm_response_generator(input_iterator):
    """
    Вход: Генератор пар (anchor_text, examples_list)
    Выход: Генератор триплетов (prompt, anchor_text, llm_answer)
    """
    print("llm_response_generator started")
    
    for anchor, examples in input_iterator:

        # 1. Формируем блок примеров (Few-Shot)
        examples_block = ""
        for ex_text, ex_label in examples:
            label_name = get_label_name(ex_label)
            examples_block += f"Text: '{ex_text}'\nLabel: {label_name}\n\n"
            
        # 2. Формируем финальный промпт
        prompt = (
            f"You are an emotion classifier. Analyze the text and return ONLY the label name.\n\n"
            f"Here are some examples:\n{examples_block}\n"
            f"Now analyze this text:\nText: '{anchor}'\nLabel:"
        )
        
        answer = call_llm_api.call_llm_api(prompt)
        
            
        yield prompt, anchor, answer