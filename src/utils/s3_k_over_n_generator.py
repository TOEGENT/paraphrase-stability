"""
Задача 3.2
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
    На выходе дать
        yield prompt,anchor,k,N
            yield -> 
                (выдаёт на каждые 25 элементов)
                    prompt1, anchor_first, answer_first,
                    prompt2, anchor_first, answer2,
                    ...
                    prompt10, anchor_first, answer1,
                    ...
                    prompt25,anchor_first,answer25
                ->
                prompt1
                    'Examples: 
                        "B": 0,
                        "C": 0,
                        "D": 1,
                        "E": 1
                    Now guess
                        "A": ?',
                anchor_first
                    "A"
                k=
                    answer_first.freq
                    = 20
                N=24

    Решение
        (1)
            Выделяем в памяти место под будущие k,N
                k = 0
                N = 0
        (2)
            запоминаем первый промпт, якорь и ответ как исходные перед перестановками
                anchor_first = anchor
                answer_first = answer
                prompt_first = prompt                
        (3)
            Для каждого prompt,anchor,answer
                прибавляем к k единицу если якорь тот же
                прибавляем к N единицу
        (4)
            Если якорь не тот же, следовательно, пошла новая группа
                выбрасываем prompt_first,answer_first,k,N
                    yield prompt_first, answer_first, k, N

"""

def s3_k_over_n_generator(llm_iterator):

    #(1)
    k = 0
    N = 0

    anchor_first = None
    answer_first = None
    prompt_first = None

  
    for prompt,anchor, answer in llm_iterator: #(3)

        if anchor_first is None:
            #(2)
            anchor_first = anchor
            answer_first = answer
            prompt_first = prompt
            k = 1
            N = 1
            continue

        if anchor != anchor_first: #(4)
            yield prompt_first, anchor, answer_first, k, N
            anchor_first = anchor
            answer_first = answer
            prompt_first = prompt
            k = 1
            N = 1
            continue

        #(3)
        N += 1
        if answer == answer_first:
            k += 1

    if N > 0:
        yield prompt_first, anchor, answer_first, k, N