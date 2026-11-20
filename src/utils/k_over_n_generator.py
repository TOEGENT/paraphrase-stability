"""
Задача:
    Принять от llm_responce_generator
        yield prompt, anchor, answer 
    На выходе дать 
        yield prompt,anchor,k,N
"""

def k_over_n_generator(llm_iterator):
    k = 0
    N = 0
    anchor_first = None
    answer_first = None

    for prompt,anchor, answer in llm_iterator:
    
        if anchor_first is None:
            anchor_first = anchor
            answer_first = answer
            k = 1
            N = 1
            continue
    
        if anchor != anchor_first:
            yield prompt, k, N
            
            anchor_first = anchor
            answer_first = answer
            k = 1
            N = 1
            continue
            
        N += 1
        if answer == answer_first:
            k += 1

    if N > 0:
        yield prompt, k, N