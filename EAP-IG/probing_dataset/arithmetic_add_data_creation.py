import pandas as pd
import math
from itertools import permutations
from transformers import AutoTokenizer
import random
random.seed(2025)


def decompose_positive(total, z):
    assert z >= 1 and total >= z, "Need total ≥ z and z ≥ 1"
    
    # Generate z−1 sorted random cut points between 1 and total−1
    cuts = sorted(random.sample(range(1, total), z - 1))
    
    # Build the parts from the differences between cut points
    parts = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, z - 1)] + [total - cuts[-1]]
    
    # Each part is guaranteed to be > 0
    return tuple(parts)


model_name = 'meta-llama/Llama-3.2-1B' # gpt2 # meta-llama/Llama-3.2-1B
tokenizer = AutoTokenizer.from_pretrained(model_name)

all_answers = [i for i in range(100, 1000)]
sampled_tuple_by_op = {}
zs = [2,3,4,5]
for z in zs:
    sampled_tuple_by_op[z] = []
    for i in range(125):
        ans = random.choice(all_answers)
        tup = decompose_positive(ans, z)
        sampled_tuple_by_op[z].append(tup)

df = pd.DataFrame(columns=["clean", "corrupted", "answer", "correct_idx", "incorrect_idx"] + [f'paraphrase{i}' for i in range(1, 10)])
for z, tuples in sampled_tuple_by_op.items():
    for operands in tuples:
        summation = sum(operands)
        clean = '+'.join(map(str, operands)) + '=' # expression
        corrupted = ''
        while True:
            corrupted_sample = random.sample(tuples, 1)[0]
            corrupted_summation = sum(corrupted_sample)
            if corrupted_summation != summation:
                corrupted = '+'.join(map(str, corrupted_sample)) + '='
                break
        assert corrupted != ''

        answer = str(summation)
        corrupted_answer = str(corrupted_summation)

        correct_idx = tokenizer.encode(answer, add_special_tokens=False)[0]
        assert isinstance(correct_idx, int)

        incorrect_idx = tokenizer.encode(corrupted_answer, add_special_tokens=False)[0]
        assert isinstance(incorrect_idx, int)

        all_reorders = list(permutations(operands))[1:]
        if len(all_reorders) > 9:
            all_reorders = random.sample(all_reorders, 9)
        else:
            all_reorders = all_reorders + [all_reorders[-1]] * (9 - len(all_reorders))

        paraphrases = ['+'.join(map(str, i)) + '=' for i in all_reorders]
        df.loc[len(df)] = [clean, corrupted, answer, correct_idx, incorrect_idx] + paraphrases

df.to_csv(f"arithmetic_add_{model_name.split('/')[-1].replace('.', '')}.csv", index=False)