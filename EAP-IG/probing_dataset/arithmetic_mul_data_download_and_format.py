import pandas as pd
import math
import argparse
from itertools import combinations, permutations
from sympy import factorint
from functools import reduce
import operator
from dotenv import load_dotenv
from transformers import AutoTokenizer
import random
load_dotenv()
random.seed(2025)


def group_factors(factors, z):
    """
    Generate all unique groupings of 'factors' into 'z' multiplicative groups.
    """
    if z == 1:
        yield [math.prod(factors)]
        return
    seen = set()

    def helper(remaining, groups):
        if len(groups) == z - 1:
            last_group = math.prod(remaining)
            final = groups + [last_group]
            key = tuple(sorted(final))
            if key not in seen:
                seen.add(key)
                yield final
            return

        for i in range(1, len(remaining)):
            for subset in set(combinations(remaining, i)):
                subset_prod = math.prod(subset)
                rest = list(remaining)
                try:
                    for s in subset:
                        rest.remove(s)
                except ValueError:
                    continue  # skip if subset elements not fully in rest
                yield from helper(rest, groups + [subset_prod])

    yield from helper(factors, [])

def all_groupings(x, z_min=2, z_max=5):
    factor_dict = factorint(x)
    factors = []
    for prime, count in factor_dict.items():
        factors += [prime] * count

    results = {}
    for z in range(z_min, min(len(factors), z_max) + 1):
        groupings = list(group_factors(factors, z))
        results[z] = groupings
    return results

ll = 0
tuple_by_op = {}
for x in range(100, 1000):
    results = all_groupings(x, z_min=2, z_max=5)

    for z in sorted(results):
        if z not in tuple_by_op:
            tuple_by_op[z] = []

        for g in results[z]:
            ll += 1
            tuple_by_op[z].append(tuple(g))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate arithmetic multiplication dataset')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name for tokenizer')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    sampled_tuple_by_op = {}
    for z, tuples in tuple_by_op.items():
        if len(tuples) > 125:
            sampled = random.sample(tuples, 125) # without replacement
        else:
            sampled = tuples # if fewer than 125, keep them all

        random.shuffle(sampled)
        sampled_tuple_by_op[z] = sampled
    # sampled_tuple_by_op = {k: v for k, v in sampled_tuple_by_op.items() if k in [4, 5]}
    df = pd.DataFrame(columns=["clean", "corrupted", "answer", "correct_idx", "incorrect_idx"] + [f'paraphrase{i}' for i in range(1, 10)])
    for z, tuples in sampled_tuple_by_op.items():
        for operands in tuples:
            product = reduce(operator.mul, operands)
            clean = '*'.join(map(str, operands)) + '=' # expression
            corrupted = ''
            while True:
                corrupted_sample = random.sample(tuples, 1)[0]
                corrupted_product = reduce(operator.mul, corrupted_sample)
                if corrupted_product != product:
                    corrupted = '*'.join(map(str, corrupted_sample)) + '='
                    break
            assert corrupted != ''

            answer = str(product)
            corrupted_answer = str(corrupted_product)

            correct_idx = tokenizer.encode(answer, add_special_tokens=False)[0]
            assert isinstance(correct_idx, int)

            incorrect_idx = tokenizer.encode(corrupted_answer, add_special_tokens=False)[0]
            assert isinstance(incorrect_idx, int)

            all_reorders = list(permutations(operands))[1:]
            if len(all_reorders) > 9:
                all_reorders = random.sample(all_reorders, 9)
            else:
                all_reorders = all_reorders + [all_reorders[-1]] * (9 - len(all_reorders))

            paraphrases = ['*'.join(map(str, i)) + '=' for i in all_reorders]
            df.loc[len(df)] = [clean, corrupted, answer, correct_idx, incorrect_idx] + paraphrases

    df.to_csv(f"arithmetic_mul_{args.model_name.split('/')[-1].replace('.', '')}.csv", index=False)