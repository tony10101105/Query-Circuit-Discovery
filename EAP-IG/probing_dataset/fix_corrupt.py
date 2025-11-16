# read in mmlu and convert to EAP format
import re
import os
import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
import ast


def mmlu_format_corrupted_prompt(row):
    question = 'Which is the most possible answer?' + '\n'
    choices = ['(A) ' + row['A'] + '\n', '(B) ' + row['B'] + '\n', '(C) ' + row['C'] + '\n', '(D) ' + row['D'] + '\n']
    suffix = 'Answer: ('

    complete_prompt = question + ''.join(choices) + suffix
    
    return complete_prompt

def arcc_format_corrupted_prompt(row):
    # print(row['choices'])
    # row['choices'] = row['choices'].replace("array(", "").replace(', dtype=object)', '')
    # print(row['choices'])
    # row['choices'] = ast.literal_eval(row['choices'])
    question = 'Which is the most possible answer?' + '\n'
    
    # print(row['choices'])
    # print(type(row['choices']))
    option_text = row['choices']['text']
    option_prefix = row['choices']['label']
    assert len(option_text) == len(option_prefix)
    choices = ['(' + option_prefix[i] + ') ' + option_text[i] + '\n' for i in range(len(option_text))]
    suffix = 'Answer: ('

    complete_prompt = question + ''.join(choices) + suffix
    
    return complete_prompt


model_name = 'meta-llama/Llama-3.2-1B' # gpt2 # meta-llama/Llama-3.2-1B
interested_categories = ['marketing', 'professional_medicine', 'astronomy', 'college_biology', 'high_school_computer_science', 'logical_fallacies', 'nutrition', 'international_law', 'management']

for interested_category in interested_categories:
    # all_data = pd.read_csv(f'mmlu_{interested_category}_Llama-32-1B_gpt4o_paraphrases_only_stem.csv')
    all_data = pd.read_csv(f'mmlu_{interested_category}_Llama-32-1B.csv')
    print(f'data number of mmlu {interested_category} test set: {len(all_data)}')

    all_data['corrupted'] = all_data.apply(mmlu_format_corrupted_prompt, axis=1)

    # all_data.to_csv(f'mmlu_{interested_category}_Llama-32-1B_gpt4o_paraphrases_only_stem.csv', index=False)
    all_data.to_csv(f'mmlu_{interested_category}_Llama-32-1B.csv', index=False)


# all_data = pd.read_csv(f'arc_challenge_Llama-32-1B_gpt4o_paraphrases_only_stem.csv')
all_data = pd.read_csv(f'arc_challenge_Llama-32-1B.csv')
print(f'data number of arc challenge test set: {len(all_data)}')

splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["test"])

all_data['corrupted'] = df.apply(arcc_format_corrupted_prompt, axis=1)

# all_data.to_csv(f'arc_challenge_Llama-32-1B_gpt4o_paraphrases_only_stem.csv', index=False)
all_data.to_csv(f'arc_challenge_Llama-32-1B.csv', index=False)