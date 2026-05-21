# read in mmlu and convert to EAP format
import re
import os
import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer


def format_clean_prompt(row):
    question = row['question'] + '\n'
    option_text = row['choices']['text']
    option_prefix = row['choices']['label']
    assert len(option_text) == len(option_prefix)
    choices = ['(' + option_prefix[i] + ') ' + option_text[i] + '\n' for i in range(len(option_text))]
    suffix = 'Answer: ('
    
    complete_prompt = question + ''.join(choices) + suffix

    return complete_prompt

def format_corrupted_prompt(row):
    question = 'Which is the most possible answer?' + '\n'
    option_text = row['choices']['text']
    option_prefix = row['choices']['label']
    assert len(option_text) == len(option_prefix)
    choices = ['(' + option_prefix[i] + ') ' + option_text[i] + '\n' for i in range(len(option_text))]
    suffix = 'Answer: ('

    complete_prompt = question + ''.join(choices) + suffix
    
    return complete_prompt


category = 'Challenge'
model_name = 'meta-llama/Llama-3.2-1B' # gpt2 # meta-llama/Llama-3.2-1B
splits = {'train': f'ARC-{category}/train-00000-of-00001.parquet', 'test': f'ARC-{category}/test-00000-of-00001.parquet', 'validation': f'ARC-{category}/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["test"])

column_names = ['question', 'choices', 'answerKey']

print(f'data number ARC-{category} test set: {len(df)}')

df["clean"] = df.apply(format_clean_prompt, axis=1)
df['corrupted'] = df.apply(format_corrupted_prompt, axis=1)

# correct_idx,incorrect_idx
tokenizer = AutoTokenizer.from_pretrained(model_name)

correct_idx = [tokenizer.encode(text, add_special_tokens=False)[0] for text in df['answerKey']]
incorrect_idx = []
for j, text in enumerate(df['answerKey']):
    row = df.iloc[j]
    wrong_choices = [i for i in row['choices']['label'] if i != text]
    wrong_choices_idx = [tokenizer.encode(i, add_special_tokens=False)[0] for i in wrong_choices]
    incorrect_idx.append(wrong_choices_idx)

df['correct_idx'] = correct_idx
df['incorrect_idx'] = incorrect_idx
df.to_csv(f"arc_{category.lower()}_{model_name.split('/')[-1].replace('.', '')}.csv", index=False)
