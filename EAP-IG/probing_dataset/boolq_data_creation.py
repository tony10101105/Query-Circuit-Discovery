# read in mmlu and convert to EAP format
import re
import os
import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
model_name = 'meta-llama/Llama-3.2-1B' # gpt2 # meta-llama/Llama-3.2-1B
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_clean_length(row):
    length = len(tokenizer.encode(row['clean'], add_special_tokens=False))
    return length

def format_clean_prompt(row):
    question = row['question']
    # print(row['choices'])
    # print(row['answerKey'])
    passage = row['passage']
    suffix = 'Answer (Yes/No): '
    complete_prompt = f"Read the passage and answer the question.\nPassage: {passage}\nQuestion: {question}\nAnswer: "
    return complete_prompt

def format_corrupted_prompt(row):
    clean = row['clean']
    
    # Regex to match everything between "Passage: " and "Question:"
    pattern = r'(Passage:)(.*?)(\nQuestion:)'

    # Replace the middle part with XXXXX
    corrupted = re.sub(pattern, r'\1 XXXXX\3', clean, flags=re.DOTALL)
    return corrupted


splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/google/boolq/" + splits["validation"])

column_names = ['question', 'passage', 'answer']

df["clean"] = df.apply(format_clean_prompt, axis=1)
df['corrupted'] = df.apply(format_corrupted_prompt, axis=1)
df['answer'] = df['answer'].astype(str)
df['length'] = df.apply(get_clean_length, axis=1)
df = df.sort_values(by='length', ascending=True)
df = df.head(500)
# correct_idx,incorrect_idx

correct_idx = [tokenizer.encode(text, add_special_tokens=False)[0] for text in df['answer']]
incorrect_idx = []
for j, text in enumerate(df['answer']):
    row = df.iloc[j]
    wrong_choices = [i for i in ['Yes', 'No'] if i != text]
    wrong_choices_idx = [tokenizer.encode(i, add_special_tokens=False)[0] for i in wrong_choices]
    incorrect_idx.append(wrong_choices_idx)

df['correct_idx'] = correct_idx
df['incorrect_idx'] = incorrect_idx
df.to_csv(f"boolq_{model_name.split('/')[-1].replace('.', '')}.csv", index=False)