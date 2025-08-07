# read in mmlu and convert to EAP format
import os
import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer


all_cat = ['security_studies', 'high_school_us_history', 'nutrition', 'business_ethics', 'miscellaneous', 'jurisprudence', 'moral_scenarios', 'high_school_geography', 'high_school_european_history', 'world_religions', 'college_chemistry', 'professional_psychology', 'philosophy', 'high_school_physics', 'machine_learning', 'electrical_engineering', 'logical_fallacies', 'sociology', 'professional_medicine', 'college_medicine', 'clinical_knowledge', 'astronomy', 'abstract_algebra', 'public_relations', 'college_computer_science', 'econometrics', 'human_aging', 'international_law', 'high_school_psychology', 'formal_logic', 'high_school_mathematics', 'professional_accounting', 'management', 'college_physics', 'medical_genetics', 'prehistory', 'anatomy', 'elementary_mathematics', 'high_school_microeconomics', 'high_school_macroeconomics', 'high_school_biology', 'professional_law', 'us_foreign_policy', 'computer_security', 'college_biology', 'high_school_chemistry', 'high_school_government_and_politics', 'marketing', 'high_school_statistics', 'global_facts', 'college_mathematics', 'conceptual_physics', 'human_sexuality', 'high_school_world_history', 'high_school_computer_science', 'moral_disputes', 'virology']

def format_clean_prompt(row):
    question = row['question'] + '\n'
    choices = ['(A) ' + row['A'] + '\n', '(B) ' + row['B'] + '\n', '(C) ' + row['C'] + '\n', '(D) ' + row['D'] + '\n']
    suffix = 'Answer: ('
    
    complete_prompt = question + ''.join(choices) + suffix
    # answer_idx = row['answer']
    
    return complete_prompt

def format_corrupted_prompt(row):
    question = 'What is the most possible answer?' + '\n'
    choices = ['(A) ' + row['A'] + '\n', '(B) ' + row['B'] + '\n', '(C) ' + row['C'] + '\n', '(D) ' + row['D'] + '\n']
    suffix = 'Answer: ('

    complete_prompt = question + ''.join(choices) + suffix
    
    return complete_prompt


model_name = 'meta-llama/Llama-3.2-3B' # gpt2 # meta-llama/Llama-3.2-1B

column_names = ['question', 'A', 'B', 'C', 'D', 'answer']
folder_path = 'mmlu_test'
dfs = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, names=column_names, na_values=[], keep_default_na=False)

        category = os.path.splitext(filename)[0].replace('_test', '')
        df['category'] = category

        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
assert len(all_data) == 14042

all_data["clean"] = all_data.apply(format_clean_prompt, axis=1)
all_data['corrupted'] = all_data.apply(format_corrupted_prompt, axis=1)

# print(all_data.head())

# correct_idx,incorrect_idx
tokenizer = AutoTokenizer.from_pretrained(model_name)

correct_idx = [tokenizer.encode(text, add_special_tokens=False)[0] for text in all_data['answer']]
incorrect_idx = []
for text in all_data['answer']:
    wrong_choices = [i for i in ['A', 'B', 'C', 'D'] if i != text]
    wrong_choices_idx = [tokenizer.encode(i, add_special_tokens=False)[0] for i in wrong_choices]
    incorrect_idx.append(wrong_choices_idx)

all_data['correct_idx'] = correct_idx
all_data['incorrect_idx'] = incorrect_idx
all_data.to_csv(f"mmlu_{model_name.split('/')[-1].replace('.', '')}.csv", index=False)
