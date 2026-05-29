import re
import os
import ast
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
load_dotenv()


def format_clean_prompt(row):
    question = row['question'] + '\n'
    choices = ['(A) ' + row['A'] + '\n', '(B) ' + row['B'] + '\n', '(C) ' + row['C'] + '\n', '(D) ' + row['D'] + '\n']
    suffix = 'Answer: ('
    
    complete_prompt = question + ''.join(choices) + suffix
    
    return complete_prompt

def format_corrupted_prompt(row):
    question = 'Which is the most possible answer?' + '\n'
    choices = ['(A) ' + row['A'] + '\n', '(B) ' + row['B'] + '\n', '(C) ' + row['C'] + '\n', '(D) ' + row['D'] + '\n']
    suffix = 'Answer: ('

    complete_prompt = question + ''.join(choices) + suffix
    
    return complete_prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and format MMLU dataset')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name for tokenizer')
    parser.add_argument('--category', type=str, default='marketing',
                        help='MMLU category') # marketing, professional_medicine, astronomy, college_biology, high_school_computer_science, logical_fallacies, nutrition, international_law, management
    parser.add_argument('--folder_path', type=str, default='mmlu_test',
                        help='Folder containing raw MMLU csv files')
    args = parser.parse_args()

    column_names = ['question', 'A', 'B', 'C', 'D', 'answer']
    dfs = []
    for filename in os.listdir(args.folder_path):
        if filename.endswith('.csv'):
            category = os.path.splitext(filename)[0].replace('_test', '')
            if category != args.category:
                continue

            file_path = os.path.join(args.folder_path, filename)
            df = pd.read_csv(file_path, names=column_names, na_values=[], keep_default_na=False)

            df['category'] = category

            dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    print(f'data number of mmlu {args.category} test set: {len(all_data)}')

    all_data["clean"] = all_data.apply(format_clean_prompt, axis=1)
    all_data['corrupted'] = all_data.apply(format_corrupted_prompt, axis=1)

    # correct_idx,incorrect_idx
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    correct_idx = [tokenizer.encode(text, add_special_tokens=False)[0] for text in all_data['answer']]
    incorrect_idx = []
    for text in all_data['answer']:
        wrong_choices = [i for i in ['A', 'B', 'C', 'D'] if i != text]
        wrong_choices_idx = [tokenizer.encode(i, add_special_tokens=False)[0] for i in wrong_choices]
        incorrect_idx.append(wrong_choices_idx)

    all_data['correct_idx'] = correct_idx
    all_data['incorrect_idx'] = incorrect_idx
    all_data.to_csv(f"mmlu_{args.category}_{args.model_name.split('/')[-1].replace('.', '')}.csv", index=False)