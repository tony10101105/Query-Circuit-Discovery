# read in mmlu and convert to EAP format
import re
import os
import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer



def format_clean_prompt(row):
    total_len = 0
    for opt in ['A', 'B', 'C', 'D']:
        length = llama_tokenizer.encode(row[opt], add_special_tokens=False)
        total_len += len(length)
    # print('total len: ', total_len)
    return 1 if total_len == 4 else 0

def format_corrupted_prompt(row):
    # Regex explanation:
    # ^(.*?)\n   -> capture the first line (the description) up to the first newline
    # (?=\(A\))  -> look ahead to ensure the next line starts with (A)
    clean = row['clean']
    pattern = r"^(.*?)\n(?=\(A\))"
    return re.sub(pattern, 'Which is the most possible answer?\n', clean, flags=re.MULTILINE)


# all_cat = ['security_studies', 'high_school_us_history', 'nutrition', 'business_ethics', 'miscellaneous', 'jurisprudence', 'moral_scenarios', 'high_school_geography', 'high_school_european_history', 'world_religions', 'college_chemistry', 'professional_psychology', 'philosophy', 'high_school_physics', 'machine_learning', 'electrical_engineering', 'logical_fallacies', 'sociology', 'professional_medicine', 'college_medicine', 'clinical_knowledge', 'astronomy', 'abstract_algebra', 'public_relations', 'college_computer_science', 'econometrics', 'human_aging', 'international_law', 'high_school_psychology', 'formal_logic', 'high_school_mathematics', 'professional_accounting', 'management', 'college_physics', 'medical_genetics', 'prehistory', 'anatomy', 'elementary_mathematics', 'high_school_microeconomics', 'high_school_macroeconomics', 'high_school_biology', 'professional_law', 'us_foreign_policy', 'computer_security', 'college_biology', 'high_school_chemistry', 'high_school_government_and_politics', 'marketing', 'high_school_statistics', 'global_facts', 'college_mathematics', 'conceptual_physics', 'human_sexuality', 'high_school_world_history', 'high_school_computer_science', 'moral_disputes', 'virology']
model_name = 'meta-llama/Llama-3.2-1B' # gpt2 # meta-llama/Llama-3.2-1B

llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_correct_idx = [llama_tokenizer.encode(text, add_special_tokens=False) for text in ['hello', 'hello world']]

interested_category = ['security_studies', 'high_school_us_history', 'nutrition', 'business_ethics', 'miscellaneous', 'jurisprudence', 'moral_scenarios', 'high_school_geography', 'high_school_european_history', 'world_religions', 'college_chemistry', 'professional_psychology', 'philosophy', 'high_school_physics', 'machine_learning', 'electrical_engineering', 'logical_fallacies', 'sociology', 'professional_medicine', 'college_medicine', 'clinical_knowledge', 'astronomy', 'abstract_algebra', 'public_relations', 'college_computer_science', 'econometrics', 'human_aging', 'international_law', 'high_school_psychology', 'formal_logic', 'high_school_mathematics', 'professional_accounting', 'management', 'college_physics', 'medical_genetics', 'prehistory', 'anatomy', 'elementary_mathematics', 'high_school_microeconomics', 'high_school_macroeconomics', 'high_school_biology', 'professional_law', 'us_foreign_policy', 'computer_security', 'college_biology', 'high_school_chemistry', 'high_school_government_and_politics', 'marketing', 'high_school_statistics', 'global_facts', 'college_mathematics', 'conceptual_physics', 'human_sexuality', 'high_school_world_history', 'high_school_computer_science', 'moral_disputes', 'virology']
column_names = ['question', 'A', 'B', 'C', 'D', 'answer']
folder_path = 'mmlu_test'
dfs = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        category = os.path.splitext(filename)[0].replace('_test', '')
        if category not in interested_category:
            continue

        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, names=column_names, na_values=[], keep_default_na=False)

        df['category'] = category

        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
print(f'data number of mmlu {interested_category} test set: {len(all_data)}')

all_data["clean"] = all_data.apply(format_clean_prompt, axis=1)
print(sum(all_data["clean"]))