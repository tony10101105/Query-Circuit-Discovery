import pandas as pd
from transformers import AutoTokenizer


gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama/Llama-3.2-1B")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

df = pd.read_csv("mmlu_gpt2.csv")

# Decode llama3 token indices to strings
# decoded_correct = []
# for idx in df['correct_idx']:
#     decoded_correct.append(tokenizer.decode([idx]))

decoded_correct = [gpt_tokenizer.decode([idx]) for idx in df['correct_idx']]
decoded_incorrect = [gpt_tokenizer.decode([idx]) for idx in df['incorrect_idx']]

# Re-tokenize using GPT2
llama_correct_idx = [llama_tokenizer.encode(text, add_special_tokens=False)[0] for text in decoded_correct]
llama_incorrect_idx = [llama_tokenizer.encode(text, add_special_tokens=False)[0] for text in decoded_incorrect]

# Add to DataFrame
df['correct_idx'] = llama_correct_idx
df['incorrect_idx'] = llama_incorrect_idx
df.to_csv("ioi_llama32.csv", index=False)
