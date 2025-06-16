import pandas as pd
from transformers import AutoTokenizer


llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

df = pd.read_csv("ioi_llama.csv")

# Decode llama3 token indices to strings
# decoded_correct = []
# for idx in df['correct_idx']:
#     decoded_correct.append(tokenizer.decode([idx]))

decoded_correct = [llama_tokenizer.decode([idx]) for idx in df['correct_idx']]
decoded_incorrect = [llama_tokenizer.decode([idx]) for idx in df['incorrect_idx']]

# Re-tokenize using GPT2
gpt2_correct_idx = [gpt2_tokenizer.encode(text, add_special_tokens=False)[0] for text in decoded_correct]
gpt2_incorrect_idx = [gpt2_tokenizer.encode(text, add_special_tokens=False)[0] for text in decoded_incorrect]

# Add to DataFrame
df['correct_idx'] = gpt2_correct_idx
df['incorrect_idx'] = gpt2_incorrect_idx
df.to_csv("ioi_gpt2.csv", index=False)
