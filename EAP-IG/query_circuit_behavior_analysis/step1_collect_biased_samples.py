import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.utils import set_seed, tokenize_plus
from save_score_matrix.models import TargetModelConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--bias_threshold', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_path', type=str, default='probing_dataset/gender_bias_gpt2.csv')
parser.add_argument('--output_path', type=str, default='query_circuit_behavior_analysis/data/biased_samples.json')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.eval()

df = pd.read_csv(args.dataset_path)

biased_samples = []
with torch.inference_mode():
    for start in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[start:start + args.batch_size]
        tokens, attention_mask, input_lengths, _ = tokenize_plus(model, batch['clean'].tolist())
        logits = model(tokens, attention_mask=attention_mask)
        last_logits = logits[torch.arange(len(batch)), input_lengths - 1]
        probs = torch.softmax(last_logits, dim=-1)

        for row_offset, (_, row) in enumerate(batch.iterrows()):
            clean_idx = int(row['clean_answer_idx'])
            corrupted_idx = int(row['corrupted_answer_idx'])
            prob_bias = (probs[row_offset, clean_idx] - probs[row_offset, corrupted_idx]).item()
            logit_bias = (last_logits[row_offset, clean_idx] - last_logits[row_offset, corrupted_idx]).item()

            if abs(prob_bias) > args.bias_threshold:
                biased_samples.append({
                    'idx': start + row_offset,
                    'clean': row['clean'],
                    'corrupted': row['corrupted'],
                    'clean_answer_idx': clean_idx,
                    'corrupted_answer_idx': corrupted_idx,
                    'prob_bias': prob_bias,
                    'logit_bias': logit_bias,
                })

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w') as f:
    json.dump(biased_samples, f, indent=2)

prob_biases = np.abs([s['prob_bias'] for s in biased_samples])
print(f'{len(biased_samples)} / {len(df)} samples with |P(he) - P(she)| > {args.bias_threshold}')
print(f'baseline probability bias: {prob_biases.mean():.3f} ± {prob_biases.std():.3f}')
print(f'saved to {args.output_path}')
