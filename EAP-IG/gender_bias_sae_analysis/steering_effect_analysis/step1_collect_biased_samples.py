import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from eap.query_circuit_utils import logit_diff, get_logit_positions
from save_score_matrix.models import TargetModelConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--bias_threshold', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_path', type=str, default='probing_dataset/gender_bias_gpt2.csv')
parser.add_argument('--output_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/biased_samples.json')
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

        clean_idxs = torch.tensor(batch['clean_answer_idx'].tolist())
        corrupted_idxs = torch.tensor(batch['corrupted_answer_idx'].tolist())
        labels = torch.stack([clean_idxs, corrupted_idxs], dim=-1)

        logit_biases = logit_diff(logits, None, input_lengths, labels, mc=False, mean=False, loss=False)

        last_logits = get_logit_positions(logits, input_lengths)
        probs = torch.softmax(last_logits, dim=-1)
        prob_biases = probs[torch.arange(len(batch)), clean_idxs] - probs[torch.arange(len(batch)), corrupted_idxs]

        for row_offset, (_, row) in enumerate(batch.iterrows()):
            clean_idx = int(row['clean_answer_idx'])
            corrupted_idx = int(row['corrupted_answer_idx'])
            prob_bias = prob_biases[row_offset].item()
            logit_bias = logit_biases[row_offset].item()

            if prob_bias > args.bias_threshold:
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

prob_biases = np.array([s['prob_bias'] for s in biased_samples])
print(f'{len(biased_samples)} / {len(df)} samples with P(stereotypical) - P(anti-stereotypical) > {args.bias_threshold}')
print(f'baseline probability bias: {prob_biases.mean():.3f} ± {prob_biases.std():.3f}')
print(f'saved to {args.output_path}')
