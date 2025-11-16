from functools import partial

import os
import sys
import ast
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute
from src.eap.utils import topn_indices, set_seed

shots = "Which is the most possible answer?\n"

os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

set_seed(2025)

def PARA_collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    # print(labels)
    labels = labels[0]
    clean = clean[0]
    corrupted = corrupted[0]
    # print('clean: ', clean)
    return clean, corrupted, labels

class PARA_EAPDataset(Dataset):
    def __init__(self, filepath, category=None, num_samples=None):
        self.df = pd.read_csv(filepath)
        if category:
            self.df = self.df[self.df['category'] == category]
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        print(f'Loaded {len(self.df)} samples from {filepath} with category {category} and {len(self.df)} samples')

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # expand each row to 10 paraphrases
        clean = []
        for i in range(10):
            if i == 0:
                clean.append(row['clean'])
            else:
                clean.append(row['paraphrase' + str(i)])
        
        # corrupted = []
        # for i in range(10):
        #     corrupted.append(row['corrupted'])
            # assert self.format_corrupted_prompt(row['clean']) == row['corrupted'], f"Corrupted prompt format mismatch.\nOriginal Corruption: {row['corrupted']}, Reg Corruption: {self.format_corrupted_prompt(row['clean'])}"
        corrupted = [row['corrupted']] * 10

        correct_idx = int(row['correct_idx'])
        incorrect_idx = ast.literal_eval(row['incorrect_idx'])
        labels = [[correct_idx, incorrect_idx]] * 10

        return clean, corrupted, labels

    # def format_corrupted_prompt(self, text):
    #     # Regex explanation:
    #     # ^(.*?)\n   -> capture the first line (the description) up to the first newline
    #     # (?=\(A\))  -> look ahead to ensure the next line starts with (A)
    #     pattern = r"^(.*?)\n(?=\(A\))"
    #     return re.sub(pattern, "Which is the most possible answer?\n", text, flags=re.MULTILINE)

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=PARA_collate_EAP)

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: list[list],             # list of [correct_idx, [wrong_idx1, wrong_idx2, ...]]
    version: int = 8,
    mean: bool = True,
    loss: bool = False,
):
    # Extract the last-token logits: [batch_size, vocab_size]
    logits = get_logit_positions(logits, input_length)
    probs = torch.softmax(logits, dim=-1)

    batch_size = logits.size(0)

    # Get correct token logits
    correct_idxs = torch.tensor([lbl[0] for lbl in labels], device=logits.device)
    correct_logits = logits[torch.arange(batch_size), correct_idxs]

    if version == 1:
        # (1) correct - second-max token over vocab (excluding correct token)
        masked_logits = logits.clone()
        masked_logits[torch.arange(batch_size), correct_idxs] = float('-inf')
        bad = masked_logits.max(dim=-1).values

    elif version == 2:
        # (2) correct - average of all *other* vocab tokens
        sums = logits.sum(dim=-1) - correct_logits
        bad = sums / (logits.shape[-1] - 1)

    elif version == 3:
        # (3) correct - second max among wrong option tokens
        bad_vals = []
        for i, (_, wrong_ids) in enumerate(labels):
            wrong_logits = logits[i, torch.tensor(wrong_ids, device=logits.device)]
            bad_vals.append(wrong_logits.max())
        bad = torch.stack(bad_vals)

    elif version == 4:
        # (4) correct - average over all wrong option tokens
        bad_vals = []
        for i, (_, wrong_ids) in enumerate(labels):
            wrong_logits = logits[i, torch.tensor(wrong_ids, device=logits.device)]
            bad_vals.append(wrong_logits.mean())
        bad = torch.stack(bad_vals)

    # ---------------------------
    # New: probability versions
    # ---------------------------
    elif version == 5:
        # (5) prob(correct) - second-max prob over full vocab (exclude correct)
        correct_probs = probs[torch.arange(batch_size), correct_idxs]
        masked_probs = probs.clone()
        # Set correct index to -1 so it can’t be the max (probs are in [0,1])
        masked_probs[torch.arange(batch_size), correct_idxs] = -1.0
        bad = masked_probs.max(dim=-1).values
        results = correct_probs - bad

    elif version == 6:
        # (6) prob(correct) - mean prob over all other vocab (exclude correct)
        correct_probs = probs[torch.arange(batch_size), correct_idxs]
        sums = probs.sum(dim=-1) - correct_probs
        bad = sums / (probs.shape[-1] - 1)
        results = correct_probs - bad

    elif version == 7:
        # (7) prob(correct) - max prob among wrong options
        correct_probs = probs[torch.arange(batch_size), correct_idxs]
        bad_vals = []
        for i, (_, wrong_ids) in enumerate(labels):
            wrong_probs = probs[i, torch.tensor(wrong_ids, device=logits.device)]
            bad_vals.append(wrong_probs.max())
        bad = torch.stack(bad_vals)
        results = correct_probs - bad

    elif version == 8:
        # (8) prob(correct) - mean prob among wrong options
        correct_probs = probs[torch.arange(batch_size), correct_idxs]
        bad_vals = []
        for i, (_, wrong_ids) in enumerate(labels):
            wrong_probs = probs[i, torch.tensor(wrong_ids, device=logits.device)]
            bad_vals.append(wrong_probs.mean())
        bad = torch.stack(bad_vals)
        results = correct_probs - bad

    elif version == 9:
        # (8) prob(correct) - sum prob among wrong options
        correct_probs = probs[torch.arange(batch_size), correct_idxs]
        bad_vals = []
        for i, (_, wrong_ids) in enumerate(labels):
            wrong_probs = probs[i, torch.tensor(wrong_ids, device=logits.device)]
            bad_vals.append(wrong_probs.sum())
        bad = torch.stack(bad_vals)
        results = correct_probs - bad
    else:
        raise ValueError(f"Invalid version: {version}")

    # For versions 1–4, compute results after 'bad' is set
    if version in (1, 2, 3, 4):
        results = correct_logits - bad

    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


# topns = [500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 32000, 32491] # 32491 for gpt2-small, 386713 for llama
topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
category = 'challenge'
metric_version = 4 # 1,2,3,4,5,6,7,8,9
rephrase_type = 'only_stem'
rephrase_model = 'gpt4o' # gpt4o-mini
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = PARA_EAPDataset(f'probing_dataset/arc_{category}_Llama-32-1B_{rephrase_model}_paraphrases_{rephrase_type}.csv', num_samples=2000)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    assert len(clean) == len(corrupted) and len(corrupted) == len(label)
    
    batch_slice_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]
    model.reset_hooks()
    g = Graph.from_model(model)
    
    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False, version=metric_version)).mean().item()
    corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False, version=metric_version), run_corrupted=True).mean().item()

    # only for padding corrupted input
    _ = evaluate_baseline(model, batch_slice_data, partial(logit_diff, loss=False, mean=False, version=metric_version), run_corrupted=True, manual_pad=True).mean().item()

    print('attributing for this single data...')
    attribute(model, g, batch_slice_data, partial(logit_diff, loss=True, mean=True, version=metric_version), method=method, ig_steps=steps, intervention=intervention, file_idx=i, cat=category)
    continue
    print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        # g.apply_greedy(topn, True) # extremely slow
        
        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False, version=metric_version), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)
        results = results.mean().item()
        circuit_results.append(results)
        
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)

        print(f"Original performance: {baseline:.2f}; circuit performance: {results:.2f}; corrupted_baseline: {corrupted_baseline:.2f}; faithfulness: {faithfulness:.2f}")

    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'topns': topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })
exit(0)
faithfulness_matrix = np.array([res['circuit_faithfulness'] for res in all_results])
average_faithfulness = faithfulness_matrix.mean(axis=0)
plt.plot(topns, average_faithfulness, marker='o', label='Avg Faithfulness')
plt.xlabel('Top-n Edges')
plt.ylabel('Average Circuit Faithfulness')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('arcc.png')

# topns = [x // 1000 for x in topns]  # Convert to 'k'
# for i, results in enumerate(all_results):
#     plt.plot(topns, results['circuit_faithfulness'], label=f'EAP-IG (step=20) on Query {i}', marker='o')
# plt.ylim(-2.1, 2.1)
# plt.axhline(y=0, linestyle='--', color='gray')
# plt.axhline(y=1, linestyle='--', color='gray')
# plt.xlabel('Number of Edges (k)', fontsize=16)
# plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(fontsize=16)
# plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
# plt.tight_layout()
# plt.savefig(f'figures/mmlu_marketing_metric{metric_version}_instability.pdf', bbox_inches='tight')
# plt.close()        
# exit(0)

with open(f'preprocessed_data/arcc_{method.lower()}_metric{metric_version}_{steps}steps.json', 'w') as f:
    json.dump(all_results, f, indent=2)
    
# topns = all_results[0]['topns']
# faithfulness_per_topn = [[] for _ in range(len(topns))]

# for d in all_results:
#     for i, val in enumerate(d['circuit_faithfulness']):
#         faithfulness_per_topn[i].append(val)

# # Compute mean and std across samples
# means = np.array([np.mean(vals) for vals in faithfulness_per_topn])
# stds  = np.array([np.std(vals)  for vals in faithfulness_per_topn])

# # Plot with error bars
# plt.figure(figsize=(7, 4.5))
# plt.errorbar(
#     topns,
#     means,
#     yerr=stds,
#     fmt='-o',
#     label='One Sample IG',
#     capsize=4,
#     elinewidth=1,
#     markeredgewidth=1
# )
# plt.ylim(-2.5, 2.1)
# # plt.xscale('log')
# plt.xlabel('Top-K Edges')
# plt.ylabel('Circuit Faithfulness')
# plt.title(f'MMLU Circuit Faithfulness vs Top-K Edges (metric {metric_version})')
# plt.legend()
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.savefig(f'mmlu_marketing_one_sample_{metric_version}.png', dpi=500, bbox_inches='tight')