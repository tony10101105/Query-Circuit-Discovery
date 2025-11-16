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

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = list(labels)
    # labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
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
        return row['clean'], row['corrupted'], [int(row['correct_idx']), ast.literal_eval(row['incorrect_idx'])]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
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
category = 'marketing'
metric_version = 8 # 1,2,3,4,5,6,7,8,9
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations # EAP
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset(f'probing_dataset/mmlu_{category}_Llama-32-1B.csv', num_samples=300)
# ds = EAPDataset(f'probing_dataset/test_paraphrase.csv', num_samples=300)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    # if i not in [11,41,49]:
    if i not in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
        continue
    single_data = [(clean, corrupted, label)]

    model.reset_hooks()
    
    g = Graph.from_model(model)

    # print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False, version=metric_version)).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False, version=metric_version), run_corrupted=True).mean().item()

    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)
    
    print(f"{i}-th sample. Original: {baseline}; corrupted: {corrupted_baseline}")
    # print('attributing for this single data...')
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True, version=metric_version), method=method, ig_steps=steps, intervention=intervention, quiet=True)
    
    # print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness_ndf, circuit_faithfulness_nfs = [], []
    for topn in topns:
        g.apply_topn(topn, True)
        # g.apply_greedy(topn, True)

        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False, version=metric_version), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        circuit_results.append(results)

        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        faithfulness_ndf = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness_ndf.append(faithfulness_ndf)

        faithfulness_nfs = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness_nfs.append(faithfulness_nfs)

        print(f"{i}-th sample. Original performance: {baseline:.2f}; circuit performance: {results:.2f}; corrupted_baseline: {corrupted_baseline:.2f}; faithfulness (NDF): {faithfulness_ndf:.2f}; faithfulness (NFS): {faithfulness_nfs:.2f}")

    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'topns': topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness_ndf': circuit_faithfulness_ndf,
        'circuit_faithfulness_nfs': circuit_faithfulness_nfs
    })


topns = [x // 1000 for x in topns]  # Convert to 'k'

fig, axes = plt.subplots(6, 3, figsize=(15, 24))  # 3 rows × 3 cols
for i, (results, ax1) in enumerate(zip(all_results, axes.flat)):
    row, col = divmod(i, 3)

    # Plot NFS (Primary Y-axis on left)
    ax1.plot(topns, results['circuit_faithfulness_nfs'],
             label='NFS', marker='s', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax1.set_ylim(-2.1, 2.1)
    ax1.axhline(y=0, linestyle='--', color='gray')
    ax1.axhline(y=1, linestyle='--', color='gray')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)

    # Add NFS label only on left-middle subplot (i == 3)
    if i == 6:
        ax1.set_ylabel("Normalized Faithfulness Score (NFS)", color='tab:red', fontsize=20)
    else:
        ax1.set_ylabel("")

    # Plot NDF (Secondary Y-axis on right)
    ax2 = ax1.twinx()
    ax2.plot(topns, results['circuit_faithfulness_ndf'],
             label='NDF (ours)', marker='o', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax2.set_ylim(-2.1, 2.1)

    # Add NDF label only on right-middle subplot (i == 5)
    if i == 8:
        ax2.set_ylabel("Normalized Deviation Faithfulness (NDF)", color='tab:blue', fontsize=20)
    else:
        ax2.set_ylabel("")

    # X-axis label on bottom-middle subplot (i == 7)
    if i == 16:
        ax1.set_xlabel("Number of Edges (k)", fontsize=20)

    ax1.set_title(f"Query {i+1}", fontsize=16)
    ax1.set_xticks([topns[0]] + topns[5:])
    ax1.tick_params(axis='x', labelsize=14)

    # Optional: legend on first subplot only
    if i == 0:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=20, loc='lower right')

fig.tight_layout()
fig.savefig(f'mmlu_marketing_metric{metric_version}_NFS_NDF_grid.pdf')