from functools import partial

import os
import sys
import ast
import json
import bisect
import numpy as np
import random
from scipy.stats import spearmanr
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

os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

set_seed(2025)

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, data_num):
        self.df = pd.read_csv(filepath)
        self.df = self.df[-data_num:]

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


data_num = 1000
topns = [500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000] # 32491 for gpt2-small, 386713 for llama
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset(f'probing_dataset/arithmetic_mul_Llama-32-1B.csv', data_num=data_num)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
all_vanilla_results = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples")):
    para_data = []
    for k in range(10):
        para_data.append(np.load(f"score_data/arithmetic_mul/{i}_{k}.npy"))

    para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)
    
    single_data = [(clean, corrupted, label)]

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')
    
    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    all_results_per_sample = []

    for j in tqdm(range(para_data.shape[0]), total=para_data.shape[0], desc="Processing paraphrases"):
        model.reset_hooks()
        
        g = Graph.from_model(model)

        g.scores = torch.from_numpy(para_data[j])

        circuit_performance, circuit_complement_performance, circuit_faithfulness, circuit_complement_faithfulness = [], [], [], []
        for topn in topns:
            g.apply_topn(topn, True)

            print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            circuit_performance.append(results)
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)
            
            print(f"Model performance: {baseline:.2f}; corrupted baseline: {corrupted_baseline:.2f}'; circuit performance: {results:.2f}; faithfulness: {faithfulness:.2f}")

        if j == 0:
            all_vanilla_results.append(circuit_faithfulness)

        all_results_per_sample.append(circuit_faithfulness)

    all_results_per_sample = np.array(all_results_per_sample)
    all_results.append(all_results_per_sample)

topns = [x // 1000 for x in topns]  # Convert to 'k'

markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+', '1']
for i in range(10, 1, -1):
    para = [np.max(k[:i], axis=0) for k in all_results]
    avg = np.mean(np.stack(para, axis=0), axis=0)
    plt.plot(topns, avg, label=f'Best-of-{i}', marker=markers[i-2])

all_vanilla_results = np.array(all_vanilla_results).mean(0)
print('all_vanilla_results: ', all_vanilla_results)
plt.plot(topns, all_vanilla_results, label='Single Query', marker='2')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges (k)', fontsize=13)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/arithmetic_mul_para_num.pdf')