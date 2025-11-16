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
        self.df = self.df[:data_num]

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
topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-xl' # meta-llama/Llama-3.2-1B-Instruct, gpt2-small
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', data_num=data_num)
dataloader = ds.to_dataloader(batch_size=1)

para_data = []
for k in range(data_num):
    para_data.append(np.load(f"score_data/ioi_{steps}steps/gpt2-xl/ioi_edge_scores_{k}.npy"))

para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)

all_best_results = []
# all_best_complement_results = []
all_vanilla_results = []
# all_vanilla_complement_results = []
all_avg_results = []
all_csm_results = []
all_ibon_results = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples", position=0)):
    single_data = [(clean, corrupted, label)]

    # print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    # best_complement_results = [-1]*len(topns)
    best_results = [-1]*len(topns)
    best_para_indices = [0]*len(topns) # keep track of the best paraphrase index for each topn

    all_indices = list(range(para_data.shape[0]))
    available_idxs = [para_idx for para_idx in all_indices if para_idx != i]
    sampled = random.sample(available_idxs, 9)
    sampled = [i] + sampled

    for j in tqdm(sampled, total=len(sampled), desc="Processing paraphrases", position=1):
        model.reset_hooks()
        
        g = Graph.from_model(model)

        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness, circuit_complement_faithfulness = [], []
        for topn in topns:
            g.apply_topn(topn, True)

            # print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
            circuit_faithfulness.append(faithfulness)

            # g.apply_topn(topn, True) # if complement is True, return M\C instead of C
            # complement_results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            # complement_results = complement_results.mean().item()

            # complement_faithfulness = 1 - min(abs((baseline - complement_results) / (baseline - corrupted_baseline)), 1)
            # circuit_complement_faithfulness.append(complement_faithfulness)
            
            # print(f"{i}-th sample; model performance: {baseline:.2f}; corrupted baseline: {corrupted_baseline:.2f}'; circuit performance: {results:.2f}; faithfulness: {faithfulness:.2f}")

        if j == i:
            all_vanilla_results.append(circuit_faithfulness)
            # all_vanilla_complement_results.append(circuit_complement_faithfulness)
        
        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_para_indices[idx] = j
                # best_complement_results[idx] = circuit_complement_faithfulness[idx]

    all_best_results.append(best_results)
    # all_best_complement_results.append(best_complement_results)

    # averaging
    model.reset_hooks()
    g = Graph.from_model(model)

    g.scores = torch.from_numpy(para_data[sampled].mean(0))
    # g.scores = torch.from_numpy(para_data.mean(0))

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)

    all_avg_results.append(circuit_faithfulness)
    
    # BoN with Constraint-adaptive Score Matrix (BoN-CSM)
    score_mat = np.full_like(para_data[j], -np.inf, dtype=float)
    tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    filled    = np.zeros_like(score_mat, dtype=bool)

    for l, topn in enumerate(topns):           # l = 0 (highest priority), 1, ...
        best_para_idx = best_para_indices[l]
        M = np.abs(para_data[best_para_idx])
        M = np.where(np.isfinite(M), M, -np.inf)
        for (a, b) in topn_indices(M, topn):
            if not filled[a, b]:
                score_mat[a, b] = M[a, b]
                tier_mat[a, b]  = l
                filled[a, b]    = True
    
    model.reset_hooks()
    g = Graph.from_model(model)
    g.scores = torch.from_numpy(score_mat)

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn_by_tier(topn, tier_mat)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)
    all_csm_results.append(circuit_faithfulness)
    
    # interpolated BoN (iBoN)
    new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]
    
    circuit_faithfulness = []
    for k, topn in enumerate(new_topns):
        model.reset_hooks()
        g = Graph.from_model(model)
    
        previous_topn, next_topn = topns[k], topns[k+1]

        score_mat = np.full_like(para_data[j], -np.inf, dtype=float)
        tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
        filled    = np.zeros_like(score_mat, dtype=bool)

        for l, top in zip([k, k+1], [previous_topn, next_topn]):
            best_para_idx = best_para_indices[l]
            M = np.abs(para_data[best_para_idx])
            M = np.where(np.isfinite(M), M, -np.inf)
            for (a, b) in topn_indices(M, top):
                if not filled[a, b]:
                    score_mat[a, b] = M[a, b]
                    tier_mat[a, b]  = l
                    filled[a, b]    = True
        
        g.scores = torch.from_numpy(score_mat)
        g.apply_topn_by_tier(topn, tier_mat)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)
    all_ibon_results.append(circuit_faithfulness)


all_best_results = np.array(all_best_results).mean(0)
all_vanilla_results = np.array(all_vanilla_results).mean(0)
all_avg_results = np.array(all_avg_results).mean(0)
all_csm_results = np.array(all_csm_results).mean(0)
all_ibon_results = np.array(all_ibon_results).mean(0)
print('all_best_results: ', all_best_results)
print('all_vanilla_results: ', all_vanilla_results)
print('all_avg_results: ', all_avg_results)
print('all_csm_results: ', all_csm_results)
print('all_ibon_results: ', all_ibon_results)
plt.plot(topns, all_vanilla_results, label='Single Query', marker='o')
plt.plot(topns, all_avg_results, label='Averaging', marker='o')
plt.plot(topns, all_best_results, label='BoN', marker='o')
plt.plot(topns, all_csm_results, label='BoN-CSM', marker='o')
plt.plot(new_topns, all_ibon_results, label='iBoN', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/ioi_gpt2-xl.pdf', bbox_inches='tight')