from functools import partial

import os
import sys
import ast
import json
import numpy as np
import random
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from venny4py.venny4py import *
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

e = [36] # in figure 4
choices = list(range(data_num)) # for rebuttal: adding 5 randomly selected samples
choices.remove(36)
nums = random.choices(choices, k=5)
e += nums
e = sorted(e)

topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
# topns = [10000, 20000, 30000, 32491] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', data_num=data_num)
dataloader = ds.to_dataloader(batch_size=1)

all_results = {}
all_best_results = []
all_vanilla_results = []
all_avg_results = []
all_avg_results_10 = []

para_data = []
for k in range(data_num):
    para_data.append(np.load(f"score_data/ioi_{steps}steps/gpt2-small/ioi_edge_scores_{k}.npy"))

para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)


for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples")):
    if i not in e:
        continue
    single_data = [(clean, corrupted, label)]
    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')
    
    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    best_results = [-1]*len(topns)
    best_para_indices = [0]*len(topns) # keep track of the best paraphrase index for each topn

    all_indices = list(range(para_data.shape[0]))
    available_idxs = [para_idx for para_idx in all_indices if para_idx != i]
    sampled = random.sample(available_idxs, 9)
    sampled = [i] + sampled
    
    for j in tqdm(sampled, total=len(sampled), desc="Processing paraphrases"):
        model.reset_hooks()
        
        g = Graph.from_model(model)

        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness = []
        for topn in topns:
            g.apply_topn(topn, True)

            print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
            # faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

            print(f"Model performance: {baseline:.2f}; corrupted baseline: {corrupted_baseline:.2f}'; circuit performance: {results:.2f}; faithfulness: {faithfulness:.2f}")
        if j == i:
            all_vanilla_results.append(circuit_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_para_indices[idx] = j
        
        if i not in all_results:
            all_results[i] = {}
        all_results[i].update({j: circuit_faithfulness})

    all_best_results.append(best_results)
    
    # average all 1000 paraphrases
    model.reset_hooks()
    g = Graph.from_model(model)

    g.scores = torch.from_numpy(para_data.mean(0))

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        # faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)

    all_avg_results.append(circuit_faithfulness)

    # average only the 10
    model.reset_hooks()
    g = Graph.from_model(model)

    g.scores = torch.from_numpy(para_data[sampled].mean(0))

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        # faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)

    all_avg_results_10.append(circuit_faithfulness)


for k, idx in enumerate(e):
    results = all_results[idx]
    avg_results = all_avg_results[k]
    # plt.plot(topns, avg_results, label='Averaging - All\n(Capability Circuit)',
    #             color="blue", linewidth=3, linestyle="--", marker="s")  # thick solid line, square markers
    
    # avg_results_10 = all_avg_results_10[k]
    # plt.plot(topns, avg_results_10, label='Averaging - 10',
    #             color="green", linewidth=3, linestyle="--", marker="s")  # thick solid line, square markers
    for i, (key, value) in enumerate(results.items()):
        if key == idx:
            label = 'Query $q$'
        else:
            print('key: ', key)
            label = f'Paraphrase $q_{i}$'

        if label == "Query $q$":
            plt.plot(topns, value, label=label,
                    color="blue", linewidth=3, linestyle="--", marker="s")  # thick dashed line, square markers
        else:
            plt.plot(topns, value, label=label,
                    linewidth=1, linestyle="-", alpha=0.6, marker="o")  # thinner, faded lines for paraphrases

    # plt.ylim(-0.1, 1.1)
    plt.ylim(-2.1, 2.1)
    plt.xlabel('Number of Edges', fontsize=15)
    # plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
    plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'figures/ioi_query{idx}_para_performance_nfs.pdf')
    plt.close()
exit(0)

# plot Venn diagram
def jaccard_similarity(list1, list2):
    A, B = set(list1), set(list2)
    if not A and not B:  # convention when both empty
        return 1.0
    return len(A & B) / len(A | B), len(A & B), len(A | B)

para2 = para_data[0]
para2 = abs(para2)
para2 = np.where(np.isinf(para2), -np.inf, para2)
para2_topn250_idx = set(topn_indices(para2, 250))
para2_topn500_idx = set(topn_indices(para2, 500))

para15 = para_data[7]
para15 = abs(para15)
para15 = np.where(np.isinf(para15), -np.inf, para15)
para15_topn250_idx = set(topn_indices(para15, 250))
para15_topn500_idx = set(topn_indices(para15, 500))

a = jaccard_similarity(para2_topn250_idx, para15_topn250_idx)
b = jaccard_similarity(para2_topn250_idx, para15_topn500_idx)
c = jaccard_similarity(para2_topn500_idx, para15_topn250_idx)
d = jaccard_similarity(para2_topn500_idx, para15_topn500_idx)

print(f'jac: {a[0]:.2f}, intersec: {a[1]:.2f}, union: {a[2]:.2f}')
print(f'jac: {b[0]:.2f}, intersec: {b[1]:.2f}, union: {b[2]:.2f}')
print(f'jac: {c[0]:.2f}, intersec: {c[1]:.2f}, union: {c[2]:.2f}')
print(f'jac: {d[0]:.2f}, intersec: {d[1]:.2f}, union: {d[2]:.2f}')

#dict of sets
sets = {
    'Paraphrase 0 - Top-250 (NDF: 0)': para2_topn250_idx,
    'Paraphrase 0 - Top-500 (NDF: 0.80)': para2_topn500_idx, # good
    'Paraphrase 7 - Top-500 (NDF: 0.28)': para15_topn500_idx,
    'Paraphrase 7 - Top-250 (NDF: 0.98)': para15_topn250_idx,} # good

venny4py(sets=sets, ext='pdf')
print(all_results[e][0])
print(all_results[e][7])