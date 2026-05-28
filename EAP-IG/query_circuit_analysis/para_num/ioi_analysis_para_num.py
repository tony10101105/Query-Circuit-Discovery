import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

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

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import topn_indices, set_seed
from eap.query_circuit_utils import get_logit_positions, logit_diff, EAPDataset


set_seed(2025)
data_num = 1000
topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small' # meta-llama/Llama-3.2-1B-Instruct, gpt2-small
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', data_num=data_num)
dataloader = ds.to_dataloader(batch_size=1)

para_data = []
for k in range(data_num):
    para_data.append(np.load(f"Query-Circuit-Dataset/score_data/ioi_{steps}steps/gpt2-small/ioi_edge_scores_{k}.npy"))

para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)

all_results = []
all_vanilla_results = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples")):
    single_data = [(clean, corrupted, label)]

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')
    
    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    all_results_per_sample = []
    
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

            # print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

            # print(f"{i}-th sample; model performance: {baseline:.2f}; corrupted baseline: {corrupted_baseline:.2f}'; circuit performance: {results:.2f}; faithfulness: {faithfulness:.2f}")

        if j == i:
            all_vanilla_results.append(circuit_faithfulness)

        all_results_per_sample.append(circuit_faithfulness)
    
    all_results_per_sample = np.array(all_results_per_sample)
    all_results.append(all_results_per_sample)

markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+', '1']
for i in range(10, 1, -1):
    para = [np.max(k[:i], axis=0) for k in all_results]
    avg = np.mean(np.stack(para, axis=0), axis=0)
    plt.plot(topns, avg, label=f'Best-of-{i}', marker=markers[i-2])

all_vanilla_results = np.array(all_vanilla_results).mean(0)
print('all_vanilla_results: ', all_vanilla_results)
plt.plot(topns, all_vanilla_results, label='Single Query', marker='2')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=13)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# handles, labels = plt.gca().get_legend_handles_labels()
# sorted_handles_labels = sorted(
#     zip(handles, labels),
#     key=lambda x: (x[1] == "Original Query", x[1])  # True sorts after False
# )
# handles_sorted, labels_sorted = zip(*sorted_handles_labels)
# print(labels_sorted)
# plt.legend(handles_sorted, labels_sorted, fontsize=10, loc='lower right')
plt.legend(fontsize=10, loc='lower right')

plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/ioi_para_num.pdf', bbox_inches='tight')