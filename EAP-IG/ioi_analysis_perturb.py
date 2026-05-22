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
from utils import get_logit_positions, logit_diff, EAPDataset


set_seed(2025)
data_num = 1000
topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
perturb_times = 1
var = [0.01, 0.001]
# var = [0.1]
replace_ratio = [0.1, 0.3]
assert len(var) == len(replace_ratio)
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

all_best_results = []
all_vanilla_results = []
all_best_random = []
all_best_perturb_all_var = []
all_best_random_drop_all = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples", position=0)):
    single_data = [(clean, corrupted, label)]

    # print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

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

        circuit_faithfulness = []
        for topn in topns:
            g.apply_topn(topn, True)

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)
        if j == i:
            all_vanilla_results.append(circuit_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_para_indices[idx] = j

    all_best_results.append(best_results)

    # random scores
    best_random = [-1]*len(topns)
    best_rand_indices = [0]*len(topns)
    for _ in range(10):
        model.reset_hooks()
        g = Graph.from_model(model)
        g.scores = torch.rand_like(torch.from_numpy(para_data[0]))
        
        circuit_faithfulness = []
        for topn in topns:
            g.apply_topn(topn, True)

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

        for idx in range(len(best_random)):
            if circuit_faithfulness[idx] > best_random[idx]:
                best_random[idx] = circuit_faithfulness[idx]
                best_rand_indices[idx] = j

    all_best_random.append(best_random)
    
    # perturb on circuit of original query
    base_score = para_data[i]
    all_best_perturb = []
    for v in range(len(var)):
        best_perturb = [-1]*len(topns)
        for p in range(10):
            model.reset_hooks()
            g = Graph.from_model(model)
            if p == 0:
                g.scores = torch.from_numpy(base_score)
            else:
                # attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-inputs-sg', ig_steps=steps, intervention=intervention, quiet=True, perturb_times=perturb_times, var=var[v])
                g.scores = torch.from_numpy(base_score) + torch.randn_like(torch.from_numpy(base_score)) * var[v]

            circuit_faithfulness = []
            for topn in topns:
                g.apply_topn(topn, True)

                results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
                results = results.mean().item()

                faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
                circuit_faithfulness.append(faithfulness)

            for idx in range(len(best_perturb)):
                if circuit_faithfulness[idx] > best_perturb[idx]:
                    best_perturb[idx] = circuit_faithfulness[idx]
        all_best_perturb.append(best_perturb)
    all_best_perturb_all_var.append(all_best_perturb)

    # perturb by random dropping
    base_score = para_data[i]
    all_best_random_drop = []
    for v in range(len(var)):
        best_random_drop = [-1]*len(topns)
        for p in range(10):
            model.reset_hooks()
            g = Graph.from_model(model)
                                 
            g.scores = torch.from_numpy(base_score)
            circuit_faithfulness = []
            for topn in topns:
                if p == 0:
                    g.apply_topn(topn, True)
                else:
                    g.apply_topn_and_rand(topn, True, replace_ratio=replace_ratio[v])

                results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
                results = results.mean().item()

                faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
                circuit_faithfulness.append(faithfulness)

            for idx in range(len(best_random_drop)):
                if circuit_faithfulness[idx] > best_random_drop[idx]:
                    best_random_drop[idx] = circuit_faithfulness[idx]
        all_best_random_drop.append(best_random_drop)
    all_best_random_drop_all.append(all_best_random_drop)


all_best_perturb_all_var1 = [all_best_perturb_all_var[i][0] for i in range(len(all_best_perturb_all_var))]
all_best_perturb_all_var2 = [all_best_perturb_all_var[i][1] for i in range(len(all_best_perturb_all_var))]
all_best_random_drop_all1 = [all_best_random_drop_all[i][0] for i in range(len(all_best_random_drop_all))]
all_best_random_drop_all2 = [all_best_random_drop_all[i][1] for i in range(len(all_best_random_drop_all))]

all_best_perturb_all_var1 = np.array(all_best_perturb_all_var1).mean(0)
all_best_perturb_all_var2 = np.array(all_best_perturb_all_var2).mean(0)
all_best_random_drop_all1 = np.array(all_best_random_drop_all1).mean(0)
all_best_random_drop_all2 = np.array(all_best_random_drop_all2).mean(0)

all_best_results = np.array(all_best_results).mean(0)
all_vanilla_results = np.array(all_vanilla_results).mean(0)
all_best_random = np.array(all_best_random).mean(0)
print('all_best_results: ', all_best_results)
print('all_vanilla_results: ', all_vanilla_results)
print('all_best_random: ', all_best_random)
print('all_best_perturb_all_var1 (0.01): ', all_best_perturb_all_var1)
print('all_best_perturb_all_var2 (0.001): ', all_best_perturb_all_var2)
print('all_best_random_drop_all1 (0.1): ', all_best_random_drop_all1)
print('all_best_random_drop_all2 (0.3): ', all_best_random_drop_all2)
plt.plot(topns, all_vanilla_results, label='Single Query', marker='o')
plt.plot(topns, all_best_results, label='BoN-Para.', marker='o')
plt.plot(topns, all_best_random, label='BoN-Random', marker='o')
plt.plot(topns, all_best_perturb_all_var1, label='BoN-SP (0.01)', marker='o')
plt.plot(topns, all_best_perturb_all_var2, label='BoN-SP (0.001)', marker='o')
plt.plot(topns, all_best_random_drop_all1, label='BoN-ER (0.1)', marker='o')
plt.plot(topns, all_best_random_drop_all2, label='BoN-ER (0.3)', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/ioi_perturb.pdf', bbox_inches='tight')