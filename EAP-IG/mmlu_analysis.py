from functools import partial

import os
import sys
import ast
import json
import bisect
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
from time import sleep

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute
from src.eap.utils import topn_indices, set_seed
from utils import get_logit_positions, logit_diff, EAPDataset


set_seed(2025)
topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
category = 'astronomy' # marketing, professional_medicine, astronomy, college_biology, high_school_computer_science, logical_fallacies, nutrition, international_law, management
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B-Instruct # meta-llama/Meta-Llama-3-8B-Instruct
# model = HookedTransformer.from_pretrained(model_name, device='cuda')
model = HookedTransformer.from_pretrained_no_processing(model_name, device='cuda', torch_dtype=torch.float16)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset(f'probing_dataset/mmlu_{category}_Llama-32-1B.csv', num_samples=500, mc=True)
dataloader = ds.to_dataloader(batch_size=1)

all_best_results = []
all_vanilla_results = []
all_avg_results = []
all_csm_results = []
all_ibon_results = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples", position=0)):
    para_data = []
    for k in range(10):
        para_data.append(np.load(f"Query-Circuit-Dataset/score_data/mmlu_{category}/llama3-8b/metric4_{i}_{k}.npy"))

    para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)
    
    single_data = [(clean, corrupted, label)]

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    best_c = [-1]*len(topns)
    best_results = [-1]*len(topns)
    # best_complement_results = [-1]*len(topns)
    best_para_indices = [0]*len(topns) # keep track of the best paraphrase index for each topn
    
    for j in tqdm(range(para_data.shape[0]), total=para_data.shape[0], desc="Processing paraphrases", position=1, leave=False):
        model.reset_hooks()
        
        g = Graph.from_model(model)

        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness, circuit_complement_faithfulness = [], []
        c = []
        for topn in topns:
            g.apply_topn(topn, True)
            # g.to_json(f'mmlu_Llama-3.2-1B-Instruct_{topn}_circuit.json')
            # gz = g.to_image(f'mmlu_{model_name}_{topn}_circuit.png')
            exit(0)
            # print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            try:
                faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            except:
                faithfulness = 0
            # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
            c.append(results)
            circuit_faithfulness.append(faithfulness)

            # g.apply_topn(topn, True, complement=True) # if complement is True, return M\C instead of C
            # complement_results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            # complement_results = complement_results.mean().item()
            
            # complement_faithfulness = 1 - min(abs((baseline - complement_results) / (baseline - corrupted_baseline)), 1)
            # circuit_complement_faithfulness.append(complement_faithfulness)

            # print(f"{i}-th sample; model performance: {baseline:.2f}; corrupted baseline: {corrupted_baseline:.2f}; circuit performance: {results:.2f}; faithfulness: {faithfulness:.2f}")

        if j == 0:
            all_vanilla_results.append(circuit_faithfulness)
            # all_vanilla_complement_results.append(circuit_complement_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                # best_complement_results[idx] = circuit_complement_faithfulness[idx]
                best_para_indices[idx] = j
                best_c[idx] = c[idx]
    # if baseline > corrupted_baseline:
    #     best_results = [round(x, 2) for x in best_results]
    #     print(f"{i}-th sample; model performance: {baseline:.2f}; corrupted: {corrupted_baseline:.2f}; prob results: {best_c}; faithfulness: {best_results}")
    #     print('clean: ', clean)
    all_best_results.append(best_results)
    # all_best_complement_results.append(best_complement_results)

    # averaging
    model.reset_hooks()
    g = Graph.from_model(model)

    g.scores = torch.from_numpy(para_data.mean(0))

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        try: 
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        except:
            faithfulness = 0
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)

    all_avg_results.append(circuit_faithfulness)
    
    # BoN with Constraint-adaptive Score Matrix (BoN-CSM)
    score_mat = np.full_like(para_data[j], 0, dtype=float)
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
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        try:
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        except:
            faithfulness = 0
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)

    all_csm_results.append(circuit_faithfulness)

    # interpolated BoN (iBoN)
    new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]
    # new_topns = sorted(new_topns+topns)
    circuit_faithfulness = []
    for k, topn in enumerate(new_topns):
        model.reset_hooks()
        g = Graph.from_model(model)
        
        score_mat = np.full_like(para_data[j], 0, dtype=float)
        tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
        filled    = np.zeros_like(score_mat, dtype=bool)
            
        idx = bisect.bisect_left(topns, topn)
        if topn not in topns:
            prev_idx, previous_topn = idx - 1, topns[idx - 1]
            next_idx, next_topn = idx, topns[idx]

            for l, top in zip([prev_idx, next_idx], [previous_topn, next_topn]):
                best_para_idx = best_para_indices[l]
                M = np.abs(para_data[best_para_idx])
                M = np.where(np.isfinite(M), M, -np.inf)
                for (a, b) in topn_indices(M, top):
                    if not filled[a, b]:
                        score_mat[a, b] = M[a, b]
                        tier_mat[a, b]  = l
                        filled[a, b]    = True
        else:
            matched_idx = idx
            best_para_idx = best_para_indices[matched_idx]
            M = np.abs(para_data[best_para_idx])
            M = np.where(np.isfinite(M), M, -np.inf)
            for (a, b) in topn_indices(M, topn):
                if not filled[a, b]:
                    score_mat[a, b] = M[a, b]
                    tier_mat[a, b]  = matched_idx
                    filled[a, b]    = True

        g.scores = torch.from_numpy(score_mat)
        g.apply_topn_by_tier(topn, tier_mat)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        try:
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        except:
            faithfulness = 0
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)
    all_ibon_results.append(circuit_faithfulness)
    

topns = [x // 1000 for x in topns]  # Convert to 'k'
new_topns = [x // 1000 for x in new_topns]  # Convert to 'k'

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
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/mmlu_{category}_llama3-8b_2.pdf')