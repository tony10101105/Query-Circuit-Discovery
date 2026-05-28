import os as _os, sys as _sys
_base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_base)
_sys.path.insert(0, _base)

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
from utils import get_logit_positions, logit_diff, EAPDataset

shots = "Which is the most possible answer?\n"


set_seed(2025)
topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
category = 'marketing'
metric_version = 4 # 1,2,3,4,5,6,7,8,9
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations # EAP
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset(f'probing_dataset/mmlu_{category}_Llama-32-1B.csv', num_samples=300, mc=True)
# ds = EAPDataset(f'probing_dataset/test_paraphrase.csv', num_samples=300)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    if i not in [11, 41, 49]:
        continue
    single_data = [(clean, corrupted, label)]

    model.reset_hooks()
    
    g = Graph.from_model(model)

    # print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True).mean().item()
    
    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)
    
    print(f"{i}-th sample. Original: {baseline}; corrupted: {corrupted_baseline}")
    # print('attributing for this single data...')
    attribute(model, g, single_data, partial(logit_diff, mc=True, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention, quiet=True)
    # x = g.scores.cpu().detach().numpy()
    # x[~g.real_edge_mask] = -np.inf
    # np.save(f'Query-Circuit-Dataset/score_data/vanilla/mmlu_{category}_metric8_edge_scores_{i}.npy', x)

    # print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness = []
    # g.scores = torch.rand(g.scores.shape, device=g.scores.device)
    for topn in topns:
        g.apply_topn(topn, True)
        # g.apply_greedy(topn, True)

        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        circuit_results.append(results)

        faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        # faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)

        print(f"{i}-th sample. Original performance: {baseline:.2f}; circuit performance: {results:.2f}; corrupted_baseline: {corrupted_baseline:.2f}; faithfulness: {faithfulness:.2f}")

    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'topns': topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })

topns = [x // 1000 for x in topns]  # Convert to 'k'
for i, results in zip(range(len([11, 44, 49])), all_results):
    plt.plot(topns, results['circuit_faithfulness'], label=f'Query {i+1}', marker='o')
plt.ylim(-2.1, 2.1)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1, linestyle='--', color='gray')
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/mmlu_marketing_metric{metric_version}_instability_11_44_49.pdf', bbox_inches='tight')