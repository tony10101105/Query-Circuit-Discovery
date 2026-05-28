from functools import partial

import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 
from utils import get_logit_positions, logit_diff, EAPDataset
topns = [50, 100, 200, 300, 400, 500, 750, 1000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/gender_bias_gpt2.csv', correct_col='clean_answer_idx', incorrect_col='corrupted_answer_idx')
dataloader = ds.to_dataloader(batch_size=10)

g = Graph.from_model(model)

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean().item()
print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
print('attributing...')
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention)
print('evaluating circuit...')
circuit_results = []
circuit_faithfulness_g, circuit_faithfulness_t = [], []
for topn in tqdm(topns):
    g.apply_topn(topn, True)
    g.prune()
    # print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)

    results = results.mean().item()
    circuit_results.append(results)
    
    faithfulness_t = (results - corrupted_baseline) / (baseline - corrupted_baseline)
    # faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
    circuit_faithfulness_t.append(round(faithfulness_t, 2))

    g.apply_greedy(topn, True)
    g.prune()

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)

    results = results.mean().item()
    circuit_results.append(results)
    
    faithfulness_g = (results - corrupted_baseline) / (baseline - corrupted_baseline)
    circuit_faithfulness_g.append(round(faithfulness_g, 2))

    # print(f"Original performance: {baseline:.2f}; circuit performance: {results:.2f}; corrupted_baseline: {corrupted_baseline:.2f}; faithfulness: {faithfulness:.2f}")

# all_results = {
#     'baseline': baseline,
#     'corrupted_baseline': corrupted_baseline,
#     'topns': topns,
#     'circuit_results': circuit_results,
#     'circuit_faithfulness': circuit_faithfulness
# }

print('topns: ', topns)
print('circuit_faithfulness_t: ', circuit_faithfulness_t)
print('circuit_faithfulness_g: ', circuit_faithfulness_g)
plt.plot(topns, circuit_faithfulness_t, label=f'Greedy Selection', marker='o')
plt.plot(topns, circuit_faithfulness_g, label=f'Dijkstra-like Construction', marker='o')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=16)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, loc='lower right')
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'gender_bias_replication.pdf', bbox_inches='tight')