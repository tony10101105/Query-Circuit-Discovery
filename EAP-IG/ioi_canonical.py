from functools import partial

import json
import matplotlib.pyplot as plt
import pygraphviz
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 
from utils import get_logit_positions, logit_diff, EAPDataset
topns = [50, 100, 250, 500, 1000, 1500, 2000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
# print(model.cfg)
# exit(0)
ds = EAPDataset('probing_dataset/ioi_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=10)

g = Graph.from_model(model)

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean().item()
print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
print('attributing...')
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=15, intervention=intervention)

print('evaluating circuit...')
circuit_results = []
circuit_faithfulness = []
for topn in topns:
    g.apply_topn(topn, True)
    g.prune()
    print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

    # g.to_json(f'ioi_{model_name}_{topn}_circuit.json')
    # gz = g.to_image(f'ioi_{model_name}_{topn}_circuit.png')

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)

    results = results.mean().item()
    circuit_results.append(results)
    
    faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
    circuit_faithfulness.append(faithfulness)
    
    print(f"Original performance was {baseline}; the circuit's performance is {results}; faithfulness is {faithfulness}")

all_results = {
    'baseline': baseline,
    'corrupted_baseline': corrupted_baseline,
    'topns': topns,
    'circuit_results': circuit_results,
    'circuit_faithfulness': circuit_faithfulness
}
with open(f'ioi_{method.lower()}_15steps_all_sample_data.json', 'w') as f:
    json.dump(all_results, f, indent=2)
