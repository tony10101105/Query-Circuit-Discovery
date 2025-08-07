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

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

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
