"""
# Finding the Greater-Than Circuit Using EAP(-IG)

First, we import various packages.
"""

from functools import partial

import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        return row['clean'], row['corrupted'], row['label']
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def get_prob_diff(tokenizer: PreTrainedTokenizer):
    year_indices = torch.tensor([tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])

    def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)[:, year_indices]

        results = []
        for prob, year in zip(probs, labels):
            results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
    
        results = torch.stack(results)
        if loss:
            results = -results
        if mean: 
            results = results.mean()
        return results
    return prob_diff

def kl_div(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results


topns = [50, 100, 200, 300, 400, 500, 750, 1000] # 32491
# topns = [4000, 10000, 15000, 20000, 25000, 30000, 32000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/greater_than_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=10)
prob_diff = get_prob_diff(model.tokenizer)

g = Graph.from_model(model)

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False), run_corrupted=True).mean().item()

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
print('attributing...')
attribute(model, g, dataloader, partial(prob_diff, loss=True, mean=True), method=method, ig_steps=5, intervention=intervention)

print('evaluating circuit...')
circuit_results = []
circuit_faithfulness_g, circuit_faithfulness_t = [], []
for topn in tqdm(topns):
    g.apply_topn(topn, True)
    g.prune()
    # print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

    # g.to_json(f'ioi_{model_name}_{topn}_circuit.json')
    # gz = g.to_image(f'ioi_{model_name}_{topn}_circuit.png')

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(prob_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)

    results = results.mean().item()
    circuit_results.append(results)
    
    faithfulness_t = (results - corrupted_baseline) / (baseline - corrupted_baseline)
    # faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
    circuit_faithfulness_t.append(round(faithfulness_t, 2))

    g.apply_greedy(topn, True)
    g.prune()

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(prob_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)

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
plt.savefig(f'gt_replication.pdf', bbox_inches='tight')

"""
# Instantiate a graph with a model
g_eap = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_eap, dataloader, partial(prob_diff, loss=True, mean=True), method='EAP')

g_eap.apply_topn(200, True)

results_eap, _, _, _ = evaluate_graph(model, g_eap, dataloader, partial(prob_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True)
results_eap = results_eap.mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_eap}")


# Instantiate a graph with a model
g_cc = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_cc, dataloader, partial(prob_diff, loss=True, mean=True), method='clean-corrupted')

g_cc.apply_topn(200, True)

results_cc, _, _, _ = evaluate_graph(model, g_cc, dataloader, partial(prob_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True)
results_cc = results_cc.mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_cc}")
"""