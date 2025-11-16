from functools import partial

import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 
import sys
sys.setrecursionlimit(1000000)  # default ~1000
def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = self.df.head(100)

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


# topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
topns = [300000] # 100000
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/ioi_llama32.csv')
dataloader = ds.to_dataloader(batch_size=1)

g = Graph.from_model(model)

print('evaluating baseline...')
# baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
# corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean().item()
# print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

print('attributing...')
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention)
print('evaluating circuit...')
circuit_results = []
circuit_faithfulness_g, circuit_faithfulness_t = [], []
times = []
for topn in tqdm(topns):
    start = time.perf_counter()
    g.apply_topn(topn, True)
    # g.apply_greedy(topn, True)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    continue
    g.prune()
    # print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

    # g.to_json(f'ioi_{model_name}_{topn}_circuit.json')
    # gz = g.to_image(f'ioi_{model_name}_{topn}_circuit.png')

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
print('times: ', sum(times)/len(times))
exit(0)
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
plt.savefig(f'ioi_replication.pdf', bbox_inches='tight')
