from functools import partial

import os
import random
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For TransformerLens:
    # Optional: sets the default generator for dropout, noise etc.
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
set_seed(2025)

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
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


data_num = 3
# topns = [50, 100, 250, 500, 1000, 1500, 2000] # 32491
topns = [500, 1000, 1500, 2000] # 32491
# topns = [10000, 20000, 30000, 32491] # 32491
method = 'EAP-IG-inputs-sg' # EAP-IG-inputs # EAP-IG-activations
steps = 20
perturb_times = 20 if method == 'EAP-IG-inputs-sg' else None
var = 0.01 if method == 'EAP-IG-inputs-sg' else None
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
total_f, total_steps = 0, 0
for clean, corrupted, label in tqdm(dataloader):
    single_data = [(clean, corrupted, label)]
    model.reset_hooks()
    
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    print('attributing for this single data...')
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention, quiet=True, perturb_times=perturb_times, var=var)

    print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        g.prune()
        
        gz = g.to_image('test.png')
        exit(0)
    
        # edge_mask = g.in_graph.cpu().numpy() # 1=has the edge
        # np.save(f'{method}_{topn}_connected_edges_mask.npy', edge_mask)
        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=intervention, quiet=True)
        results = results.mean().item()
        circuit_results.append(results)
        
        faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)

        print(f"Original performance: {baseline:.3f}; circuit performance: {results:.3f}; corrupted_baseline: {corrupted_baseline:.3f}; faithfulness: {faithfulness:.3f}")
        total_steps += 1
        total_f += faithfulness
        
    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'topns': topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })

print(f'avg faithfulness: {total_f / total_steps:.3f} over {total_steps} steps')

# with open(f'ioi_{method.lower()}_{steps}steps_one_sample_data.json', 'w') as f:
#     json.dump(all_results, f, indent=2)

method2 = 'eap-ig-inputs' # eap-ig-activations # eap-ig-inputs

one_sample_data1 = all_results

one_sample_filepath2 = f'ioi_{method2}_{steps}steps_one_sample_data.json'
with open(one_sample_filepath2, 'r') as file:
    one_sample_data2 = json.load(file)
one_sample_data2 = one_sample_data2[:data_num]
assert len(one_sample_data1) == len(one_sample_data2), "Data length mismatch between two methods."

one_sample_faithfulness1 = [d['circuit_faithfulness'] for d in one_sample_data1]
one_sample_faithfulness1 = np.mean(one_sample_faithfulness1, axis=0)
one_sample_faithfulness1 = one_sample_faithfulness1.tolist()

one_sample_faithfulness2 = [d['circuit_faithfulness'] for d in one_sample_data2]
one_sample_faithfulness2 = np.mean(one_sample_faithfulness2, axis=0)
one_sample_faithfulness2 = one_sample_faithfulness2.tolist()

assert one_sample_data1[0]['topns'] == one_sample_data2[0]['topns'], "Top-n values do not match between one-sample and all-sample experiments."

plt.plot(one_sample_data1[0]['topns'], one_sample_faithfulness1, label='One Sample IG-SG', marker='o')
plt.plot(one_sample_data2[0]['topns'], one_sample_faithfulness2, label='One Sample IG', marker='s')
plt.ylim(-0.1, 1.1)
plt.xlabel('Top-K Edges')
plt.ylabel('Circuit Faithfulness')
plt.title(f'IOI Circuit Faithfulness vs Top-K Edges')
plt.legend()

plt.savefig(f'ioi_{data_num}_{var}_one_sample_faithfulness.png', dpi=500)