# %%
"""
# Finding the Greater-Than Circuit Using EAP(-IG)

First, we import various packages.
"""

# %%
from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 
from src.eap.utils import topn_indices, set_seed

import os
import time

os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

set_seed(2025)

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, num_samples=None):
        self.df = pd.read_csv(filepath)
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        print(f'Loaded {len(self.df)} samples from {filepath}')


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


"""
## Performing EAP-IG

First, we load the model, data, and metric.
"""

num_samples = 20

# model_name = 'meta-llama/Llama-3.2-3B-Instruct'
model_name = 'gpt2'
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
    dtype=torch.float16
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', num_samples=num_samples)
dataloader = ds.to_dataloader(batch_size=1)


# eap
# Instantiate a graph with a model
g_eap = Graph.from_model(model)

start = time.time()
# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_eap, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP')
end = time.time()
elapsed = end - start
avg_elapsed = elapsed / num_samples
print(f"Elapsed time: {elapsed:.1f} seconds")
print(f"Average time per sample: {avg_elapsed:.1f} seconds")
exit(0)

g_eap.apply_topn(20000, True)

results_eap = evaluate_graph(model, g_eap, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_eap}")


# eap-ig
# Instantiate a graph with a model
g = Graph.from_model(model)

start = time.time()
# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)
end = time.time()
elapsed = end - start
avg_elapsed = elapsed / num_samples
print(f"Elapsed time: {elapsed:.1f} seconds")
print(f"Average time per sample: {avg_elapsed:.1f} seconds")
exit(0)


g.apply_topn(20000, True)
g.to_pt('ioi_graph.pt')


baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results}")



"""
We can also test other EAP-IG variants:
"""

# Instantiate a graph with a model
g_cc = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_cc, dataloader, partial(logit_diff, loss=True, mean=True), method='clean-corrupted')

g_cc.apply_topn(20000, True)

results_cc = evaluate_graph(model, g_cc, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_cc}")