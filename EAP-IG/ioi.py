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

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 

# %%
"""
## Dataset and Metrics

This package expects data to come from a dataloader. Each item consists of clean and corrupted paired inputs (strings), as well as a label (encoded as a token id). For convenience, we've included a dataset in that form as a CSV (more to come with the full code of the paper).

A metric takes in the model's (possibly corrupted) logits, clean logits, input lengths, and labels. It computes a metric value for each batch item; this can either be used as is, or turned into a loss (lower is better), or meaned.
"""

# %%
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

# %%
"""
## Performing EAP-IG

First, we load the model, data, and metric.
"""

# %%
model_name = 'meta-llama/Meta-Llama-3-8B'
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

ds = EAPDataset('ioi_llama.csv')
dataloader = ds.to_dataloader(10)

# %%
"""
Then, we perform EAP! We instantiate an unscored graph from the model, and use the attribute method to score it. This requires a model, graph, dataloader, and loss. We set `method='EAP-IG'`, and set the number of iterations via `ig_steps`.
"""

# %%
# Instantiate a graph with a model
g = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)

# %%
"""
We can now apply greedy search to the scored graph to find a circuit! We prune dead nodes, and export the circuit.
"""

# %%
g.apply_topn(20000, True)
g.to_pt('ioi_graph.pt')

# %%
"""
We then evaluate our model's metric score as opposed to a baseline.
"""

# %%
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results}")

exit(0)

# %%
"""
We can now compare that to a circuit found with vanilla EAP.
"""

# %%
# Instantiate a graph with a model
g_eap = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_eap, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP')

g_eap.apply_topn(20000, True)

results_eap = evaluate_graph(model, g_eap, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_eap}")

# %%
"""
We can also test other EAP-IG variants:
"""

# %%
# Instantiate a graph with a model
g_cc = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_cc, dataloader, partial(logit_diff, loss=True, mean=True), method='clean-corrupted')

g_cc.apply_topn(20000, True)

results_cc = evaluate_graph(model, g_cc, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_cc}")