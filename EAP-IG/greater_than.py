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


# %%
"""
## Performing EAP-IG

First, we load the model, data, and metric.
"""

# %%
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('greater_than_data.csv')
dataloader = ds.to_dataloader(120)
prob_diff = get_prob_diff(model.tokenizer)

# %%
"""
Then, we perform EAP! We instantiate an unscored graph from the model, and use the attribute method to score it. This requires a model, graph, dataloader, and loss. We set `method='EAP-IG'`, and set the number of iterations via `ig_steps`.
"""

# %%
# Instantiate a graph with a model
g = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g, dataloader, partial(prob_diff, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)

# %%
"""
We can now apply greedy search to the scored graph to find a circuit! We prune dead nodes, and export the circuit.
"""

# %%
g.apply_topn(200, True)
g.to_json('graph.json')

# %%
"""
We can then convert our circuit into a visualization!
"""

# %%
try:
    import pygraphviz
    gz = g.to_image(f'graph.png')
except ImportError:
    print("No pygraphviz installed; skipping this part")

# %%
"""
We then evaluate our model's metric score as opposed to a baseline.
"""

# %%
baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
results = evaluate_graph(model, g, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results}")

# %%
g.count_included_nodes(), g.count_included_edges()

# %%
"""
We can now compare that to a circuit found with vanilla EAP.
"""

# %%
# Instantiate a graph with a model
g_eap = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_eap, dataloader, partial(prob_diff, loss=True, mean=True), method='EAP')

g_eap.apply_topn(200, True)

results_eap = evaluate_graph(model, g_eap, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_eap}")

# %%
"""
We can also test other EAP-IG variants:
"""

# %%
# Instantiate a graph with a model
g_cc = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_cc, dataloader, partial(prob_diff, loss=True, mean=True), method='clean-corrupted')

g_cc.apply_topn(200, True)

results_cc = evaluate_graph(model, g_cc, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_cc}")