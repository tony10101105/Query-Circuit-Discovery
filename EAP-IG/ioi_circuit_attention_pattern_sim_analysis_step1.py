from functools import partial

import pygraphviz
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline

import numpy as np
from tqdm import tqdm

from similarity_calculation import get_js_div


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
        self.df = self.df.data_num(frac=1)

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


ana_only_att = True # if True, filter out mlp nodes.
eval_baseline = False # evaluate baseline complete-circuit model

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=10)

g1 = Graph.from_json('preprocessed_circuit/ioi_gpt2_canonical_circuit.json')
g2 = Graph.from_json('preprocessed_circuit/ioi_gpt2_canonical_circuit.json')
# g2 = Graph.from_json('ioi_gpt2-small_500_circuit.json')
# print(f'g1 node, edge number: {g1.count_included_nodes()}, {g1.count_included_edges()}')
# print(f'g2 node, edge number: {g2.count_included_nodes()}, {g2.count_included_edges()}')
# for i in range(1, 13):
#     g1.in_graph[i*13, :] = False
#     g2.in_graph[i*13, :] = False
# g1.prune()
# g2.prune()
print(f'g1 node, edge number: {g1.count_included_nodes()}, {g1.count_included_edges()}')
print(f'g2 node, edge number: {g2.count_included_nodes()}, {g2.count_included_edges()}')

# load all nodes' names
g1_node_name = list(g1.nodes.keys())[:-1] # remove the logits node
g2_node_name = list(g2.nodes.keys())[:-1]

# indices of nodes that are in the graph
g1_node_idx = g1.nodes_in_graph.nonzero(as_tuple=True)[0].tolist()
g2_node_idx = g2.nodes_in_graph.nonzero(as_tuple=True)[0].tolist()

### for all-node plot
g1_node_idx = [i for i in range(0, len(g1.nodes_in_graph), 1)]
g2_node_idx = [i for i in range(0, len(g2.nodes_in_graph), 1)]
###

if ana_only_att:
    # indices of att nodes
    g1_node_idx = [i for i in g1_node_idx if 'a' in g1_node_name[i]]
    g2_node_idx = [i for i in g2_node_idx if 'a' in g2_node_name[i]]
    # only keep the names of these nodes
    g1_node_name = [g1_node_name[i] for i in g1_node_idx]
    g2_node_name = [g2_node_name[i] for i in g2_node_idx]


print('evaluating circuit...')
results_g1, _, _, all_node_pattern_g1 = evaluate_graph(model, g1, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True)
results_g1 = results_g1.mean().item()
results_g2, _, _, all_node_pattern_g2 = evaluate_graph(model, g2, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True)
results_g2 = results_g2.mean().item()

if eval_baseline:
    print('evaluating baseline...')
    baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
    print(f"Original performance: {baseline}, g1: {results_g1}, g2: {results_g2}")

# below assume same-model analysis, i.e., same token space
all_node_pattern_g1 = all_node_pattern_g1.transpose(0, 1).cpu()  # (node_num_g1, data_num, n_pos, n_pos)
all_node_pattern_g2 = all_node_pattern_g2.transpose(0, 1).cpu()  # (node_num_g2, data_num, n_pos, n_pos)
node_num_g1 = all_node_pattern_g1.shape[0]
node_num_g2 = all_node_pattern_g2.shape[0]

print('start calculating pairwise distance')
feature_distances = np.empty((node_num_g1, node_num_g2), dtype=object)
for i in tqdm(g1_node_idx, desc='Outer'):
    X = all_node_pattern_g1[i]
    for j in tqdm(g2_node_idx, desc='Inner', leave=False):
        Y = all_node_pattern_g2[j]

        # print('X shape: ', X.shape) # (data_num, n_pos, n_pos)
        # print('Y shape: ', Y.shape) # (data_num, n_pos, n_pos)

        avg_js_div = get_js_div(X, Y)
        # print('avg_js_div: ', avg_js_div)
        feature_distances[i, j] = avg_js_div


np.savez("preprocessed_data/ioi_cost_matrix_all_atthead_pattern.npz", 
        feature_distances=feature_distances, 
        g1_node_name=g1_node_name, 
        g2_node_name=g2_node_name,
        g1_node_idx=g1_node_idx,
        g2_node_idx=g2_node_idx)