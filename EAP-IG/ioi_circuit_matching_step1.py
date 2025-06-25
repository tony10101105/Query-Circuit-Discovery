"""
this script loads the greater-than circuit manually identified by its original paper
"""

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
import matplotlib.pyplot as plt
# from svcca import cca_core
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, frac):
        self.df = pd.read_csv(filepath)
        self.df = self.df.sample(frac=frac, random_state=2025).reset_index(drop=True)

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


ana_only_att = True # if True, filter out mlp nodes.
data_frac = 1 # for fast experiment.
eval_baseline = False # evaluate baseline complete-circuit model
n_cca_component = 2

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('ioi_gpt2.csv', data_frac)
dataloader = ds.to_dataloader(batch_size=10)

g1 = Graph.from_json('ioi_gpt2_canonical_circuit.json')
g2 = Graph.from_json('ioi_gpt2_canonical_circuit.json')
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
results_g1, all_node_rep_g1, all_layer_rep_g1 = evaluate_graph(model, g1, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True)
results_g1 = results_g1.mean().item()

results_g2, all_node_rep_g2, all_layer_rep_g2 = evaluate_graph(model, g2, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True)
results_g2 = results_g2.mean().item()

# ### for layer-wise analysis
# all_node_rep_g1 = all_layer_rep_g1
# all_node_rep_g2 = all_layer_rep_g2
# g1_node_idx = [i for i in range(12)]
# g2_node_idx = [i for i in range(12)]
# g1_node_name = [str(i) for i in range(12)]
# g2_node_name = [str(i) for i in range(12)]
# ###

if eval_baseline:
    print('evaluating baseline...')
    baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
    print(f"Original performance: {baseline}, g1: {results_g1}, g2: {results_g2}")

# below assume same-model analysis, i.e., same token space
n_forward_g1, d_model_g1 = all_node_rep_g1[0].size(2), all_node_rep_g1[0].size(3)
n_forward_g2, d_model_g2 = all_node_rep_g2[0].size(2), all_node_rep_g2[0].size(3)

# merge token dim and d_model dim
all_node_rep_g1 = [i.transpose(1, 2) for i in all_node_rep_g1]
all_node_rep_g2 = [i.transpose(1, 2) for i in all_node_rep_g2]

print('all_node_rep_g1: ', all_node_rep_g1[0].shape) # (bs, node_num_g1, n_pos, 768)
print('all_node_rep_g2: ', all_node_rep_g2[0].shape)

# TODO: some are 14592 (19 tokens), but some are 15360 and 16128. We should make them flexible in the future
all_node_rep_g1 = [i.reshape(i.size(0), n_forward_g1, -1)[:, g1_node_idx, :14592].cpu().numpy() for i in all_node_rep_g1]
all_node_rep_g2 = [i.reshape(i.size(0), n_forward_g2, -1)[:, g2_node_idx, :14592].cpu().numpy() for i in all_node_rep_g2]
print('all_node_rep_g1: ', all_node_rep_g1[0].shape) # (bs, node_num_g1, n_pos, 768)
print('all_node_rep_g2: ', all_node_rep_g2[0].shape)

all_node_rep_g1 = np.vstack(all_node_rep_g1).transpose(1, 0, 2)
all_node_rep_g2 = np.vstack(all_node_rep_g2).transpose(1, 0, 2)
print('all_node_rep_g1: ', all_node_rep_g1.shape) # (bs, node_num_g1, n_pos, 768)
print('all_node_rep_g2: ', all_node_rep_g2.shape)

node_num_g1, node_num_g2 = all_node_rep_g1.shape[0], all_node_rep_g2.shape[0]

# pca for reduce dimension
x_pca = []
for i in tqdm(range(node_num_g1)):
    pca = PCA(n_components=300, random_state=2025)
    reduced = pca.fit_transform(all_node_rep_g1[i])  # Shape: (499, 300)
    x_pca.append(reduced)
all_node_rep_g1 = np.stack(x_pca)  # Shape: (27, 499, 300)

x_pca = []
for i in tqdm(range(node_num_g2)):
    pca = PCA(n_components=300, random_state=2025)
    reduced = pca.fit_transform(all_node_rep_g2[i])  # Shape: (499, 300)
    x_pca.append(reduced)
all_node_rep_g2 = np.stack(x_pca)  # Shape: (27, 499, 300)


print('start calculating pairwise distance')
cca_distances = np.zeros((node_num_g1, node_num_g2))
for i in tqdm(range(node_num_g1), desc='Outer'):
    X = all_node_rep_g1[i, :, :]
    for j in tqdm(range(node_num_g2), desc='Inner', leave=False):
        Y = all_node_rep_g2[j, :, :]

        print('X shape: ', X.shape)
        print('Y shape: ', Y.shape)
        exit(0)
        cca = CCA(n_components=n_cca_component, max_iter=500, scale=True)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)

        # corr = [np.corrcoef(X_c[:, k].T, Y_c[:, k].T)[0, 1] for k in range(n_cca_component)]
        # cca_distances[i, j] = 1 - sum(corr)/len(corr)
        corr = np.corrcoef(X_c[:, 0].T, Y_c[:, 0].T)[0, 1]
        cca_distances[i, j] = 1 - corr
        print('ori corr: ', corr)

        # cca = CCA(n_components=n_cca_component, max_iter=500, scale=True)
        # cca.fit(Y, X)
        # X_c, Y_c = cca.transform(X, Y)
        # corr = np.corrcoef(X_c[:, 0].T, Y_c[:, 0].T)[0, 1]
        # cca_distances[i, j] = 1 - corr

        # print('new corr: ', corr)
        
        if i == j: # exactly the same features. Should have ~1 corr
            assert corr > 0.95, print(corr)

np.savez("cost_matrix_all_atthead(mlp_pruned).npz", 
        cca_distances=cca_distances, 
        g1_node_name=g1_node_name, 
        g2_node_name=g2_node_name,
        g1_node_idx=g1_node_idx,
        g2_node_idx=g2_node_idx)

# now you get pairwise node distance matrix, which can serve as the cost matrix for bipartite matching
# we split the graph matching to ioi_circuit_matching_step2.py