from functools import partial

import pygraphviz
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline

import ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA

from similarity_calculation import get_representation_similarity


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = list(labels)
    # labels = torch.tensor(labels)
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
        return ast.literal_eval(row['clean']), ast.literal_eval( row['corrupted']), [int(row['label']), ast.literal_eval(row['incorrect_label'])]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    incorrect_label = [j for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample
    # print('after logits: ', logits.shape) # torch.Size([bs, 50257])
    
    # print(correct_label.shape) # torch.Size([bs])
    good = torch.gather(logits, -1, correct_label).squeeze(1)
    bad = []
    for i, sample_label in enumerate(incorrect_label):
        sample_label = torch.tensor(sample_label, device=logits.device)
        # print('sample_label: ', sample_label.shape) # torch.Size([n])
        sample_bad = torch.gather(logits[i], -1, sample_label)
        # print('sample_bad: ', sample_bad.shape)
        bad.append(sample_bad.mean())
    bad = torch.tensor(bad, device=logits.device)
    # print('bad: ', bad.shape) # torch.Size([bs])
    # good_bad = torch.gather(logits, -1, labels.to(logits.device))
    # print('good: ', good.shape)
    # results = good_bad[:, 0] - good_bad[:, 1]
    results = good - bad
    # print('results: ', results.shape) # torch.Size([bs])

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    incorrect_label = [j for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample
    probs = torch.softmax(logits, dim=-1)
    
    # print(correct_label.shape) # torch.Size([bs])
    good = torch.gather(probs, -1, correct_label).squeeze(1)
    bad = []
    for i, sample_label in enumerate(incorrect_label):
        sample_label = torch.tensor(sample_label, device=logits.device)
        sample_bad = torch.gather(probs[i], -1, sample_label)
        bad.append(sample_bad.mean())
    bad = torch.tensor(bad, device=logits.device)

    results = good - bad

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def log_prob(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    good = torch.gather(log_probs, -1, correct_label).squeeze(1)
    results = good

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def prob(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)

    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample

    probs = torch.softmax(logits, dim=-1)

    good = torch.gather(probs, -1, correct_label).squeeze(1)
    results = good

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


ana_only_att = True # if True, filter out mlp nodes.
eval_baseline = False # evaluate baseline complete-circuit model
similarity_method = ['cca', 'svcca', 'pwcca', 'linear cka', 'rbf cka']

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/induction_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=10)

# g = Graph.from_model(model)
g1 = Graph.from_json('preprocessed_circuit/gpt2.json')
g2 = Graph.from_json('preprocessed_circuit/gpt2.json')

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
results_g1, all_node_rep_g1, _, _ = evaluate_graph(model, g1, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, induction=True)
results_g1 = results_g1.mean().item()

# results_g2, all_node_rep_g2, _, _ = evaluate_graph(model, g2, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, induction=True)
# results_g2 = results_g2.mean().item()
results_g2 = results_g1

if eval_baseline:
    print('evaluating baseline...')
    baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), induction=True).mean().item()
    print(f"Original performance: {baseline}, g1: {results_g1}, g2: {results_g2}")

# below assume same-model analysis, i.e., same token space

all_node_rep_g1 = all_node_rep_g1.permute(2, 0, 1, 3).cpu().numpy() # (node_num_g1, data_num, n_pos, 768)
# all_node_rep_g2 = all_node_rep_g2.permute(2, 0, 1, 3).cpu().numpy()
all_node_rep_g2 = all_node_rep_g1
torch.cuda.empty_cache()

print('all_node_rep_g1: ', all_node_rep_g1.shape) # (157, 500, 40, 768)
print('all_node_rep_g2: ', all_node_rep_g2.shape)

# ### for truncation
# seq_len = 20
# all_node_rep_g1 = all_node_rep_g1[:, :, :seq_len, :]
# all_node_rep_g2 = all_node_rep_g2[:, :, :seq_len, :]

node_num_g1, bs_g1, n_pos_g1, d_model_g1 = all_node_rep_g1.shape
node_num_g2, bs_g2, n_pos_g2, d_model_g2 = all_node_rep_g2.shape

all_node_rep_g1 = all_node_rep_g1.reshape(node_num_g1, bs_g1, n_pos_g1 * d_model_g1)
all_node_rep_g2 = all_node_rep_g2.reshape(node_num_g2, bs_g2, n_pos_g2 * d_model_g2)

# pca for reduce dimension
pca_g1 = []
for i in tqdm(range(node_num_g1)):
    pca = PCA(n_components=300, random_state=2025)
    reduced = pca.fit_transform(all_node_rep_g1[i])  # Shape: (data_num, 300)
    pca_g1.append(reduced)
all_node_rep_g1 = np.stack(pca_g1)  # Shape: (node_num_g1, data_num, 300)

pca_g2 = []
for i in tqdm(range(node_num_g2)):
    pca = PCA(n_components=300, random_state=2025)
    reduced = pca.fit_transform(all_node_rep_g2[i])  # Shape: (data_num, 300)
    pca_g2.append(reduced)
all_node_rep_g2 = np.stack(pca_g2)  # Shape: (node_num_g2, data_num, 300)

print('start calculating pairwise distance')
feature_distances = np.empty((node_num_g1, node_num_g2), dtype=object)
for i in tqdm(range(node_num_g1), desc='Outer'):
    X = all_node_rep_g1[i]
    for j in tqdm(range(node_num_g2), desc='Inner', leave=False):
        Y = all_node_rep_g2[j]

        # print('X shape: ', X.shape) # (sample, feature)
        # print('Y shape: ', Y.shape) # (sample, feature)

        results = get_representation_similarity(X.T, Y.T, methods=similarity_method)
        feature_distances[i, j] = results


np.savez("preprocessed_data/induction_cost_matrix_all_atthead_rep.npz", 
        feature_distances=feature_distances, 
        g1_node_name=g1_node_name, 
        g2_node_name=g2_node_name,
        g1_node_idx=g1_node_idx,
        g2_node_idx=g2_node_idx)
