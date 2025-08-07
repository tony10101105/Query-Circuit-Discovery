"""
this script loads the greater-than circuit manually identified by its original paper
"""

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
from tqdm import tqdm
import seaborn as sns

from preprocessed_circuit.gt_circuit_constant import GT_CIRCUIT, head_2_layer, GT_CIRCUIT_GROUP_A, GT_CIRCUIT_GROUP_B, GROUP_A_IDX_2_CLASS, GROUP_B_IDX_2_CLASS


metric = 'Linear CKA' # Linear CKA, RBF CKA, PWCCA
data = np.load('preprocessed_data/gt_cost_matrix_all_atthead_rep.npz', allow_pickle=True)
g1_node_name = data['g1_node_name'].tolist() # 144 attention head names
g2_node_name = data['g2_node_name'].tolist()
assert g1_node_name == g2_node_name
g1_node_idx = data['g1_node_idx'].tolist() # 144 attention head idx
g2_node_idx = data['g2_node_idx'].tolist()
g1_mlp_idx = [i for i in list(range(0, 157)) if i not in g1_node_idx and i != 0]
g2_mlp_idx = [i for i in list(range(0, 157)) if i not in g2_node_idx and i != 0]
assert g1_node_idx == g2_node_idx

feature_distances = np.array([[d[metric.lower()] for d in row] for row in data['feature_distances']])
att_feature_distances = feature_distances[np.ix_(g1_node_idx, g2_node_idx)]
mlp_feature_distances = feature_distances[np.ix_(g1_mlp_idx, g2_mlp_idx)]
print('feature_distances shape: ', att_feature_distances.shape)
data = np.load('preprocessed_data/gt_cost_matrix_all_atthead_pattern.npz', allow_pickle=True)
pattern_distances = np.array(data['feature_distances'][np.ix_(data['g1_node_idx'], data['g2_node_idx'])], dtype=np.float32)
print('pattern_distances shape: ', pattern_distances.shape)
    
L = 12
N = 144
# lambda_decay = np.linspace(0, 5, 101)
lambda_decay = [0.5]

best_alpha = None
best_acc = -float('inf')
alpha_values = np.linspace(0, 1, 11)


### attention head matching

# for alpha in tqdm(alpha_values):
#     beta = 1 - alpha
for lamb in tqdm(lambda_decay):
    adaptive_alphas = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            l_i = i // 12
            l_j = j // 12
            avg_layer = (l_i + l_j) / 2
            # alpha = (1 - avg_layer / (L - 1))**3
            alpha = np.exp(-lamb * avg_layer)
            adaptive_alphas[i, j] = np.clip(alpha, 0, 1)

    # node_distances = alpha * att_feature_distances + beta * pattern_distances
    node_distances = adaptive_alphas * att_feature_distances + (1 - adaptive_alphas) * pattern_distances

    group_a_idx_tup = list(GROUP_A_IDX_2_CLASS.keys())
    group_a_idx_tup = [i for i in group_a_idx_tup if i[1] is not None]
    group_b_idx_tup = list(GROUP_B_IDX_2_CLASS.keys())
    group_b_idx_tup = [i for i in group_b_idx_tup if i[1] is not None]
    group_a_idx = [i[0]*12 + i[1] for i in group_a_idx_tup]
    group_b_idx = [i[0]*12 + i[1] for i in group_b_idx_tup]

    node_distances_a_b = node_distances[np.ix_(group_b_idx, group_a_idx)]

    row_max_indices = np.argmin(node_distances_a_b, axis=1)

    meta_project = {'01': 'readin', '0305': 'readin', 'MEARLY': 'readin', 'AMID': 'spike', 'MLATE': 'boost'}
    cnt = 0
    for i, idx in enumerate(row_max_indices):
        b_node_cls = GROUP_B_IDX_2_CLASS[group_b_idx_tup[i]]
        a_node_cls = GROUP_A_IDX_2_CLASS[group_a_idx_tup[idx]]
        # print(f'i: {i}, gt: {b_node_cls}, pred: {a_node_cls}')
        b_node_cls = meta_project[b_node_cls]
        a_node_cls = meta_project[a_node_cls]
        if b_node_cls == a_node_cls:
            cnt += 1
        else:
            print(f'i: {i}, gt: {b_node_cls}, pred: {a_node_cls}')

    acc = cnt / len(row_max_indices)
    print(f'lambda: {lamb}, acc: {acc}')
    # print(f'alpha: {alpha}, acc: {acc}')
    if acc > best_acc:
        best_acc = acc
        best_l = lamb
        # best_alpha = alpha

print(f'best_acc: {best_acc}')
print(f'best_lambda: {best_l}')
# print(f'best_alpha: {best_alpha}')


### MLP matching
node_distances = mlp_feature_distances # (12, 12)

group_a_idx_tup = list(GROUP_A_IDX_2_CLASS.keys())
group_a_idx_tup = [i for i in group_a_idx_tup if i[1] is None]
group_b_idx_tup = list(GROUP_B_IDX_2_CLASS.keys())
group_b_idx_tup = [i for i in group_b_idx_tup if i[1] is None]
group_a_idx = [i[0] for i in group_a_idx_tup]
group_b_idx = [i[0] for i in group_b_idx_tup]

node_distances_a_b = node_distances[np.ix_(group_b_idx, group_a_idx)]

row_max_indices = np.argmin(node_distances_a_b, axis=1)

cnt = 0
for i, idx in enumerate(row_max_indices):
    b_node_cls = GROUP_B_IDX_2_CLASS[group_b_idx_tup[i]]
    a_node_cls = GROUP_A_IDX_2_CLASS[group_a_idx_tup[idx]]
    print(f'i: {i}, gt: {b_node_cls}, pred: {a_node_cls}')
    if b_node_cls == a_node_cls:
        cnt += 1
        
acc = cnt / len(row_max_indices)
print(f'mlp acc: {acc}')