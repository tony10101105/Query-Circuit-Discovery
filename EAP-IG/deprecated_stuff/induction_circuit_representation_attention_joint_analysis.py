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

from preprocessed_circuit.induction_circuit_constant import INDUCTION_CIRCUIT, head_2_layer, INDUCTION_CIRCUIT_GROUP_A, INDUCTION_CIRCUIT_GROUP_B, GROUP_A_IDX_2_CLASS, GROUP_B_IDX_2_CLASS


best_acc = -float('inf')
L = 12
N = 144
best_l = -1
lambda_decay = np.linspace(0, 5, 101)

best_alpha = -1
alpha_values = np.linspace(0, 1, 11)

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
            
    scale = False
    metric = 'Linear CKA' # Linear CKA, RBF CKA, PWCCA
    data = np.load('preprocessed_data/induction_cost_matrix_all_atthead_rep_truncated.npz', allow_pickle=True)
    g1_node_name = data['g1_node_name'].tolist() # 144 attention head names
    g2_node_name = data['g2_node_name'].tolist()
    assert g1_node_name == g2_node_name
    g1_node_idx = data['g1_node_idx'].tolist()
    g2_node_idx = data['g2_node_idx'].tolist()
    assert g1_node_idx == g2_node_idx

    feature_distances = np.array([[d[metric.lower()] for d in row] for row in data['feature_distances']])
    feature_distances = feature_distances[np.ix_(g1_node_idx, g2_node_idx)]
    # print('feature_distances shape: ', feature_distances.shape)
    data = np.load('preprocessed_data/induction_cost_matrix_all_atthead_pattern_truncated.npz', allow_pickle=True)
    pattern_distances = np.array(data['feature_distances'][np.ix_(data['g1_node_idx'], data['g2_node_idx'])], dtype=np.float32)
    # print('pattern_distances shape: ', pattern_distances.shape)

    if scale:
        feature_distances = (feature_distances - feature_distances.min()) / (feature_distances.max() - feature_distances.min())
        pattern_distances = (pattern_distances - pattern_distances.min()) / (pattern_distances.max() - pattern_distances.min())

    # node_distances = alpha * feature_distances + beta * pattern_distances
    node_distances = adaptive_alphas * feature_distances + (1 - adaptive_alphas) * pattern_distances


    plt.figure(figsize=(10, 8))
    sns.heatmap(node_distances, cmap='viridis', vmin=0, vmax=1, xticklabels=g2_node_name, yticklabels=g1_node_name)
    plt.title("Distance Heatmap")
    plt.xlabel("Layers")
    plt.ylabel("Layers")
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    # plt.savefig("figures/ooo.png", dpi=500, bbox_inches='tight')

    block_size = 12
    n_blocks = 12
    diagonal_blocks = []
    for i in range(n_blocks):
        # Extract the i-th diagonal block
        row_start = i * block_size
        row_end = (i + 1) * block_size
        col_start = i * block_size
        col_end = (i + 1) * block_size
        block = node_distances[row_start:row_end, col_start:col_end]
        avg = np.sum(block)/(144-12)
        diagonal_blocks.append(avg)

    plt.figure(figsize=(10, 4))  # Adjust figure size if needed
    plt.plot(diagonal_blocks, 'o-', color='blue', linewidth=2, markersize=8)  # 'o-' for line + markers
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Node Distance", fontsize=12)
    plt.ylim(0, 0.8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(12))
    # plt.savefig("figures/ensemble_layer_vs_atthead_feature_distance_test.png", dpi=500, bbox_inches='tight')


    # shallow-to-deep heads
    std_head = list(head_2_layer.keys())
    head_scores = []
    for h in std_head:
        head_ids = INDUCTION_CIRCUIT[h]
        head_ids = [f'a{i[0]}.h{i[1]}' for i in head_ids]
        head_array_idx = [g1_node_name.index(i) for i in head_ids]
        block = node_distances[np.ix_(head_array_idx, head_array_idx)]
        avg = np.sum(block)/(len(head_array_idx)*(len(head_array_idx)-1))
        head_scores.append(avg)

    layer_scores = []
    for h in std_head:
        layers = head_2_layer[h]
        head_ids = []
        for l in layers:
            head_ids += [f'a{l}.h{i}' for i in range(12)]
        head_array_idx = [g1_node_name.index(i) for i in head_ids]
        block = node_distances[np.ix_(head_array_idx, head_array_idx)]
        avg = np.sum(block)/(len(head_array_idx)*(len(head_array_idx)-1))
        layer_scores.append(avg)

    plt.figure(figsize=(10, 4))  # Adjust figure size if needed
    plt.plot(layer_scores, 'o-', color='blue', linewidth=2, markersize=8, label='Corresponding Layers\' Average')  # 'o-' for line + markers
    plt.plot(head_scores, 'o-', color='red', linewidth=2, markersize=8, label='Specialized Head')  # 'o-' for line + markers
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Node Distance", fontsize=12)
    plt.ylim(0, 0.8)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
    plt.xticks(range(len(std_head)), labels=std_head)  # Ensure all 12 indices are shown
    # plt.savefig("figures/ensemble_testtesttest.png", dpi=500, bbox_inches='tight')

    plt.close()

    # now we conduct experimetns of gpt2-small vs. gpt2-small. Assume circuit a is well-analyzed and circuit b is unknown
    group_a_idx_tup = list(GROUP_A_IDX_2_CLASS.keys())
    group_b_idx_tup = list(GROUP_B_IDX_2_CLASS.keys())
    group_a_idx = [i[0]*12 + i[1] for i in group_a_idx_tup]
    group_b_idx = [i[0]*12 + i[1] for i in group_b_idx_tup]

    node_distances_a_b = node_distances[np.ix_(group_b_idx, group_a_idx)]
    # print(node_distances_a_b)

    row_max_indices = np.argmin(node_distances_a_b, axis=1)
    # print(row_max_indices)

    cnt = 0
    for i, idx in enumerate(row_max_indices):
        b_node_cls = GROUP_B_IDX_2_CLASS[group_b_idx_tup[i]]
        a_node_cls = GROUP_A_IDX_2_CLASS[group_a_idx_tup[idx]]
        if b_node_cls == a_node_cls:
            cnt += 1
        else:
            print(f'i: {i}, b_head: {group_b_idx_tup[i]}, a_head: {group_a_idx_tup[idx]}, gt: {b_node_cls}, pred: {a_node_cls}')

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