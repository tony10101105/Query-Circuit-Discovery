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

from preprocessed_circuit.induction_circuit_constant import INDUCTION_CIRCUIT, head_2_layer

scale = False
force_symmetric = False
metric = 'Linear CKA' # Linear CKA, RBF CKA, PWCCA
data = np.load('preprocessed_data/induction_cost_matrix_all_atthead_rep.npz', allow_pickle=True)
g1_node_name = data['g1_node_name'].tolist()
g2_node_name = data['g2_node_name'].tolist()
assert g1_node_name == g2_node_name
feature_distances = data['feature_distances']
print('feature_distances shape: ', feature_distances.shape)
feature_distances = np.array([[d[metric.lower()] for d in row] for row in feature_distances])

print('min: ', feature_distances.min())
print('max: ', feature_distances.max())
if scale:
    feature_distances = (feature_distances - feature_distances.min()) / (feature_distances.max() - feature_distances.min())
if force_symmetric:
    feature_distances = (feature_distances + feature_distances.T) / 2

plt.figure(figsize=(10, 8))
sns.heatmap(feature_distances, cmap='viridis', vmin=0, vmax=1, xticklabels=g2_node_name, yticklabels=g1_node_name)
plt.title(f"{metric} Distance Heatmap")
plt.xlabel("Layers")
plt.ylabel("Layers")
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig("figures/induction/rep_heatmap.png", dpi=500, bbox_inches='tight')


block_size = 12
n_blocks = 12
diagonal_blocks = []
for i in range(n_blocks):
    # Extract the i-th diagonal block
    row_start = i * block_size
    row_end = (i + 1) * block_size
    col_start = i * block_size
    col_end = (i + 1) * block_size
    block = feature_distances[row_start:row_end, col_start:col_end]
    avg = np.sum(block)/(144-12)
    diagonal_blocks.append(avg)

plt.figure(figsize=(10, 4))  # Adjust figure size if needed
plt.plot(diagonal_blocks, 'o-', color='blue', linewidth=2, markersize=8)  # 'o-' for line + markers
plt.xlabel("Layer", fontsize=12)
plt.ylabel(f"{metric} Feature Distance", fontsize=12)
plt.ylim(0, 0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(np.arange(12))
plt.savefig("figures/induction/rep_layer_vs_atthead_feature_distance_test.png", dpi=500, bbox_inches='tight')


# shallow-to-deep heads
std_head = list(head_2_layer.keys())
head_scores = []
for h in std_head:
    head_ids = INDUCTION_CIRCUIT[h]
    head_ids = [f'a{i[0]}.h{i[1]}' for i in head_ids]
    head_array_idx = [g1_node_name.index(i) for i in head_ids]
    block = feature_distances[np.ix_(head_array_idx, head_array_idx)]
    avg = np.sum(block)/(len(head_array_idx)*(len(head_array_idx)-1))
    head_scores.append(avg)
    
plt.figure(figsize=(10, 4))  # Adjust figure size if needed
plt.plot(head_scores, 'o-', color='blue', linewidth=2, markersize=8)  # 'o-' for line + markers
plt.xlabel("Layer", fontsize=12)
plt.ylabel(f"{metric} Feature Distance", fontsize=12)
plt.ylim(0, 0.8)
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
plt.xticks(range(len(std_head)), labels=std_head)  # Ensure all 12 indices are shown
plt.savefig("figures/induction/rep_test22.png", dpi=500, bbox_inches='tight')



# shallow-to-deep heads
layer_scores = []
for h in std_head:
    layers = head_2_layer[h]
    head_ids = []
    for l in layers:
        head_ids += [f'a{l}.h{i}' for i in range(12)]
    head_array_idx = [g1_node_name.index(i) for i in head_ids]
    block = feature_distances[np.ix_(head_array_idx, head_array_idx)]
    avg = np.sum(block)/(len(head_array_idx)*(len(head_array_idx)-1))
    layer_scores.append(avg)

plt.figure(figsize=(10, 4))  # Adjust figure size if needed
plt.plot(layer_scores, 'o-', color='blue', linewidth=2, markersize=8, label='Corresponding Layers\' Average')  # 'o-' for line + markers
plt.plot(head_scores, 'o-', color='red', linewidth=2, markersize=8, label='Specialized Head')  # 'o-' for line + markers
plt.xlabel("Layer", fontsize=12)
plt.ylabel(f"{metric} Feature Distance", fontsize=12)
plt.ylim(0, 0.8)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
plt.xticks(range(len(std_head)), labels=std_head)  # Ensure all 12 indices are shown
plt.savefig("figures/induction/rep_testtesttest.png", dpi=500, bbox_inches='tight')