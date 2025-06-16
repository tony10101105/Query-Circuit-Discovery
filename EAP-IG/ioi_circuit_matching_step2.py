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
from svcca import cca_core
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import seaborn as sns

scale = True
force_symmetric = True
data = np.load('cost_matrix_layerwise_pca.npz')
g1_node_name = data['g1_node_name'].tolist()[2:]
g2_node_name = data['g2_node_name'].tolist()[2:]
cca_distances = data['cca_distances'][2: , 2:]
print('min: ', cca_distances.min())
print('max: ', cca_distances.max())
if scale:
    cca_distances = (cca_distances - cca_distances.min()) / (cca_distances.max() - cca_distances.min())
if force_symmetric:
    cca_distances = (cca_distances + cca_distances.T) / 2

plt.figure(figsize=(10, 8))
sns.heatmap(cca_distances, cmap='viridis', vmin=0, vmax=1, xticklabels=g2_node_name, yticklabels=g1_node_name)
plt.title("CCA Distance Heatmap")
plt.xlabel("Layers")
plt.ylabel("Layers")
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig("cca_distance_heatmap_layerwise_pca.pdf", dpi=500, bbox_inches='tight')
exit(0)


block_size = 12
n_blocks = 12
diagonal_blocks = []
for i in range(n_blocks):
    # Extract the i-th diagonal block
    row_start = i * block_size
    row_end = (i + 1) * block_size
    col_start = i * block_size
    col_end = (i + 1) * block_size
    block = cca_distances[row_start:row_end, col_start:col_end]
    avg = np.sum(block)/(144-12)
    diagonal_blocks.append(avg)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))  # Adjust figure size if needed
plt.plot(diagonal_blocks, 'o-', color='blue', linewidth=2, markersize=8)  # 'o-' for line + markers
plt.xlabel("Layer", fontsize=12)
plt.ylabel("1 - Attention Heads Feature Correlation", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
plt.xticks(np.arange(12))  # Ensure all 12 indices are shown
plt.savefig("layer_vs_atthead_feature_distance.pdf", dpi=500, bbox_inches='tight')


IOI_CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}



assert g1_node_name == g2_node_name
# shallow-to-deep heads
std_head = ["duplicate token", "previous token", "induction", "s2 inhibition", "name mover", "backup name mover", "negative"]
head_scores = []
for h in std_head:
    head_ids = IOI_CIRCUIT[h]
    head_ids = [f'a{i[0]}.h{i[1]}' for i in head_ids]
    head_array_idx = [g1_node_name.index(i) for i in head_ids]
    block = cca_distances[np.ix_(head_array_idx, head_array_idx)]
    avg = np.sum(block)/(len(head_array_idx)*(len(head_array_idx)-1))
    head_scores.append(avg)
    
plt.figure(figsize=(10, 4))  # Adjust figure size if needed
plt.plot(head_scores, 'o-', color='blue', linewidth=2, markersize=8)  # 'o-' for line + markers
plt.xlabel("Layer", fontsize=12)
plt.ylabel("1 - Attention Heads Feature Correlation", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
plt.xticks(range(7), labels=std_head)  # Ensure all 12 indices are shown
plt.savefig("test.pdf", dpi=500, bbox_inches='tight')
