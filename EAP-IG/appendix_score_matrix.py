from functools import partial

import os
import sys
import ast
import json
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.utils import topn_indices, set_seed


set_seed(2025)

dataset = 'ioi' # mmlu_astronomy # ioi
i = 1
para_data = []
if 'mmlu' in dataset:
    for k in range(10):
        para_data.append(np.load(f"Query-Circuit-Dataset/score_data/mmlu_astronomy/metric4_{i}_{k}.npy"))
elif 'ioi' in dataset:
    for i in range(0, 10):
        para_data.append(np.load(f"Query-Circuit-Dataset/score_data/ioi_20steps/llama32-1b/ioi_edge_scores_{i}.npy"))

para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)

# Flatten all arrays in para_data to compute global percentiles
ALL_SCORES = np.concatenate([arr.flatten() for arr in para_data])
ALL_SCORES = ALL_SCORES[~np.isinf(ALL_SCORES)]  # drop inf
ALL_SCORES = ALL_SCORES[~np.isnan(ALL_SCORES)]  # drop nan if needed

vmax = np.nanpercentile(ALL_SCORES, 97)
vmin = np.nanpercentile(ALL_SCORES, 3)

fig, axes = plt.subplots(4, 3, figsize=(15, 9))  # 4 rows × 3 cols
axes = axes.flatten()

for j in range(10):  # 10 plots
    arr = para_data[j]
    arr = arr[6*(32+1)+1:10*(32+1)+1, 7*(32*3+1)+1:11*(32*3+1)+1]
    arr[np.isinf(arr)] = np.nan
    if j == 0:
        idx = 0
    else:
        idx = j + 2
    im = axes[idx].imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])
    if j == 0:
        axes[idx].set_title(f"Original Query (id=1)", fontsize=16)
    else:
        axes[idx].set_title(f"Paraphrase {j}", fontsize=16)

# Add ONE shared colorbar outside the grid
cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                    fraction=0.02, pad=0.02)  # shrink + spacing
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Edge Score", fontsize=16)

# Hide unused subplots
axes[1].axis("off")
axes[2].axis("off")

plt.tight_layout(rect=[0, 0, 0.88, 1])  # leave space on the right for cbar

plt.savefig(f'figures/{dataset}_heatmap_grid.pdf', bbox_inches="tight")