from functools import partial

import os
import sys
import ast
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute
from src.eap.utils import topn_indices, set_seed
from utils import get_logit_positions, logit_diff, PARAEAPDataset


set_seed(2025)
data_num = 500
topns = [500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000] # 32491 for gpt2-small, 386713 for llama
# topns = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000] # 386713 for llama
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations # EAP
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = PARAEAPDataset(f'probing_dataset/arithmetic_add_Llama-32-1B.csv', data_num=data_num, simple=True)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples")):
    assert len(clean) == len(corrupted) and len(corrupted) == len(label)
    batch_slice_data = [([clean[0][j]], [corrupted[0][j]], torch.tensor([label[0][j]])) for j in range(len(clean[0]))]

    model.reset_hooks()
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True).mean().item()

    print('attributing for this single data and save the score matrix...')
    attribute(model, g, batch_slice_data, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention, file_idx=i)
