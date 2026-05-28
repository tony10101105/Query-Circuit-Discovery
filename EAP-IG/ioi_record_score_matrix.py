from functools import partial

import os
import random
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute
from src.eap.utils import set_seed
from utils import get_logit_positions, logit_diff, EAPDataset
set_seed(2025)
data_num = 1000
topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
# topns = [10000, 20000, 30000, 32491] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
perturb_times = 50 if method == 'EAP-IG-inputs-sg' else None
var = 0.1 if method == 'EAP-IG-inputs-sg' else None
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-xl' # meta-llama/Llama-3.2-1B-Instruct, gpt2-small
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', data_num=data_num)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    single_data = [(clean, corrupted, label)]
    
    model.reset_hooks()
    
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    print('attributing for this single data...')
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention, quiet=True, perturb_times=perturb_times, var=var)
    
    x = g.scores.cpu().detach().numpy()
    x[~g.real_edge_mask] = -np.inf
    np.save(f'Query-Circuit-Dataset/score_data/ioi_{steps}steps/{model_name}/ioi_edge_scores_{i}.npy', x)