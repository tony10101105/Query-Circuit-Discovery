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

os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

set_seed(2025)

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    # labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, data_num):
        self.df = pd.read_csv(filepath)
        self.df = self.df[:data_num]

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    # def __getitem__(self, index):
    #     row = self.df.iloc[index]
    #     print(row['clean'])
    #     print([row['correct_idx'], row['incorrect_idx']])
    #     exit(0)
    #     return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # expand each row to 10 paraphrases
        clean = []
        for i in range(10):
            if i == 0:
                clean.append(row['clean'])
            else:
                clean.append(row['paraphrase' + str(i)])
        
        corrupted = [row['corrupted']] * 10

        correct_idx = int(row['correct_idx'])
        incorrect_idx = int(row['incorrect_idx'])
        labels = [[correct_idx, incorrect_idx]] * 10
        return clean, corrupted, labels
    
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

ds = EAPDataset(f'probing_dataset/arithmetic_add_Llama-32-1B.csv', data_num=data_num)
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
