from functools import partial

import os
import sys
import re
import ast
import json
import numpy as np
import time
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

def PARA_collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    # print(labels)
    labels = labels[0]
    clean = clean[0]
    corrupted = corrupted[0]
    # print('clean: ', clean)
    return clean, corrupted, labels

class PARA_EAPDataset(Dataset):
    def __init__(self, filepath, category=None, num_samples=None):
        self.df = pd.read_csv(filepath)
        if category:
            self.df = self.df[self.df['category'] == category]
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        print(f'Loaded {len(self.df)} samples from {filepath} with category {category} and {len(self.df)} samples')

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
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
        incorrect_idx = ast.literal_eval(row['incorrect_idx'])
        labels = [[correct_idx, incorrect_idx]] * 10

        return clean, corrupted, labels

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=PARA_collate_EAP)

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: list[list],             # list of [correct_idx, [wrong_idx1, wrong_idx2, ...]]
    mean: bool = True,
    loss: bool = False,
):
    # Extract the last-token logits: [batch_size, vocab_size]
    logits = get_logit_positions(logits, input_length)
    probs = torch.softmax(logits, dim=-1)

    batch_size = logits.size(0)

    # Get correct token logits
    correct_idxs = torch.tensor([lbl[0] for lbl in labels], device=logits.device)
    correct_logits = logits[torch.arange(batch_size), correct_idxs]

    # (4) correct - average over all wrong option tokens
    bad_vals = []
    for i, (_, wrong_ids) in enumerate(labels):
        wrong_logits = logits[i, torch.tensor(wrong_ids, device=logits.device)]
        bad_vals.append(wrong_logits.mean())
    bad = torch.stack(bad_vals)

    results = correct_logits - bad

    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
category = 'astronomy' # marketing, professional_medicine, astronomy, college_biology, high_school_computer_science, logical_fallacies, nutrition, international_law, management
rephrase_type = 'only_stem'
rephrase_model = 'gpt4o' # gpt4o-mini
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = PARA_EAPDataset(f'probing_dataset/mmlu_{category}_Llama-32-1B_{rephrase_model}_paraphrases_{rephrase_type}.csv', num_samples=500)
dataloader = ds.to_dataloader(batch_size=1)
print(f'Number of samples in dataloader: {len(dataloader)}')

all_results = []
all_best_results = []
start = time.perf_counter()
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    batch_slice_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]
    batch_slice_data = batch_slice_data[:2]
    baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    # only for padding corrupted input
    _ = evaluate_baseline(model, batch_slice_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    best_results = [-1]*len(topns)
    best_para_indices = [0]*len(topns) # keep track of the best paraphrase index for each topn

    for j, single_data in enumerate(batch_slice_data):
        model.reset_hooks()
        g = Graph.from_model(model)
        
        attribute(model, g, [single_data], partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention, file_idx=i, cat=category, quiet=True)

        circuit_results = []
        circuit_faithfulness = []
        for topn in topns:
            g.apply_topn(topn, True)
            
            results, _, _, _ = evaluate_graph(model, g, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            circuit_results.append(results)
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_para_indices[idx] = j

    all_best_results.append(best_results)

end = time.perf_counter()
print(f'Total time: {end - start:.2f} seconds')