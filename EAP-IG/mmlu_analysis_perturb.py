from functools import partial

import os
import sys
import ast
import json
import bisect
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
from time import sleep

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
    labels = list(labels)
    # labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, category=None, num_samples=None):
        self.df = pd.read_csv(filepath)
        if category:
            self.df = self.df[self.df['category'] == category]
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        self.df = self.df.iloc[110:]
        print(f'Loaded {len(self.df)} samples from {filepath} with category {category} and {len(self.df)} samples')

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [int(row['correct_idx']), ast.literal_eval(row['incorrect_idx'])]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

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
    # results = probs[torch.arange(batch_size), correct_idxs]

    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
category = 'astronomy' # marketing, professional_medicine, astronomy, college_biology, high_school_computer_science, logical_fallacies, nutrition, international_law, management
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
perturb_times = 1
var = [0.01, 0.001]
replace_ratio = [0.1, 0.3]
assert len(var) == len(replace_ratio)
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset(f'probing_dataset/mmlu_{category}_Llama-32-1B.csv', num_samples=500)
dataloader = ds.to_dataloader(batch_size=1)

all_best_results = []
all_vanilla_results = []
all_best_random = []
all_best_perturb_all_var = []
all_best_random_drop_all = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples", position=0)):
    para_data = []
    for k in range(10):
        para_data.append(np.load(f"score_data//mmlu_{category}/metric4_{i+110}_{k}.npy"))

    para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)
    
    single_data = [(clean, corrupted, label)]

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

    # only for padding corrupted input
    _ = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

    best_results = [-1]*len(topns)
    best_para_indices = [0]*len(topns) # keep track of the best paraphrase index for each topn
    
    for j in tqdm(range(para_data.shape[0]), total=para_data.shape[0], desc="Processing paraphrases", position=1, leave=False):
        model.reset_hooks()
        g = Graph.from_model(model)
        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness, circuit_complement_faithfulness = [], []
        c = []
        for topn in topns:
            g.apply_topn(topn, True)
            
            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

        if j == 0:
            all_vanilla_results.append(circuit_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_para_indices[idx] = j

    all_best_results.append(best_results)

    # random scores
    best_random = [-1]*len(topns)
    best_rand_indices = [0]*len(topns)
    for _ in range(10):
        model.reset_hooks()
        g = Graph.from_model(model)
        g.scores = torch.rand_like(torch.from_numpy(para_data[0]))
        
        circuit_faithfulness = []
        for topn in topns:
            g.apply_topn(topn, True)

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

        for idx in range(len(best_random)):
            if circuit_faithfulness[idx] > best_random[idx]:
                best_random[idx] = circuit_faithfulness[idx]
                best_rand_indices[idx] = j

    all_best_random.append(best_random)
    
    # perturb on circuit of original query
    base_score = para_data[0]
    all_best_perturb = []
    for v in range(len(var)):
        best_perturb = [-1]*len(topns)
        for p in range(10):
            model.reset_hooks()
            g = Graph.from_model(model)
            if p == 0:
                g.scores = torch.from_numpy(base_score)
            else:
                # attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-inputs-sg', ig_steps=steps, intervention=intervention, quiet=True, perturb_times=perturb_times, var=var[v])
                g.scores = torch.from_numpy(base_score) + torch.randn_like(torch.from_numpy(base_score)) * var[v]

            circuit_faithfulness = []
            for topn in topns:
                g.apply_topn(topn, True)

                results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
                results = results.mean().item()

                faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
                circuit_faithfulness.append(faithfulness)

            for idx in range(len(best_perturb)):
                if circuit_faithfulness[idx] > best_perturb[idx]:
                    best_perturb[idx] = circuit_faithfulness[idx]
        all_best_perturb.append(best_perturb)
    all_best_perturb_all_var.append(all_best_perturb)

    # perturb by random dropping
    base_score = para_data[0]
    all_best_random_drop = []
    for v in range(len(var)):
        best_random_drop = [-1]*len(topns)
        for p in range(10):
            model.reset_hooks()
            g = Graph.from_model(model)
                                 
            g.scores = torch.from_numpy(base_score)
            circuit_faithfulness = []
            for topn in topns:
                if p == 0:
                    g.apply_topn(topn, True)
                else:
                    g.apply_topn_and_rand(topn, True, replace_ratio=replace_ratio[v])

                results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
                results = results.mean().item()

                faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
                circuit_faithfulness.append(faithfulness)

            for idx in range(len(best_random_drop)):
                if circuit_faithfulness[idx] > best_random_drop[idx]:
                    best_random_drop[idx] = circuit_faithfulness[idx]
        all_best_random_drop.append(best_random_drop)
    all_best_random_drop_all.append(all_best_random_drop)


all_best_perturb_all_var1 = [all_best_perturb_all_var[i][0] for i in range(len(all_best_perturb_all_var))]
all_best_perturb_all_var2 = [all_best_perturb_all_var[i][1] for i in range(len(all_best_perturb_all_var))]
all_best_random_drop_all1 = [all_best_random_drop_all[i][0] for i in range(len(all_best_random_drop_all))]
all_best_random_drop_all2 = [all_best_random_drop_all[i][1] for i in range(len(all_best_random_drop_all))]

all_best_perturb_all_var1 = np.array(all_best_perturb_all_var1).mean(0)
all_best_perturb_all_var2 = np.array(all_best_perturb_all_var2).mean(0)
all_best_random_drop_all1 = np.array(all_best_random_drop_all1).mean(0)
all_best_random_drop_all2 = np.array(all_best_random_drop_all2).mean(0)

all_best_results = np.array(all_best_results).mean(0)
all_vanilla_results = np.array(all_vanilla_results).mean(0)
all_best_random = np.array(all_best_random).mean(0)
print('all_best_results: ', all_best_results)
print('all_vanilla_results: ', all_vanilla_results)
print('all_best_random: ', all_best_random)
print('all_best_perturb_all_var1 (0.01): ', all_best_perturb_all_var1)
print('all_best_perturb_all_var2 (0.001): ', all_best_perturb_all_var2)
print('all_best_random_drop_all1 (0.1): ', all_best_random_drop_all1)
print('all_best_random_drop_all2 (0.3): ', all_best_random_drop_all2)
plt.plot(topns, all_vanilla_results, label='Single Query', marker='o')
plt.plot(topns, all_best_results, label='BoN-Para.', marker='o')
plt.plot(topns, all_best_random, label='BoN-Random', marker='o')
plt.plot(topns, all_best_perturb_all_var1, label='BoN-SP (0.01)', marker='o')
plt.plot(topns, all_best_perturb_all_var2, label='BoN-SP (0.001)', marker='o')
plt.plot(topns, all_best_random_drop_all1, label='BoN-ER (0.1)', marker='o')
plt.plot(topns, all_best_random_drop_all2, label='BoN-ER (0.3)', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/mmlu_{category}_perturb_1.pdf')