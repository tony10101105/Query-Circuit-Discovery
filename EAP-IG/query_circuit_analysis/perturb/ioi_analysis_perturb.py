import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, EAPDataset, ndf, nfs
from save_score_matrix.models import DatasetConfig, TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--topns', type=int, nargs='+', default=[50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
parser.add_argument('--score_matrix_dir', type=str, default='probing_dataset/Query-Circuit-Dataset/score_matrix/ioi/gpt2-small')
parser.add_argument('--dataset_path', type=str, default='probing_dataset/ioi_gpt2.csv')
parser.add_argument('--output_figure', type=str, default='figures/ioi_perturb.pdf')
parser.add_argument('--faithfulness_metric', type=str, default='NDF', choices=['NDF', 'NFS'])
args = parser.parse_args()

dataset_cfg = DatasetConfig(num_samples=args.num_samples)
model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()
faithfulness_fn = ndf if args.faithfulness_metric == 'NDF' else nfs

var = [0.01, 0.001]
replace_ratio = [0.1, 0.3]

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = EAPDataset(args.dataset_path, num_samples=dataset_cfg.num_samples)
dataloader = ds.to_dataloader(batch_size=1)

para_data = []
for k in range(args.num_samples):
    para_data.append(np.load(f"{args.score_matrix_dir}/{k}.npy"))

para_data = np.stack(para_data, axis=0)   # shape: (num_samples, rows, cols)

all_best_results = []
all_vanilla_results = []
all_best_random = []
all_best_perturb_all_var = []
all_best_random_drop_all = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples", position=0)):
    single_data = [(clean, corrupted, label)]

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    pad_corrupted_to_clean(model, single_data)

    best_results = [-1]*len(args.topns)

    all_indices = list(range(para_data.shape[0]))
    available_idxs = [para_idx for para_idx in all_indices if para_idx != i]
    sampled = random.sample(available_idxs, 9)
    sampled = [i] + sampled

    for j in tqdm(sampled, total=len(sampled), desc="Processing paraphrases", position=1):
        model.reset_hooks()
        g = Graph.from_model(model)
        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness = []
        for topn in args.topns:
            g.apply_topn(topn, True)

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
            results = results.mean().item()
            circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))

        if j == i:
            all_vanilla_results.append(circuit_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]

    all_best_results.append(best_results)

    # random scores
    best_random = [-1]*len(args.topns)
    for _ in range(10):
        model.reset_hooks()
        g = Graph.from_model(model)
        g.scores = torch.rand_like(torch.from_numpy(para_data[0]))

        circuit_faithfulness = []
        for topn in args.topns:
            g.apply_topn(topn, True)

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
            results = results.mean().item()
            circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))

        for idx in range(len(best_random)):
            if circuit_faithfulness[idx] > best_random[idx]:
                best_random[idx] = circuit_faithfulness[idx]

    all_best_random.append(best_random)

    # perturb on circuit of original query
    base_score = para_data[i]
    all_best_perturb = []
    for v in range(len(var)):
        best_perturb = [-1]*len(args.topns)
        for p in range(10):
            model.reset_hooks()
            g = Graph.from_model(model)
            if p == 0:
                g.scores = torch.from_numpy(base_score)
            else:
                g.scores = torch.from_numpy(base_score) + torch.randn_like(torch.from_numpy(base_score)) * var[v]

            circuit_faithfulness = []
            for topn in args.topns:
                g.apply_topn(topn, True)

                results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
                results = results.mean().item()
                circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))

            for idx in range(len(best_perturb)):
                if circuit_faithfulness[idx] > best_perturb[idx]:
                    best_perturb[idx] = circuit_faithfulness[idx]
        all_best_perturb.append(best_perturb)
    all_best_perturb_all_var.append(all_best_perturb)

    # perturb by random dropping
    base_score = para_data[i]
    all_best_random_drop = []
    for v in range(len(var)):
        best_random_drop = [-1]*len(args.topns)
        for p in range(10):
            model.reset_hooks()
            g = Graph.from_model(model)

            g.scores = torch.from_numpy(base_score)
            circuit_faithfulness = []
            for topn in args.topns:
                if p == 0:
                    g.apply_topn(topn, True)
                else:
                    g.apply_topn_and_rand(topn, True, replace_ratio=replace_ratio[v])

                results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
                results = results.mean().item()
                circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))

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

plt.plot(args.topns, all_vanilla_results, label='Single Query', marker='o')
plt.plot(args.topns, all_best_results, label='BoN-Para.', marker='o')
plt.plot(args.topns, all_best_random, label='BoN-Random', marker='o')
plt.plot(args.topns, all_best_perturb_all_var1, label='BoN-GP ($\sigma$=0.01)', marker='o')
plt.plot(args.topns, all_best_perturb_all_var2, label='BoN-GP ($\sigma$=0.001)', marker='o')
plt.plot(args.topns, all_best_random_drop_all1, label='BoN-ER ($t$=0.1)', marker='o')
plt.plot(args.topns, all_best_random_drop_all2, label='BoN-ER ($t$=0.3)', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(args.output_figure, bbox_inches='tight')
