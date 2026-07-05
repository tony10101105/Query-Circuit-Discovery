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
from eap.utils import topn_indices, set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, EAPDataset, ndf, nfs
from save_score_matrix.models import DatasetConfig, TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--topns', type=int, nargs='+', default=[50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
parser.add_argument('--score_matrix_dir', type=str, default='probing_dataset/Query-Circuit-Dataset/score_matrix/ioi/gpt2-small')
parser.add_argument('--dataset_path', type=str, default='probing_dataset/ioi_gpt2.csv')
parser.add_argument('--output_figure', type=str, default='figures/ioi_complement.pdf')
parser.add_argument('--faithfulness_metric', type=str, default='NDF', choices=['NDF', 'NFS'])
args = parser.parse_args()

dataset_cfg = DatasetConfig(num_samples=args.num_samples)
model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()
faithfulness_fn = ndf if args.faithfulness_metric == 'NDF' else nfs

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
all_best_complement_results = []
all_vanilla_results = []
all_vanilla_complement_results = []
all_avg_results = []
all_csm_results = []
all_ibon_results = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples", position=0)):
    single_data = [(clean, corrupted, label)]

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    torch.cuda.synchronize()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
    torch.cuda.synchronize()

    pad_corrupted_to_clean(model, single_data)
    torch.cuda.synchronize()

    best_complement_results = [-1]*len(args.topns)
    best_results = [-1]*len(args.topns)
    best_para_indices = [0]*len(args.topns)

    all_indices = list(range(para_data.shape[0]))
    available_idxs = [para_idx for para_idx in all_indices if para_idx != i]
    sampled = random.sample(available_idxs, 9)
    sampled = [i] + sampled

    for j in tqdm(sampled, total=len(sampled), desc="Processing paraphrases", position=1):
        model.reset_hooks()

        g = Graph.from_model(model)
        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness, circuit_complement_faithfulness = [], []
        for topn in args.topns:
            g.apply_topn(topn, True)
            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
            torch.cuda.synchronize()
            results = results.mean().item()
            torch.cuda.synchronize()
            circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))

            g.apply_topn(topn, True, complement=True)
            complement_results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
            torch.cuda.synchronize()
            complement_results = complement_results.mean().item()
            torch.cuda.synchronize()
            circuit_complement_faithfulness.append(faithfulness_fn(complement_results, baseline, corrupted_baseline))

        if j == i:
            all_vanilla_results.append(circuit_faithfulness)
            all_vanilla_complement_results.append(circuit_complement_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_complement_results[idx] = circuit_complement_faithfulness[idx]
                best_para_indices[idx] = j

    all_best_results.append(best_results)
    all_best_complement_results.append(best_complement_results)

    # averaging
    model.reset_hooks()
    g = Graph.from_model(model)
    g.scores = torch.from_numpy(para_data[sampled].mean(0))

    circuit_faithfulness = []
    for topn in args.topns:
        g.apply_topn(topn, True, complement=True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
        torch.cuda.synchronize()
        results = results.mean().item()
        circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))

    all_avg_results.append(circuit_faithfulness)

    # BoN with Constraint-adaptive Score Matrix (BoN-CSM)
    score_mat = np.full_like(para_data[j], -np.inf, dtype=float)
    tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    filled    = np.zeros_like(score_mat, dtype=bool)

    for l, topn in enumerate(args.topns):
        best_para_idx = best_para_indices[l]
        M = np.abs(para_data[best_para_idx])
        M = np.where(np.isfinite(M), M, -np.inf)
        for (a, b) in topn_indices(M, topn):
            if not filled[a, b]:
                score_mat[a, b] = M[a, b]
                tier_mat[a, b]  = l
                filled[a, b]    = True

    model.reset_hooks()
    g = Graph.from_model(model)
    g.scores = torch.from_numpy(score_mat)

    circuit_faithfulness = []
    for topn in args.topns:
        g.apply_topn_by_tier(topn, tier_mat, complement=True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
        torch.cuda.synchronize()
        results = results.mean().item()
        circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))
    all_csm_results.append(circuit_faithfulness)

    # interpolated BoN (iBoN)
    new_topns = [int((args.topns[k]+args.topns[k-1])/2) for k in range(1, len(args.topns))]
    circuit_faithfulness = []
    for k, topn in enumerate(new_topns):
        model.reset_hooks()
        g = Graph.from_model(model)

        previous_topn, next_topn = args.topns[k], args.topns[k+1]

        score_mat = np.full_like(para_data[j], -np.inf, dtype=float)
        tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
        filled    = np.zeros_like(score_mat, dtype=bool)

        for l, top in zip([k, k+1], [previous_topn, next_topn]):
            best_para_idx = best_para_indices[l]
            M = np.abs(para_data[best_para_idx])
            M = np.where(np.isfinite(M), M, -np.inf)
            for (a, b) in topn_indices(M, top):
                if not filled[a, b]:
                    score_mat[a, b] = M[a, b]
                    tier_mat[a, b]  = l
                    filled[a, b]    = True

        g.scores = torch.from_numpy(score_mat)
        g.apply_topn_by_tier(topn, tier_mat, complement=True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
        torch.cuda.synchronize()
        results = results.mean().item()
        circuit_faithfulness.append(faithfulness_fn(results, baseline, corrupted_baseline))
    all_ibon_results.append(circuit_faithfulness)

topns = [x / 1000 for x in args.topns]  # Convert to 'k'
new_topns = [x / 1000 for x in new_topns]  # Convert to 'k'

all_best_complement_results = np.array(all_best_complement_results).mean(0)
all_vanilla_complement_results = np.array(all_vanilla_complement_results).mean(0)
all_avg_results = np.array(all_avg_results).mean(0)
all_csm_results = np.array(all_csm_results).mean(0)
all_ibon_results = np.array(all_ibon_results).mean(0)
print('all_vanilla_complement_results: ', all_vanilla_complement_results)
print('all_best_complement_results: ', all_best_complement_results)
print('all_avg_results: ', all_avg_results)
print('all_csm_results: ', all_csm_results)
print('all_ibon_results: ', all_ibon_results)
plt.plot(topns, all_vanilla_complement_results, label='Single Query', marker='o')
plt.plot(topns, all_avg_results, label='Averaging', marker='o')
plt.plot(topns, all_best_complement_results, label='BoN', marker='o')
plt.plot(topns, all_csm_results, label='BoN-CSM', marker='o')
plt.plot(new_topns, all_ibon_results, label='iBoN', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(args.output_figure, bbox_inches='tight')
