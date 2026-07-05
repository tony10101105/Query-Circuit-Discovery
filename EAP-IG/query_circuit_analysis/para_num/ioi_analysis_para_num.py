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
parser.add_argument('--output_figure', type=str, default='figures/ioi_para_num.pdf')
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

all_results = []
all_vanilla_results = []

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples")):
    single_data = [(clean, corrupted, label)]

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    pad_corrupted_to_clean(model, single_data)

    all_results_per_sample = []

    all_indices = list(range(para_data.shape[0]))
    available_idxs = [para_idx for para_idx in all_indices if para_idx != i]
    sampled = random.sample(available_idxs, 9)
    sampled = [i] + sampled

    for j in tqdm(sampled, total=len(sampled), desc="Processing paraphrases"):
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

        all_results_per_sample.append(circuit_faithfulness)

    all_results_per_sample = np.array(all_results_per_sample)
    all_results.append(all_results_per_sample)

markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+', '1']
for i in range(10, 1, -1):
    para = [np.max(k[:i], axis=0) for k in all_results]
    avg = np.mean(np.stack(para, axis=0), axis=0)
    plt.plot(args.topns, avg, label=f'Best-of-{i}', marker=markers[i-2])

all_vanilla_results = np.array(all_vanilla_results).mean(0)
print('all_vanilla_results: ', all_vanilla_results)
plt.plot(args.topns, all_vanilla_results, label='Single Query', marker='2')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=13)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(args.output_figure, bbox_inches='tight')