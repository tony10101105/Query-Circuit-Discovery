import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, EAPDataset, nfs
from save_score_matrix.models import DatasetConfig, TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='marketing')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--topns', type=int, nargs='+', default=[500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000])
parser.add_argument('--dataset_path', type=str, default='probing_dataset/mmlu_marketing_Llama-32-1B.csv')
parser.add_argument('--sample_indices', type=int, nargs='+', default=[11, 41, 49])
parser.add_argument('--output_figure', type=str, default='figures/mmlu_marketing_instability_11_41_49.pdf')
args = parser.parse_args()

dataset_cfg = DatasetConfig(category=args.category, num_samples=args.num_samples)
model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()

model = HookedTransformer.from_pretrained_no_processing(model_cfg.model_name, device=model_cfg.device, torch_dtype=torch.float16)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = EAPDataset(args.dataset_path, num_samples=dataset_cfg.num_samples, mc=True)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    if i not in args.sample_indices:
        continue
    single_data = [(clean, corrupted, label)]

    model.reset_hooks()

    g = Graph.from_model(model)

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True).mean().item()

    pad_corrupted_to_clean(model, single_data)

    print(f"{i}-th sample. Original: {baseline}; corrupted: {corrupted_baseline}")
    attribute(model, g, single_data, partial(logit_diff, mc=True, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, quiet=True)

    circuit_results = []
    circuit_faithfulness = []
    for topn in args.topns:
        g.apply_topn(topn, True)

        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention, quiet=True)
        results = results.mean().item()
        circuit_results.append(results)

        faithfulness = nfs(results, baseline, corrupted_baseline)
        circuit_faithfulness.append(faithfulness)

        print(f"{i}-th sample. Original performance: {baseline:.2f}; circuit performance: {results:.2f}; corrupted_baseline: {corrupted_baseline:.2f}; faithfulness: {faithfulness:.2f}")

    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })

topns = [x / 1000 for x in args.topns]  # Convert to 'k'
for i, results in zip(args.sample_indices, all_results):
    plt.plot(topns, results['circuit_faithfulness'], label=f'Query {i}', marker='o')
plt.ylim(-2.1, 2.1)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1, linestyle='--', color='gray')
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(args.output_figure, bbox_inches='tight')