import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial
import json

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, ndf
from save_score_matrix.models import TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--topn', type=int, default=150)
parser.add_argument('--num_additional', type=int, default=9)
parser.add_argument('--ndf_threshold', type=float, default=0.8)
parser.add_argument('--biased_samples_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/biased_samples.json')
parser.add_argument('--circuit_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/paired_source_circuits')
parser.add_argument('--output_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/paired_circuits')
parser.add_argument('--summary_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/paired_summary.json')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

with open(args.biased_samples_path, 'r') as f:
    biased_samples = json.load(f)

os.makedirs(args.circuit_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

circuits = {}
baselines = {}
for sample in tqdm(biased_samples, desc='discovery'):
    label = torch.tensor([[sample['clean_answer_idx'], sample['corrupted_answer_idx']]])
    single_data = [([sample['clean']], [sample['corrupted']], label)]
    model.reset_hooks()

    g = Graph.from_model(model)

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    pad_corrupted_to_clean(model, single_data)
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, quiet=True)

    g.apply_topn(args.topn, True)
    g.prune()

    circuit_info = {
        'idx': sample['idx'],
        'topn': args.topn,
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'nodes_in_graph': g.nodes_in_graph.cpu().numpy().tolist(),
        'in_graph': g.in_graph.cpu().numpy().tolist(),
    }
    with open(os.path.join(args.circuit_dir, f'sample_{sample["idx"]}.json'), 'w') as f:
        json.dump(circuit_info, f)

    circuits[sample['idx']] = g
    baselines[sample['idx']] = (baseline, corrupted_baseline)

rng = np.random.default_rng(2025)
all_idxs = [s['idx'] for s in biased_samples]

summary = []
for sample in tqdm(biased_samples, desc='cross-evaluation'):
    target_idx = sample['idx']
    other_idxs = [i for i in all_idxs if i != target_idx]
    source_idxs = [target_idx] + rng.choice(other_idxs, size=args.num_additional, replace=False).tolist()

    label = torch.tensor([[sample['clean_answer_idx'], sample['corrupted_answer_idx']]])
    single_data = [([sample['clean']], [sample['corrupted']], label)]
    pad_corrupted_to_clean(model, single_data)
    baseline, corrupted_baseline = baselines[target_idx]

    candidates = []
    for source_idx in source_idxs:
        model.reset_hooks()
        results, _, _, _ = evaluate_graph(model, circuits[source_idx], single_data, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=alg_cfg.intervention, quiet=True)
        candidates.append({'source_idx': source_idx, 'ndf': ndf(results.mean().item(), baseline, corrupted_baseline)})

    best = max(candidates, key=lambda c: c['ndf'])
    worst = min(candidates, key=lambda c: c['ndf'])

    pair_info = {
        'idx': target_idx,
        'candidates': candidates,
        'best_source_idx': best['source_idx'],
        'best_ndf': best['ndf'],
        'worst_source_idx': worst['source_idx'],
        'worst_ndf': worst['ndf'],
        'selected': best['ndf'] > args.ndf_threshold,
    }
    with open(os.path.join(args.output_dir, f'sample_{target_idx}.json'), 'w') as f:
        json.dump(pair_info, f, indent=2)

    summary.append(pair_info)

with open(args.summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

selected = [s for s in summary if s['selected']]
print(f'{len(selected)} / {len(summary)} samples with best-circuit NDF > {args.ndf_threshold}')
