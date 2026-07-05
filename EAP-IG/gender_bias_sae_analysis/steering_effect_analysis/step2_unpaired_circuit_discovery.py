import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial
import json

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
parser.add_argument('--topn', type=int, default=200)
parser.add_argument('--ndf_threshold', type=float, default=0.8)
parser.add_argument('--biased_samples_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/biased_samples.json')
parser.add_argument('--output_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/unpaired_circuits')
parser.add_argument('--summary_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/unpaired_summary.json')
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

os.makedirs(args.output_dir, exist_ok=True)

summary = []
for sample in tqdm(biased_samples):
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

    results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=alg_cfg.intervention, quiet=True)
    results = results.mean().item()

    faithfulness = ndf(results, baseline, corrupted_baseline)

    circuit_info = {
        'idx': sample['idx'],
        'topn': args.topn,
        'ndf': faithfulness,
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'results': results,
        'nodes_in_graph': g.nodes_in_graph.cpu().numpy().tolist(),
        'in_graph': g.in_graph.cpu().numpy().tolist(),
    }
    with open(os.path.join(args.output_dir, f'sample_{sample["idx"]}.json'), 'w') as f:
        json.dump(circuit_info, f)

    summary.append({'idx': sample['idx'], 'ndf': faithfulness, 'group': 'high' if faithfulness > args.ndf_threshold else 'low'})

with open(args.summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

high = [s for s in summary if s['group'] == 'high']
low = [s for s in summary if s['group'] == 'low']
print(f'high-NDF group (NDF > {args.ndf_threshold}): {len(high)} samples')
print(f'low-NDF group (NDF <= {args.ndf_threshold}): {len(low)} samples')
