import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.utils import set_seed
from eap.sae_utils import load_sae_suite, load_sae_explanations, is_gender_feature, get_circuit_sae_features
from save_score_matrix.models import TargetModelConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--keywords', type=str, nargs='+', default=None)
parser.add_argument('--sae_data_dir', type=str, default='sae_data/gpt2-small/32k')
parser.add_argument('--biased_samples_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/biased_samples.json')
parser.add_argument('--paired_circuit_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/paired_source_circuits')
parser.add_argument('--unpaired_circuit_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/unpaired_circuits')
parser.add_argument('--paired_summary_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/paired_summary.json')
parser.add_argument('--unpaired_summary_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/unpaired_summary.json')
parser.add_argument('--output_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/circuit_features')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention
model.eval()

with open(args.biased_samples_path, 'r') as f:
    biased_samples = {s['idx']: s for s in json.load(f)}
with open(args.paired_summary_path, 'r') as f:
    paired_summary = json.load(f)
with open(args.unpaired_summary_path, 'r') as f:
    unpaired_summary = json.load(f)

layers = list(range(model.cfg.n_layers))
sae_suite = load_sae_suite(layers, model_cfg.device)
explanations = load_sae_explanations(args.sae_data_dir, layers)

circuit_cache = {}
def load_circuit(circuit_dir, source_idx):
    if (circuit_dir, source_idx) not in circuit_cache:
        with open(os.path.join(circuit_dir, f'sample_{source_idx}.json'), 'r') as f:
            circuit_info = json.load(f)
        g = Graph.from_model(model)
        g.nodes_in_graph[:] = torch.tensor(circuit_info['nodes_in_graph'], dtype=torch.bool)
        g.in_graph[:] = torch.tensor(circuit_info['in_graph'], dtype=torch.bool)
        g.prune()
        circuit_cache[(circuit_dir, source_idx)] = g
    return circuit_cache[(circuit_dir, source_idx)]


def analyze_circuit_on_target(target_idx, source_idx, circuit_dir, output_name):
    sample = biased_samples[target_idx]
    graph = load_circuit(circuit_dir, source_idx)
    nodes_in_graph = graph.nodes_in_graph.cpu().numpy()

    circuit_features = get_circuit_sae_features(model, graph, sae_suite, sample['clean'], sample['corrupted'], nodes_in_graph, args.top_k)

    enriched_features = {}
    gender_features = {}
    for layer_name, tokens_feature_list in circuit_features.items():
        enriched_tokens = []
        gender_idxs = set()
        for token_features in tokens_feature_list:
            enriched_token_features = []
            for feature_idx, activation_val in token_features:
                description = None
                if layer_name in explanations and str(feature_idx) in explanations[layer_name]:
                    description = explanations[layer_name][str(feature_idx)].get('description', None)
                enriched_token_features.append({
                    'feature_index': feature_idx,
                    'activation': activation_val,
                    'description': description,
                })
                if activation_val > 0 and is_gender_feature(description, args.keywords):
                    gender_idxs.add(feature_idx)
            enriched_tokens.append(enriched_token_features)
        enriched_features[layer_name] = enriched_tokens
        gender_features[layer_name] = sorted(gender_idxs)

    num_gender = sum(len(v) for v in gender_features.values())
    num_gender_mlp = sum(len(v) for k, v in gender_features.items() if k.endswith('-mlp'))
    result = {
        'target_idx': target_idx,
        'source_idx': source_idx,
        'num_gender_features': num_gender,
        'num_gender_features_mlp': num_gender_mlp,
        'gender_features': gender_features,
        'circuit_features': enriched_features,
    }
    with open(os.path.join(args.output_dir, f'{output_name}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


os.makedirs(args.output_dir, exist_ok=True)

groups = {'paired_best': [], 'paired_worst': [], 'unpaired_high': [], 'unpaired_low': []}

for pair in tqdm(paired_summary, desc='paired'):
    if not pair['selected']:
        continue
    best = analyze_circuit_on_target(pair['idx'], pair['best_source_idx'], args.paired_circuit_dir, f'paired_target{pair["idx"]}_best')
    worst = analyze_circuit_on_target(pair['idx'], pair['worst_source_idx'], args.paired_circuit_dir, f'paired_target{pair["idx"]}_worst')
    groups['paired_best'].append(best['num_gender_features'])
    groups['paired_worst'].append(worst['num_gender_features'])

for entry in tqdm(unpaired_summary, desc='unpaired'):
    result = analyze_circuit_on_target(entry['idx'], entry['idx'], args.unpaired_circuit_dir, f'unpaired_target{entry["idx"]}')
    groups[f'unpaired_{entry["group"]}'].append(result['num_gender_features'])

stats = {name: {'mean': float(np.mean(counts)), 'std': float(np.std(counts)), 'n': len(counts)} for name, counts in groups.items()}
with open(os.path.join(args.output_dir, 'gender_feature_stats.json'), 'w') as f:
    json.dump({'stats': stats, 'counts': groups}, f, indent=2)

keys = ['paired_best', 'paired_worst', 'unpaired_high', 'unpaired_low']
for name in keys:
    print(f'{name}: {stats[name]["mean"]:.2f} ± {stats[name]["std"]:.2f} (n={stats[name]["n"]})')
