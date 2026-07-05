import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.utils import set_seed
from eap.sae_utils import load_sae_suite, load_sae_explanations, get_circuit_sae_features
from save_score_matrix.models import TargetModelConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--layers', type=int, nargs='+', default=list(range(12)))
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--sae_data_dir', type=str, default='sae_data/gpt2-small/32k')
parser.add_argument('--graph_data_dir', type=str, default='gender_bias_sae_analysis/graph_data/top100')
parser.add_argument('--output_dir', type=str, default='gender_bias_sae_analysis/sae_analysis_data/top100')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention
model.eval()


def load_graph_data_samples(graph_data_dir):
    graph_path = Path(graph_data_dir)
    if not graph_path.exists():
        print(f"Graph data directory not found: {graph_data_dir}")
        return []
    json_files = sorted(graph_path.glob("gender_bias_*.json"))
    print(f"Loading data from {len(json_files)} graph_data files...")
    samples = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            samples.append({
                'idx': data['idx'],
                'clean': data['clean'],
                'corrupted': data['corrupted'],
                'nodes_in_graph': np.array(data['nodes_in_graph']),
                'in_graph': np.array(data['in_graph']),
                'clean_label': data['clean_label'],
                'corrupted_label': data['corrupted_label'],
            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    samples.sort(key=lambda x: x['idx'])
    return samples


def analyze_sample(sae_suite, explanations, sample_data, top_k):
    clean_text = sample_data['clean']
    corrupted_text = sample_data['corrupted']

    graph = Graph.from_model(model.cfg)
    graph.nodes_in_graph[:] = torch.tensor(sample_data['nodes_in_graph'], dtype=torch.bool)
    graph.in_graph[:] = torch.tensor(sample_data['in_graph'], dtype=torch.bool).reshape(graph.in_graph.shape)
    graph.prune()

    circuit_features = get_circuit_sae_features(
        model, graph, sae_suite, clean_text, corrupted_text, sample_data['nodes_in_graph'], top_k
    )

    enriched_features = {}
    for layer_name, tokens_feature_list in circuit_features.items():
        enriched_tokens = []
        for token_features in tokens_feature_list:
            enriched_token_features = []
            for feature_idx, activation_val in token_features:
                feature_info = {
                    'layer': layer_name,
                    'feature_index': feature_idx,
                    'activation': activation_val,
                    'description': None,
                }
                if layer_name in explanations and str(feature_idx) in explanations[layer_name]:
                    feature_info['description'] = explanations[layer_name][str(feature_idx)].get('description', 'No description')
                enriched_token_features.append(feature_info)
            enriched_tokens.append(enriched_token_features)
        enriched_features[layer_name] = enriched_tokens

    return {
        'sample_idx': sample_data['idx'],
        'clean_text': clean_text,
        'corrupted_text': corrupted_text,
        'clean_label': model.tokenizer.decode([sample_data['clean_label']]),
        'corrupted_label': model.tokenizer.decode([sample_data['corrupted_label']]),
        'circuit_features': enriched_features,
    }


os.makedirs(args.output_dir, exist_ok=True)

samples = load_graph_data_samples(args.graph_data_dir)
if not samples:
    print("No graph data files found.")
    raise SystemExit(1)

print(f"Loading SAE suite for layers {args.layers}...")
sae_suite = load_sae_suite(args.layers, model_cfg.device)
explanations = load_sae_explanations(args.sae_data_dir, args.layers)

print(f"Analyzing {len(samples)} samples...")
for sample in tqdm(samples):
    result = analyze_sample(sae_suite, explanations, sample, args.top_k)
    with open(os.path.join(args.output_dir, f'sample_{sample["idx"]}.json'), 'w') as f:
        json.dump(result, f, indent=2)

print(f"Analysis complete. Results saved to {args.output_dir}/")