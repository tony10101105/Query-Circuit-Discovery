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
from einops import einsum
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE

from eap.graph import Graph, AttentionNode
from eap.utils import set_seed, tokenize_plus
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


def load_sae_suite(layers):
    sae_suite = {}
    for layer in layers:
        sae_att = SAE.from_pretrained(
            release="gpt2-small-attn-out-v5-32k",
            sae_id=f'blocks.{layer}.hook_attn_out',
            device=model_cfg.device,
        )
        sae_att.use_error_term = True
        sae_suite[f"{layer}-att"] = sae_att

        sae_mlp = SAE.from_pretrained(
            release="gpt2-small-mlp-out-v5-32k",
            sae_id=f'blocks.{layer}.hook_mlp_out',
            device=model_cfg.device,
        )
        sae_mlp.use_error_term = True
        sae_suite[f"{layer}-mlp"] = sae_mlp
    return sae_suite


def load_sae_explanations(base_dir, layers):
    base_path = Path(base_dir)
    explanations = {}
    for layer in layers:
        for suffix in ('att', 'mlp'):
            key = f'{layer}-{suffix}'
            exp_dir = base_path / f"{key}_32k-oai" / "explanations"
            if exp_dir.exists():
                explanations[key] = {}
                for jsonl_file in sorted(exp_dir.glob("*.jsonl")):
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                explanations[key][data["index"]] = data
    return explanations


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


def get_circuit_sae_features(graph, sae_suite, clean_text, corrupted_text, nodes_in_graph, top_k):
    clean_tokens, attention_mask, _, n_pos = tokenize_plus(model, [clean_text])
    corrupted_tokens, _, _, _ = tokenize_plus(model, [corrupted_text])

    circuit_activations = {}

    def make_capture_hook(layer_name):
        def hook_fn(activations, hook):
            circuit_activations[layer_name] = activations.detach().clone()
        return hook_fn

    capture_hooks = []
    for layer_idx in range(model.cfg.n_layers):
        att_start = 1 + layer_idx * 13
        if np.any(nodes_in_graph[att_start:att_start + 12]):
            capture_hooks.append((f'blocks.{layer_idx}.hook_attn_out', make_capture_hook(f'{layer_idx}-att')))
        if nodes_in_graph[att_start + 12]:
            capture_hooks.append((f'blocks.{layer_idx}.hook_mlp_out', make_capture_hook(f'{layer_idx}-mlp')))

    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    neuron_matrix = None
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
        neuron_matrix = 1 - neuron_matrix
    in_graph_matrix = 1 - in_graph_matrix

    activation_difference = torch.zeros(
        (1, n_pos, graph.n_forward, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
    )

    def make_fwd_hook_corrupted(src_index, node):
        def hook_fn(activations, hook):
            if isinstance(node, AttentionNode):
                activation_difference[:, :, src_index] += activations[:, :, node.head, :]
            else:
                activation_difference[:, :, src_index] += activations
        return hook_fn

    def make_fwd_hook_clean(src_index, node):
        def hook_fn(activations, hook):
            if isinstance(node, AttentionNode):
                activation_difference[:, :, src_index] -= activations[:, :, node.head, :]
            else:
                activation_difference[:, :, src_index] -= activations
        return hook_fn

    fwd_hooks_corrupted, fwd_hooks_clean = [], []
    for node in graph.nodes.values():
        if node.out_hook != '':
            src_idx = graph.forward_index(node, attn_slice=False)
            fwd_hooks_corrupted.append((node.out_hook, make_fwd_hook_corrupted(src_idx, node)))
            fwd_hooks_clean.append((node.out_hook, make_fwd_hook_clean(src_idx, node)))

    def make_input_construction_hook(in_graph_vector, neuron_matrix_vec):
        def hook_fn(activations, hook):
            if neuron_matrix_vec is not None:
                update = einsum(
                    activation_difference[:, :, :len(in_graph_vector)],
                    neuron_matrix_vec[:len(in_graph_vector)],
                    in_graph_vector,
                    'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden',
                )
            else:
                update = einsum(
                    activation_difference[:, :, :len(in_graph_vector)],
                    in_graph_vector,
                    'batch pos previous hidden, previous ... -> batch pos ... hidden',
                )
            activations += update
            return activations
        return hook_fn

    input_construction_hooks = []
    for layer in range(model.cfg.n_layers):
        if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{layer}.h0']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                input_construction_hooks.append((
                    node.qkv_inputs[i],
                    make_input_construction_hook(in_graph_matrix[:prev_index, bwd_index], neuron_matrix),
                ))
        if graph.nodes[f'm{layer}'].in_graph:
            node = graph.nodes[f'm{layer}']
            prev_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_construction_hooks.append((
                node.in_hook,
                make_input_construction_hook(in_graph_matrix[:prev_index, bwd_index], neuron_matrix),
            ))

    with torch.inference_mode():
        with model.hooks(fwd_hooks_corrupted):
            model(corrupted_tokens, attention_mask=attention_mask)
        with model.hooks(fwd_hooks_clean + input_construction_hooks + capture_hooks):
            model(clean_tokens, attention_mask=attention_mask)

    results = {}
    for layer_name, sae in sae_suite.items():
        if layer_name not in circuit_activations:
            continue
        activations = circuit_activations[layer_name]
        _, seq_len, d_model = activations.shape
        sae_features = sae.encode(activations.reshape(-1, d_model)).reshape(1, seq_len, -1)
        token_features = []
        for t in range(seq_len):
            top_vals, top_idxs = torch.topk(sae_features[0, t], top_k)
            token_features.append([(int(idx), float(val)) for idx, val in zip(top_idxs.cpu(), top_vals.cpu())])
        results[layer_name] = token_features
    return results


def analyze_sample(sae_suite, explanations, sample_data, top_k):
    clean_text = sample_data['clean']
    corrupted_text = sample_data['corrupted']

    graph = Graph.from_model(model.cfg)
    graph.nodes_in_graph[:] = torch.tensor(sample_data['nodes_in_graph'], dtype=torch.bool)
    graph.in_graph[:] = torch.tensor(sample_data['in_graph'], dtype=torch.bool).reshape(graph.in_graph.shape)
    graph.prune()

    circuit_features = get_circuit_sae_features(
        graph, sae_suite, clean_text, corrupted_text, sample_data['nodes_in_graph'], top_k
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
sae_suite = load_sae_suite(args.layers)
explanations = load_sae_explanations(args.sae_data_dir, args.layers)

print(f"Analyzing {len(samples)} samples...")
for sample in tqdm(samples):
    result = analyze_sample(sae_suite, explanations, sample, args.top_k)
    with open(os.path.join(args.output_dir, f'sample_{sample["idx"]}.json'), 'w') as f:
        json.dump(result, f, indent=2)

print(f"Analysis complete. Results saved to {args.output_dir}/")
