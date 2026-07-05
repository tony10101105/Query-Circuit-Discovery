"""
Shared utilities for SAE-based circuit analysis and feature steering.

Exports:
  load_sae_suite              -- load attn-out / mlp-out SAEs for the given layers
  load_sae_explanations       -- load auto-interp feature explanations from disk
  GENDER_KEYWORDS             -- default keyword list for gender-related features
  is_gender_feature           -- regex keyword match on a feature description
  get_circuit_sae_features    -- top-k SAE features per token for nodes in a circuit
  run_with_sae_feature_ablation -- last-token logits with selected SAE features zeroed
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from einops import einsum
from sae_lens import SAE

from .graph import AttentionNode
from .utils import tokenize_plus


GENDER_KEYWORDS = [
    'gender', 'male', 'female', 'males', 'females', 'man', 'men', 'woman', 'women',
    'he', 'she', 'her', 'hers', 'his', 'him', 'himself', 'herself',
    'masculine', 'feminine', 'boy', 'girl', 'boys', 'girls',
]


def load_sae_suite(layers, device):
    sae_suite = {}
    for layer in layers:
        sae_att = SAE.from_pretrained(
            release="gpt2-small-attn-out-v5-32k",
            sae_id=f'blocks.{layer}.hook_attn_out',
            device=device,
        )
        sae_att.use_error_term = True
        sae_suite[f"{layer}-att"] = sae_att

        sae_mlp = SAE.from_pretrained(
            release="gpt2-small-mlp-out-v5-32k",
            sae_id=f'blocks.{layer}.hook_mlp_out',
            device=device,
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


def is_gender_feature(description, keywords=None):
    if not description:
        return False
    if keywords is None:
        keywords = GENDER_KEYWORDS
    pattern = r'\b(' + '|'.join(keywords) + r')\b'
    return re.search(pattern, description.lower()) is not None


def get_circuit_sae_features(model, graph, sae_suite, clean_text, corrupted_text, nodes_in_graph, top_k):
    """Run the circuit (patching intervention) on clean_text and return top-k SAE features per token for each circuit node layer."""
    clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, [clean_text])
    corrupted_tokens, _, _, _ = tokenize_plus(model, [corrupted_text], manual_pad_to_length=input_lengths.cpu().tolist())

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


def run_with_sae_feature_ablation(model, sae_suite, text, ablate_features):
    """Return last-token logits on text with the given SAE features (dict of sae_suite key -> feature indices) zero-ablated."""
    tokens, attention_mask, input_lengths, _ = tokenize_plus(model, [text])

    def make_splice_hook(sae, feature_idxs):
        def hook_fn(activations, hook):
            features = sae.encode(activations)
            error = activations - sae.decode(features)
            features[..., feature_idxs] = 0
            return sae.decode(features) + error
        return hook_fn

    hooks = []
    for key, feature_idxs in ablate_features.items():
        if not feature_idxs:
            continue
        layer, suffix = key.split('-')
        hook_name = f'blocks.{layer}.hook_attn_out' if suffix == 'att' else f'blocks.{layer}.hook_mlp_out'
        hooks.append((hook_name, make_splice_hook(sae_suite[key], list(feature_idxs))))

    with torch.inference_mode():
        with model.hooks(hooks):
            logits = model(tokens, attention_mask=attention_mask)
    return logits[0, input_lengths[0] - 1]
