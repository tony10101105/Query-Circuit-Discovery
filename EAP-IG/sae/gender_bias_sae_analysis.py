"""
Gender Bias SAE Analysis
Analyzes gender bias samples using GPT-2 Small with SAE feature extraction
"""

import os as _os, sys as _sys
_base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_base)
_sys.path.insert(0, _base)

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Import EAP-IG graph and evaluation utilities
import sys
sys.path.append('src')
from eap.graph import Graph
from eap.utils import tokenize_plus


def set_seed(seed: int = 2025):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gpt2_subject() -> HookedTransformer:
    """Load GPT-2 Small model using TransformerLens with required configs"""
    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained(
        'gpt2-small',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Set required configurations for circuit evaluation
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def load_sae_suite(device: torch.device, layers: List[int]) -> Dict[str, SAE]:
    """
    Load SAE suite for GPT-2 Small
    
    Args:
        device: torch device to load SAEs onto
        layers: List of layer indices to load SAEs for
    
    Returns:
        Dictionary mapping layer names to SAE objects
    """
    print(f"Loading SAE suite for layers {layers}...")
    sae_suite = {}
    
    for layer in layers:
        # Load attention SAE
        att_id = f'blocks.{layer}.hook_attn_out'
        sae_att = SAE.from_pretrained(
            release="gpt2-small-attn-out-v5-32k",
            sae_id=att_id,
            device=str(device),
        )
        sae_att.use_error_term = True
        sae_suite[f"{layer}-att"] = sae_att
        
        # Load MLP SAE
        mlp_id = f'blocks.{layer}.hook_mlp_out'
        sae_mlp = SAE.from_pretrained(
            release="gpt2-small-mlp-out-v5-32k",
            sae_id=mlp_id,
            device=str(device),
        )
        sae_mlp.use_error_term = True
        sae_suite[f"{layer}-mlp"] = sae_mlp
    
    print(f"Loaded {len(sae_suite)} SAEs")
    return sae_suite


def load_sae_explanations(base_dir: str, layers: List[int]) -> Dict[str, Dict[str, Any]]:
    """
    Load SAE feature explanations from JSONL files
    
    Args:
        base_dir: Base directory containing SAE data (e.g., "sae_data/gpt2-small/32k")
        layers: List of layer indices to load
    
    Returns:
        Dictionary mapping layer names to explanation data
    """
    print(f"Loading SAE explanations from {base_dir}...")
    base_path = Path(base_dir)
    explanations = {}
    
    for layer in layers:
        # Load attention explanations
        att_dir = base_path / f"{layer}-att_32k-oai" / "explanations"
        if att_dir.exists():
            explanations[f"{layer}-att"] = {}
            for jsonl_file in sorted(att_dir.glob("*.jsonl")):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            explanations[f"{layer}-att"][data["index"]] = data
        
        # Load MLP explanations
        mlp_dir = base_path / f"{layer}-mlp_32k-oai" / "explanations"
        if mlp_dir.exists():
            explanations[f"{layer}-mlp"] = {}
            for jsonl_file in sorted(mlp_dir.glob("*.jsonl")):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            explanations[f"{layer}-mlp"][data["index"]] = data
    
    print(f"Loaded explanations for {len(explanations)} layer types")
    return explanations


def load_graph_data_samples(graph_data_dir: str) -> List[Dict[str, Any]]:
    """
    Load samples directly from graph_data JSON files
    
    Args:
        graph_data_dir: Directory containing *.json files
    
    Returns:
        List of sample dictionaries containing all necessary graph data
    """
    graph_path = Path(graph_data_dir)
    samples = []
    
    if not graph_path.exists():
        print(f"Graph data directory not found: {graph_data_dir}")
        return samples
    
    json_files = sorted(graph_path.glob("gender_bias_*.json"))
    print(f"Loading data from {len(json_files)} graph_data files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Load all necessary data including circuit graph structure
                samples.append({
                    'idx': data['idx'],
                    'clean': data['clean'],
                    'corrupted': data['corrupted'],
                    'nodes_in_graph': np.array(data['nodes_in_graph']),
                    'in_graph': np.array(data['in_graph']),
                    'clean_label': data['clean_label'],
                    'corrupted_label': data['corrupted_label']
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    samples.sort(key=lambda x: x['idx'])
    print(f"Loaded {len(samples)} samples from graph_data")
    return samples


def get_circuit_sae_features(
    model: HookedTransformer,
    graph: Graph,
    sae_suite: Dict[str, SAE],
    clean_text: str,
    corrupted_text: str,
    nodes_in_graph: np.ndarray,
    top_k: int = 20
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Get top-k activated SAE features when running the circuit with activation patching
    Only processes nodes that are in the circuit.
    
    Args:
        model: HookedTransformer model
        graph: Graph object with circuit structure
        sae_suite: Dictionary of SAE objects
        clean_text: Clean input text
        corrupted_text: Corrupted input text
        nodes_in_graph: Boolean array indicating which nodes are in circuit (157 elements)
        top_k: Number of top features to return per layer
    
    Returns:
        Dictionary mapping layer names to lists of (feature_idx, activation_value) tuples
    """
    from einops import einsum
    
    # Tokenize inputs
    clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, [clean_text])
    corrupted_tokens, _, _, _ = tokenize_plus(model, [corrupted_text])
    
    # Prepare activation storage
    circuit_activations = {}
    
    def make_activation_capture_hook(layer_name):
        """Create hook to capture activations at a specific layer"""
        def hook_fn(activations, hook):
            # Store all token activations
            circuit_activations[layer_name] = activations.detach().clone()
        return hook_fn
    
    # Create hooks to capture activations at attention and MLP outputs
    # Only for nodes that are in the circuit
    # nodes_in_graph structure: [input(1), layer0_attn(12), layer0_mlp(1), ..., layer11_attn(12), layer11_mlp(1)]
    capture_hooks = []
    for layer_idx in range(model.cfg.n_layers):
        # Check attention heads for this layer
        # Attention head indices: 1 + layer_idx*13 to 1 + layer_idx*13 + 11
        att_start_idx = 1 + layer_idx * 13
        att_end_idx = att_start_idx + 12
        any_att_in_circuit = np.any(nodes_in_graph[att_start_idx:att_end_idx])
        
        if any_att_in_circuit:
            att_hook_name = f'blocks.{layer_idx}.hook_attn_out'
            capture_hooks.append((att_hook_name, make_activation_capture_hook(f'{layer_idx}-att')))
        
        # Check MLP for this layer
        # MLP index: 1 + layer_idx*13 + 12
        mlp_idx = 1 + layer_idx * 13 + 12
        if nodes_in_graph[mlp_idx]:
            mlp_hook_name = f'blocks.{layer_idx}.hook_mlp_out'
            capture_hooks.append((mlp_hook_name, make_activation_capture_hook(f'{layer_idx}-mlp')))
    
    # Now set up circuit evaluation hooks (similar to evaluate_graph)
    # Construct in_graph matrix
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    neuron_matrix = None
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
        neuron_matrix = 1 - neuron_matrix
    
    # Invert for corruption
    in_graph_matrix = 1 - in_graph_matrix
    
    # Create activation difference matrix
    batch_size = 1
    activation_difference = torch.zeros(
        (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
        device=model.cfg.device, 
        dtype=model.cfg.dtype
    )
    
    # Create forward hooks to capture activation differences
    from eap.graph import AttentionNode
    
    def make_fwd_hook_corrupted(src_index, node):
        def hook_fn(activations, hook):
            if isinstance(node, AttentionNode):
                # Attention nodes: (batch, pos, head, d_model) -> need to index head dimension
                activation_difference[:, :, src_index] += activations[:, :, node.head, :]
            else:
                # MLP/Input nodes: (batch, pos, d_model)
                activation_difference[:, :, src_index] += activations
        return hook_fn
    
    def make_fwd_hook_clean(src_index, node):
        def hook_fn(activations, hook):
            if isinstance(node, AttentionNode):
                # Attention nodes: (batch, pos, head, d_model) -> need to index head dimension
                activation_difference[:, :, src_index] -= activations[:, :, node.head, :]
            else:
                # MLP/Input nodes: (batch, pos, d_model)
                activation_difference[:, :, src_index] -= activations
        return hook_fn
    
    fwd_hooks_corrupted = []
    fwd_hooks_clean = []
    
    # Add hooks for all nodes
    for node in graph.nodes.values():
        if node.out_hook != '':
            src_idx = graph.forward_index(node, attn_slice=False)
            fwd_hooks_corrupted.append((node.out_hook, make_fwd_hook_corrupted(src_idx, node)))
            fwd_hooks_clean.append((node.out_hook, make_fwd_hook_clean(src_idx, node)))
    
    # Create input construction hooks for circuit evaluation
    def make_input_construction_hook(in_graph_vector, neuron_matrix_vec):
        def input_construction_hook(activations, hook):
            if neuron_matrix_vec is not None:
                update = einsum(
                    activation_difference[:, :, :len(in_graph_vector)],
                    neuron_matrix_vec[:len(in_graph_vector)],
                    in_graph_vector,
                    'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden'
                )
            else:
                update = einsum(
                    activation_difference[:, :, :len(in_graph_vector)],
                    in_graph_vector,
                    'batch pos previous hidden, previous ... -> batch pos ... hidden'
                )
            activations += update
            return activations
        return input_construction_hook
    
    input_construction_hooks = []
    for layer in range(model.cfg.n_layers):
        # Attention hooks
        if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{layer}.h0']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                hook_fn = make_input_construction_hook(
                    in_graph_matrix[:prev_index, bwd_index],
                    neuron_matrix
                )
                input_construction_hooks.append((node.qkv_inputs[i], hook_fn))
        
        # MLP hooks
        if graph.nodes[f'm{layer}'].in_graph:
            node = graph.nodes[f'm{layer}']
            prev_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            hook_fn = make_input_construction_hook(
                in_graph_matrix[:prev_index, bwd_index],
                neuron_matrix
            )
            input_construction_hooks.append((node.in_hook, hook_fn))
    
    # Run circuit evaluation with activation capture
    with torch.inference_mode():
        # First pass: get corrupted activations
        with model.hooks(fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)
        
        # Second pass: run circuit with patching and capture activations
        with model.hooks(fwd_hooks_clean + input_construction_hooks + capture_hooks):
            _ = model(clean_tokens, attention_mask=attention_mask)
    
    # Now pass captured activations through SAEs
    results = {}
    for layer_name, sae in sae_suite.items():
        if layer_name not in circuit_activations:
            continue
        
        activations = circuit_activations[layer_name]  # (batch, seq_len, d_model)
        batch_size, seq_len, d_model = activations.shape
        
        # Flatten to (batch*seq_len, d_model) for SAE encoding
        flat_acts = activations.reshape(-1, d_model)
        
        # Pass through SAE
        with torch.no_grad():
            sae_features = sae.encode(flat_acts)  # (batch*seq_len, n_features)
        
        # Reshape back to (batch, seq_len, n_features)
        sae_features = sae_features.reshape(batch_size, seq_len, -1)
        
        # Get top-k features for each token position
        token_features_list = []
        for token_idx in range(seq_len):
            token_features = sae_features[0, token_idx, :]  # (n_features,)
            top_values, top_indices = torch.topk(token_features, top_k)
            
            token_features_list.append([
                (int(idx), float(val))
                for idx, val in zip(top_indices.cpu(), top_values.cpu())
            ])
        
        results[layer_name] = token_features_list  # List of lists, one per token
    
    return results


def analyze_sample(
    model: HookedTransformer,
    sae_suite: Dict[str, SAE],
    explanations: Dict[str, Dict[str, Any]],
    sample_data: Dict[str, Any],
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Analyze a single gender bias sample using circuit evaluation
    
    Args:
        model: HookedTransformer model
        sae_suite: Dictionary of SAE objects
        explanations: SAE feature explanations
        sample_data: Dictionary containing circuit graph data and texts
        top_k: Number of top features to analyze
    
    Returns:
        Dictionary containing analysis results
    """
    clean_text = sample_data['clean']
    corrupted_text = sample_data['corrupted']
    sample_idx = sample_data['idx']
    
    # Create Graph object and load circuit structure
    graph = Graph.from_model(model.cfg)
    
    # Set which nodes and edges are in the circuit
    graph.nodes_in_graph[:] = torch.tensor(sample_data['nodes_in_graph'], dtype=torch.bool)
    graph.in_graph[:] = torch.tensor(sample_data['in_graph'], dtype=torch.bool).reshape(graph.in_graph.shape)
    
    # Prune graph to make it valid
    graph.prune()
    
    # Get SAE features from circuit activations
    circuit_features = get_circuit_sae_features(
        model=model,
        graph=graph,
        sae_suite=sae_suite,
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        nodes_in_graph=sample_data['nodes_in_graph'],
        top_k=top_k
    )
    
    # Enrich features with explanations
    # circuit_features is now {layer_name: [[token0_features], [token1_features], ...]}
    enriched_features = {}
    for layer_name, tokens_feature_list in circuit_features.items():
        enriched_tokens = []
        
        # Process each token's features
        for token_idx, token_features in enumerate(tokens_feature_list):
            enriched_token_features = []
            
            for feature_idx, activation_val in token_features:
                feature_info = {
                    'layer': layer_name,
                    'feature_index': feature_idx,
                    'activation': activation_val,
                    'description': None,
                    # 'max_activation': None
                }
                
                # Look up explanation
                if layer_name in explanations and str(feature_idx) in explanations[layer_name]:
                    exp_data = explanations[layer_name][str(feature_idx)]
                    feature_info['description'] = exp_data.get('description', 'No description')
                    # feature_info['max_activation'] = exp_data.get('maxActApprox', None)
                
                enriched_token_features.append(feature_info)
            
            enriched_tokens.append(enriched_token_features)
        
        enriched_features[layer_name] = enriched_tokens
    
    # Decode labels from token IDs to words
    clean_label_token = sample_data['clean_label']
    corrupted_label_token = sample_data['corrupted_label']
    clean_label_word = model.tokenizer.decode([clean_label_token])
    corrupted_label_word = model.tokenizer.decode([corrupted_label_token])
    
    return {
        'sample_idx': sample_idx,
        'clean_text': clean_text,
        'corrupted_text': corrupted_text,
        'clean_label': clean_label_word,
        # 'clean_label_token': clean_label_token,
        'corrupted_label': corrupted_label_word,
        # 'corrupted_label_token': corrupted_label_token,
        'circuit_features': enriched_features
    }


def main():
    """Main analysis function"""
    # Set seed for reproducibility
    set_seed(2025)
    
    # Configuration
    LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # All GPT-2 Small layers
    TOP_K = 5
    SAE_DATA_DIR = 'sae_data/gpt2-small/32k'
    GRAPH_DATA_DIR = 'graph_data/top100'
    OUTPUT_DIR = 'gender_bias_sae_analysis_results/top100'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load samples from graph_data
    samples = load_graph_data_samples(GRAPH_DATA_DIR)
    if not samples:
        print("No graph data files found. Exiting.")
        return
    
    # Load model
    model = load_gpt2_subject()
    
    # Load SAE suite
    sae_suite = load_sae_suite(device, LAYERS)
    
    # Load SAE explanations
    explanations = load_sae_explanations(SAE_DATA_DIR, LAYERS)
    
    # Analyze each sample
    print(f"\nAnalyzing {len(samples)} samples with circuit evaluation...")
    all_results = []
    
    for sample in tqdm(samples):
        idx = sample['idx']
        
        result = analyze_sample(
            model=model,
            sae_suite=sae_suite,
            explanations=explanations,
            sample_data=sample,
            top_k=TOP_K
        )
        
        all_results.append(result)
        
        # Save individual result
        output_file = os.path.join(OUTPUT_DIR, f'sample_{idx}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"\n✓ Analysis complete! Results saved to {OUTPUT_DIR}/")
    print(f"  - Individual samples: {OUTPUT_DIR}/sample_*.json")


if __name__ == "__main__":
    main()
