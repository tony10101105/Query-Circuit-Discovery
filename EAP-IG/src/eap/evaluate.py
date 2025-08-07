from typing import Callable, List, Union, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations, no_tokenize_plus
from .graph import Graph, AttentionNode


def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                   metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]], 
                   quiet=False, intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   intervention_dataloader: Optional[DataLoader]=None, skip_clean:bool=True, 
                   hook_rep:bool=False, hook_layer:bool=False, hook_pattern:bool=False, induction=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune 
        beforehand to make sure your circuit is valid.

    Args:
        model (HookedTransformer): The model to run the circuit on 
        graph (Graph): The circuit to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): The metric(s) to evaluate with respect to
        quiet (bool, optional): Whether to silence the tqdm progress bar. Defaults to False.
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): Which ablation to evaluate with respect to. 
            'patching' is an interchange intervention; mean-positional takes the positional mean over the given dataset. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): The dataset to take the mean over. Must be set if intervention is mean or mean-positional. Defaults to None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of faithfulness scores; if a list, each list entry 
            corresponds to a metric in the input list
    """
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
        
    assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"
    
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    # This step cleans up the graph, removing components until it's fully connected
    graph.prune()

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    
    # same thing but for neurons
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use it as a mask to specify 
    # which edges we want to corrupt
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix
        
    if model.cfg.use_normalization_before_and_after: # default false
        # If the model also normalizes the outputs of attention heads, we'll need to take that into account when evaluating the graph.
        attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device='cuda', dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    # Ours
    all_node_rep = [] if hook_rep else None
    all_layer_rep = [] if hook_layer else None
    all_node_pattern = [] if hook_pattern else None

    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph 
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # print('hook: ', hook.name)
            # Case where layernorm is applied after attention (gemma only)
            if model.cfg.use_normalization_before_and_after: # default false
                activation_differences = activation_matrix[0] - activation_matrix[1]
                
                # get the clean outputs of the attention heads that came before
                clean_attention_results = einsum(activation_matrix[1, :, :, :len(in_graph_vector)], 
                                                 attention_head_mask[:len(in_graph_vector)], 
                                                 'batch pos previous hidden, previous layer -> batch pos layer hidden')
                
                # get the update corresponding to non-attention heads, and the difference between clean and corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                  neuron_matrix[:len(in_graph_vector)], 
                                                  in_graph_vector, 
                                                  non_attention_head_mask[:len(in_graph_vector)], 
                                                  'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                            neuron_matrix[:len(in_graph_vector)], 
                                                            in_graph_vector, 
                                                            attention_head_mask[:len(in_graph_vector)], 
                                                            'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden')                    
                else:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                  in_graph_vector, 
                                                  non_attention_head_mask[:len(in_graph_vector)], 
                                                  'batch pos previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                            in_graph_vector, 
                                                            attention_head_mask[:len(in_graph_vector)], 
                                                            'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden')
                
                # add the biases to the attention results, and compute the corrupted attention results using the difference
                # we process all the attention heads at once; this is how we can tell if we're doing that
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                
                # pass both the clean and corrupted attention results through the layernorm and 
                # add the difference to the update
                update = non_attention_update
                valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])                        
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])
                        
            else:
                # In the non-gemma case, things are easy!
                activation_differences = activation_matrix
                # print('activation_differences: ', activation_differences.shape) # torch.Size([10, 21, 157, 768])
                # The ... here is to account for a potential head dimension, when constructing a whole attention layer's input
                if neuron_matrix is not None:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                else:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous ... -> batch pos ... hidden')
                    # print('in_graph_vector: ', in_graph_vector.shape) # torch.Size([157])
                    # print('b: ', activation_differences[:, :, :len(in_graph_vector)].shape) # torch.Size([10, 21, 157, 768]) 

            activations += update # torch.Size([bs, pos, 1 or 12, d_model])
            return activations
        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in the layer is in the graph, just construct the input for the entire layer
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)) and \
                not (neuron_matrix is None and all(parent_edge.in_graph for head in range(model.cfg.n_heads) for parent_edge in graph.nodes[f'a{layer}.h{head}'].parent_edges)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))
                    
            # add MLP hook if MLP in graph
            if graph.nodes[f'm{layer}'].in_graph and \
                not (neuron_matrix is None and all(parent_edge.in_graph for parent_edge in graph.nodes[f'm{layer}'].parent_edges)):
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))
                    
        # Always add the logits hook
        if not (neuron_matrix is None and all(parent_edge.in_graph for parent_edge in graph.nodes['logits'].parent_edges)):
            node = graph.nodes['logits']
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks
    
    # convert metrics to list if it's not already
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]

    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    max_n_pos = -1
    for clean, corrupted, label in dataloader:
        # clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        # corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        if not induction:
            clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
            corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        else: # when induction, the input is list of tokens, not string
            clean_tokens, attention_mask, input_lengths, n_pos = no_tokenize_plus(model, clean)
            corrupted_tokens, _, _, _ = no_tokenize_plus(model, corrupted)
        max_n_pos = max(max_n_pos, n_pos)
        # fwd_hooks_corrupted adds in corrupted acts to activation_difference
        # fwd_hooks_clean subtracts out clean acts from activation_difference
        # activation difference is of size (batch, pos, src_nodes, hidden)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference, node_representation, layer_representation, node_pattern = make_hooks_and_matrices(model, graph, len(clean), n_pos, None, hook_rep=hook_rep, hook_layer=hook_layer, hook_pattern=hook_pattern)
        
        input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            else:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                if 'mean' in intervention:
                    activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = None if skip_clean else model(clean_tokens, attention_mask=attention_mask)
            with model.hooks(fwd_hooks_clean + input_construction_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                if hook_rep or hook_layer or hook_pattern:
                    mask = torch.arange(n_pos, device=activation_difference.device)[None, :] < input_lengths[:, None] # [bs, n_pos]
                    if hook_rep:
                        node_representation = node_representation * mask[:, :, None, None] # [bs, n_pos, node_num, d_model]
                        all_node_rep.append(node_representation)
                    if hook_layer:
                        layer_representation = layer_representation * mask[:, :, None, None] # [bs, n_pos, n_layer, d_model]
                        all_layer_rep.append(layer_representation)
                    if hook_pattern:
                        node_pattern = node_pattern * mask[:, None, :, None] # [bs, node_num, n_pos, n_pos]
                        all_node_pattern.append(node_pattern)

        for i, metric in enumerate(metrics):
            r = metric(logits, clean_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    if len(results) == 1:
        results = results[0]

    if hook_rep:
        all_node_rep_padded = []
        for rep in all_node_rep:
            current_size = rep.size(1)
            padding_needed = max(0, max_n_pos - current_size)
            rep = F.pad(rep, (0, 0, 0, 0, 0, padding_needed))
            all_node_rep_padded.append(rep)
        all_node_rep = torch.cat(all_node_rep_padded, dim=0)

    if hook_layer:
        all_layer_rep_padded = []
        for rep in all_layer_rep:
            current_size = rep.size(1)
            padding_needed = max(0, max_n_pos - current_size)
            rep = F.pad(rep, (0, 0, 0, 0, 0, padding_needed))
            all_layer_rep_padded.append(rep)
        all_layer_rep = torch.cat(all_layer_rep_padded, dim=0)

    if hook_pattern:
        all_node_pattern_padded = []
        for pattern in all_node_pattern:
            current_size = pattern.size(-2)
            padding_needed = max(0, max_n_pos - current_size)
            pattern = F.pad(pattern, (0, padding_needed, 0, padding_needed))
            all_node_pattern_padded.append(pattern)
        all_node_pattern = torch.cat(all_node_pattern_padded, dim=0)

    return results, all_node_rep, all_layer_rep, all_node_pattern


def evaluate_baseline(model: HookedTransformer, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], 
                      run_corrupted=False, quiet=False, induction=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluates the model on the given dataloader, without any intervention. This is useful for computing the baseline performance of the model.

    Args:
        model (HookedTransformer): The model to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (List[Callable[[Tensor], Tensor]]): The metrics to evaluate with respect to
        run_corrupted (bool, optional): Whether to evaluate on corrupted examples instead. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of performance scores; if a list, each list entry corresponds to a metric in the input list
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    
    results = [[] for _ in metrics]
    if not quiet:
        dataloader = tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, _ = tokenize_plus(model, clean)
        corrupted_tokens, attention_mask_corrupted, input_lengths_corrupted, _ = tokenize_plus(model, corrupted)

        def input_perturbation_hook(var: float):
            def hook_fn(activations, hook):
                noise = torch.randn_like(activations) * var
                new_input = activations + noise
                new_input.requires_grad = True 
                return new_input
            return hook_fn
            
        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask_corrupted)
            # with model.hooks(fwd_hooks=[('hook_embed', input_perturbation_hook(0.01))]): # 3.731813907623291 # 0.01 std is good
            logits = model(clean_tokens, attention_mask=attention_mask)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths_corrupted, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results
