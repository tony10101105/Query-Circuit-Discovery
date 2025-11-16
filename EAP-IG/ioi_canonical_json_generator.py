import torch
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set


# cfg = {'NTK_by_parts_factor': 8.0,                                                                                                                             
#  'NTK_by_parts_high_freq_factor': 4.0,                                                                                                                   
#  'NTK_by_parts_low_freq_factor': 1.0,                                                                                                                    
#  'NTK_original_ctx_len': 8192,                                                                                                                           
#  'act_fn': 'gelu_new',                                                                                                                                   
#  'attention_dir': 'causal',                                                                                                                              
#  'attn_only': False,                                                                                                                                     
#  'attn_scale': 8.0,                                                                                                                                      
#  'attn_scores_soft_cap': -1.0,                                                                                                                           
#  'attn_types': None,                                                                                                                                     
#  'checkpoint_index': None,                                                                                                                               
#  'checkpoint_label_type': None,                                                                                                                          
#  'checkpoint_value': None,                                                                                                                               
#  'd_head': 64,                                                                                                                                           
#  'd_mlp': 3072,                                                                                                                                          
#  'd_model': 768,                                                                                                                                         
#  'd_vocab': 50257,                                                                                                                                       
#  'd_vocab_out': 50257,
#  'dtype': torch.float32,                   
#  'decoder_start_token_id': None,       
#  'default_prepend_bos': True,          
#  'device': 'cuda',                                  
#  'eps': 1e-05,                         
#  'experts_per_token': None,            
#  'final_rms': False,                   
#  'from_checkpoint': False,             
#  'gated_mlp': False,                   
#  'init_mode': 'gpt2',                  
#  'init_weights': False,                
#  'initializer_range': 0.02886751345948129,                                     
#  'load_in_4bit': False,                
#  'model_name': 'gpt2',                 
#  'n_ctx': 1024,                        
#  'n_devices': 1,                       
#  'n_heads': 12,                        
#  'n_key_value_heads': None,            
#  'n_layers': 12,                       
#  'n_params': 84934656,                 
#  'normalization_type': 'LNPre',        
#  'num_experts': None,                  
#  'original_architecture': 'GPT2LMHeadModel',                                   
#  'output_logits_soft_cap': -1.0,       
#  'parallel_attn_mlp': False,           
#  'positional_embedding_type': 'standard',                                      
#  'post_embedding_ln': False,           
#  'relative_attention_max_distance': None,                                      
#  'relative_attention_num_buckets': None, 
#   'rotary_adjacent_pairs': False,                                                                                                                              
#  'rotary_base': 10000,                                                                                                                                        
#  'rotary_dim': None,                                                                                                                                          
#  'scale_attn_by_inverse_layer_idx': False,                                                                                                                    
#  'seed': None,                                                                                                                                                
#  'tie_word_embeddings': False,                                                                                                                                
#  'tokenizer_name': 'gpt2',                                                                                                                                    
#  'tokenizer_prepends_bos': False,                                                                                                                             
#  'trust_remote_code': False,                                                                                                                                  
#  'ungroup_grouped_query_attention': False,                                                                                                                    
#  'use_NTK_by_parts_rope': False,                                                                                                                              
#  'use_attn_in': False,                                                                                                                                        
#  'use_attn_result': True,                                                                                                                                     
#  'use_attn_scale': True,                                                                                                                                      
#  'use_hook_mlp_in': True,                                                                                                                                     
#  'use_hook_tokens': False,                                                                                                                                    
#  'use_local_attn': False,                                                                                                                                     
#  'use_normalization_before_and_after': False,                                                                                                                 
#  'use_split_qkv_input': True,                                                                                                                                 
#  'window_size': None
#  }



# --- Read base file ---
with open("preprocessed_circuit/graph_base.json", "r") as f:
    base_data = json.load(f)
    

# --- Define circuit structure ---

@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: Tuple[str, ...]

IOI_CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

special_connections: Set[Conn] = {
    Conn("input", "previous token", ("q", "k", "v")),
    Conn("input", "duplicate token", ("q", "k", "v")),
    Conn("input", "s2 inhibition", ("q",)),
    Conn("input", "negative", ("k", "v")),
    Conn("input", "name mover", ("k", "v")),
    Conn("input", "backup name mover", ("k", "v")),
    Conn("previous token", "induction", ("k", "v")),
    Conn("induction", "s2 inhibition", ("k", "v")),
    Conn("duplicate token", "s2 inhibition", ("k", "v")),
    Conn("s2 inhibition", "negative", ("q",)),
    Conn("s2 inhibition", "name mover", ("q",)),
    Conn("s2 inhibition", "backup name mover", ("q",)),
    Conn("negative", "OUTPUT", ()),
    Conn("name mover", "OUTPUT", ()),
    Conn("backup name mover", "OUTPUT", ()),
}

# --- Build the graph ---
t_to_name = {}
for name, heads in IOI_CIRCUIT.items():
    for t in heads:
        t_to_name[t] = name

present_heads = set(t_to_name.keys())
conn_map = {(c.inp, c.out): "".join(c.qkv) for c in special_connections}

def node_name(ntype: str, layer: int = None, head: int = None):
    if ntype == "MLP":
        return f"m{layer}"
    elif ntype == "logits":
        return "logits"
    elif ntype == "input":
        return "input"
    elif ntype == "attn_out":
        return f"a{layer}.h{head}"
    elif ntype.startswith("attn_"):
        return f"a{layer}.h{head}<{ntype[-1]}>"
    else:
        raise ValueError(f"Unknown node type: {ntype}")

def add_edge(src: str, dst: str):
    base_data['edges'][f"{src}->{dst}"] = {"score": 0.0, "in_graph": True}
    if '>' in dst:
        dst = dst[:-3]
    base_data['nodes'][src] = {"in_graph": True}
    base_data['nodes'][dst] = {"in_graph": True}


# Add input -> logits
add_edge("input", "logits")

# MLPs
for i in range(base_data['cfg']["n_layers"]):
    add_edge("input", node_name("MLP", i))
    add_edge(node_name("MLP", i), "logits")
    for j in range(i):
        add_edge(node_name("MLP", j), node_name("MLP", i))

# Heads
for (layer, head) in present_heads:
    group = t_to_name[(layer, head)]
    attn_out = node_name("attn_out", layer, head)

    # input -> qkv
    for letter in "qkv":
        if ("input", group) in conn_map and letter in conn_map[("input", group)]:
            add_edge("input", node_name(f"attn_{letter}", layer, head))

    # attn_out -> logits
    if (group, "OUTPUT") in conn_map:
        add_edge(attn_out, "logits")

    # MLP -> qkv
    for i in range(layer):
        for letter in "qkv":
            add_edge(node_name("MLP", i), node_name(f"attn_{letter}", layer, head))

    # attn_out -> MLP
    for i in range(layer, base_data['cfg']["n_layers"]):
        add_edge(attn_out, node_name("MLP", i))


    # attn_out -> attn_out
    # for (layer, head) in present_heads:
    src_group = t_to_name[(layer, head)]
    attn_out = node_name("attn_out", layer, head)
    for (layer2, head2) in present_heads:
        dst_group = t_to_name[(layer2, head2)]
        if (src_group, dst_group) in conn_map:
            for letter in "qkv":
                if letter in conn_map[(src_group, dst_group)]:
                    add_edge(attn_out, node_name(f"attn_{letter}", layer2, head2))


with open("preprocessed_circuit/ioi_gpt2_canonical_circuit.json", "w") as f:
    json.dump(base_data, f, indent=2)