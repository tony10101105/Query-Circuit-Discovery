"""
read the base json file and generate the json file of canonical circuit of gpt2-small
"""


import torch
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `TorchIndex`s are essentially indices that say which part of the tensor is being affected. 

    EXAMPLES: Initialise [:, :, 3] with TorchIndex([None, None, 3]) and [:] with TorchIndex([None])    

    Also we want to be able to call e.g `my_dictionary[my_torch_index]` hence the hashable tuple stuff
    
    Note: ideally this would be integrated with transformer_lens.utils.Slice in future; they are accomplishing similar but different things"""

    def __init__(
        self, 
        list_of_things_in_tuple: List,
    ):
        # check correct types
        for arg in list_of_things_in_tuple:
            if type(arg) in [type(None), int]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        # make an object that can be indexed into a tensor
        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])

        # make an object that can be hashed (so used as a dictionary key)
        self.hashable_tuple = tuple(list_of_things_in_tuple)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    # some graphics things

    def __repr__(self, use_actual_colon=True) -> str: # graphviz, an old library used to dislike actual colons in strings, but this shouldn't be an issue anymore
        ret = "["
        for idx, x in enumerate(self.hashable_tuple):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += ":" if use_actual_colon else "COLON"
            elif type(x) == int:
                ret += str(x)
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret

    def graphviz_index(self, use_actual_colon=True) -> str:
        return self.__repr__(use_actual_colon=use_actual_colon)

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


# --- Load base graph template ---
with open("preprocessed_circuit/graph_base.json", "r") as f:
    base_data = json.load(f)

@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: Tuple[str, ...]

CIRCUIT = {
    "0305": [(0, 3), (0, 5)],
    "01": [(0, 1)],
    "MEARLY": [(0, None), (1, None), (2, None), (3, None)],
    "AMID": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)],
    "MLATE": [(8, None), (9, None), (10, None), (11, None)],
}

SPECIAL_CONNECTIONS: Set[Conn] = {
    Conn("input", "0305", ("q", "k", "v")),
    Conn("input", "01", ("q", "k", "v")),
    Conn("input", "MEARLY", ("q", "k", "v")),
    Conn("0305", "AMID", ("q", "k", "v")),
    Conn("01", "MEARLY", ("q", "k", "v")),
    Conn("01", "AMID", ("q", "k", "v")),
    Conn("MEARLY", "AMID", ("q", "k", "v")),
    Conn("AMID", "MLATE", ("q", "k", "v")),
    Conn("AMID", "OUTPUT", ()),
    Conn("MLATE", "OUTPUT", ()),
}

connected_pairs = [
    ("01", "MEARLY"),
    ("01", "AMID"),
    ("0305", "AMID"),
    ("MEARLY", "AMID"),
    ("AMID", "MLATE"),
]

# --- Helper Functions ---
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

def tuple_to_hooks(layer_idx, head_idx, outp=False):
    if outp:
        if head_idx is None:
            return [(f"blocks.{layer_idx}.hook_mlp_out", TorchIndex([None]))]
        else:
            return [(f"blocks.{layer_idx}.attn.hook_result", TorchIndex([None, None, head_idx]))]

    else:
        if head_idx is None:
            return [(f"blocks.{layer_idx}.hook_mlp_in", TorchIndex([None]))]
        else:
            ret = []
            for letter in "qkv":
                ret.append((f"blocks.{layer_idx}.hook_{letter}_input", TorchIndex([None, None, head_idx])))
            return ret
    
    
# --- Build Graph ---
add_edge("input", "logits")

# Attach input connections for early groups
for GROUP in ["0305", "01", "MEARLY"]:
    for layer, head in CIRCUIT[GROUP]:
        # for hook_name, idx in tuple_to_hooks(layer, head, outp=False):
            # Connect model input residual to head/MLP input
            # add_edge("input", node_name(hook_name.split('.')[-3], layer, head) if 'attn' in hook_name else node_name("MLP", layer))
        if head is None: # MLP
            add_edge("input", node_name("MLP", layer))
        else: # attention
            for letter in "qkv":
                add_edge("input", node_name(f"attn_{letter}", layer, head))

# Attach output connections for later groups
for GROUP in ["AMID", "MLATE"]:
    for layer, head in CIRCUIT[GROUP]:
        add_edge(node_name("attn_out", layer, head) if head is not None else node_name("MLP", layer), "logits")

# Interconnect MLPs within each group (only MLP groups)
for GROUP, nodes in CIRCUIT.items():
    if nodes[0][1] is not None:
        continue
    for i1, _ in nodes:
        for i2, _ in nodes:
            if i1 >= i2:
                continue
            add_edge(node_name("MLP", i1), node_name("MLP", i2))

# Connect groups according to connected_pairs
for GROUP1, GROUP2 in connected_pairs:
    for i1, j1 in CIRCUIT[GROUP1]:
        for i2, j2 in CIRCUIT[GROUP2]:
            # if GROUP1 == '01' and GROUP2 == 'MEARLY':
            #     print(f'{j1}, {j2}')
            if i1 > i2 or (i1 == i2 and j1 is None and j2 is not None): # ?
                continue
            # Connect output of GROUP1 to input of GROUP2
            # for src_hook, src_idx in tuple_to_hooks(i1, j1, outp=True):
                # for dst_hook, dst_idx in tuple_to_hooks(i2, j2, outp=False):
            for letter in "qkv":
                add_edge(
                    node_name("attn_out", i1, j1) if j1 is not None else node_name("MLP", i1),
                    node_name(f"attn_{letter}", i2, j2) if j2 is not None else node_name("MLP", i2)
                )

# # Connect qkv flows within heads
# for layer, head in sum(CIRCUIT.values(), start=[]):
#     if head is None:
#         continue
#     for letter in "qkv":
#         # attn_out -> hook_{letter}
#         add_edge(
#             node_name("attn_out", layer, head),
#             node_name(f"attn_{letter}", layer, head)
#         )

with open("preprocessed_circuit/gt_gpt2_canonical_circuit.json", "w") as f:
    json.dump(base_data, f, indent=2)