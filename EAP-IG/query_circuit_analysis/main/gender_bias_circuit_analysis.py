"""
Analyze Gender Bias Circuit - Sample 107
Input: "The doctor said that"
Output: "he" (biased towards male pronoun)
"""

import os as _os, sys as _sys
_base = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_base)
_sys.path.insert(0, _base)

import json
from pathlib import Path

# Load the circuit graph and SAE features
with open('graph_data/top100/gender_bias_107_top100.json', 'r') as f:
    graph_data = json.load(f)

with open('gender_bias_sae_analysis_results/top100/sample_107.json', 'r') as f:
    sae_features = json.load(f)

# Node index mapping: 
# Index 0: input
# Indices 1 + L*13 + H: layer L attention head H (H=0-11)
# Indices 1 + L*13 + 12: layer L MLP
def index_to_node(idx):
    """Convert node index to node name"""
    if idx == 0:
        return 'input'
    idx -= 1
    layer = idx // 13
    pos = idx % 13
    if pos < 12:
        return f'a{layer}.h{pos}'
    else:
        return f'm{layer}'

def node_to_index(node_name):
    """Convert node name to index"""
    if node_name == 'input':
        return 0
    if node_name.startswith('a'):
        # Format: a{layer}.h{head}
        layer, head = node_name[1:].split('.h')
        return 1 + int(layer) * 13 + int(head)
    elif node_name.startswith('m'):
        # Format: m{layer}
        layer = int(node_name[1:])
        return 1 + layer * 13 + 12
    return None

# Get active nodes
nodes_in_graph = graph_data['nodes_in_graph']
active_indices = [i for i, active in enumerate(nodes_in_graph) if active]
active_nodes = [index_to_node(i) for i in active_indices]

print("="*80)
print("GENDER BIAS CIRCUIT ANALYSIS - Sample 107")
print("="*80)
print(f"\nInput: '{sae_features['clean_text']}'")
print(f"Output: '{sae_features['clean_label']}' (biased - should be gender-neutral)")
print(f"Corrupted input: '{sae_features['corrupted_text']}' → '{sae_features['corrupted_label']}'")
print(f"\nActive nodes in circuit: {len(active_nodes)}")
print(f"Nodes: {', '.join(sorted(active_nodes, key=lambda x: node_to_index(x)))}")

# Analyze features in each active node
print("\n" + "="*80)
print("INFORMATION FLOW ANALYSIS")
print("="*80)

# Map diagram nodes to layers
node_layers = {}
for node in active_nodes:
    if node == 'input':
        node_layers.setdefault('input', []).append(node)
    elif node.startswith('a'):
        layer = int(node.split('.')[0][1:])
        node_layers.setdefault(f'layer_{layer}', []).append(node)
    elif node.startswith('m'):
        layer = int(node[1:])
        node_layers.setdefault(f'layer_{layer}', []).append(node)

# Analyze key nodes with gender/doctor features
print("\n📍 EARLY LAYERS (0-2): Detecting 'doctor' and processing context")
print("-"*80)

# Layer 0 - where "doctor" token is processed
if '0-att' in sae_features['circuit_features']:
    print("\n🔍 Layer 0 Attention (a0.h3, a0.h4, a0.h10):")
    for token_idx, token_features in enumerate(sae_features['circuit_features']['0-att']):
        if token_features:
            print(f"\n   Token position {token_idx}:")
            # Look for doctor/gender related features
            for feat in token_features[:5]:  # Show top 5
                if feat.get('description') and any(word in feat['description'].lower() 
                    for word in ['doctor', 'medical', 'professional', 'women', 'gender', 'male']):
                    print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                    print(f"     Activation: {feat['activation']:.3f}")

if '0-mlp' in sae_features['circuit_features']:
    print("\n🔍 Layer 0 MLP (m0):")
    for token_idx, token_features in enumerate(sae_features['circuit_features']['0-mlp']):
        if token_features:
            print(f"\n   Token position {token_idx}:")
            for feat in token_features[:5]:
                if feat.get('description') and any(word in feat['description'].lower() 
                    for word in ['doctor', 'medical', 'said', 'the']):
                    print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                    print(f"     Activation: {feat['activation']:.3f}")

# Layer 1-2 MLPs
for layer in [1, 2]:
    if f'{layer}-mlp' in sae_features['circuit_features']:
        print(f"\n🔍 Layer {layer} MLP (m{layer}):")
        for token_idx, token_features in enumerate(sae_features['circuit_features'][f'{layer}-mlp']):
            if token_features:
                print(f"\n   Token position {token_idx}:")
                for feat in token_features[:5]:
                    if feat.get('description') and any(word in feat['description'].lower() 
                        for word in ['doctor', 'medical', 'said', 'professional']):
                        print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                        print(f"     Activation: {feat['activation']:.3f}")

print("\n" + "="*80)
print("📍 MIDDLE LAYERS (3-7): Building gender associations")
print("-"*80)

for layer in [3, 4, 5, 6, 7]:
    layer_key_att = f'{layer}-att'
    layer_key_mlp = f'{layer}-mlp'
    
    has_features = False
    if layer_key_att in sae_features['circuit_features']:
        for token_features in sae_features['circuit_features'][layer_key_att]:
            if token_features:
                has_features = True
                break
    if layer_key_mlp in sae_features['circuit_features']:
        for token_features in sae_features['circuit_features'][layer_key_mlp]:
            if token_features:
                has_features = True
                break
    
    if has_features:
        print(f"\n🔍 Layer {layer}:")
        
        if layer_key_att in sae_features['circuit_features']:
            for token_idx, token_features in enumerate(sae_features['circuit_features'][layer_key_att]):
                if token_features:
                    print(f"   Attention - Token {token_idx}:")
                    for feat in token_features[:3]:
                        if feat.get('description') and any(word in feat['description'].lower() 
                            for word in ['doctor', 'medical', 'male', 'female', 'gender', 'women', 'men', 'he', 'she', 'said']):
                            print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                            print(f"     Activation: {feat['activation']:.3f}")
        
        if layer_key_mlp in sae_features['circuit_features']:
            for token_idx, token_features in enumerate(sae_features['circuit_features'][layer_key_mlp]):
                if token_features:
                    print(f"   MLP - Token {token_idx}:")
                    for feat in token_features[:3]:
                        if feat.get('description') and any(word in feat['description'].lower() 
                            for word in ['doctor', 'medical', 'male', 'female', 'gender', 'women', 'men', 'he', 'she', 'said']):
                            print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                            print(f"     Activation: {feat['activation']:.3f}")

print("\n" + "="*80)
print("📍 LATE LAYERS (8-11): Finalizing gender prediction")
print("-"*80)

for layer in [8, 9, 10, 11]:
    layer_key_att = f'{layer}-att'
    layer_key_mlp = f'{layer}-mlp'
    
    has_features = False
    if layer_key_att in sae_features['circuit_features']:
        for token_features in sae_features['circuit_features'][layer_key_att]:
            if token_features:
                has_features = True
                break
    if layer_key_mlp in sae_features['circuit_features']:
        for token_features in sae_features['circuit_features'][layer_key_mlp]:
            if token_features:
                has_features = True
                break
    
    if has_features:
        print(f"\n🔍 Layer {layer}:")
        
        if layer_key_att in sae_features['circuit_features']:
            for token_idx, token_features in enumerate(sae_features['circuit_features'][layer_key_att]):
                if token_features:
                    print(f"   Attention - Token {token_idx}:")
                    for feat in token_features[:5]:
                        if feat.get('description'):
                            print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                            print(f"     Activation: {feat['activation']:.3f}")
        
        if layer_key_mlp in sae_features['circuit_features']:
            for token_idx, token_features in enumerate(sae_features['circuit_features'][layer_key_mlp]):
                if token_features:
                    print(f"   MLP - Token {token_idx}:")
                    for feat in token_features[:5]:
                        if feat.get('description'):
                            print(f"   • Feature {feat['feature_index']}: {feat['description']}")
                            print(f"     Activation: {feat['activation']:.3f}")

print("\n" + "="*80)
print("SUMMARY: Why the circuit outputs 'he' instead of 'she'")
print("="*80)
print("""
Based on the SAE feature analysis, the gender bias emerges through:

1. DOCTOR DETECTION (Layers 0-2):
   - Early layers detect "doctor" as a medical professional
   - Features activate for medical contexts and professional roles

2. IMPLICIT GENDER ASSOCIATION (Layers 3-7):
   - Middle layers build associations between "doctor" and male gender
   - Statistical bias from training data: doctors historically male-dominated
   - Features for "said" and reported speech link to attribution patterns

3. PRONOUN PREDICTION (Layers 8-11):
   - Late layers consolidate the gender prediction
   - Circuit strongly predicts "he" based on the accumulated gender association
   - The bias is encoded in the feature activations throughout the network

The circuit essentially learns: "doctor" → "male professional" → "he"
This reflects societal biases present in the training data.
""")
