import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--sample_idx', type=int, default=107)
parser.add_argument('--topn', type=int, default=100)
parser.add_argument('--graph_data_dir', type=str, default='gender_bias_sae_analysis/graph_data')
parser.add_argument('--sae_data_dir', type=str, default='gender_bias_sae_analysis/sae_analysis_data')
args = parser.parse_args()

graph_path = os.path.join(args.graph_data_dir, f'top{args.topn}', f'gender_bias_{args.sample_idx}_top{args.topn}.json')
sae_path = os.path.join(args.sae_data_dir, f'top{args.topn}', f'sample_{args.sample_idx}.json')

with open(graph_path, 'r') as f:
    graph_data = json.load(f)

with open(sae_path, 'r') as f:
    sae_features = json.load(f)


def index_to_node(idx):
    if idx == 0:
        return 'input'
    idx -= 1
    layer, pos = divmod(idx, 13)
    return f'a{layer}.h{pos}' if pos < 12 else f'm{layer}'


def node_to_index(node_name):
    if node_name == 'input':
        return 0
    if node_name.startswith('a'):
        layer, head = node_name[1:].split('.h')
        return 1 + int(layer) * 13 + int(head)
    if node_name.startswith('m'):
        return 1 + int(node_name[1:]) * 13 + 12
    return None


def has_layer_features(layer):
    for suffix in ('att', 'mlp'):
        key = f'{layer}-{suffix}'
        if key in sae_features['circuit_features'] and any(sae_features['circuit_features'][key]):
            return True
    return False


def print_layer_features(layer, keywords=None, top_k=5):
    for suffix, label in (('att', 'Attention'), ('mlp', 'MLP')):
        key = f'{layer}-{suffix}'
        if key not in sae_features['circuit_features']:
            continue
        for token_idx, token_features in enumerate(sae_features['circuit_features'][key]):
            if not token_features:
                continue
            print(f'   {label} - Token {token_idx}:')
            for feat in token_features[:top_k]:
                desc = feat.get('description')
                if not desc:
                    continue
                if keywords and not any(w in desc.lower() for w in keywords):
                    continue
                print(f"   - Feature {feat['feature_index']}: {desc}")
                print(f"     Activation: {feat['activation']:.3f}")


active_nodes = [index_to_node(i) for i, active in enumerate(graph_data['nodes_in_graph']) if active]

print("=" * 80)
print(f"GENDER BIAS CIRCUIT ANALYSIS - Sample {args.sample_idx}")
print("=" * 80)
print(f"\nInput: '{sae_features['clean_text']}'")
print(f"Output: '{sae_features['clean_label']}'")
print(f"Corrupted input: '{sae_features['corrupted_text']}' -> '{sae_features['corrupted_label']}'")
print(f"\nActive nodes in circuit: {len(active_nodes)}")
print(f"Nodes: {', '.join(sorted(active_nodes, key=node_to_index))}")

EARLY_KEYWORDS = ['doctor', 'medical', 'professional', 'women', 'gender', 'male', 'said', 'the']
MID_KEYWORDS = ['doctor', 'medical', 'male', 'female', 'gender', 'women', 'men', 'he', 'she', 'said']

print("\n" + "=" * 80)
print("EARLY LAYERS (0-2): Detecting 'doctor' and processing context")
print("-" * 80)
for layer in range(3):
    if has_layer_features(layer):
        print(f"\nLayer {layer}:")
        print_layer_features(layer, keywords=EARLY_KEYWORDS)

print("\n" + "=" * 80)
print("MIDDLE LAYERS (3-7): Building gender associations")
print("-" * 80)
for layer in range(3, 8):
    if has_layer_features(layer):
        print(f"\nLayer {layer}:")
        print_layer_features(layer, keywords=MID_KEYWORDS, top_k=3)

print("\n" + "=" * 80)
print("LATE LAYERS (8-11): Finalizing gender prediction")
print("-" * 80)
for layer in range(8, 12):
    if has_layer_features(layer):
        print(f"\nLayer {layer}:")
        print_layer_features(layer)