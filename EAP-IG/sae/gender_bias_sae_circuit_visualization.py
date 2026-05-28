"""
Circuit Topology Visualization
Visualizes circuit structure from graph data
"""

import os as _os, sys as _sys
_base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_base)
_sys.path.insert(0, _base)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class CircuitTopologyVisualizer:
    """Visualizes circuit topology from graph data"""
    
    def __init__(self, sample_idx: int, graph_data_dir: str = "graph_data"):
        self.sample_idx = sample_idx
        self.graph_data_dir = Path(graph_data_dir)
        
        # Load data
        self.load_sample_data()
        self.parse_circuit_structure()
        
    def load_sample_data(self):
        """Load graph data for this sample"""
        # Load graph circuit data
        graph_file = self.graph_data_dir / f"gender_bias_{self.sample_idx}_top400.json"
        with open(graph_file, 'r') as f:
            self.graph_data = json.load(f)
        
        self.nodes_in_graph = np.array(self.graph_data['nodes_in_graph'])
        self.in_graph = np.array(self.graph_data['in_graph'])
        
        print(f"Loaded sample {self.sample_idx}")
        
    def parse_circuit_structure(self):
        """Parse which nodes and edges are in the circuit"""
        # Node structure: [input(1), layer0_attn(12), layer0_mlp(1), ..., layer11_attn(12), layer11_mlp(1)]
        self.n_layers = 12
        self.n_heads = 12
        
        # Parse nodes
        self.circuit_nodes = {
            'input': self.nodes_in_graph[0],
            'attention': {},  # {layer: [heads in circuit]}
            'mlp': {}  # {layer: bool}
        }
        
        for layer in range(self.n_layers):
            att_start = 1 + layer * 13
            # att_end = att_start + 12
            mlp_idx = att_start + 12
            
            # Which attention heads are in circuit
            attn_heads = [head for head in range(12) if self.nodes_in_graph[att_start + head]]
            self.circuit_nodes['attention'][layer] = attn_heads
            self.circuit_nodes['mlp'][layer] = bool(self.nodes_in_graph[mlp_idx])
        
        # Count edges
        self.n_edges_in_circuit = int(self.in_graph.sum())
        self.n_nodes_in_circuit = int(self.nodes_in_graph.sum())
        
        print(f"\nCircuit Statistics:")
        print(f"  Nodes: {self.n_nodes_in_circuit} / 157")
        print(f"  Edges: {self.n_edges_in_circuit} / 32491")
    
    def visualize_circuit_topology(self, output_path: str = None, drop_isolated: bool = True):
        """Visualize the circuit as a layer-by-layer graph"""
        fig, ax = plt.subplots(figsize=(36, 28))
        
        # Layer positions
        layer_spacing = 1.5
        x_positions = [i * layer_spacing for i in range(self.n_layers + 1)]
        
        # Build nodes
        node_positions = {}
        node_to_index = {}  # Map (type, layer, head) to source index
        node_labels = {}
        node_types = {}
        colors = {'attention': '#3498db', 'mlp': '#e74c3c', 'input': '#2ecc71'}
        
        # Input node
        node_positions[('input', 0)] = (0, 0.5)
        node_to_index[('input', 0)] = 0
        node_labels[('input', 0)] = 'Input'
        node_types[('input', 0)] = 'input'
        
        # Layer nodes
        for layer in range(self.n_layers):
            x = x_positions[layer + 1]
            y_positions = list(np.linspace(0.2, 1.6, 12))
            
            # Attention heads
            attn_heads = self.circuit_nodes['attention'][layer]
            if attn_heads:
                # n_heads = len(attn_heads)
                for head in attn_heads:
                    y = y_positions[head]
                    node_key = ('attention', layer, head)
                    node_positions[node_key] = (x, y)
                    node_to_index[node_key] = 1 + layer * 13 + head
                    node_labels[node_key] = f'A{layer}.{head}'
                    node_types[node_key] = 'attention'
            
            # MLP node
            if self.circuit_nodes['mlp'][layer]:
                y_mlp = 0.05
                node_key = ('mlp', layer)
                node_positions[node_key] = (x, y_mlp)
                node_to_index[node_key] = 1 + layer * 13 + 12
                node_labels[node_key] = f'M{layer}'
                node_types[node_key] = 'mlp'
        
        # Draw edges
        print(f"Drawing edges for circuit...")
        edge_count = 0
        
        # Helper function to get destination indices for a node
        def get_dest_indices(node_key):
            """Get all possible destination indices for a node in the in_graph matrix"""
            # Destination structure (445 total):
            # 0: input
            # 1-36: layer 0 attention heads (12 heads * 3 QKV = 36)
            # 37: layer 0 MLP
            # 38-73: layer 1 attention heads (12 heads * 3 QKV = 36)
            # 74: layer 1 MLP
            # ... pattern repeats for 12 layers
            # Structure per layer: 12*3 (attention QKV) + 1 (MLP) = 37 indices
            
            if node_key[0] == 'input':
                return [0]
            elif node_key[0] == 'attention':
                layer, head = node_key[1], node_key[2]
                # Each layer starts at: 1 + layer * 37
                # Each head has 3 indices (Q, K, V): head * 3, head * 3 + 1, head * 3 + 2
                layer_start = 1 + layer * 37
                head_base = layer_start + head * 3
                return [head_base, head_base + 1, head_base + 2]  # Q, K, V indices
            elif node_key[0] == 'mlp':
                layer = node_key[1]
                # MLP is at position: 1 + layer * 37 + 36 (after all attention heads)
                mlp_idx = 1 + layer * 37 + 36
                return [mlp_idx]
            return []
        
        # Build edges based on in_graph matrix
        edge_list = []
        for src_key, (src_x, src_y) in node_positions.items():
            if src_key not in node_to_index:
                continue
            src_idx = node_to_index[src_key]
            
            # Get source layer
            if src_key[0] == 'input':
                src_layer = -1
            else:
                src_layer = src_key[1]
            
            # Get all edges from this source
            for dst_key, (dst_x, dst_y) in node_positions.items():
                if dst_key not in node_to_index:
                    continue
                if src_key == dst_key:
                    continue
                
                # Get destination layer
                if dst_key[0] == 'input':
                    dst_layer = -1
                else:
                    dst_layer = dst_key[1]
                
                # Only allow forward edges (to later layers)
                if dst_layer <= src_layer:
                    continue
                
                dst_idx = node_to_index[dst_key]
                
                # Get all possible destination indices for the destination node
                dest_indices = get_dest_indices(dst_key)
                
                # Check if there's an edge from src to any representation of dst
                has_edge = False
                for dest_idx in dest_indices:
                    if dest_idx < self.in_graph.shape[1] and self.in_graph[src_idx, dest_idx]:
                        has_edge = True
                        break
                
                if has_edge:
                    edge_list.append((src_key, dst_key))
                    edge_count += 1
        
        print(f"Drew {edge_count} edges (from in_graph matrix)")
        
        # Debug: Check which nodes have no connections
        nodes_with_edges = set()
        for src_key, dst_key in edge_list:
            nodes_with_edges.add(src_key)
            nodes_with_edges.add(dst_key)

        isolated_nodes = set(node_positions.keys()) - nodes_with_edges
        if isolated_nodes:
            print(f"Warning: {len(isolated_nodes)} nodes have no edges:")
            for node in sorted(isolated_nodes):
                print(f"  {node}")

        # Optionally drop isolated nodes from the plot
        if drop_isolated and isolated_nodes:
            for node in isolated_nodes:
                node_positions.pop(node, None)
                node_to_index.pop(node, None)
                node_labels.pop(node, None)
                node_types.pop(node, None)

        # Draw edges
        for src_key, dst_key in edge_list:
            if drop_isolated and (src_key not in node_positions or dst_key not in node_positions):
                continue
            src_x, src_y = node_positions[src_key]
            dst_x, dst_y = node_positions[dst_key]
            ax.plot([src_x, dst_x], [src_y, dst_y],
                    color='gray', alpha=0.4, linewidth=1.5, zorder=1)

        # Draw nodes
        for node_key, (x, y) in node_positions.items():
            node_type = node_types[node_key]
            ax.scatter([x], [y], s=1500, c=colors[node_type], alpha=0.8,
                      edgecolors='black', linewidth=2.5 if node_type != 'input' else 3, zorder=3)
            ax.text(x, y, node_labels[node_key], ha='center', va='center',
                    fontsize=11 if node_type == 'input' else 9, fontweight='bold')
        
        # Add layer labels
        for layer in range(self.n_layers):
            ax.text(x_positions[layer + 1], -0.05, f'L{layer}', 
                   ha='center', va='top', fontsize=11, fontweight='bold')
        
        ax.set_xlim(-0.8, x_positions[-1] + 0.8)
        ax.set_ylim(-0.1, 1.8)
        ax.axis('off')
        
        # Title
        # title = f"Circuit Topology - Sample {self.sample_idx}\n"
        # title += f"{self.n_nodes_in_circuit} nodes, {self.n_edges_in_circuit} edges"
        # ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=colors['input'], label='Input'),
            mpatches.Patch(color=colors['attention'], label='Attention'),
            mpatches.Patch(color=colors['mlp'], label='MLP')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=500, bbox_inches='tight')
            print(f"Saved circuit topology to {output_path}")
        plt.show()


def visualize_sample(sample_idx: int, output_dir: str = "circuit_topology_output"):
    """Visualize circuit topology for a sample"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Visualizing Circuit Topology - Sample {sample_idx}")
    print(f"{'='*60}")
    
    visualizer = CircuitTopologyVisualizer(sample_idx)
    visualizer.visualize_circuit_topology(
        output_path / f"sample_{sample_idx}_topology.png"
    )
    
    print(f"\n{'='*60}")
    print(f"Visualization saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Visualize sample 2 as an example
    visualize_sample(sample_idx=2)
    
    # You can visualize other samples too
    # visualize_sample(sample_idx=0)
    # visualize_sample(sample_idx=4)
