import json
from pathlib import Path

# Load the existing graph JSON
with open("graph.json", "r") as f:
    graph_data = json.load(f)

# Set all nodes' "in_graph" to False
for node in graph_data["nodes"].values():
    node["in_graph"] = False

# Set all edges' "in_graph" to False and "score" to 0.0
for edge in graph_data["edges"].values():
    edge["in_graph"] = False
    edge["score"] = 0.0

# Save as base file
base_file_path = "graph_base.json"
with open(base_file_path, "w") as f:
    json.dump(graph_data, f, indent=2)