import pygraphviz as pgv
import networkx as nx

# Create a directed graph
G = nx.DiGraph()
G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("A", "D")])

# Compute topological sort
try:
    topological_order = list(nx.topological_sort(G))
    print("Topological order:", topological_order)  # Example: ['A', 'B', 'D', 'C']
except nx.NetworkXUnfeasible:
    print("Graph has a cycle - no topological sort exists")
    exit()

# Convert to PyGraphviz
P = pgv.AGraph(directed=True, strict=False)
P.graph_attr.update(
    rankdir="TB",  # Top-to-bottom layout
    nodesep="0.8",  # Vertical spacing between nodes
    ranksep="0.5",  # Horizontal spacing between ranks
    splines="spline"
)

# Add all nodes (without rank assignment)
for node in G.nodes():
    P.add_node(node)

# **CRITICAL FIX: Enforce strict ordering with invisible edges**
for i in range(len(topological_order) - 1):
    # if (topological_order[i], topological_order[i+1]) not in G.edges():
    P.add_edge(
        topological_order[i],
        topological_order[i + 1],
        key="invisible",
        style="invis",  # Invisible edge forces ordering
        weight="1000",   # High weight ensures this edge dominates layout
    )

# Add original edges
for u, v in G.edges():
    P.add_edge(u, v, key="visible")

# Draw
P.layout(prog="dot")
P.draw("sss.png")