"""Layouts of networkx graphs with MAGNET -> benchmarks/plots/graph_demo.png"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from magnet import MAGNET

rng = np.random.default_rng(0)
panels = []

# 1. communities of very different sizes, weak inter-community links
sizes = [3000, 1500, 800, 400, 200, 100, 50, 25]
G = nx.random_partition_graph(sizes, 0.02, 0.0003, seed=0)
labels = np.array([G.nodes[n]["block"] for n in G.nodes])
panels.append(("random partition graph (8 blocks of 25 to 3000 nodes)", G, labels))

# 2. power-law degrees with triangles (Holme-Kim), coloured by log degree
H = nx.powerlaw_cluster_graph(8000, 3, 0.5, seed=0)
deg = np.array([d for _, d in H.degree()])
panels.append(("power-law cluster graph (8k nodes, hubs)", H, np.log(deg)))

# 3. a grid with a few random long-range shortcuts (small world): manifold-like graph
W = nx.grid_2d_graph(60, 60)
W = nx.convert_node_labels_to_integers(W)
coords = np.array([(i // 60, i % 60) for i in range(3600)])
panels.append(("60x60 grid (manifold-like), coloured by row", W, coords[:, 0]))

# 4. weighted graph: les miserables (weights = number of co-occurrences = similarities)
L = nx.les_miserables_graph()
comm = {n: i for i, c in enumerate(nx.community.greedy_modularity_communities(L, weight="weight")) for n in c}
panels.append(("Les Miserables (77 nodes, weighted, coloured by modularity community)", L, np.array([comm[n] for n in L.nodes])))

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
for ax, (title, G, c) in zip(axes.ravel(), panels):
    t = time.time()
    Y = MAGNET(random_state=0).fit_transform(G)
    dt = time.time() - t
    pos = {n: Y[i] for i, n in enumerate(G.nodes)}
    small = G.number_of_nodes() < 500
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3 if small else 0.03, width=0.8 if small else 0.2)
    ax.scatter(Y[:, 0], Y[:, 1], s=40 if small else 4, c=c, cmap="tab10" if len(np.unique(c)) <= 10 else "viridis", zorder=3)
    if small:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    ax.set_title(f"{title}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges - {dt:.1f} s", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out = os.path.join(HERE, "plots", "graph_demo.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
