"""Quick demo of MAGNET v3 on a graph and on a data matrix."""
import time
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from magnet import MAGNET

# ---------------------------------------------------------------- graph input
n_nodes = 5000
try:
    G = pickle.load(open("graph.pkl", "rb"))
except FileNotFoundError:
    G = nx.random_partition_graph([n_nodes] * 6, 0.01, 0.00008)
    pickle.dump(G, open("graph.pkl", "wb"))
print(G)

start = time.time()
X = MAGNET(n_epochs=200, verbose=True).fit_transform(G)   # rows follow list(G.nodes)
print(f"graph: {time.time() - start:.2f}s")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
labels = np.array([n // n_nodes for n in G.nodes])
axes[0].scatter(X[:, 0], X[:, 1], s=1, c=labels, cmap="tab10")
axes[0].set_title("random partition graph")

# ---------------------------------------------------------------- data input
data = np.load("data.npy").astype(np.float32)          # (10000, 200) blobs
start = time.time()
Y = MAGNET(verbose=True).fit_transform(data)             # row i == data[i]
print(f"data: {time.time() - start:.2f}s")
axes[1].scatter(Y[:, 0], Y[:, 1], s=1)
axes[1].set_title("blobs (data.npy)")
plt.show()
