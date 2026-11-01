"""SBM graph (30k nodes, 782k edges, 6 blocks): UMAP under three protocols against MAGNET on the same graph.
  rows      : the benchmark table's protocol so far: adjacency rows (1/w) fed as a sparse feature matrix, euclidean
  precomp   : metric="precomputed" on the sparse distance matrix (1 on every edge, the rest unknown)
  graph     : umap.umap_.simplicial_set_embedding on the adjacency itself (UMAP's own layout of a given graph,
              spectral init, 200 epochs, a/b of its defaults, parallel=True as in the table)
Metrics: 10-NN label accuracy (5-fold CV on 10k nodes, as in the table), silhouette, and edge recall@15
(share of a node's 15 nearest layout neighbours that are graph neighbours, 5000 nodes).
Usage: python benchmarks/sbm_umap_fair.py"""
import os, sys, time, json, warnings
import numpy as np, scipy.sparse as sp, networkx as nx
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from datasets import sbm_graph
from metrics import evaluate, _idx
from sklearn.neighbors import NearestNeighbors
import umap, umap.umap_ as U
from magnet import MAGNET

G, y, name = sbm_graph()
A = sp.csr_matrix(nx.adjacency_matrix(G).astype(np.float32)); A.sort_indices()
n = A.shape[0]

def edge_recall(Y, k=15, m=5000):
    q = _idx(n, m, 0)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y[q], return_distance=False)[:, 1:]
    hits = sum(len(np.intersect1d(A.indices[A.indptr[i]:A.indptr[i + 1]], row)) for i, row in zip(q, nn))
    return hits / (m * k)

def report(label, Y, dt, cfg):
    Y = np.asarray(Y, np.float32)
    r = evaluate(None, Y, y, cache_key=name); r["edge_recall15"] = edge_recall(Y); r["time"] = round(dt, 2); r["config"] = cfg
    print(f"{label:<28} {dt:7.1f} s  10-NN acc {r['knn_acc']:.3f}  silhouette {r['silhouette']:+.3f}  edge recall@15 {r['edge_recall15']:.3f}", flush=True)
    return r

out = {}
# numba warm-ups outside the clock (same settings as the timed calls)
Xw = sp.random(6000, 6000, density=0.003, format="csr", dtype=np.float32); Xw.data[:] = 1; Xw.setdiag(0); Xw.eliminate_zeros()
umap.UMAP(n_epochs=20).fit_transform(Xw)
umap.UMAP(n_epochs=20, metric="precomputed").fit_transform(Xw + Xw.T)
Aw = ((Xw + Xw.T) > 0).astype(np.float32).tocsr()
a, b = U.find_ab_params(1.0, 0.1)
U.simplicial_set_embedding(Aw, Aw, 2, 1.0, a, b, 1.0, 5, 20, "spectral", np.random.RandomState(0), "euclidean", {}, False, {}, False, parallel=True)

# 1. rows as features (table protocol so far)
D = A.copy(); D.data = 1.0 / D.data
t = time.time(); Y = umap.UMAP().fit_transform(D); out["umap_rows"] = report("UMAP, rows as features", Y, time.time() - t, "UMAP() on the sparse 1/w adjacency rows, euclidean")
# 2. precomputed sparse distances
t = time.time(); Y = umap.UMAP(metric="precomputed").fit_transform(D); out["umap_precomputed"] = report("UMAP, precomputed sparse", Y, time.time() - t, "UMAP(metric='precomputed') on the sparse distance matrix (1 per edge)")
# 3. UMAP's layout of the graph itself
t = time.time()
Y, _ = U.simplicial_set_embedding(A, A, 2, 1.0, a, b, 1.0, 5, 200, "spectral", np.random.RandomState(0), "euclidean", {}, False, {}, False, parallel=True)
out["umap_graph"] = report("UMAP, simplicial_set_embedding", Y, time.time() - t, "simplicial_set_embedding(graph=A, spectral init, 200 epochs, a/b of min_dist 0.1, parallel)")
# 4. MAGNET on the graph
t = time.time(); Y = MAGNET(random_state=0).fit_transform(G); out["magnet"] = report("MAGNET (defaults)", Y, time.time() - t, "MAGNET(random_state=0).fit_transform(G)")
json.dump(out, open(os.path.join(HERE, "cache", "sbm_umap_fair.json"), "w"), indent=1)
