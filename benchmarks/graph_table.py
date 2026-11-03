"""Graph table of the paper: MAGNET against UMAP's own layout of a given graph (umap.umap_.simplicial_set_embedding on
the adjacency: spectral init, 500 epochs for graphs below 10k nodes and 200 above, UMAP's defaults, parallel; the
metric="precomputed" route refuses rows with fewer than n_neighbors entries, i.e. any graph with low-degree nodes) on
Cora (2708 nodes), the US and Europe airport graphs, the SBM + chains graph and the SBM 30k benchmark graph; 3 seeds
for MAGNET (UMAP without random_state so that its layout stays parallel). Metrics: edge preservation (share of a node's
graph neighbours among its nearest layout neighbours, averaged over nodes), 10-NN label accuracy where labels exist,
and wall-clock time, all cores. -> benchmarks/cache/graph_table.json, paper/tables/graphs.tex.
Usage: python benchmarks/graph_table.py"""
import os, sys, json, time, pickle, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp, networkx as nx
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from magnet import MAGNET
import umap, umap.umap_ as U
AB = U.find_ab_params(1.0, 0.1)
def umap_graph(A, n_epochs):
    Y, _ = U.simplicial_set_embedding(A, A, 2, 1.0, AB[0], AB[1], 1.0, 5, n_epochs, "spectral", np.random.RandomState(0), "euclidean", {}, False, {}, False, parallel=True)
    return np.asarray(Y, dtype=np.float32)

def edge_pres(A, Y, m=5000, seed=0):
    A = sp.csr_matrix(A); n = A.shape[0]; deg = np.diff(A.indptr); k = int(min(50, deg.max()))
    q = np.arange(n) if n <= m else np.random.default_rng(seed).choice(n, m, replace=False)
    lnn = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y[q], return_distance=False)[:, 1:]
    out = [len(np.intersect1d(A.indices[A.indptr[i]:A.indptr[i + 1]], lnn[j, :deg[i]])) / deg[i] for j, i in enumerate(q) if deg[i]]
    return float(np.mean(out))

def knn_acc(Y, y):
    m = y != -1
    if m.sum() < 50 or len(np.unique(y[m])) < 2: return float("nan")
    q = np.arange(len(Y))[m]
    if len(q) > 10000: q = np.random.default_rng(2).choice(q, 10000, replace=False)
    return float(cross_val_score(KNeighborsClassifier(10), Y[q], y[q], cv=5).mean())

def read_cora():
    d = "/home/maixent/stellargraph-datasets/cora"; ids, labs = [], []
    for line in open(os.path.join(d, "cora.content")):
        p = line.split(); ids.append(p[0]); labs.append(p[-1])
    idx = {k: i for i, k in enumerate(ids)}; rows, cols = [], []
    for line in open(os.path.join(d, "cora.cites")):
        a, b = line.split()
        if a in idx and b in idx: rows.append(idx[a]); cols.append(idx[b])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids))); A = ((A + A.T) > 0).astype(np.float32).tocsr()
    _, y = np.unique(np.array(labs), return_inverse=True)
    return A, y

def read_airports(region):
    RW = "/home/maixent/Reputation Squad Dropbox/Maixent Chenebaux/DATAPROJECTS/lib/rolewalk/benchmarks/data"
    G = nx.read_edgelist(os.path.join(RW, f"{region}-airports.edgelist"), nodetype=int); lab = {}
    for line in open(os.path.join(RW, f"labels-{region}-airports.txt")):
        p = line.split()
        if p[0] != "node": lab[int(p[0])] = int(p[1])
    nodes = sorted(G.nodes()); A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes), dtype=np.float32)
    return A, np.array([lab.get(v, -1) for v in nodes])

def sbm_chains():
    sizes = [300] * 5; p = [[0.002] * 5 for _ in range(5)]
    for i in range(5): p[i][i] = 0.05
    G = nx.stochastic_block_model(sizes, p, seed=0); n0 = G.number_of_nodes()
    for b in range(5):
        start = n0 + 300 * b; nx.add_path(G, range(start, start + 300)); G.add_edge(sum(sizes[:b]), start)
    A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes())), dtype=np.float32)
    y = np.concatenate([np.repeat(np.arange(5), 300), np.repeat(np.arange(5, 10), 300)])
    return A, y

def sbm30k():
    G = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "cmagnet", "graph.pkl"), "rb"))
    nodes = list(G.nodes()); A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes), dtype=np.float32)
    from datasets import sbm_graph
    _, y, _ = sbm_graph()
    return A, np.asarray(y)

GRAPHS = [("Cora (2.7k, 5.3k edges)", read_cora), ("US airports (1.2k, 13.6k)", lambda: read_airports("usa")),
          ("Europe airports (0.4k, 6k)", lambda: read_airports("europe")), ("SBM + chains (3k)", sbm_chains), ("SBM 30k (782k edges)", sbm30k)]
SMOKE = "--smoke" in sys.argv
if not SMOKE:
    W = sp.random(3000, 3000, density=0.005, format="csr", dtype=np.float32); W = (W + W.T).tocsr(); W.setdiag(0); W.eliminate_zeros(); W.data[:] = 1.0
    umap_graph(W, 20)   # numba warm-up outside the clock
out = {}
SEEDS = 1 if SMOKE else 3
if "--table-only" in sys.argv:   # regenerate the .tex from the cached JSON
    out = {k: {m: {kk: tuple(v) for kk, v in r.items()} for m, r in d.items()} for k, d in json.load(open(os.path.join(HERE, "cache", "graph_table.json"))).items()}
for name, load in ([] if "--table-only" in sys.argv else (GRAPHS[:1] if SMOKE else GRAPHS)):
    A, y = load(); A.data[:] = 1.0; n = A.shape[0]
    rows = {}
    for s in range(SEEDS):
        t = time.time(); Y = MAGNET(random_state=s).fit_transform(A); dt = time.time() - t
        rows.setdefault("MAGNET", []).append(dict(time=dt, edge=edge_pres(A, Y), acc=knn_acc(Y, y)))
    for s in range(0 if SMOKE else SEEDS):
        t = time.time(); Y = umap_graph(A, 500 if n < 10000 else 200); dt = time.time() - t
        rows.setdefault("UMAP", []).append(dict(time=dt, edge=edge_pres(A, Y), acc=knn_acc(Y, y)))
    out[name] = {m: {k: (float(np.mean([r[k] for r in R])), float(np.std([r[k] for r in R]))) for k in ("time", "edge", "acc")} for m, R in rows.items()}
    if SMOKE: out[name]["UMAP"] = out[name]["MAGNET"]
    for m, r in out[name].items():
        print(f"{name:<28}{m:<8} time {r['time'][0]:6.1f} ± {r['time'][1]:.1f}   edge {r['edge'][0]:.3f} ± {r['edge'][1]:.3f}   acc {r['acc'][0]:.3f}", flush=True)
if "--table-only" not in sys.argv: json.dump(out, open(os.path.join(HERE, "cache", "graph_table.json"), "w"), indent=1)
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"graph & \multicolumn{2}{c}{time (s)} & \multicolumn{2}{c}{edge preservation} & \multicolumn{2}{c}{10-NN acc.} \\",
     r" & \magnet{} & UMAP & \magnet{} & UMAP & \magnet{} & UMAP \\", r"\midrule"]
for name, r in out.items():
    a, u = r["MAGNET"], r["UMAP"]
    def b(x, yv, hi=True):
        s1, s2 = f"{x[0]:.3f}", f"{yv[0]:.3f}"
        if np.isnan(x[0]): return "--", "--"
        if abs(x[0] - yv[0]) < 5e-4: return s1, s2
        return (r"\textbf{" + s1 + "}", s2) if (x[0] > yv[0]) == hi else (s1, r"\textbf{" + s2 + "}")
    t1, t2 = (r"\textbf{" + f"{a['time'][0]:.1f}" + "}", f"{u['time'][0]:.1f}") if a["time"][0] < u["time"][0] else (f"{a['time'][0]:.1f}", r"\textbf{" + f"{u['time'][0]:.1f}" + "}")
    e1, e2 = b(a["edge"], u["edge"]); c1, c2 = b(a["acc"], u["acc"])
    L.append(f"{name} & {t1} & {t2} & {e1} & {e2} & {c1} & {c2} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "..", "paper", "tables", "graphs.tex"), "w").write("\n".join(L) + "\n")
print("table written")
