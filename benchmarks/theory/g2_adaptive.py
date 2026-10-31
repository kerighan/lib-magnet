"""Adaptive per-point attraction exponent (kernel_b='adaptive': crowding read off the layout at 40% of the epochs,
L2 for the uncrowded points, L1 for the others, Cauchy repulsion, one run, no restart) against the constants and
the oracle where it exists. Data: COIL-20, Cora, SBM + chains, USA airports, the synthetic mixtures, MNIST 20k, Paul15."""
import sys, os, time, glob, re, numpy as np, scipy.sparse as sp
REPO, DATA = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def knn15_fn(X):
    hnn = NearestNeighbors(n_neighbors=16, algorithm="brute").fit(X).kneighbors(X, return_distance=False)[:, 1:]
    def f(Y):
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        return np.array([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15
    return f
def edge_pres_fn(A):
    A = sp.csr_matrix(A); deg = np.diff(A.indptr); k = int(min(50, deg.max()))
    def f(Y):
        lnn = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        out = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            if deg[i]: out[i] = len(np.intersect1d(A.indices[A.indptr[i]:A.indptr[i + 1]], lnn[i, :deg[i]])) / deg[i]
        return out
    return f

def bench(name, X, groups, metric, oracle=None, **kw):
    print(f"== {name}  n={len(groups)}", flush=True)
    confs = [("const 0.5", dict(kernel_b=0.5)), ("const 0.6", dict(kernel_b=0.6)), ("const 1.0", dict(kernel_b=1.0))]
    if oracle is not None: confs.append(("oracle", dict(kernel_b=oracle)))
    if not sp.issparse(X): confs.append(("local (dimprint)", dict(kernel_b="local")))
    confs += [("adapt .5@.6/0", dict(kernel_b="adaptive", adaptive=(0.5, 1.0, 0.6, 0.0, 2.0, 5))), ("adapt .5@.6/.05", dict(kernel_b="adaptive", adaptive=(0.5, 1.0, 0.6, 0.05, 2.0, 5))),
              ("adapt .5@.75/0", dict(kernel_b="adaptive", adaptive=(0.5, 1.0, 0.75, 0.0, 2.0, 5))), ("adapt .6@.6/0", dict(kernel_b="adaptive", adaptive=(0.6, 1.0, 0.6, 0.0, 2.0, 5))),
              ("adapt .5@.6/0 d10", dict(kernel_b="adaptive", adaptive=(0.5, 1.0, 0.6, 0.0, 2.0, 10)))]
    gl = [g for g in np.unique(groups)]
    for label, c in confs:
        scores, times, fr = [], [], []
        for seed in (0, 1):
            t = time.time(); M = MAGNET(n_threads=4, random_state=seed, **c, **kw).fit(X); times.append(time.time() - t)
            scores.append(metric(M.embedding_))
            if M.node_b_ is not None and np.ndim(M.node_b_) == 1: fr.append(M.node_b_ == 1.0)
        e = np.mean(scores, 0)
        per = " ".join(f"{g}:{e[groups==g].mean():.3f}" for g in gl[:12])
        extra = ("  | b=1: " + " ".join(f"{g}:{np.mean(fr, 0)[groups==g].mean():.2f}" for g in gl[:12])) if fr else ""
        print(f"  {label:16s} T={np.mean(times):4.1f}s score {e.mean():.4f}   {per}{extra}", flush=True)

from PIL import Image
files = sorted(glob.glob(os.path.join(DATA, "coil20", "coil-20-proc", "*.png")))
X = np.stack([np.asarray(Image.open(f).convert("L").resize((64, 64)), dtype=np.float32).ravel() / 255 for f in files])
lab = np.array([int(re.search(r"obj(\d+)__", os.path.basename(f)).group(1)) for f in files])
bench("coil20", X, lab, knn15_fn(X))

def mixture(n, qs, d=50, seed=0):
    r = np.random.default_rng(seed); y = r.integers(0, len(qs), n); centers = r.normal(0, 8, (len(qs), d)).astype(np.float32); X = np.empty((n, d), np.float32)
    for c, q in enumerate(qs):
        idx = np.where(y == c)[0]; basis = np.linalg.qr(r.normal(size=(d, q)))[0].T.astype(np.float32)
        X[idx] = centers[c] + r.normal(size=(len(idx), q)).astype(np.float32) @ basis
    return X, np.array(qs)[y]
X, q = mixture(6000, [1, 2, 10, 10]); bench("mix 1/2/10/10 6k", X, q, knn15_fn(X), oracle=np.where(q <= 2, 1.0, 0.5).astype(np.float32))
X, q = mixture(20000, [2] * 5 + [10] * 5); bench("mix 2-10 20k", X, q, knn15_fn(X), oracle=np.where(q <= 2, 1.0, 0.5).astype(np.float32))

from datasets import mnist
X, y, _ = mnist(20000); bench("mnist 20k", X, y, knn15_fn(X))

import h5py
f = h5py.File(os.path.join(DATA, "paul15", "paul15.h5"), "r")
counts = f["data"][()].T.astype(np.float32); genes = f["info.genes_codes"][()].astype(int); genes = genes - 1 if genes.min() == 1 else genes
cl = f["cluster.id"][()].ravel().astype(int); Xp = np.log1p(counts / np.maximum(counts.sum(1, keepdims=True), 1) * 1e4)[:, genes]
Mp = MAGNET(pca=20, n_threads=4, random_state=0).fit(Xp); bench("paul15 pca20", Xp, cl, knn15_fn(np.asarray(Mp._prepared_, dtype=np.float32)), pca=20)

def read_cora():
    d = "/home/maixent/stellargraph-datasets/cora"; ids, labs = [], []
    for line in open(os.path.join(d, "cora.content")):
        p = line.split(); ids.append(p[0]); labs.append(p[-1])
    idx = {k: i for i, k in enumerate(ids)}; rows, cols = [], []
    for line in open(os.path.join(d, "cora.cites")):
        a, b = line.split()
        if a in idx and b in idx: rows.append(idx[a]); cols.append(idx[b])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids))); A = ((A + A.T) > 0).astype(np.float32).tocsr()
    return A, np.array(labs)
A, labs = read_cora(); bench("cora graph", A, labs, edge_pres_fn(A))

RW = "/home/maixent/Reputation Squad Dropbox/Maixent Chenebaux/DATAPROJECTS/lib/rolewalk/benchmarks/data"
G = nx.read_edgelist(os.path.join(RW, "usa-airports.edgelist"), nodetype=int); lab = {}
for line in open(os.path.join(RW, "labels-usa-airports.txt")):
    p = line.split()
    if p[0] != "node": lab[int(p[0])] = int(p[1])
nodes = sorted(G.nodes()); A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes)).astype(np.float32); labels = np.array([lab.get(v, -1) for v in nodes])
bench("usa airports", A, labels, edge_pres_fn(A))

sizes = [300] * 5; p = [[0.002] * 5 for _ in range(5)]
for i in range(5): p[i][i] = 0.05
G = nx.stochastic_block_model(sizes, p, seed=0); n0 = G.number_of_nodes()
for b in range(5):
    start = n0 + 300 * b; nx.add_path(G, range(start, start + 300)); G.add_edge(sum(sizes[:b]), start)
kind = np.array(["block"] * n0 + ["chain"] * 1500); A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()))).astype(np.float32)
bench("SBM + chains", A, kind, edge_pres_fn(A), oracle=np.where(kind == "chain", 1.0, 0.5).astype(np.float32))
