"""Gate: distribution of the per-node crowding (share of a node's graph edges that end beyond the kernel scale in the
default 2-D layout) on real data where uncrowded regions should exist: COIL-20 (20 objects x 72 angles: 1-D circles),
Paul15 (single-cell differentiation trajectories), citation and airport graphs, an SBM with chains attached; the
synthetic mixtures and MNIST as references. For data matrices, the payoff of kernel_b='local' is measured too."""
import sys, os, time, glob, re, numpy as np, scipy.sparse as sp
REPO, DATA = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from magnet.roles import local_intrinsic_dimension
from sklearn.neighbors import NearestNeighbors
import networkx as nx
rng = np.random.default_rng(0)

def crowding(M, Y):
    P = M.graph_.tocsr(); dk = M.a_ ** (-1.0 / (2 * M.b_))
    i = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr)); j = P.indices
    far = (np.sqrt(((Y[i] - Y[j]) ** 2).sum(1)) > dk).astype(np.float64)
    return np.bincount(i, weights=far, minlength=P.shape[0]) / np.maximum(np.diff(P.indptr), 1)

def knn15(X, Y):
    hnn = NearestNeighbors(n_neighbors=16, algorithm="brute").fit(X).kneighbors(X, return_distance=False)[:, 1:]
    lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return np.array([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15

def edge_pres(A, Y):
    A = sp.csr_matrix(A); deg = np.diff(A.indptr); k = int(min(50, deg.max()))
    lnn = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    out = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        if deg[i] == 0: continue
        out[i] = len(np.intersect1d(A.indices[A.indptr[i]:A.indptr[i + 1]], lnn[i, :deg[i]])) / deg[i]
    return out

def report(name, M, Y, labels, groups=None):
    c = crowding(M, Y)
    line = f"{name:18s} n={len(Y):6d}  crowd median {np.median(c):.2f} p10 {np.percentile(c,10):.2f} p25 {np.percentile(c,25):.2f} p75 {np.percentile(c,75):.2f}  frac<=0.05 {np.mean(c<=0.05):.2f} <=0.1 {np.mean(c<=0.1):.2f} >=0.25 {np.mean(c>=0.25):.2f}"
    print(line, flush=True)
    if labels is not None:
        labs = [l for l in np.unique(labels)][:24]
        print("    per label median crowding: " + " ".join(f"{l}:{np.median(c[labels==l]):.2f}" for l in labs), flush=True)
    return c

def run_data(name, X, labels, **kw):
    t = time.time(); M = MAGNET(n_threads=4, random_state=0, **kw).fit(X); Y = M.embedding_
    c = report(name, M, Y, labels)
    Xd = M._prepared_ if hasattr(M, "_prepared_") and M._prepared_ is not None else X
    q = local_intrinsic_dimension(np.asarray(Xd, dtype=np.float32), 15)
    print(f"    local MLE dim median {np.median(q):.1f} p10 {np.percentile(q,10):.1f}  (T={time.time()-t:.1f}s)")
    res = {}
    for kb in (0.5, 0.6, 1.0, "local"):
        sc = []
        for seed in (0, 1):
            Mk = MAGNET(kernel_b=kb, n_threads=4, random_state=seed, **kw).fit(X)
            sc.append(knn15(np.asarray(Xd, dtype=np.float32), Mk.embedding_))
        res[kb] = np.mean(sc, 0)
        extra = f"  (b=1 on {np.mean(Mk.node_b_==1):.2f} of points)" if kb == "local" else ""
        print(f"    kernel_b={str(kb):6s} 15-NN kept {res[kb].mean():.4f}" + extra, flush=True)
    return c, res

def run_graph(name, A, labels, oracle=None):
    A = sp.csr_matrix(A); A = ((A + A.T) > 0).astype(np.float32)
    M = MAGNET(n_threads=4, random_state=0).fit(A); Y = M.embedding_
    c = report(name, M, Y, labels)
    for label, kb in [("0.5", 0.5), ("0.6", 0.6), ("1.0", 1.0)] + ([("oracle", oracle)] if oracle is not None else []):
        sc = [edge_pres(A, MAGNET(kernel_b=kb, n_threads=4, random_state=s).fit_transform(A)) for s in (0, 1)]
        e = np.mean(sc, 0)
        per = " ".join(f"{l}:{e[labels==l].mean():.3f}" for l in np.unique(labels)[:12]) if labels is not None else ""
        print(f"    kernel_b={label:6s} edge preservation {e.mean():.4f}   per label {per}", flush=True)

# ---------------- COIL-20
from PIL import Image
files = sorted(glob.glob(os.path.join(DATA, "coil20", "coil-20-proc", "*.png")))
X = np.stack([np.asarray(Image.open(f).convert("L").resize((64, 64)), dtype=np.float32).ravel() / 255 for f in files])
lab = np.array([int(re.search(r"obj(\d+)__", os.path.basename(f)).group(1)) for f in files])
run_data("coil20 (1-D circles)", X, lab)

# ---------------- Paul15
import h5py
f = h5py.File(os.path.join(DATA, "paul15", "paul15.h5"), "r")
counts = f["data"][()].T.astype(np.float32)             # cells x genes
genes = f["info.genes_codes"][()].astype(int); genes = genes - 1 if genes.min() == 1 else genes
cl = f["cluster.id"][()].ravel().astype(int)
Xp = counts / np.maximum(counts.sum(1, keepdims=True), 1) * 1e4; Xp = np.log1p(Xp[:, genes])
run_data("paul15 (trajectories)", Xp, cl, pca=50)

# ---------------- references
sys.path.insert(0, os.path.join(REPO, "benchmarks", "theory"))
def mixture(n, qs, d=50, seed=0):
    r = np.random.default_rng(seed); y = r.integers(0, len(qs), n); centers = r.normal(0, 8, (len(qs), d)).astype(np.float32); X = np.empty((n, d), np.float32)
    for c, q in enumerate(qs):
        idx = np.where(y == c)[0]; basis = np.linalg.qr(r.normal(size=(d, q)))[0].T.astype(np.float32)
        X[idx] = centers[c] + r.normal(size=(len(idx), q)).astype(np.float32) @ basis
    return X, np.array(qs)[y]
X, q = mixture(6000, [1, 2, 10, 10]); run_data("mix 1/2/10/10-D ref", X, q)
from datasets import mnist
X, y, _ = mnist(6000); run_data("mnist 6k ref", X, y)

# ---------------- graphs
def read_cora():
    d = os.path.join("/home/maixent/stellargraph-datasets/cora")
    ids, labs = [], []
    for line in open(os.path.join(d, "cora.content")):
        p = line.split(); ids.append(p[0]); labs.append(p[-1])
    idx = {k: i for i, k in enumerate(ids)}; labs = np.array(labs)
    rows, cols = [], []
    for line in open(os.path.join(d, "cora.cites")):
        a, b = line.split()
        if a in idx and b in idx: rows.append(idx[a]); cols.append(idx[b])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
    return A, labs
A, labs = read_cora(); run_graph("cora graph", A, labs)

RW = "/home/maixent/Reputation Squad Dropbox/Maixent Chenebaux/DATAPROJECTS/lib/rolewalk/benchmarks/data"
for region in ("usa", "europe"):
    G = nx.read_edgelist(os.path.join(RW, f"{region}-airports.edgelist"), nodetype=int)
    lab = {}
    for line in open(os.path.join(RW, f"labels-{region}-airports.txt")):
        p = line.split()
        if p[0] != "node": lab[int(p[0])] = int(p[1])
    nodes = sorted(G.nodes()); A = nx.to_scipy_sparse_array(G, nodelist=nodes); labels = np.array([lab.get(v, -1) for v in nodes])
    run_graph(f"{region} airports", A, labels)

# SBM with chains
sizes = [300] * 5; p = [[0.002] * 5 for _ in range(5)]
for i in range(5): p[i][i] = 0.05
G = nx.stochastic_block_model(sizes, p, seed=0); n0 = G.number_of_nodes(); kind = np.array(["block"] * n0)
for b in range(5):
    start = n0 + 300 * b; nx.add_path(G, range(start, start + 300)); G.add_edge(sum(sizes[:b]), start)
kind = np.concatenate([kind, np.array(["chain"] * 1500)])
A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()))
run_graph("SBM + chains", A, kind, oracle=np.where(kind == "chain", 1.0, 0.5).astype(np.float32))
