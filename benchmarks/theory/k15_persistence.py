"""k15: does the full-batch pass keep the start's global structure, or has it just not had time to forget it?
MNIST 70k. (a) The stiff-spring SGD start (mid 0.5, seed 0) refined for 3000 iterations, metrics every 100 (15-NN on
the 5000 benchmark queries, distance Spearman and triplets on the 2000-point sample, centroid Spearman, the KL
objective on the graph edges). (b) The PCA-2 start refined for 1200 iterations, same trace. All cores.
-> k15_persistence.json / .log. Usage: python benchmarks/theory/k15_persistence.py"""
import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from metrics import _idx, high_dim_pdist, high_dim_neighbors
from magnet import MAGNET
from magnet.tsne import optimize_tsne, FFTRepulsion2D
import scipy.sparse as sp

X, y, key = mnist(70000); n = len(X); y = np.asarray(y).astype(int)
q = _idx(n, 5000, 0); hnn = high_dim_neighbors(X, q, 15, key)
g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1); dh = Dh[iu]
rng = np.random.default_rng(0); i, j, l = rng.integers(0, len(g), (3, 200000)); ok = (i != j) & (i != l) & (j != l); i, j, l = i[ok], j[ok], l[ok]
Ch = np.array([X[y == c].mean(0) for c in range(10)]); Dch = np.sqrt(((Ch[:, None] - Ch[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]

def metrics(Y, P, rep):
    lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y[q], return_distance=False)
    lnn = np.array([row[row != a][:15] for a, row in zip(q, lnn)])
    knn = float(np.mean([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15)
    Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); dl = Dl[iu]
    Cl = np.array([Y[y == c].mean(0) for c in range(10)]); Dcl = np.sqrt(((Cl[:, None] - Cl[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]
    # KL(P || Q) on the edges: sum p log p - sum p log q, q_ij = (1 + d^2)^-1 / Z, Z from the FFT sum (self terms removed)
    sumQ, _ = rep(Y); Z = sumQ
    C = P.tocoo(); d2 = ((Y[C.row] - Y[C.col]) ** 2).sum(1); logq = -np.log1p(d2) - np.log(Z)
    kl = float((C.data * (np.log(C.data) - logq)).sum())
    return dict(knn15=knn, sp=float(spearmanr(dh, dl).correlation), trip=float(np.mean(np.sign(Dh[i, j] - Dh[i, l]) == np.sign(Dl[i, j] - Dl[i, l]))),
                cent=float(spearmanr(Dch, Dcl).correlation), kl=kl, std=float(Y[:, 0].std()))

m = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X)
P = sp.csr_matrix(m.graph_, dtype=np.float32); P = P / P.sum()
rep = FFTRepulsion2D(n_interp=3, box_width=1.0, n_threads=-1)
out = {}
for label, Y0, n_iter in (("SGD mid 0.5 (hybrid start)", m.embedding_.astype(np.float32), 3000), ("PCA-2", PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32), 1200)):
    Y = Y0.copy(); Y -= Y.mean(0); Y *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y[:, 0].std()), 1e-12)
    trace = [dict(it=0, t=0.0, **metrics(Y, P, rep))]; t0 = time.time()
    print(f"\n{label}\n{'it':>5}{'t(s)':>7}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'KL':>9}{'std':>7}", flush=True)
    def show(r): print(f"{r['it']:>5}{r['t']:>7.1f}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{r['kl']:>9.4f}{r['std']:>7.1f}", flush=True)
    show(trace[0])
    done = 0
    while done < n_iter:
        step = 100
        tt = time.time(); Y = optimize_tsne(P, Y, n_iter=step, early_iter=0, n_threads=-1)
        elapsed = time.time() - tt; done += step
        r = dict(it=done, t=trace[-1]["t"] + elapsed, **metrics(Y, P, rep)); trace.append(r); show(r)
    out[label] = trace
    json.dump(out, open(os.path.join(HERE, "k15_persistence.json"), "w"), indent=1)
