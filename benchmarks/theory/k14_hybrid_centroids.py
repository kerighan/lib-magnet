"""k14: why does the hybrid's full-batch pass lose the arrangement of the class centroids (MNIST, Spearman of the 45
inter-centroid distances 0.635 after the SGD pass -> 0.421 after the refinement) while keeping the pairwise Spearman?
One SGD layout (loss="umap", mid_weight 0.5, as under loss="hybrid"), then the refinement pass re-run from a copy of it
under variants: iterations, hand-over scale, mid-range spring kept on, learning rate, kernel dof. For each: 15-NN
preservation, pairwise Spearman (all / short / intermediate / long), centroid Spearman, time of the pass.
Usage: python benchmarks/theory/k14_hybrid_centroids.py [n=70000]  (4 threads, ~1.5 GB)"""
import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from metrics import _idx, high_dim_pdist, high_dim_neighbors
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from sklearn.neighbors import NearestNeighbors

n = int(sys.argv[1]) if len(sys.argv) > 1 else 70000
X, y, key = mnist(n); n = len(X)
q = _idx(n, 5000, 0); hnn = high_dim_neighbors(X, q, 15, key)
g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1); dh = Dh[iu]
t1, t2 = np.quantile(dh, [1 / 3, 2 / 3])
Ch = np.array([X[y == c].mean(0) for c in range(10)])
Dch = np.sqrt(((Ch[:, None] - Ch[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]

def metrics(Y):
    lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y[q], return_distance=False)
    lnn = np.array([row[row != i][:15] for i, row in zip(q, lnn)])
    knn = float(np.mean([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15)
    dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1))[iu]
    Cl = np.array([Y[y == c].mean(0) for c in range(10)])
    Dcl = np.sqrt(((Cl[:, None] - Cl[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]
    return dict(knn15=knn, sp=float(spearmanr(dh, dl).correlation), short=float(spearmanr(dh[dh <= t1], dl[dh <= t1]).correlation),
                mid=float(spearmanr(dh[(dh > t1) & (dh <= t2)], dl[(dh > t1) & (dh <= t2)]).correlation),
                long=float(spearmanr(dh[dh > t2], dl[dh > t2]).correlation), cent=float(spearmanr(Dch, Dcl).correlation),
                std=float(Y[:, 0].std()))

m = MAGNET(loss="umap", mid_weight=0.5, random_state=0, n_threads=4).fit(X)
Y0 = m.embedding_.astype(np.float32); P = m.graph_; pairs = m._mid_pairs_
rows = [("SGD pass (start)", metrics(Y0), 0.0)]
print(f"{'variant':<42}{'knn15':>7}{'sp':>7}{'short':>7}{'mid':>7}{'long':>7}{'cent':>7}{'std':>7}{'t(s)':>7}")
def show(name, r, t):
    print(f"{name:<42}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['short']:>7.3f}{r['mid']:>7.3f}{r['long']:>7.3f}{r['cent']:>7.3f}{r['std']:>7.1f}{t:>7.1f}", flush=True)
show(*rows[0])

def refine(name, scale=31.0, n_iter=300, mid_w=0.0, mid_phase=0.5, lr=None, dof=1.0, early=0, exag=1.0, momentum=(0.5, 0.8)):
    Y = Y0.copy(); Y -= Y.mean(0); Y *= scale * np.sqrt(n / 70000.0) / max(float(Y[:, 0].std()), 1e-12)
    t = time.time()
    Y = optimize_tsne(P, Y, n_iter=n_iter, early_iter=early, exaggeration=exag, learning_rate=lr, n_threads=4,
                      mid_pairs=pairs if mid_w > 0 else None, mid_weight=3.0 * mid_w, mid_phase=mid_phase, dof=dof, momentum=momentum)
    r = metrics(Y); rows.append((name, r, time.time() - t)); show(name, r, time.time() - t)
    if name == "hybrid default (31, 300 it)":
        np.savez_compressed(os.path.join(HERE, "..", "cache", "k14_hybrid_default.npz"), Y=Y)

refine("hybrid default (31, 300 it)")
for it in (25, 75, 150):
    refine(f"iterations {it}", n_iter=it)
for s in (15.0, 62.0, 124.0):
    refine(f"hand-over scale {s:.0f}", scale=s)
for w, ph in ((0.1, 1.0), (0.5, 1.0), (0.5, 0.5), (2.0, 1.0)):
    refine(f"mid spring {w}, phase {ph}", mid_w=w, mid_phase=ph)
refine("learning rate 1000", lr=1000.0)
refine("learning rate 20000", lr=20000.0)
refine("momentum 0.5 only", momentum=(0.5, 0.5))
refine("dof 0.5", dof=0.5)
refine("scale 62 + mid 0.5 phase 1", scale=62.0, mid_w=0.5, mid_phase=1.0)
refine("scale 62 + 150 it", scale=62.0, n_iter=150)
json.dump([dict(name=a, time=c, **b) for a, b, c in rows], open(os.path.join(HERE, "k14_hybrid_centroids.json"), "w"), indent=1)
