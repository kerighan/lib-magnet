"""Extra local metrics on the six cached MNIST 70k layouts (paper/figures/cache/): kNN preservation at
k = 5 / 15 / 50 / 100 and continuity (trustworthiness with the roles of the two spaces swapped), on the
same 5000-point query subsample as the benchmark table. Usage: python benchmarks/extra_metrics.py"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from datasets import mnist
from metrics import _idx, high_dim_neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

X, y, key = mnist(70000)
n = len(X); q = _idx(n, 5000, 0)
hnn = high_dim_neighbors(X, q, 100, key)              # cached as hnn_mnist70000_k100_q5000.npy
Xq = X[q]
names = {"umap0512": "UMAP 0.5.12", "opentsnefitsne": "openTSNE (FIt-SNE)", "pacmap": "PaCMAP",
         "magnetsgd": "MAGNET SGD", "magnetlosstsne": "MAGNET full-batch", "magnetlosshybrid": "MAGNET hybrid"}
out = {}
print(f"{'method':<20} {'k=5':>6} {'k=15':>6} {'k=50':>6} {'k=100':>6} {'trust15':>8} {'cont15':>7} {'trust50':>8} {'cont50':>7} {'time':>6}")
for slug, label in names.items():
    f = os.path.join(HERE, "..", "paper", "figures", "cache", f"mnist_{slug}.npz")
    if not os.path.exists(f):
        print(label, "missing"); continue
    z = np.load(f); Y, dt = z["Y"], float(z["dt"])
    lnn = NearestNeighbors(n_neighbors=101).fit(Y).kneighbors(Y[q], return_distance=False)
    lnn = np.array([row[row != i][:100] for i, row in zip(q, lnn)])
    r = {"time": dt}
    for k in (5, 15, 50, 100):
        r[f"knn{k}"] = float(np.mean([len(np.intersect1d(a[:k], b[:k])) for a, b in zip(hnn, lnn)]) / k)
    Yq = Y[q]
    for k in (15, 50):
        r[f"trust{k}"] = float(trustworthiness(Xq, Yq, n_neighbors=k))
        r[f"cont{k}"] = float(trustworthiness(Yq, Xq, n_neighbors=k))
    out[label] = r
    print(f"{label:<20} {r['knn5']:>6.3f} {r['knn15']:>6.3f} {r['knn50']:>6.3f} {r['knn100']:>6.3f} {r['trust15']:>8.3f} {r['cont15']:>7.3f} {r['trust50']:>8.3f} {r['cont50']:>7.3f} {dt:>6.1f}", flush=True)
json.dump(out, open(os.path.join(HERE, "cache", "extra_metrics_mnist.json"), "w"), indent=1)
