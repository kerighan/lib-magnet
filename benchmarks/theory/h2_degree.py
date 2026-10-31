"""Does the attraction optimum depend on the node's in-degree in the kNN graph (how many points list it as a
neighbour)? Per-node 15-NN preservation by in-degree quintile for each exponent, on 20 newsgroups and MNIST 20k,
then a per-node exponent driven by the in-degree against the constants."""
import sys, os, time, numpy as np, scipy.sparse as sp
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist, news20
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
for name, (X, y, key), kw in (("news20", news20(), {"metric": "cosine"}), ("mnist20k", mnist(20000), {})):
    Xn = normalize(X) if kw.get("metric") == "cosine" else X
    hnn = NearestNeighbors(n_neighbors=16, algorithm="brute").fit(Xn).kneighbors(Xn, return_distance=False)[:, 1:]
    def knn15(Y):
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        return np.array([len(np.intersect1d(a, c)) for a, c in zip(hnn, lnn)]) / 15
    indeg = np.bincount(hnn.ravel(), minlength=len(X))
    edges = np.quantile(indeg, [0.2, 0.4, 0.6, 0.8]); binid = np.searchsorted(edges, indeg, side="right")
    labels = [f"<= {edges[0]:.0f}", f"..{edges[1]:.0f}", f"..{edges[2]:.0f}", f"..{edges[3]:.0f}", f"> {edges[3]:.0f}"]
    print(f"== {name}: in-degree quintile edges {edges}, share with in-degree < 5: {np.mean(indeg < 5):.2f}, < 10: {np.mean(indeg < 10):.2f}", flush=True)
    per = {}
    for b in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        sc = [knn15(MAGNET(kernel_b=b, n_threads=4, random_state=s, **kw).fit_transform(X)) for s in (0, 1)]
        e = np.mean(sc, 0); per[b] = e
        print(f"   b={b:<4} all {e.mean():.4f}  by in-degree quintile: " + "  ".join(f"{l}: {e[binid==k].mean():.4f}" for k, l in enumerate(labels)), flush=True)
    best = {k: max(per, key=lambda b: per[b][binid == k].mean()) for k in range(5)}
    print("   best exponent per quintile:", " ".join(f"{labels[k]}: {best[k]}" for k in range(5)), flush=True)
    for label, law in (("oracle per quintile", lambda d: np.array([best[k] for k in binid])),
                       ("in-degree law 0.8/0.6/0.5", lambda d: np.where(d < 5, 0.8, np.where(d < 15, 0.6, 0.5))),
                       ("in-degree law 0.7/0.5", lambda d: np.where(d < 10, 0.7, 0.5))):
        nb = law(indeg).astype(np.float32)
        sc = [knn15(MAGNET(kernel_b=nb, n_threads=4, random_state=s, **kw).fit_transform(X)) for s in (0, 1)]
        e = np.mean(sc, 0)
        print(f"   {label:26s} all {e.mean():.4f}  by quintile: " + "  ".join(f"{e[binid==k].mean():.4f}" for k in range(5)) + f"   (mean b {nb.mean():.2f})", flush=True)
