"""Does the optimal kernel exponent b follow the intrinsic dimension of the data?
Synthetic Gaussian clusters of known intrinsic dimension embedded in 50-D (+ the real data sets):
TwoNN estimate of the intrinsic dimension, sweep of kernel_b, best b by 15-NN preservation."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from metrics import evaluate
from magnet import MAGNET
from magnet.knn import knn_nndescent


def twonn(X, n_sample=5000, seed=0):
    """TwoNN intrinsic dimension estimator (Facco et al. 2017) from the two nearest neighbours."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X[idx])
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    F = np.arange(1, len(mu) + 1) / len(mu)
    keep = slice(0, int(0.9 * len(mu)))  # discard the tail as in the paper
    x = np.log(mu[keep]); y = -np.log(1 - F[keep])
    return float((x * y).sum() / (x * x).sum())


def synthetic(n, m, d=50, clusters=10, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, clusters, n)
    centers = rng.normal(0, 8, (clusters, d)).astype(np.float32)
    X = np.empty((n, d), np.float32)
    for c in range(clusters):
        idx = np.where(y == c)[0]
        basis = np.linalg.qr(rng.normal(size=(d, m)))[0].T.astype(np.float32)   # orthonormal m-dim subspace
        X[idx] = centers[c] + rng.normal(size=(len(idx), m)).astype(np.float32) @ basis + rng.normal(0, 0.02, (len(idx), d)).astype(np.float32)
    return X, y


def main():
    bs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    rows = []
    cases = [(f"gauss m={m}", *synthetic(20000, m)) for m in (2, 4, 8, 16, 32)]
    from datasets import mnist, fmnist, news20
    cases += [("MNIST 20k", *mnist(20000)[:2]), ("Fashion 20k", *fmnist(20000)[:2]), ("20 newsgroups", *news20()[:2])]
    for name, X, y in cases:
        key = name.replace(" ", "_").replace("=", "")
        ID = twonn(X)
        res = {}
        for b in bs:
            kw = {"metric": "cosine"} if "newsgroups" in name else {}
            Y = MAGNET(kernel_b=b, random_state=0, **kw).fit_transform(X)
            r = evaluate(X, Y, y, cache_key=key)
            res[b] = (r["knn15_preservation"], r["trustworthiness"], r["dist_spearman"])
        best = max(bs, key=lambda b: res[b][0])
        rows.append((name, ID, best, res))
        print(f"{name:16s} TwoNN ID={ID:5.1f}  best b (15-NN)={best}  " + " ".join(f"b{b}:{res[b][0]:.3f}" for b in bs), flush=True)
    print("\nsummary: name, intrinsic dim, best b")
    for name, ID, best, _ in rows:
        print(f"  {name:16s} {ID:5.1f} {best}")


if __name__ == "__main__":
    main()
