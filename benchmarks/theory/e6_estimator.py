"""E6: production estimator of the per-node attraction exponent: neighbour-smoothed MLE local dimension q_i,
b_i = 1 when q_i <= m + 0.5 (not crowded: L2), 0.5 otherwise (crowded: L1); repulsion Cauchy. 2-D and 3-D mixtures."""
import sys, os, time, numpy as np, scipy.sparse as sp
SCR = sys.argv[1]; REPO = sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "benchmarks")); sys.path.insert(0, SCR)
from magnet import MAGNET
from magnet.knn import knn_nndescent
from sklearn.neighbors import NearestNeighbors
def mixture(n, qs, d=50, seed=0):
    rng = np.random.default_rng(seed); y = rng.integers(0, len(qs), n); centers = rng.normal(0, 8, (len(qs), d)).astype(np.float32); X = np.empty((n, d), np.float32)
    for c, q in enumerate(qs):
        idx = np.where(y == c)[0]; basis = np.linalg.qr(rng.normal(size=(d, q)))[0].T.astype(np.float32)
        X[idx] = centers[c] + rng.normal(size=(len(idx), q)).astype(np.float32) @ basis
    return X, y, np.array(qs)[y]
def estimator(X, m, k=15, smooth=True):
    ind, dist = knn_nndescent(X, k + 1, random_state=0)
    d = np.maximum(dist[:, 1:], 1e-12); q = (k - 1) / np.maximum(np.log(d[:, -1:] / d[:, :-1]).sum(1), 1e-12)
    if smooth:  # average log q over the neighbourhood (self + k neighbours)
        lq = np.log(q); q = np.exp((lq + lq[ind[:, 1:]].sum(1)) / (k + 1))
    return q, np.where(q <= m + 0.5, 1.0, 0.5).astype(np.float32)
for m, qs in ((2, [2] * 5 + [10] * 5), (3, [3] * 5 + [10] * 5), (2, [2] * 3 + [3] * 3 + [5] * 2 + [10] * 2)):
    X, y, qpt = mixture(20000, qs)
    hnn = NearestNeighbors(n_neighbors=16, algorithm="brute").fit(X).kneighbors(X, return_distance=False)[:, 1:]
    def knn15(Y):
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        inter = np.array([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15
        return inter.mean(), {q: inter[qpt == q].mean() for q in sorted(set(qs))}
    for smooth in (False, True):
        qh, est = estimator(X, m, smooth=smooth)
        print(f"m={m} qs={sorted(set(qs))} smooth={smooth}: " + " ".join(f"q{q}: est median {np.median(qh[qpt==q]):.1f} frac b=1 {np.mean(est[qpt==q]==1):.2f}" for q in sorted(set(qs))))
    confs = [("global 0.5", 0.5), ("global 0.6", 0.6), ("global 1.0", 1.0), ("oracle", np.where(qpt <= m, 1.0, 0.5).astype(np.float32)), ("estimated", est)]
    for label, kb in confs:
        for seed in (0, 1):
            t = time.time(); Y = MAGNET(n_components=m, kernel_b=kb, n_threads=4, random_state=seed).fit_transform(X); dt = time.time() - t
            a, per = knn15(Y)
            print(f"  {label:10s} s={seed} T={dt:4.1f}s knn15 all={a:.4f} " + " ".join(f"q{q}={v:.4f}" for q, v in per.items()), flush=True)
