"""E7: per-node attraction exponent from the EMBEDDING-side crowding: run a first phase with the global exponent,
measure for each node the share of its graph edges that end beyond the kernel scale, set b_i = 1 for the nodes that
are not crowded (share below a threshold) and 0.5 for the others, finish from the phase-1 layout. Repulsion Cauchy."""
import sys, os, time, numpy as np, scipy.sparse as sp
SCR = sys.argv[1]; REPO = sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "benchmarks")); sys.path.insert(0, SCR)
from magnet import MAGNET
import _magnet; assert _magnet.__file__.startswith(SCR)
from datasets import mnist
from sklearn.neighbors import NearestNeighbors
def mixture(n, qs, d=50, seed=0):
    rng = np.random.default_rng(seed); y = rng.integers(0, len(qs), n); centers = rng.normal(0, 8, (len(qs), d)).astype(np.float32); X = np.empty((n, d), np.float32)
    for c, q in enumerate(qs):
        idx = np.where(y == c)[0]; basis = np.linalg.qr(rng.normal(size=(d, q)))[0].T.astype(np.float32)
        X[idx] = centers[c] + rng.normal(size=(len(idx), q)).astype(np.float32) @ basis
    return X, y, np.array(qs)[y]
def crowding(M, Y):
    P = M.graph_.tocsr(); dk = M.a_ ** (-1.0 / (2 * M.b_))
    i = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr)); j = P.indices
    far = (np.sqrt(((Y[i] - Y[j]) ** 2).sum(1)) > dk).astype(np.float64)
    return np.bincount(i, weights=far, minlength=P.shape[0]) / np.maximum(np.diff(P.indptr), 1)
def two_phase(X, m, seed, thr, frac=0.4, b_low=0.5, b_high=1.0, total=None, base=0.6):
    total = total or (500 if len(X) <= 10000 else 300)
    n1 = int(frac * total)
    M1 = MAGNET(n_components=m, kernel_b=base, n_epochs=n1, n_threads=4, random_state=seed).fit(X)
    Y1 = M1.embedding_; cr = crowding(M1, Y1)
    nb = np.where(cr <= thr, b_high, b_low).astype(np.float32)
    M2 = MAGNET(n_components=m, kernel_b=nb, n_epochs=total - n1, init=Y1.copy(), n_threads=4, random_state=seed).fit(X)
    return M2.embedding_, cr, nb
for name, m, (X, y, qpt) in [("mix2-10 2D", 2, mixture(20000, [2] * 5 + [10] * 5)), ("mix3-10 3D", 3, mixture(20000, [3] * 5 + [10] * 5)),
                              ("mix2-3-5-10 2D", 2, mixture(20000, [2] * 3 + [3] * 3 + [5] * 2 + [10] * 2)), ("mnist20k 2D", 2, mnist(20000)[:2] + (None,))]:
    print(f"== {name}")
    hnn = NearestNeighbors(n_neighbors=16, algorithm="brute").fit(X).kneighbors(X, return_distance=False)[:, 1:]
    def knn15(Y):
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        inter = np.array([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15
        return inter.mean(), ({q: inter[qpt == q].mean() for q in sorted(set(qpt))} if qpt is not None else {})
    for label, kb in [("global 0.5", 0.5), ("global 0.6", 0.6)] + ([("oracle", np.where(qpt <= m, 1.0, 0.5).astype(np.float32))] if qpt is not None else []):
        for seed in (0, 1):
            t = time.time(); Y = MAGNET(n_components=m, kernel_b=kb, n_threads=4, random_state=seed).fit_transform(X); dt = time.time() - t
            a, per = knn15(Y); print(f"  {label:14s} s={seed} T={dt:4.1f}s knn15 all={a:.4f} " + " ".join(f"q{q}={v:.4f}" for q, v in per.items()), flush=True)
    for thr in (0.05, 0.1, 0.15):
        for seed in (0, 1):
            t = time.time(); Y, cr, nb = two_phase(X, m, seed, thr); dt = time.time() - t
            a, per = knn15(Y)
            extra = " ".join(f"q{q}: crowd med {np.median(cr[qpt==q]):.2f} b=1 {np.mean(nb[qpt==q]==1):.2f}" for q in sorted(set(qpt))) if qpt is not None else f"crowd med {np.median(cr):.2f} b=1 frac {np.mean(nb==1):.2f}"
            print(f"  2-phase thr={thr:<4} s={seed} T={dt:4.1f}s knn15 all={a:.4f} " + " ".join(f"q{q}={v:.4f}" for q, v in per.items()) + "  | " + extra, flush=True)
