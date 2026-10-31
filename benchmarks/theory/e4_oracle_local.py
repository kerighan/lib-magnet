"""E4: per-node attraction exponent with an oracle (true local dimension) on a 2-D + 10-D cluster mixture,
repulsion Cauchy (decoupled) everywhere. Upper bound of what an adaptive exponent can gain."""
import sys, os, time, numpy as np
SCR = sys.argv[1]; REPO = sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "benchmarks")); sys.path.insert(0, SCR)
from magnet import MAGNET
from magnet.roles import local_intrinsic_dimension
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng(0)
n, d, qs = 20000, 50, [2] * 5 + [10] * 5
y = rng.integers(0, 10, n); centers = rng.normal(0, 8, (10, d)).astype(np.float32); X = np.empty((n, d), np.float32)
for c, q in enumerate(qs):
    idx = np.where(y == c)[0]; basis = np.linalg.qr(rng.normal(size=(d, q)))[0].T.astype(np.float32)
    X[idx] = centers[c] + rng.normal(size=(len(idx), q)).astype(np.float32) @ basis
qpt = np.array(qs)[y]
hnn = NearestNeighbors(n_neighbors=16, algorithm="brute").fit(X).kneighbors(X, return_distance=False)[:, 1:]
def knn15(Y):
    lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    inter = np.array([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15
    return inter.mean(), inter[qpt == 2].mean(), inter[qpt == 10].mean()
idim = local_intrinsic_dimension(X, 15)
print("local idim estimate: q=2 points median %.1f, q=10 points median %.1f" % (np.median(idim[qpt == 2]), np.median(idim[qpt == 10])))
est = np.where(idim <= 4, 1.0, 0.5).astype(np.float32)
print("estimator picks b=1 for %.2f of q=2 points and %.2f of q=10 points" % ((est[qpt == 2] == 1).mean(), (est[qpt == 10] == 1).mean()))
confs = [("global 0.5", 0.5), ("global 0.6", 0.6), ("global 0.8", 0.8), ("global 1.0", 1.0),
         ("oracle 1.0/0.5", np.where(qpt == 2, 1.0, 0.5).astype(np.float32)), ("oracle 1.0/0.4", np.where(qpt == 2, 1.0, 0.4).astype(np.float32)),
         ("estimated 1.0/0.5", est)]
for label, kb in confs:
    for seed in (0, 1):
        t = time.time(); Y = MAGNET(kernel_b=kb, n_threads=4, random_state=seed).fit_transform(X); dt = time.time() - t
        a, a2, a10 = knn15(Y)
        print(f"{label:18s} s={seed} T={dt:4.1f}s knn15 all={a:.4f} q2={a2:.4f} q10={a10:.4f}", flush=True)
