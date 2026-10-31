"""E2: how many kNN edges end up beyond the kernel scale (the 'contamination' of the attraction), per dataset,
at the default settings; to be compared with the measured attraction optimum."""
import sys, os, numpy as np, scipy.sparse as sp
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from magnet.roles import local_intrinsic_dimension
from datasets import mnist, fmnist, news20
from tail_law import synthetic
from metrics import evaluate
sets = [("mnist5k", lambda: mnist(5000), {}), ("mnist20k", lambda: mnist(20000), {}), ("fmnist20k", lambda: fmnist(20000), {}),
        ("news20", news20, {"metric": "cosine"})]
for q in (2, 3, 5, 10, 20):
    sets.append((f"gauss_q{q}", (lambda q=q: synthetic(20000, q) + (f"gaussq{q}_clean",)), {}))
print(f"{'dataset':10s} {'n':>6s} {'idim':>5s} {'knn15':>6s} {'d_kern':>6s} {'med_d':>6s} {'p>dk':>6s} {'p>1':>6s} {'p>2':>6s} {'near_grip/far_drag(b=.5)':>8s}")
for name, load, kw in sets:
    X, y, key = load()
    M = MAGNET(n_threads=4, random_state=0, **kw); Y = M.fit_transform(X)
    P = sp.triu(M.graph_.tocoo(), k=1)
    d = np.sqrt(((Y[P.row] - Y[P.col]) ** 2).sum(1))
    a, b = M.a_, M.b_
    dk = a ** (-1.0 / (2 * b))
    r = evaluate(X, Y, y, cache_key=key)
    idim = float(np.median(local_intrinsic_dimension(X[:5000] if not sp.issparse(X) else X[:5000].toarray(), 15)))
    print(f"{name:10s} {len(X):6d} {idim:5.1f} {r['knn15_preservation']:6.3f} {dk:6.2f} {np.median(d):6.3f} {np.mean(d > dk):6.3f} {np.mean(d > 1):6.3f} {np.mean(d > 2):6.3f}", flush=True)
