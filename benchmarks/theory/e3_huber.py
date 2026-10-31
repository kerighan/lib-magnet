"""E3: Huber attraction (L2 below c, L1 above, L1 tail) against the power kernels, repulsion Cauchy everywhere."""
import sys, os, time, numpy as np
SCR = sys.argv[1]; REPO = sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "benchmarks")); sys.path.insert(0, SCR)
from magnet import MAGNET
import _magnet; assert _magnet.__file__.startswith(SCR), _magnet.__file__
from datasets import mnist, fmnist
from tail_law import synthetic
from metrics import evaluate
assert MAGNET.__module__.startswith("magnet") and os.path.dirname(os.path.dirname(sys.modules["magnet"].__file__)) == SCR
sets = [("mnist20k", lambda: mnist(20000)), ("gauss_q2", lambda: synthetic(20000, 2) + ("gaussq2_clean",)),
        ("gauss_q10", lambda: synthetic(20000, 10) + ("gaussq10_clean",)), ("fmnist20k", lambda: fmnist(20000))]
confs = [("power", 0.5), ("power", 0.6), ("power", 1.0)] + [("huber", c) for c in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)]
for name, load in sets:
    X, y, key = load()
    for kern, b in confs:
        for seed in (0, 1):
            t = time.time()
            M = MAGNET(kernel=kern, kernel_b=b, n_threads=4, random_state=seed); Y = M.fit_transform(X); dt = time.time() - t
            r = evaluate(X, Y, y, cache_key=key)
            print(f"{name:10s} {kern:6s} b/c={b:<5} s={seed} a={M.a_:7.2f} T={dt:4.1f}s knn15={r['knn15_preservation']:.4f} trust={r['trustworthiness']:.4f} spear={r['dist_spearman']:.4f} sil={r['silhouette']:.3f}", flush=True)
