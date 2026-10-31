"""E5: attraction optimum vs q/m with Cauchy repulsion, to fix the law b(q, m) of the per-node exponent."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from tail_law import synthetic
from metrics import evaluate
for m, qs in ((2, (3, 5)), (3, (3, 5, 10))):
    for q in qs:
        X, y = synthetic(20000, q)
        for b in (0.4, 0.5, 0.6, 0.8, 1.0):
            t = time.time(); Y = MAGNET(n_components=m, kernel_b=b, n_threads=4, random_state=0).fit_transform(X); dt = time.time() - t
            r = evaluate(X, Y, y, cache_key=f"gaussq{q}_clean")
            print(f"m={m} q={q:2d} b={b:<4} T={dt:4.1f}s knn15={r['knn15_preservation']:.4f} trust={r['trustworthiness']:.4f} spear={r['dist_spearman']:.4f}", flush=True)
