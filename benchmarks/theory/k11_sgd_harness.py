"""Times the native SGD alone (graph, schedule and mid pairs built once; m._optimize on a fresh init per variant).
Usage: python benchmarks/theory/k11_sgd_harness.py (from the repo root); MAGNET_PROFILE=1 prints busy / wait per call."""
"""Times the native SGD alone: graph, schedule and mid pairs built once, then m._optimize on a fresh init per variant.
Usage: python sgd_harness.py [path of the _magnet build dir to test first] """
import sys, os, time
if len(sys.argv) > 1: sys.path.insert(0, sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(1, os.path.dirname(B)); sys.path.insert(2, B)
import numpy as np
import _magnet; print("_magnet:", _magnet.__file__)
from datasets import LOADERS
from magnet import MAGNET
data, labels, name = LOADERS["mnist"](70000)
m = MAGNET(random_state=0, n_epochs=1).fit(data)
P = m.graph_; n = P.shape[0]
EPOCHS = 60
head, tail, eps = m._edge_schedule(P, EPOCHS)
rng = np.random.default_rng(0); Y0 = (rng.normal(size=(n, 2)) * 10).astype(np.float32)
def run(nm, threads=20, mid=0.1, reps=2, **attrs):
    saved = {k: getattr(m, k) for k in attrs}
    for k, v in attrs.items(): setattr(m, k, v)
    m.n_threads = threads
    best = 1e9
    for _ in range(reps):
        Y = Y0.copy(); t = time.time(); m._optimize(Y, head, tail, eps, P, EPOCHS, n_fixed=0, seed=0, mid_weight=mid); best = min(best, time.time() - t)
    for k, v in saved.items(): setattr(m, k, v)
    pos = np.minimum(1.0, 1.0 / eps).sum() * EPOCHS
    print(f"{nm:34s} {best:6.2f}s  ({best * threads / (pos * 6.25) * 3.5e9:5.0f} cycles/force x threads)", flush=True)
    return best
if __name__ == "__main__":
    for t in (1, 4, 10, 20):
        run(f"hogwild, {t} threads", threads=t, parallel="hogwild")
        run(f"owner, {t} threads", threads=t, parallel="owner")
    run("1 thread, b=1 (no pow)", threads=1, b_=1.0)
    run("1 thread, neg 1", threads=1, negative_sample_rate=1)
    run("1 thread, neg 10", threads=1, negative_sample_rate=10)
    run("1 thread, mid 0", threads=1, mid=0.0)
    run("1 thread, mid walks", threads=1, mid_source="walk", _mid_pairs_=None)
    run("20 threads, mid 0", threads=20, mid=0.0)
    run("20 threads, b=1", threads=20, b_=1.0)
