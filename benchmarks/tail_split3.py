"""Decoupled exponents on synthetic data with and without crowding (q = 2 and q = 10 in 2-D), then a clean timing
of the new default against the coupled kernel on MNIST 70k."""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from metrics import evaluate
from magnet import MAGNET
from tail_law import synthetic
from datasets import mnist
OUT = os.path.join(HERE, "results_tail_law.jsonl")
for q in (2, 10):
    X, y = synthetic(20000, q); key = f"gaussq{q}_clean"
    for ab in (0.4, 0.5, 0.6, 0.8, 1.0):
        for rb in (0.5, 1.0, 1.5, 2.0):
            Y = MAGNET(kernel_b=ab, repulsion_b=rb, random_state=0).fit_transform(X)
            r = evaluate(X, Y, y, cache_key=key)
            open(OUT, "a").write(json.dumps({"stage": "G", "dataset": f"gauss_q{q}", "n": 20000, "m": 2, "q": q, "b": ab, "rep_b": rb, "seed": 0, **r}) + "\n")
            print(f"G q={q} att_b={ab} rep_b={rb} knn15={r['knn15_preservation']:.4f} trust={r['trustworthiness']:.4f} spear={r['dist_spearman']:.4f}", flush=True)
print("== timing MNIST 70k, 3 seeds")
X, y, key = mnist(70000)
for label, kw in (("coupled 0.6/0.6", {"repulsion_b": None}), ("default 0.6/1.0", {}), ("0.5/1.0", {"kernel_b": 0.5})):
    ts, ks, ss, sil = [], [], [], []
    for s in range(3):
        t = time.time(); Y = MAGNET(random_state=s, **kw).fit_transform(X); ts.append(time.time() - t)
        r = evaluate(X, Y, y, cache_key=key); ks.append(r["knn15_preservation"]); ss.append(r["dist_spearman"]); sil.append(r["silhouette"])
    print(f"{label:18s} T={np.mean(ts):.2f}±{np.std(ts):.2f}s knn15={np.mean(ks):.4f} spear={np.mean(ss):.4f} sil={np.mean(sil):.3f}", flush=True)
