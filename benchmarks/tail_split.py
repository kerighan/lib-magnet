"""Decoupled exponents: attraction kernel_b x repulsion_b. Which one sets the local optimum, which one the global?"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from metrics import evaluate
from magnet import MAGNET
from datasets import mnist, fmnist
OUT = os.path.join(HERE, "results_tail_law.jsonl")

X, y, key = mnist(5000)
Y0 = MAGNET(random_state=0, n_threads=1).fit_transform(X); Y1 = MAGNET(random_state=0, n_threads=1, repulsion_b=0.6).fit_transform(X)
print("identity check (1 thread, repulsion_b == kernel_b):", np.array_equal(Y0, Y1), flush=True)

for name, (X, y, key) in [("mnist", mnist(20000)), ("fmnist", fmnist(20000))]:
    for m in (2, 3):
        for ab in (0.4, 0.5, 0.6, 0.8):
            for rb in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
                t = time.time()
                Y = MAGNET(n_components=m, kernel_b=ab, repulsion_b=rb, random_state=0, n_threads=10).fit_transform(X)
                dt = time.time() - t
                r = evaluate(X, Y, y, cache_key=key)
                rec = {"stage": "D", "dataset": name, "n": int(len(X)), "m": m, "q": None, "b": ab, "rep_b": rb, "seed": 0, "time": dt, **r}
                open(OUT, "a").write(json.dumps(rec) + "\n")
                print(f"D {name:7s} m={m} att_b={ab} rep_b={rb:<4} T={dt:4.1f}s knn15={r['knn15_preservation']:.4f} trust={r['trustworthiness']:.4f} spear={r['dist_spearman']:.4f} trip={r['triplet_acc']:.4f} sil={r['silhouette']:.3f}", flush=True)
