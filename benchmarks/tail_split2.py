"""Repulsion exponent law vs target dimension at fixed attraction exponent 1/2, plus 3-seed candidates for the default."""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from metrics import evaluate
from magnet import MAGNET
from datasets import mnist, fmnist, news20
OUT = os.path.join(HERE, "results_tail_law.jsonl")

def run(stage, name, X, y, key, m, ab, rb, seed, **kw):
    t = time.time(); Y = MAGNET(n_components=m, kernel_b=ab, repulsion_b=rb, random_state=seed, n_threads=10, **kw).fit_transform(X); dt = time.time() - t
    r = evaluate(X, Y, y, cache_key=key)
    open(OUT, "a").write(json.dumps({"stage": stage, "dataset": name, "n": int(len(X)), "m": m, "q": None, "b": ab, "rep_b": rb, "seed": seed, "time": dt, **r}) + "\n")
    print(f"{stage} {name:7s} m={m} att_b={ab} rep_b={rb:<4} s={seed} T={dt:4.1f}s knn15={r['knn15_preservation']:.4f} trust={r['trustworthiness']:.4f} spear={r['dist_spearman']:.4f} trip={r['triplet_acc']:.4f} sil={r['silhouette']:.3f}", flush=True)

data = {"mnist": mnist(20000), "fmnist": fmnist(20000)}
for name in ("mnist", "fmnist"):
    X, y, key = data[name]
    for m in (1, 4, 6):
        for rb in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0):
            run("E", name, X, y, key, m, 0.5, rb, 0)
data["news20"] = news20()
for name in ("mnist", "fmnist", "news20"):
    X, y, key = data[name]; kw = {"metric": "cosine"} if name == "news20" else {}
    for m in (2, 3):
        for ab, rb in ((0.5, 0.5), (0.5, 1.0), (0.5, 1.5), (0.6, 0.6), (0.6, 1.0), (0.8, 0.8)):
            for seed in range(3):
                run("F", name, X, y, key, m, ab, rb, seed, **kw)
