"""Multi-seed comparison of methods/configs on a dataset: mean +- std of every metric.
Usage: python benchmarks/compare.py DATASET N SEEDS name=method:json_kwargs [name=method:json_kwargs ...]
Example: python benchmarks/compare.py mnist 20000 3 umap=umap:{} v3=magnet:{} mid=magnet:'{"mid_weight":0.1,"walk_len":4}'"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bench import run

KEYS = ["time", "knn15_preservation", "trustworthiness", "dist_spearman", "triplet_acc", "knn_acc", "silhouette"]

if __name__ == "__main__":
    dataset, n, seeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    specs = []
    for a in sys.argv[4:]:
        name, rest = a.split("=", 1)
        method, js = rest.split(":", 1)
        specs.append((name, method, json.loads(js)))
    table = {}
    for name, method, kw in specs:
        rows = []
        for s in range(seeds):
            k = dict(kw)
            if method == "magnet":
                k["random_state"] = s
            rows.append(run(method, dataset, n=n, tag=f"{name}_s{s}", plot=(s == 0), **k))
        table[name] = rows
    print("\n" + "config".ljust(14) + "".join(k[:11].rjust(18) for k in KEYS))
    for name, rows in table.items():
        cells = []
        for k in KEYS:
            v = [r[k] for r in rows if k in r]
            cells.append(f"{np.mean(v):8.4f}±{np.std(v):.4f}" if v else " " * 15)
        print(name.ljust(14) + "".join(c.rjust(18) for c in cells))
