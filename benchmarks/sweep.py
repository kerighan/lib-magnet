"""Run several MAGNET configurations on one dataset and print a comparison table.
Usage: python benchmarks/sweep.py DATASET N  (configs are defined below)"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bench import run

CONFIGS = {
    "default": {},
    "mid_w0.1_wl2": dict(mid_weight=0.1, walk_len=2, mid_rate=0.25, mid_phase=0.5),
    "mid_w0.3_wl2": dict(mid_weight=0.3, walk_len=2, mid_rate=0.25, mid_phase=0.5),
    "mid_w0.1_wl4": dict(mid_weight=0.1, walk_len=4, mid_rate=0.25, mid_phase=0.5),
    "mid_w0.3_wl4": dict(mid_weight=0.3, walk_len=4, mid_rate=0.5, mid_phase=0.5),
    "mid_w1_wl3": dict(mid_weight=1.0, walk_len=3, mid_rate=0.5, mid_phase=0.3),
}

if __name__ == "__main__":
    dataset, n = sys.argv[1], int(sys.argv[2])
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CONFIGS)
    rows = []
    for name in names:
        r = run("magnet", dataset, n=n, tag=name, plot=True, **CONFIGS[name])
        rows.append((name, r))
    keys = ["time", "knn15_preservation", "trustworthiness", "dist_spearman", "triplet_acc", "knn_acc", "silhouette"]
    print("\n" + "config".ljust(16) + "".join(k[:12].rjust(13) for k in keys))
    for name, r in rows:
        print(name.ljust(16) + "".join(f"{r.get(k, float('nan')):13.4f}" for k in keys))
