"""Does a per-node diffusion scale for the mid-range attraction help? fixed walk length vs the 2-step spread proxy
(a per-role variant from structural-role signatures was also tried and removed: it changed nothing either).
Usage: python benchmarks/mid_scale_study.py [n]"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from compare import KEYS
from bench import run

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
seeds = 2
configs = [("fixed_wl4", {}), ("no_mid", {"mid_weight": 0.0}), ("spread", {"mid_scale": "spread"}),
           ("spread_w0.3", {"mid_scale": "spread", "mid_weight": 0.3})]
table = {}
for name, kw in configs:
    rows = []
    for s in range(seeds):
        rows.append(run("magnet", "mnist", n=n, tag=f"rw_{name}_s{s}", plot=False, random_state=s, **kw))
    table[name] = rows
print("\n" + "config".ljust(16) + "".join(k[:11].rjust(18) for k in KEYS))
for name, rows in table.items():
    print(name.ljust(16) + "".join(f"{np.mean([r[k] for r in rows]):8.4f}±{np.std([r[k] for r in rows]):.4f}".rjust(18) for k in KEYS))
