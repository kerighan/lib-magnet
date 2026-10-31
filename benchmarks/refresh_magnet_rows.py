"""Re-run only the MAGNET (SGD, default) rows of the full benchmark table after a change of defaults.
Same datasets, seeds and tags as full_table.py; make_table.py keeps the latest run per tag."""
import sys, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from bench import run
from full_table import DATASETS

for key, n, seeds, label in DATASETS:
    for s in range(seeds):
        kw = {"random_state": s}
        if key == "news20":
            kw["metric"] = "cosine"
        r = run("magnet", key, n=n, tag=f"full_MAGNET_sgd_{s}", plot=False, **kw)
        print(f"{label} seed {s}: T={r['time']:.1f}s knn15={r['knn15_preservation']:.3f} spear={r.get('dist_spearman', float('nan')):.3f} sil={r.get('silhouette', float('nan')):.3f}", flush=True)
