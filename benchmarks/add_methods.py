"""Run the methods added to the benchmark (densMAP, PHATE, h-NNE, t-SNE-CUDA) on the four data sets with the
seeds of BENCHMARK.md, appending to results.jsonl with the tags make_table.py expects.
Usage: LD_LIBRARY_PATH=<nvidia pip libs> python benchmarks/add_methods.py [method ...]"""
import sys, os, traceback, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from bench import run
from full_table import METHODS, DATASETS
wanted = sys.argv[1:] or ["densmap", "hnne", "tsnecuda", "phate"]
for key, n, seeds, label in DATASETS:
    for name, method, kw in METHODS:
        if method not in wanted:
            continue
        kw = dict(kw)
        if key == "news20":
            if method in ("umap", "opentsne", "magnet", "densmap", "tsnecuda"): kw["metric"] = "cosine"
            elif method == "phate": kw["knn_dist"] = "cosine"
            elif method == "pacmap": kw["distance"] = "angular"
        for s in range(seeds):
            try:
                run(method, key, n=n, tag=f"full_{name.split()[0]}_{s}", plot=False, **kw)
            except Exception as e:
                print(f"[{name} on {key}] failed: {e}", file=sys.stderr); traceback.print_exc(); break
