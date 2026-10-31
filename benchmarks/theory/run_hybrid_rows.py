"""Benchmark rows of loss="hybrid" on the four data sets with the seeds of BENCHMARK.md."""
import sys, os, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
from bench import run
from full_table import DATASETS
for key, n, seeds, label in DATASETS:
    kw = {"loss": "hybrid"}
    if key == "news20": kw["metric"] = "cosine"
    for s in range(seeds):
        run("magnet", key, n=n, tag=f"full_MAGNET_{s}", plot=False, random_state=s, **kw)
