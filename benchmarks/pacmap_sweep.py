"""PaCMAP's mid-near knobs on MNIST 70k, to pick a 'stiff' PaCMAP start for the staged 2 x 2 design with the same
tuning budget as MAGNET's spring (one sweep on MNIST, frozen before Fashion-MNIST): MN_ratio x phase lengths, 1 run
each, 15-NN / Spearman / centroids; also PaCMAP's time with and without random_state (it changes the parallelism).
-> benchmarks/cache/pacmap_sweep.json (key "stiff" = the setting with the best Spearman). Usage: python benchmarks/pacmap_sweep.py"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, pacmap
from scipy.stats import spearmanr
from datasets import mnist
from metrics import evaluate
X, y, name = mnist(70000); y = np.asarray(y).astype(int)
def cent(Y):
    C = np.array([X[y == c].mean(0) for c in range(10)]); L = np.array([Y[y == c].mean(0) for c in range(10)]); iu = np.triu_indices(10, 1)
    return float(spearmanr(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu], np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1))[iu]).correlation)
out = {"runs": []}
pacmap.PaCMAP(num_iters=(5, 5, 5)).fit_transform(X[:6000])   # numba warm-up
for rs in (None, 0):
    t = time.time(); Y = pacmap.PaCMAP(random_state=rs).fit_transform(X); dt = time.time() - t
    r = evaluate(X, Y, y, cache_key=name); print(f"random_state={rs}: {dt:.1f} s, 15-NN {r['knn15_preservation']:.3f}, Spearman {r['dist_spearman']:.3f}", flush=True)
    out[f"time_rs_{rs}"] = dt
for mn in (0.5, 1.0, 2.0, 4.0):
    for iters in ((100, 100, 250), (250, 250, 250)):
        t = time.time(); Y = pacmap.PaCMAP(MN_ratio=mn, num_iters=iters).fit_transform(X); dt = time.time() - t
        r = evaluate(X, Y, y, cache_key=name); r["centroid_spearman"] = cent(Y)
        row = dict(MN_ratio=mn, num_iters=iters, time=dt, **{k: r[k] for k in ("knn15_preservation", "dist_spearman", "triplet_acc", "centroid_spearman")})
        out["runs"].append(row); print(json.dumps(row), flush=True)
best = max(out["runs"], key=lambda r: r["dist_spearman"]); out["stiff"] = dict(MN_ratio=best["MN_ratio"], num_iters=best["num_iters"])
print("stiff =", out["stiff"]); json.dump(out, open(os.path.join(HERE, "cache", "pacmap_sweep.json"), "w"), indent=1)
