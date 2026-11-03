"""Seed dispersion of the decisive rows: MAGNET SGD / full-batch / hybrid, UMAP, openTSNE, PaCMAP on MNIST and
Fashion-MNIST 70k, 5 seeds each. Quality metrics only (15-NN preservation, trustworthiness, distance Spearman, triplet
accuracy, 10-NN accuracy, plus the Spearman of the 45 inter-centroid distances). threads=4 (default) pins every library
to 4 threads so that the runs can share the machine (times then not comparable to the table); threads=all runs the
benchmark's protocol (all cores, numba warm-up before the clock) and the times are comparable.
Results appended to benchmarks/results.jsonl with tag seeds_<dataset>_<seed>; summary by seeds_summary.py.
Usage: python benchmarks/seeds.py [datasets=mnist,fmnist] [seeds=5] [methods=sgd,tsne,hybrid,umap,opentsne,pacmap] [threads=4|all]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
_kw = dict(a.split("=", 1) for a in sys.argv[1:])
if _kw.get("threads", "4") != "all":
    os.environ.setdefault("NUMBA_NUM_THREADS", _kw.get("threads", "4")); os.environ.setdefault("OMP_NUM_THREADS", _kw.get("threads", "4"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import spearmanr
from datasets import LOADERS
from metrics import evaluate

kw = dict(a.split("=", 1) for a in sys.argv[1:])
datasets = kw.get("datasets", "mnist,fmnist").split(",")
seeds = int(kw.get("seeds", 5))
methods = kw.get("methods", "sgd,tsne,hybrid,umap,opentsne,pacmap").split(",")
T = -1 if kw.get("threads", "4") == "all" else int(kw.get("threads", "4"))

def centroid_spearman(X, Y, y):
    C = np.array([X[y == c].mean(0) for c in np.unique(y)]); L = np.array([Y[y == c].mean(0) for c in np.unique(y)])
    iu = np.triu_indices(len(C), 1)
    return float(spearmanr(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu], np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1))[iu]).correlation)

def fit(method, X, s):
    if method in ("sgd", "tsne", "hybrid"):
        from magnet import MAGNET
        return MAGNET(loss={"sgd": "umap"}.get(method, method), random_state=s, n_threads=T).fit_transform(X)
    if method == "umap":
        import umap
        return umap.UMAP().fit_transform(X)     # no random_state (it would single-thread the layout); numba's own RNG differs per run
    if method == "opentsne":
        from openTSNE import TSNE
        return np.asarray(TSNE(n_jobs=T, random_state=s).fit(X))
    if method == "pacmap":
        import pacmap
        return pacmap.PaCMAP(random_state=s).fit_transform(X)
    if method == "sktsne":
        from sklearn.manifold import TSNE
        return TSNE(n_jobs=-1, init="pca", random_state=s).fit_transform(X)
    raise ValueError(method)

for ds in datasets:
    X, y, name = LOADERS[ds](70000); y = np.asarray(y).astype(int)
    if "umap" in methods:
        import umap; umap.UMAP(n_epochs=20).fit_transform(np.random.rand(6000, 10).astype(X.dtype))
    for method in methods:
        for s in range(seeds):
            t = time.time(); Y = np.asarray(fit(method, X, s), dtype=np.float32); dt = time.time() - t
            r = evaluate(X, Y, y, cache_key=name)
            r["centroid_spearman"] = centroid_spearman(X, Y, y)
            r.update(method={"sgd": "magnet_sgd", "tsne": "magnet_tsne", "hybrid": "magnet_hybrid"}.get(method, method), dataset=name,
                     time=round(dt, 2), params={"random_state": s, "threads": "all" if T == -1 else T}, tag=f"seeds_{ds}_{s}" + ("" if T == -1 else f"_t{T}"))
            print(json.dumps(r), flush=True)
            with open(os.path.join(HERE, "results.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
