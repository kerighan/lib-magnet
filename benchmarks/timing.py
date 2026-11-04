"""Timing benchmark on a quiet machine (the round-6 reviewer's condition for precise speed claims): fits only, no
evaluation, all cores, numba warm-up for UMAP before the clock, the configurations interleaved and repeated so that a
load change hits every row alike. MAGNET sampled / full-batch / sequence / sequence + post-process, UMAP, PaCMAP,
openTSNE, scikit-learn's Barnes-Hut t-SNE; MNIST and Fashion-MNIST 70k; 3 rounds. Records the load average before
each fit. -> benchmarks/cache/timing_quiet.json. Usage: python benchmarks/timing.py [rounds=3] [datasets=mnist,fmnist]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from datasets import LOADERS
kw = dict(a.split("=", 1) for a in sys.argv[1:]); ROUNDS = int(kw.get("rounds", 3)); DS = kw.get("datasets", "mnist,fmnist").split(",")
OUT = os.path.join(HERE, "cache", "timing_quiet.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def fit(method, X, s):
    from magnet import MAGNET
    if method == "MAGNET sampled": return MAGNET(random_state=s).fit_transform(X)
    if method == "MAGNET full-batch": return MAGNET(loss="tsne", random_state=s).fit_transform(X)
    if method == "MAGNET sequence": return MAGNET(loss="hybrid", random_state=s).fit_transform(X)
    if method == "MAGNET sequence + post": return MAGNET(loss="hybrid", contraction=0.7, random_state=s).fit_transform(X)
    if method == "UMAP": import umap; return umap.UMAP().fit_transform(X)
    if method == "PaCMAP": import pacmap; return pacmap.PaCMAP(random_state=s).fit_transform(X)
    if method == "openTSNE": from openTSNE import TSNE; return np.asarray(TSNE(n_jobs=-1, random_state=s).fit(X))
    if method == "Barnes-Hut t-SNE": from sklearn.manifold import TSNE; return TSNE(n_jobs=-1, init="pca", random_state=s).fit_transform(X)
METHODS = ["MAGNET sampled", "MAGNET full-batch", "MAGNET sequence", "MAGNET sequence + post", "UMAP", "PaCMAP", "openTSNE", "Barnes-Hut t-SNE"]
for ds in DS:
    X, y, name = LOADERS[ds](70000); X = np.asarray(X, dtype=np.float32)
    import umap; umap.UMAP(n_epochs=20).fit_transform(np.random.rand(6000, 10).astype(X.dtype))   # numba warm-up on the data's dtype
    fit("MAGNET sampled", X[:5000], 0); fit("MAGNET full-batch", X[:5000], 0)                      # extension / FFT plan warm-up
    print(f"\n=== {ds}: load {os.getloadavg()}", flush=True)
    for r in range(ROUNDS):
        for m in METHODS:
            k_ = f"{ds}/{m}/{r}"
            if k_ in res: continue
            load = os.getloadavg()[0]; t = time.time(); fit(m, X, r); dt = time.time() - t
            res[k_] = dict(t=dt, load=load); json.dump(res, open(OUT, "w"), indent=1)
            print(f"  round {r} {m:<24} {dt:7.1f} s   (load before {load:.1f})", flush=True)
print("\n==== summary: mean (min-max) over rounds, seconds")
for ds in DS:
    print(ds)
    for m in METHODS:
        v = [res[f"{ds}/{m}/{r}"]["t"] for r in range(ROUNDS) if f"{ds}/{m}/{r}" in res]
        if v: print(f"  {m:<24} {np.mean(v):7.1f}  ({min(v):.1f}-{max(v):.1f})  load {np.mean([res[f'{ds}/{m}/{r}']['load'] for r in range(ROUNDS) if f'{ds}/{m}/{r}' in res]):.1f}")
