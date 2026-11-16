"""k49: supervised and semi-supervised layouts, fit(X, y) with -1 for unlabelled points (UMAP's categorical
intersection, target_weight 0.5 for both libraries). MNIST 70k; a random fraction of the labels is kept (seed 0).
Axes: acc_unl = 10-NN classification of the unlabelled points in the layout from the labelled ones (the
semi-supervised use: does the layout propagate the few labels?), nn15 = 15-NN overlap over all points (the cost in
local fidelity, 5000 query points), ari = k-means (k = 10, 10 inits) ARI against the true labels. Three runs
(seeds 0-2); the unsupervised layouts are scored against the labelled set of every fraction, so that the rows of one
fraction share their labelled points. With labels MAGNET drops its data-only mid-range attraction by default.
Usage: python benchmarks/theory/k49_semi_supervised.py"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist
from sklearn.neighbors import NearestNeighbors
from magnet import MAGNET
T = 8
OUT = os.path.join(HERE, "k49_semi_supervised.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
X, y, _ = mnist(70000)
rng = np.random.default_rng(0); u = rng.random(len(X))
Q = np.sort(rng.choice(len(X), 5000, replace=False))   # 15-NN overlap on 5000 query points, neighbours among all
IDX_HD = NearestNeighbors(n_neighbors=16).fit(X).kneighbors(X[Q], return_distance=False)[:, 1:]
def nn15(Y):
    I = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y[Q], return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / 15 for a, b in zip(IDX_HD, I)]))
FRACS = [0.01, 0.1, 1.0]
def acc(Y, lab):
    unl = ~lab
    return float((KNeighborsClassifier(10).fit(Y[lab], y[lab]).predict(Y[unl]) == y[unl]).mean()) if unl.any() else None
def axes(Y):
    return dict(nn15=nn15(Y), ari=float(adjusted_rand_score(y, KMeans(10, n_init=10, random_state=0).fit_predict(Y))))
def magnet(**k): return lambda t, s: MAGNET(random_state=s, n_threads=T, **k).fit_transform(X, t)
def umap_(t, s):
    import umap; return umap.UMAP(random_state=s, n_jobs=T).fit_transform(X, t)
M = {"MAGNET SGD": magnet(), "MAGNET hybrid": magnet(loss="hybrid"), "MAGNET hybrid + contraction": magnet(loss="hybrid", contraction=0.7), "UMAP": umap_}
for seed in range(3):
    for frac in [0.0] + FRACS:
        lab = u < frac; t = np.where(lab, y, -1) if frac > 0 else None
        for name, f in M.items():
            key = f"{name} | {frac:g} | {seed}"
            if key in res: continue
            t0 = time.time(); Y = f(t, seed); s = time.time() - t0
            r = axes(Y); r["fit_s"] = round(s, 1)
            if frac > 0: r["acc_unl"] = acc(Y, lab)
            else: r.update({f"acc_unl@{g:g}": acc(Y, u < g) for g in FRACS})
            res[key] = r; print(key, r, flush=True); json.dump(res, open(OUT, "w"), indent=1)

# summary: mean +- std over the seeds; the unsupervised rows read at each fraction's labelled set
rows = {}
for key, r in res.items():
    if key.count("|") != 2: continue
    name, frac, seed = [p.strip() for p in key.split("|")]
    if frac == "0":
        for g in FRACS:
            rows.setdefault((name, f"{g:g}", "unsup"), []).append((r[f"acc_unl@{g:g}"], r["nn15"], r["ari"]))
    else:
        rows.setdefault((name, frac, "labels"), []).append((r["acc_unl"], r["nn15"], r["ari"]))
fmt = lambda v: "-" if v[0] is None else f"{np.mean(v):.3f} +- {np.std(v):.3f}"
print(f"{'method':30s} {'frac':5s} {'':7s} {'acc_unl':16s} {'nn15':16s} {'ari':16s}")
for (name, frac, kind), v in sorted(rows.items(), key=lambda t: (float(t[0][1]), t[0][0], t[0][2] != "unsup")):
    a = [x[0] for x in v]; print(f"{name:30s} {frac:5s} {kind:7s} {fmt(a):16s} {fmt([x[1] for x in v]):16s} {fmt([x[2] for x in v]):16s}  (n={len(v)})")
