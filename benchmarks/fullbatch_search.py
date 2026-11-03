"""Parameter search of the full-batch mode (loss="tsne") on four data sets at once, scored by the composite of three axes:
15-NN preservation (local), distance Spearman (global), k-means ARI with k = number of classes (separability), each
divided by the best value seen on that data set (baselines UMAP / PaCMAP / PCA-2 / cached openTSNE and every configuration
of the search), then geometric mean ("geo") and minimum ("min"). Stage 1: one parameter at a time around the default;
stage 2: combinations of the stage-1 winners (run with stage=2 after stage 1). Resumable; 1 run per cell; 8 threads.
Data: MNIST 70k, Fashion-MNIST 70k, letter 20k, covertype 20k (the last two were used as held-out in k27; this search
makes them development data). -> benchmarks/cache/fullbatch_search.json / .log
Usage: python benchmarks/fullbatch_search.py [stage=1|2] [datasets=mnist,fmnist,letter,covertype]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "theory"))
import numpy as np, hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); STAGE = int(kw.get("stage", 1)); DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(",")
OUT = os.path.join(HERE, "cache", "fullbatch_search.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(out, open(OUT, "w"), indent=1)

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    X, y, key = load27(name, 20000); return X, y, key

def scorer(X, y, key):
    n = len(X); s = _idx(n, 10000, 2); k = len(np.unique(y))
    def score(Y):
        r = evaluate(X, Y, y, cache_key=key); Zs = Y[s]
        return dict(knn15=r["knn15_preservation"], sp=r["dist_spearman"], acc=r["knn_acc"], ari=float(adjusted_rand_score(y[s], KMeans(k, n_init=5, random_state=0).fit_predict(Zs))))
    return score

DEFAULT = dict(loss="tsne", random_state=0, n_threads=T)
GRID1 = [("default", {}),
         ("n_epochs 450", dict(n_epochs=450)), ("n_epochs 900", dict(n_epochs=900)), ("n_epochs 1200", dict(n_epochs=1200)),
         ("early 100", dict(tsne_early=100)), ("early 300", dict(tsne_early=300)),
         ("early exag 4", dict(tsne_early_exaggeration=4.0)), ("early exag 8", dict(tsne_early_exaggeration=8.0)), ("early exag 24", dict(tsne_early_exaggeration=24.0)),
         ("exag 1.25", dict(exaggeration=1.25)), ("exag 1.5", dict(exaggeration=1.5)), ("exag 2", dict(exaggeration=2.0)),
         ("tail 100 at 2", dict(hybrid_tail=(100, 2.0))), ("tail 50 at 4", dict(hybrid_tail=(50, 4.0))), ("tail 100 at 1.5", dict(hybrid_tail=(100, 1.5))),
         ("k 10", dict(n_neighbors=10)), ("k 20", dict(n_neighbors=20)), ("k 30", dict(n_neighbors=30)),
         ("dof 0.7", dict(tsne_dof=0.7)), ("dof 1.5", dict(tsne_dof=1.5)),
         ("mid 0", dict(mid_weight=0.0)), ("mid 0.3", dict(mid_weight=0.3)), ("mid 1.0", dict(mid_weight=1.0)),
         ("mid phase 0.3", dict(mid_phase=0.3)), ("mid phase 0.8", dict(mid_phase=0.8)),
         ("box 0.75", dict(fft_box=0.75)),
         ("lr n/24", dict(tsne_learning_rate="n/24")), ("lr n/6", dict(tsne_learning_rate="n/6")),
         ("init spectral", dict(init="spectral"))]
GRID = GRID1 if STAGE == 1 else [(l, c) for l, c in json.load(open(os.path.join(HERE, "cache", "fullbatch_stage2.json")))]

for ds in DS:
    X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(X); score = scorer(X, y, key)
    print(f"\n=== {key}", flush=True)
    # baselines for the references
    base = {"PCA-2": lambda: PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32)}
    import umap, pacmap
    umap.UMAP(n_epochs=20).fit_transform(np.random.rand(6000, X.shape[1]).astype(np.float32)); pacmap.PaCMAP(num_iters=(5, 5, 5)).fit_transform(X[:6000])
    base["UMAP"] = lambda: umap.UMAP().fit_transform(X); base["PaCMAP"] = lambda: pacmap.PaCMAP(random_state=0).fit_transform(X)
    if ds in ("mnist", "fmnist"): base["openTSNE"] = lambda: np.load(os.path.join(HERE, "..", "paper", "figures", "cache", f"{ds}_opentsnefitsne.npz"))["Y"]
    for lab, f in base.items():
        k_ = f"{key}/baseline/{lab}"
        if k_ in out: continue
        t = time.time(); Y = np.asarray(f(), dtype=np.float32); dt = time.time() - t; r = score(Y); r["t"] = dt; out[k_] = r; save()
        print(f"{lab:<22}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{dt:>8.1f}", flush=True)
    for lab, cfg in GRID:
        k_ = f"{key}/stage{STAGE}/{lab}"
        if k_ in out: continue
        c = dict(DEFAULT, **cfg)
        if isinstance(c.get("tsne_learning_rate"), str): c["tsne_learning_rate"] = n / float(c["tsne_learning_rate"].split("/")[1])
        t = time.time(); Y = MAGNET(**c).fit_transform(X).astype(np.float32); dt = time.time() - t
        r = score(Y); r["t"] = dt; r["cfg"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}; out[k_] = r; save()
        print(f"{lab:<22}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{dt:>8.1f}", flush=True)

# ---- composite over everything seen
print("\n" + "=" * 60)
per_ds = {}
for k_, r in out.items():
    ds = k_.split("/")[0]; per_ds.setdefault(ds, {})[k_.split("/", 1)[1]] = r
ref = {ds: {m: max(r[m] for r in rows.values()) for m in ("knn15", "sp", "ari")} for ds, rows in per_ds.items()}
# the separability axis only means something where the classes are clusters: drop it where the panel's best ARI < 0.3
AXES = {ds: (("knn15", "sp", "ari") if ref[ds]["ari"] >= 0.3 else ("knn15", "sp")) for ds in per_ds}
print("axes per data set:", {ds: len(a) for ds, a in AXES.items()})
labels = sorted({k_.split("/", 1)[1] for k_ in out})
rows = []
for lab in labels:
    geos, mins, present = [], [], []
    for ds, rr in per_ds.items():
        if lab in rr:
            sh = [rr[lab][m] / ref[ds][m] for m in AXES[ds]]; geos.append(np.prod(sh) ** (1 / len(sh))); mins.append(min(sh)); present.append(ds)
    if len(present) == len(per_ds): rows.append((lab, np.mean(geos), np.mean(mins), geos, mins))
rows.sort(key=lambda t: -t[1])
print(f"{'configuration':<32}{'mean geo':>9}{'mean min':>9}   per data set geo " + " ".join(f"{d[:9]:>9}" for d in per_ds))
for lab, g, m, geos, mins in rows:
    print(f"{lab:<32}{g:>9.3f}{m:>9.3f}   " + " ".join(f"{x:>9.3f}" for x in geos))
json.dump({"ref": ref, "ranking": [(lab, g, m) for lab, g, m, _, _ in rows]}, open(os.path.join(HERE, "cache", "fullbatch_search_summary.json"), "w"), indent=1)
