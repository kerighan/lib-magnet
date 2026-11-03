"""k30: geometric ceiling of the three-axis composite. Label-aware (oracle) transforms of cached layouts: per-class
contraction towards the class centroid (c = 1, 0.8, 0.6, 0.4), class centroids moved to the classical MDS of the
input class means (global oracle), both. Measures 15-NN / Spearman / k-means ARI and the composite against the panel
bests of composite_table.json. Answers: are the three axes jointly reachable by ANY layout close to ours, or is the
plateau structural? Usage: python benchmarks/theory/k30_oracle_axes.py [datasets=mnist,fmnist]"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist").split(",")
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k30_oracle_axes.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

def score(X, Y, y, key, rf):
    e = evaluate(X, Y, y, cache_key=key); s = _idx(len(Y), 10000, 2); kcl = len(np.unique(y))
    ari = float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s])))
    r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=ari)
    sh = [r[a] / rf[a] for a in ("knn15", "sp", "ari")]; r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh)); return r

def contract(Y, y, c):
    Z = Y.copy()
    for cl in np.unique(y):
        m = y == cl; mu = Y[m].mean(0); Z[m] = mu + c * (Y[m] - mu)
    return Z

def mds_means(X, y, Y):
    """classical MDS (2-D) of the input class means, Procrustes-aligned (rotation + isotropic scale) to the layout's centroids."""
    cls = np.unique(y); M = np.stack([np.asarray(X[y == cl].mean(0)).ravel() for cl in cls])
    D2 = ((M[:, None] - M[None]) ** 2).sum(-1); J = np.eye(len(cls)) - 1.0 / len(cls); Bm = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(Bm); Zm = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))
    C = np.stack([Y[y == cl].mean(0) for cl in cls])
    A, Cc = Zm - Zm.mean(0), C - C.mean(0); U, S, Vt = np.linalg.svd(A.T @ Cc); R = U @ Vt
    scale = S.sum() / (A ** 2).sum(); return C.mean(0) + scale * (A @ R), cls

def move_centroids(Y, y, target, cls):
    Z = Y.copy()
    for t, cl in zip(target, cls):
        m = y == cl; Z[m] += t - Y[m].mean(0)
    return Z

for ds in DS:
    X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = ref(ds)
    print(f"\n{ds}: panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}")
    print(f"{'layout':<22}{'transform':<28}{'knn15':>7}{'sp':>7}{'ari':>7}{'geo':>7}{'min':>6}")
    for lay, f in (("hybrid", f"{ds}_magnetlosshybrid.npz"), ("full-batch", f"{ds}_magnetlosstsne.npz"),
                   ("BH t-SNE", os.path.join(B, "cache", "layouts", f"{ds}_sklearn_t-SNE_BH.npz"))):
        p = f if os.path.isabs(f) else os.path.join(B, "..", "paper", "figures", "cache", f); Y = np.load(p)["Y"].astype(np.float32)
        tgt, cls = mds_means(X, y, Y)
        variants = [("identity", lambda: Y)]
        variants += [(f"contract {c}", (lambda c=c: contract(Y, y, c))) for c in (0.8, 0.6, 0.4)]
        variants += [("centroids -> MDS(means)", lambda: move_centroids(Y, y, tgt, cls))]
        variants += [(f"MDS + contract {c}", (lambda c=c: contract(move_centroids(Y, y, tgt, cls), y, c))) for c in (0.8, 0.6)]
        for name, g in variants:
            k_ = f"{ds}/{lay}/{name}"
            if k_ not in res:
                res[k_] = score(X, np.ascontiguousarray(g(), dtype=np.float32), y, key, rf); json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"{lay:<22}{name:<28}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
