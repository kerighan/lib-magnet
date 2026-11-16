"""k51: the label-free soft community weighting of k50 on the whole panel. The graph edges that cross a community of
the post-process partition (Louvain at resolution 0.5 + interface merge, no labels) are multiplied by 0.082 =
exp(-2.5) (UMAP's factor for differently labelled points at target_weight 0) before the layout; the contraction 0.7
follows. Compared with the current defaults, with and without the contraction, in the SGD and the sequence (hybrid)
modes; MNIST, Fashion-MNIST, letter, covertype (k41 references) and the held-out PBMC 68k / Tasic 2018 (k47
references), five seeds, the k41 scoring (15-NN, distance Spearman, k-means ARI on 10 000 points, 5 inits;
composite = geometric mean and minimum of the ratios to the frozen references; ARI dropped on letter / covertype).
The summary prints the paired difference soft - default per mode, with and without the contraction.
Usage: python benchmarks/theory/k51_soft_community_panel.py [datasets=...] [seeds=5] [part=run,summary]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import communities, interface_merge
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:])
DS = kw.get("datasets", "mnist,fmnist,letter,covertype,pbmc68k,tasic2018").split(","); SEEDS = int(kw.get("seeds", 5)); PART = kw.get("part", "run,summary").split(",")
OUT = os.path.join(HERE, "k51_soft_community_panel.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
REF = {"mnist": dict(knn15=0.356, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.330, sp=0.873, ari=0.483),
       "letter": dict(knn15=0.642, sp=0.716), "covertype": dict(knn15=0.632, sp=0.755)}
REF.update(json.load(open(os.path.join(HERE, "k47_scrna_heldout.json")))["_refs"])

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    if name in ("pbmc68k", "tasic2018"):
        z = np.load(os.path.join(B, "cache", "scrna", f"{name}.npz")); return z["X"], z["y"], f"scrna_{name}"
    from k27_transfer import load as load27
    return load27(name, 20000)

class SoftGraph(MAGNET):
    """MAGNET whose graph edges across the label-free communities are multiplied by `factor`."""
    def __init__(self, factor=0.082, **k):
        super().__init__(**k); self.factor = factor
    def build_graph(self, X):
        P = super().build_graph(X); c = interface_merge(P, communities(P, 0.5)); self.n_comm_ = int(c.max() + 1)
        C = P.tocoo(); f = np.where(c[C.row] == c[C.col], 1.0, self.factor).astype(np.float32)
        Q = sp.csr_matrix((C.data * f, (C.row, C.col)), shape=P.shape); Q.sort_indices(); return Q

VARIANTS = {}
for loss, tag in (("umap", "SGD"), ("hybrid", "hybrid")):
    for c in (None, 0.7):
        suf = " + contraction" if c else ""
        VARIANTS[f"{tag} default{suf}"] = (lambda loss, c: lambda s: MAGNET(loss=loss, contraction=c, random_state=s, n_threads=T))(loss, c)
        VARIANTS[f"{tag} soft{suf}"] = (lambda loss, c: lambda s: SoftGraph(0.082, loss=loss, contraction=c, random_state=s, n_threads=T))(loss, c)

def composite(r, ds):
    sh = [r[a] / REF[ds][a] for a in REF[ds]]; return float(np.prod(sh) ** (1 / len(sh))), float(min(sh))

if "run" in PART:
    for ds in DS:
        X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); s10 = _idx(n, 10000, 2)
        def score(Y):
            e = evaluate(X, Y, y, cache_key=key)
            r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"],
                     ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
            r["geo"], r["min"] = composite(r, ds); return r
        for s in range(SEEDS):
            for name, mk in VARIANTS.items():
                k = f"{ds} | {name} | {s}"
                if k in res: continue
                m = mk(s); t0 = time.time(); Y = m.fit_transform(X); t = time.time() - t0
                r = score(Y); r["fit_s"] = round(t, 1); r["n_comm"] = getattr(m, "n_comm_", None); res[k] = r
                print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True)
                json.dump(res, open(OUT, "w"), indent=1)

if "summary" in PART:
    for ds in DS:
        print(f"\n=== {ds} (refs {REF[ds]})")
        for name in VARIANTS:
            v = [res[f"{ds} | {name} | {s}"] for s in range(SEEDS) if f"{ds} | {name} | {s}" in res]
            if not v: continue
            print(f"  {name:28s} " + "  ".join(f"{a} {np.mean([x[a] for x in v]):.3f}" for a in ("knn15", "sp", "ari", "geo", "min")) + f"  (n={len(v)})")
        for base in ("SGD default", "SGD default + contraction", "hybrid default", "hybrid default + contraction"):
            soft = base.replace("default", "soft"); d = []
            for s in range(SEEDS):
                a, b = res.get(f"{ds} | {soft} | {s}"), res.get(f"{ds} | {base} | {s}")
                if a and b: d.append([a[x] - b[x] for x in ("knn15", "sp", "ari", "geo")])
            if d:
                d = np.array(d); print(f"  {soft} - {base}: " + "  ".join(f"{a} {m:+.3f} +- {sd:.3f}" for a, m, sd in zip(("knn15", "sp", "ari", "geo"), d.mean(0), d.std(0))))
