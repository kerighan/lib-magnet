"""k52: mid-range attraction kept inside the label-free communities, the global arrangement left to the placement.
k50 (MNIST, SGD): redirecting the mid-near pairs that cross a community of the post-process partition (Louvain at
resolution 0.5 + interface merge) to the point's strongest neighbour raises the k-means ARI from 0.75 to 0.93 and
drops the distance Spearman from 0.42 to 0.29. The placement of the post-process moves whole communities by rigid
translations towards a higher sample Spearman and does not touch their borders: does it give the Spearman back?
One fit per (mode, seed), scored three times: as is, after the placement alone (separation 1), after the placement
and the contraction 0.7. Baselines: the default rows of k51 (same seeds, same scoring: 15-NN, distance Spearman,
k-means ARI on 10 000 points, 5 inits; composites against the frozen references). The summary prints the paired
differences with the k51 defaults, with and without the contraction.
Usage: python benchmarks/theory/k52_mid_in_comm_placement.py [datasets=mnist] [seeds=5] [part=run,summary]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET, hierarchical_contraction
from magnet.contraction import communities, interface_merge
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:])
DS = kw.get("datasets", "mnist,fmnist,letter,covertype,pbmc68k,tasic2018").split(","); SEEDS = int(kw.get("seeds", 5)); PART = kw.get("part", "run,summary").split(",")
OUT = os.path.join(HERE, "k52_mid_in_comm_placement.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
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

class MidInComm(MAGNET):
    """MAGNET whose mid-near pairs across the label-free communities are redirected to the strongest neighbour."""
    def _label_mid_pairs(self, pairs, P):
        if pairs is None: return pairs
        self._y_fit_ = interface_merge(P, communities(P, 0.5)); self.n_comm_ = int(self._y_fit_.max() + 1)
        try: return super()._label_mid_pairs(pairs, P)
        finally: self._y_fit_ = None

MODES = {"SGD": "umap", "hybrid": "hybrid"}
STEPS = {"": None, " + placement": 1.0, " + contraction": 0.7}

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
            for tag, loss in MODES.items():
                if all(f"{ds} | {tag} mid_in_comm{suf} | {s}" in res for suf in STEPS): continue
                m = MidInComm(loss=loss, random_state=s, n_threads=T); t0 = time.time(); Y = m.fit_transform(X); t = time.time() - t0
                for suf, g in STEPS.items():
                    t1 = time.time()
                    Z = Y if g is None else hierarchical_contraction(Y, m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=g)[0]
                    r = score(Z); r["fit_s"] = round(t + (time.time() - t1 if g is not None else 0), 1); r["n_comm"] = getattr(m, "n_comm_", None)
                    k = f"{ds} | {tag} mid_in_comm{suf} | {s}"; res[k] = r
                    print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True)
                json.dump(res, open(OUT, "w"), indent=1)

if "summary" in PART:
    base = json.load(open(os.path.join(HERE, "k51_soft_community_panel.json")))
    AX = ("knn15", "sp", "ari", "geo")
    for ds in DS:
        print(f"\n=== {ds} (refs {REF[ds]})")
        for tag in MODES:
            for name, src in ((f"{tag} default", base), (f"{tag} default + contraction", base)) + tuple((f"{tag} mid_in_comm{suf}", res) for suf in STEPS):
                v = [src[f"{ds} | {name} | {s}"] for s in range(SEEDS) if f"{ds} | {name} | {s}" in src]
                if v: print(f"  {name:34s} " + "  ".join(f"{a} {np.mean([x[a] for x in v]):.3f}" for a in AX + ("min",)) + f"  (n={len(v)})")
            for suf in ("", " + contraction"):
                d = [[res[f"{ds} | {tag} mid_in_comm{suf} | {s}"][x] - base[f"{ds} | {tag} default{suf} | {s}"][x] for x in AX]
                     for s in range(SEEDS) if f"{ds} | {tag} mid_in_comm{suf} | {s}" in res and f"{ds} | {tag} default{suf} | {s}" in base]
                if d:
                    d = np.array(d); print(f"  {tag} mid_in_comm{suf} - default{suf}: " + "  ".join(f"{a} {m:+.3f} +- {sd:.3f}" for a, m, sd in zip(AX, d.mean(0), d.std(0))))
