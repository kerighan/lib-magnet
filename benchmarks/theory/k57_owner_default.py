"""k57: can the deterministic SGD (parallel="owner") become the default? The lock-free default (hogwild) gives a
different layout on every run with the same seed; the owner mode is bit-reproducible at a fixed n_threads (the
full-batch pass and the post-process already are). The owner mode shifts the attraction / mid-range balance, so
the mid-range stiffness is swept with it: sampled mode 0.05 / 0.1 (hogwild default 0.1), sequence 0.1 / 0.2 / 0.3
(hogwild default 0.2). Scoring as k53: 15-NN, distance Spearman, k-means ARI on 10 000 points, composites against
the frozen references, as is and after the post-process (separation 0.7); MNIST, Fashion-MNIST, letter, covertype,
PBMC 68k, Tasic 2018, five seeds, 8 threads. The summary prints paired differences with the hogwild default.
A third mode, parallel="serial" (the hogwild updates on one thread, everything else multithreaded), is measured too.
Result (k57_summary.txt): the owner mode is NOT a default candidate, it blurs the clusters at every stiffness
(k-means ARI, sampled mode: MNIST -0.21, Fashion -0.12, Tasic -0.14; sequence: MNIST -0.11 to -0.28; mean composite
-0.029 sampled, -0.012 sequence at 0.2, -0.019 after the post-process); the serial mode is the default's quality
(mean composite -0.004 as is, +0.001 / -0.001 after the post-process, every axis within the seed-to-seed spread)
for +7.6 to +10 s at 70 000 points and +1.2 to +1.7 s at 20 000. Hogwild stays the default; "serial" is the
reproducible option.
Usage: python benchmarks/theory/k57_owner_default.py [datasets=...] [seeds=5] [part=run,summary] [out=...]"""
import os, sys, json, time, glob, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET, hierarchical_contraction
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:])
DS = kw.get("datasets", "mnist,fmnist,letter,covertype,pbmc68k,tasic2018").split(","); SEEDS = int(kw.get("seeds", 5)); PART = kw.get("part", "run,summary").split(",")
OUT = os.path.join(HERE, kw.get("out", "k57_owner_default.json")); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
REF = {"mnist": dict(knn15=0.356, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.330, sp=0.873, ari=0.483),
       "letter": dict(knn15=0.642, sp=0.716), "covertype": dict(knn15=0.632, sp=0.755)}
REF.update(json.load(open(os.path.join(HERE, "k47_scrna_heldout.json")))["_refs"])
CONF = [("sgd hogwild 0.1", dict(parallel="hogwild", mid_weight=0.1)), ("sgd owner 0.1", dict(parallel="owner", mid_weight=0.1)),
        ("sgd owner 0.05", dict(parallel="owner", mid_weight=0.05)),
        ("hybrid hogwild 0.2", dict(loss="hybrid", parallel="hogwild", mid_weight=0.2))] + \
       [(f"hybrid owner {k}", dict(loss="hybrid", parallel="owner", mid_weight=k)) for k in (0.1, 0.2, 0.3)] + \
       [("sgd serial 0.1", dict(parallel="serial", mid_weight=0.1)), ("hybrid serial 0.2", dict(loss="hybrid", parallel="serial", mid_weight=0.2))]
BASE = {"sgd": "sgd hogwild 0.1", "hybrid": "hybrid hogwild 0.2"}

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    if name in ("pbmc68k", "tasic2018"):
        z = np.load(os.path.join(B, "cache", "scrna", f"{name}.npz")); return z["X"], z["y"], f"scrna_{name}"
    from k27_transfer import load as load27
    return load27(name, 20000)

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
            for name, c in CONF:
                ka, kb = f"{ds} | {name} | {s}", f"{ds} | {name} + post | {s}"
                if ka in res and kb in res: continue
                m = MAGNET(random_state=s, n_threads=T, **c); t0 = time.time(); Y = m.fit_transform(X); t = time.time() - t0
                t1 = time.time(); Z = hierarchical_contraction(Y, m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=0.7)[0]; tc = time.time() - t1
                for k, L, tt in ((ka, Y, t), (kb, Z, t + tc)):
                    r = score(L); r["fit_s"] = round(tt, 2); res[k] = r
                    print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True)
                json.dump(res, open(OUT, "w"), indent=1)

if "summary" in PART:
    res = {}
    for f in sorted(glob.glob(os.path.join(HERE, "k57_owner_default*.json"))): res.update(json.load(open(f)))
    AX = ("knn15", "sp", "ari", "geo", "fit_s")
    for suf in ("", " + post"):
        for fam in ("sgd", "hybrid"):
            print(f"\n######## {fam}{suf or ' (as is)'}: mean of the base, then paired difference (config - {BASE[fam]}), mean +- sd over seeds")
            tot = {}
            for ds in DS:
                seeds = [s for s in range(SEEDS) if f"{ds} | {BASE[fam]}{suf} | {s}" in res]
                if not seeds: continue
                b = {a: np.array([res[f"{ds} | {BASE[fam]}{suf} | {s}"][a] for s in seeds]) for a in AX}
                print(f"{ds:10s} {BASE[fam]:20s} " + "  ".join(f"{a} {b[a].mean():.3f}+-{b[a].std():.3f}" for a in AX))
                for name, _ in CONF:
                    if not name.startswith(fam) or name == BASE[fam]: continue
                    ok = [s for s in seeds if f"{ds} | {name}{suf} | {s}" in res]
                    if not ok: continue
                    d = {a: np.array([res[f"{ds} | {name}{suf} | {s}"][a] - res[f"{ds} | {BASE[fam]}{suf} | {s}"][a] for s in ok]) for a in AX}
                    sd = np.std([res[f"{ds} | {name}{suf} | {s}"]["ari"] for s in ok])
                    print(f"{'':10s} {name:20s} " + "  ".join(f"{a} {d[a].mean():+.3f}+-{d[a].std():.3f}" for a in AX) + f"   (ARI sd of the config {sd:.3f})")
                    tot.setdefault(name, []).append(d["geo"].mean())
            for name, v in tot.items(): print(f"mean over {len(v)} sets, geo difference: {name:20s} {np.mean(v):+.4f}  worst {np.min(v):+.4f}")
