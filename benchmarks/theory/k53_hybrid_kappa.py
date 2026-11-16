"""k53: the stiffness of the mid-range spring in the sequence (hybrid) mode. The shipped default is 0.5, chosen on
MNIST (k8 / k9); the transfer study k27 (covertype, letter, RCV1; single runs) found 0.1 at least as good outside
MNIST. Here: mid_weight 0.1 / 0.2 / 0.3 against the 0.5 rows of k51 (same seeds, same scoring: 15-NN, distance
Spearman, k-means ARI on 10 000 points, 5 inits; composites against the frozen references), on MNIST,
Fashion-MNIST, letter, covertype, PBMC 68k and Tasic 2018, five seeds. One fit per (stiffness, seed), scored as is
and after the post-process (contraction 0.7). The summary prints the paired differences with 0.5.
Usage: python benchmarks/theory/k53_hybrid_kappa.py [datasets=...] [seeds=5] [kappas=0.1,0.2,0.3] [part=run,summary]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
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
KAPPAS = [float(v) for v in kw.get("kappas", "0.1,0.2,0.3").split(",")]
OUT = os.path.join(HERE, "k53_hybrid_kappa.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
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
            for kap in KAPPAS:
                ka, kb = f"{ds} | hybrid mid {kap} | {s}", f"{ds} | hybrid mid {kap} + contraction | {s}"
                if ka in res and kb in res: continue
                m = MAGNET(loss="hybrid", mid_weight=kap, random_state=s, n_threads=T); t0 = time.time(); Y = m.fit_transform(X); t = time.time() - t0
                t1 = time.time(); Z = hierarchical_contraction(Y, m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=0.7)[0]; tc = time.time() - t1
                for k, L, tt in ((ka, Y, t), (kb, Z, t + tc)):
                    r = score(L); r["fit_s"] = round(tt, 1); res[k] = r
                    print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True)
                json.dump(res, open(OUT, "w"), indent=1)

if "summary" in PART:
    base = json.load(open(os.path.join(HERE, "k51_soft_community_panel.json")))
    AX = ("knn15", "sp", "ari", "geo")
    for suf in ("", " + contraction"):
        print(f"\n######## hybrid{suf or ' (no post-process)'}")
        for ds in DS:
            print(f"=== {ds}")
            rows = [(0.5, [base.get(f"{ds} | hybrid default{suf} | {s}") for s in range(SEEDS)])] + [(kap, [res.get(f"{ds} | hybrid mid {kap}{suf} | {s}") for s in range(SEEDS)]) for kap in KAPPAS]
            for kap, v in rows:
                w = [x for x in v if x]
                if w: print(f"  mid {kap:<4} " + "  ".join(f"{a} {np.mean([x[a] for x in w]):.3f}" for a in AX + ("min",)) + f"  (n={len(w)})")
            for kap, v in rows[1:]:
                d = np.array([[a[x] - b[x] for x in AX] for a, b in zip(v, rows[0][1]) if a and b])
                if len(d): print(f"  {kap} - 0.5: " + "  ".join(f"{a} {m:+.3f} +- {sd:.3f}" for a, m, sd in zip(AX, d.mean(0), d.std(0))))
