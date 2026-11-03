"""k38: robustness of the packaged placement + separation step (MAGNET(contraction=0.7)). (a) 3 seeds of the full-batch
and hybrid modes on MNIST / Fashion, with and without the step (mean +- sd of the three axes and of the step's time);
(b) generality: the same step applied to the cached BH t-SNE, openTSNE, UMAP, PaCMAP layouts (their own graph is not
available: MAGNET's fuzzy graph P is used for the communities); (c) scaling: covertype 250k, time of the step.
Usage: python benchmarks/theory/k38_contraction_robustness.py [part=seeds,generality,scaling]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import hierarchical_contraction
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); PART = kw.get("part", "seeds,generality,scaling").split(",")
OUT = os.path.join(HERE, "k38_contraction_robustness.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
REF = {"mnist": dict(knn15=0.352, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.328, sp=0.873, ari=0.483)}

def score(X, Y, y, key, rf):
    e = evaluate(X, Y, y, cache_key=key); s = _idx(len(Y), 10000, 2); kcl = len(np.unique(y))
    r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], trip=e["triplet_acc"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))))
    if rf: sh = [r[a] / rf[a] for a in ("knn15", "sp", "ari")]; r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh))
    return r

def fmt(r): return f"{r['knn15']:.3f} / {r['sp']:.3f} / {r['ari']:.3f}  geo {r.get('geo', float('nan')):.3f}"

if "seeds" in PART:
    print("(a) seeds: MAGNET full-batch / hybrid, with and without contraction=0.7 (3 seeds)")
    for ds in ("mnist", "fmnist"):
        X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = REF[ds]
        for mode in ("tsne", "hybrid"):
            for seed in (0, 1, 2):
                k_ = f"seeds/{ds}/{mode}/{seed}"
                if k_ not in res:
                    t = time.time(); m = MAGNET(loss=mode, random_state=seed, n_threads=T).fit(X); t_fit = time.time() - t
                    Y = m.embedding_; r0 = score(X, Y, y, key, rf)
                    t = time.time(); Z, log = hierarchical_contraction(Y, m.graph_, X=X, random_state=seed); t_c = time.time() - t
                    r1 = score(X, Z, y, key, rf); res[k_] = dict(plain=r0, contracted=r1, t_fit=t_fit, t_contraction=t_c, communities=int(log["communities"]), placement=log.get("placement"))
                    json.dump(res, open(OUT, "w"), indent=1)
                r = res[k_]; print(f"  {ds} {mode} seed {seed}: plain {fmt(r['plain'])} | +contraction {fmt(r['contracted'])} | {r['communities']} comm, fit {r['t_fit']:.1f} s + step {r['t_contraction']:.1f} s", flush=True)
            for which in ("plain", "contracted"):
                rows = [res[f"seeds/{ds}/{mode}/{s}"][which] for s in (0, 1, 2)]
                print(f"  {ds} {mode} {which:<10} mean: " + "  ".join(f"{a} {np.mean([r[a] for r in rows]):.3f} +- {np.std([r[a] for r in rows]):.3f}" for a in ("knn15", "sp", "ari", "geo")), flush=True)

if "generality" in PART:
    print("(b) generality: the step on other methods' layouts (communities from MAGNET's graph)")
    for ds in ("mnist", "fmnist"):
        X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = REF[ds]; P = sp.load_npz(os.path.join(B, "cache", f"P_{ds}.npz"))
        layouts = {"sklearn t-SNE (BH)": os.path.join(B, "cache", "layouts", f"{ds}_sklearn_t-SNE_BH.npz"), "openTSNE": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_opentsnefitsne.npz"),
                   "UMAP": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_umap0512.npz"), "PaCMAP": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_pacmap.npz")}
        for name, p in layouts.items():
            k_ = f"generality/{ds}/{name}"
            if k_ not in res:
                Y = np.load(p)["Y"].astype(np.float32); r0 = score(X, Y, y, key, rf); t = time.time(); Z, log = hierarchical_contraction(Y, P, X=X); t_c = time.time() - t
                res[k_] = dict(plain=r0, contracted=score(X, Z, y, key, rf), t_contraction=t_c); json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"  {ds} {name:<20} plain {fmt(r['plain'])} | +contraction {fmt(r['contracted'])} | step {r['t_contraction']:.1f} s", flush=True)

if "scaling" in PART:
    print("(c) scaling: covertype 250k, full-batch + contraction")
    from k27_transfer import load as load27
    X, y, key = load27("covertype", 250000); y = np.asarray(y).astype(int)
    k_ = "scaling/covertype250k"
    if k_ not in res:
        t = time.time(); m = MAGNET(loss="tsne", random_state=0, n_threads=T).fit(X); t_fit = time.time() - t
        r0 = score(X, m.embedding_, y, key, None); t = time.time(); Z, log = hierarchical_contraction(m.embedding_, m.graph_, X=X); t_c = time.time() - t
        res[k_] = dict(plain=r0, contracted=score(X, Z, y, key, None), t_fit=t_fit, t_contraction=t_c, communities=int(log["communities"])); json.dump(res, open(OUT, "w"), indent=1)
    r = res[k_]; print(f"  covertype 250k: plain {r['plain']['knn15']:.3f} / {r['plain']['sp']:.3f} | +contraction {r['contracted']['knn15']:.3f} / {r['contracted']['sp']:.3f} | {r['communities']} comm, fit {r['t_fit']:.1f} s + step {r['t_contraction']:.1f} s", flush=True)
