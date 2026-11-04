"""k42: making the community detection of the post-process tighter. Louvain's random visiting order splits a class in
some runs (MNIST seed 0: 11 communities, label ARI 0.89 against 0.92; modularity does not single the run out, k38 /
docs 6.12). Variants, on the cached fuzzy graphs of MNIST, Fashion, letter, covertype 20k (and the MNIST graphs of
two other NN-descent seeds): (a) Louvain, random order, 10 seeds; (b) deterministic orders (strength descending /
ascending, index); (c) medoid consensus of 5 random runs; (d) Leiden modularity (igraph), 5 seeds; (e) Leiden CPM,
resolution scan; (f) modularity resolution scan. Reported per variant: number of communities, label ARI (mean +- sd),
stability (mean pairwise ARI between runs), time. Then downstream: the post-process on the three cached MNIST
full-batch layouts with each candidate's communities, 15-NN / Spearman / ARI.
Usage: python benchmarks/theory/k42_community_stability.py [part=partition,downstream] [datasets=mnist,fmnist,letter,covertype]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import communities, _igraph, _leiden, hierarchical_contraction
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); PART = kw.get("part", "partition,downstream").split(","); DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(",")
OUT = os.path.join(HERE, "k42_community_stability.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    return load27(name, 20000)

def report(name, runs, y, dt):
    k = [int(r.max() + 1) for r in runs]; ari = [adjusted_rand_score(y, r) for r in runs] if y is not None else [float("nan")]
    stab = [adjusted_rand_score(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))] or [1.0]
    r = dict(n=len(runs), comm=k, label_ari=float(np.mean(ari)), label_ari_sd=float(np.std(ari)), label_ari_min=float(np.min(ari)), stability=float(np.mean(stab)), stability_min=float(np.min(stab)), t=dt / len(runs))
    print(f"  {name:<34} comm {min(k)}-{max(k)}  label ARI {r['label_ari']:.3f} +- {r['label_ari_sd']:.3f} (min {r['label_ari_min']:.3f})  stability {r['stability']:.3f} (min {r['stability_min']:.3f})  {r['t']:.2f} s/run", flush=True)
    return r

if "partition" in PART:
    for ds in DS:
        X, y, key = load(ds); y = np.asarray(y).astype(int)
        graphs = {0: sp.load_npz(os.path.join(B, "cache", f"P_{ds}.npz"))}
        if ds == "mnist":
            for gs in (1, 2): graphs[gs] = sp.csr_matrix(MAGNET(random_state=gs, n_threads=T).fit(X).graph_, dtype=np.float32)
        for gs, P in graphs.items():
            print(f"\n=== {ds}, graph seed {gs}: {P.shape[0]} nodes, {len(np.unique(y))} classes", flush=True)
            def run(name, f, seeds):
                k_ = f"{ds}/{gs}/{name}"
                if k_ in res: r = res[k_]; print(f"  {name:<34} (cached) comm {min(r['comm'])}-{max(r['comm'])}  label ARI {r['label_ari']:.3f} +- {r['label_ari_sd']:.3f} (min {r['label_ari_min']:.3f})  stability {r['stability']:.3f}  {r['t']:.2f} s/run", flush=True); return
                t = time.time(); runs = [f(s) for s in seeds]; res[k_] = report(name, runs, y, time.time() - t); json.dump(res, open(OUT, "w"), indent=1)
            run("louvain random r0.3", lambda s: communities(P, 0.3, seed=s), range(10))
            run("louvain strength r0.3", lambda s: communities(P, 0.3, seed=s, order="strength"), [0])
            run("louvain strength_asc r0.3", lambda s: communities(P, 0.3, seed=s, order="strength_asc"), [0])
            run("louvain index r0.3", lambda s: communities(P, 0.3, seed=s, order="index"), [0])
            run("louvain consensus5 r0.3", lambda s: communities(P, 0.3, seed=10 * s, consensus=5), range(4))
            run("louvain consensus9 r0.3", lambda s: communities(P, 0.3, seed=10 * s, consensus=9), range(3))
            for r_ in (0.2, 0.5, 1.0): run(f"louvain random r{r_}", lambda s, r_=r_: communities(P, r_, seed=s), range(5))
            try:
                g = _igraph(P)
                run("leiden modularity r0.3", lambda s: _leiden(g, 0.3, s), range(5))
                for gam in (2e-4, 5e-4, 1e-3, 2e-3): run(f"leiden CPM g{gam:g}", lambda s, gam=gam: _leiden(g, gam, s, objective="CPM"), range(3))
            except Exception as e: print("  igraph unavailable:", e)

if "downstream" in PART:
    ds = "mnist"; X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(y); P = sp.load_npz(os.path.join(B, "cache", f"P_{ds}.npz"))
    excl = np.unique(np.concatenate([_idx(n, 2000, 1), _idx(n, 5000, 0), _idx(n, 10000, 2)])); s10 = _idx(n, 10000, 2)
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key); return dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=float(adjusted_rand_score(y[s10], KMeans(10, n_init=5, random_state=0).fit_predict(Y[s10]))))
    print("\n=== downstream: post-process on the three MNIST full-batch layouts, communities from each variant", flush=True)
    for name, opts in (("louvain random", dict()), ("louvain strength", dict(order="strength")), ("louvain index", dict(order="index")), ("louvain consensus5", dict(consensus=5)), ("leiden modularity", dict(backend="leiden"))):
        rows = []
        for seed in (0, 1, 2):
            k_ = f"downstream/{name}/{seed}"
            if k_ not in res:
                Y = np.load(os.path.join(B, "cache", "layouts", f"mnist_fb_seed{seed}.npz"))["Y"].astype(np.float32)
                t = time.time(); Z, log = hierarchical_contraction(Y, P, X=X, random_state=seed, exclude=excl, n_threads=T, **opts); dt = time.time() - t
                r = score(np.ascontiguousarray(Z, dtype=np.float32)); r["t"] = dt; r["comm"] = int(log["communities"]); res[k_] = r; json.dump(res, open(OUT, "w"), indent=1)
            rows.append(res[k_])
        print(f"  {name:<20} " + "  ".join(f"s{i}: {r['knn15']:.3f}/{r['sp']:.3f}/{r['ari']:.3f} ({r['comm']} comm, {r['t']:.1f} s)" for i, r in enumerate(rows)) + f"  | ARI mean {np.mean([r['ari'] for r in rows]):.3f}", flush=True)
