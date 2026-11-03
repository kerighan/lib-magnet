"""k32: hierarchical, label-free contraction. Leiden communities of the fuzzy graph P at several resolutions (coarse
-> fine); at each level every community is contracted rigidly towards its centroid by a factor c. (1) line search of c
per level on the distance Spearman (2000-point sample, same as the benchmark) and on k-means ARI, applied post hoc to the
cached full-batch and hybrid layouts; (2) nested contraction (coarse level at its best c, then a fine level);
(3) contraction applied at the hand-over (SGD start) before the 300-iteration pass instead of after it.
Usage: python benchmarks/theory/k32_hierarchical_contraction.py [datasets=mnist,fmnist] [part=post,nested,handover]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate, high_dim_pdist
from magnet import MAGNET
from magnet.tsne import optimize_tsne
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist").split(","); PART = kw.get("part", "post,nested,handover").split(",")
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k32_hierarchical_contraction.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
RES = (0.02, 0.05, 0.1, 0.3); CS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

def score(X, Y, y, key, rf):
    e = evaluate(X, Y, y, cache_key=key); s = _idx(len(Y), 10000, 2); kcl = len(np.unique(y))
    ari = float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s])))
    r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], trip=e["triplet_acc"], ari=ari)
    sh = [r[a] / rf[a] for a in ("knn15", "sp", "ari")]; r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh)); return r

def contract(Y, lab, c):
    if c == 1.0: return Y.copy()
    Z = Y.copy()
    for cl in np.unique(lab):
        m = lab == cl; mu = Y[m].mean(0); Z[m] = mu + c * (Y[m] - mu)
    return Z

def leiden(P, resolution, seed=0):
    import igraph as ig, leidenalg as la
    C = sp.triu(P, 1).tocoo(); g = ig.Graph(n=P.shape[0], edges=np.c_[C.row, C.col].tolist()); g.es["weight"] = C.data.astype(float).tolist()
    return np.asarray(la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed, n_iterations=2).membership)

for ds in DS:
    X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = ref(ds); n = len(X)
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    if os.path.exists(pc): P = sp.load_npz(pc)
    else: P = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32); sp.save_npz(pc, P)
    g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1); dh = Dh[iu]
    s10 = _idx(n, 10000, 2); kcl = len(np.unique(y))
    def sp_fast(Y):
        Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); return float(spearmanr(dh, Dl[iu]).correlation)
    def ari_fast(Y): return float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=3, random_state=0).fit_predict(Y[s10])))
    labs = {r_: leiden(P, r_) for r_ in RES}
    print(f"\n{ds}: panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}; leiden: " +
          ", ".join(f"r={r_} -> {l.max() + 1} comm (ARI {adjusted_rand_score(y, l):.2f})" for r_, l in labs.items()), flush=True)
    layouts = {"full-batch": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosstsne.npz"),
               "hybrid": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosshybrid.npz")}
    if "post" in PART:
        print("line search per level (post hoc): sp / ARI as a function of c")
        for lay, p in layouts.items():
            Y0 = np.load(p)["Y"].astype(np.float32)
            for r_, lab in labs.items():
                k_ = f"{ds}/{lay}/post/r{r_}"
                if k_ not in res:
                    res[k_] = {str(c): [sp_fast(contract(Y0, lab, c)), ari_fast(contract(Y0, lab, c))] for c in CS}; json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {lay:<11} r={r_:<5} " + "  ".join(f"c{c}: {v[0]:.3f}/{v[1]:.3f}" for c, v in res[k_].items()), flush=True)
    if "nested" in PART:
        print("nested contraction (coarse at its best-Spearman c, then finer levels at c2), full scores")
        print(f"{'layout':<11}{'scheme':<40}{'knn15':>7}{'sp':>7}{'trip':>7}{'ari':>7}{'geo':>7}{'min':>6}")
        for lay, p in layouts.items():
            Y0 = np.load(p)["Y"].astype(np.float32)
            best = {r_: max(CS, key=lambda c: res[f"{ds}/{lay}/post/r{r_}"][str(c)][0]) for r_ in RES}
            schemes = [("identity", [])]
            schemes += [(f"r{r_} c{best[r_]}", [(r_, best[r_])]) for r_ in RES]
            schemes += [(f"r0.02 c{best[0.02]} > r0.3 c{c2}", [(0.02, best[0.02]), (0.3, c2)]) for c2 in (0.8, 0.6, 0.4)]
            schemes += [(f"r0.02 c{best[0.02]} > r0.1 c{best[0.1]} > r0.3 c{c2}", [(0.02, best[0.02]), (0.1, best[0.1]), (0.3, c2)]) for c2 in (0.8, 0.6)]
            schemes += [(f"all levels at their best c", [(r_, best[r_]) for r_ in RES])]
            for name, steps in schemes:
                k_ = f"{ds}/{lay}/nested/{name}"
                if k_ not in res:
                    Y = Y0
                    for r_, c in steps: Y = contract(Y, labs[r_], c)
                    res[k_] = score(X, np.ascontiguousarray(Y, dtype=np.float32), y, key, rf); json.dump(res, open(OUT, "w"), indent=1)
                r = res[k_]; print(f"{lay:<11}{name:<40}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
    if "handover" in PART:
        print("contraction at the hand-over (SGD start of the hybrid), then the 300-iteration pass")
        print(f"{'scheme':<40}{'knn15':>7}{'sp':>7}{'trip':>7}{'ari':>7}{'geo':>7}{'min':>6}{'t':>6}")
        S0 = np.load(os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetsgd.npz"))["Y"].astype(np.float32)
        schemes = [("pass only", []), ("r0.02 c0.6 > pass", [(0.02, 0.6)]), ("r0.02 c0.4 > pass", [(0.02, 0.4)]),
                   ("r0.3 c0.6 > pass", [(0.3, 0.6)]), ("r0.02 c0.6 > r0.3 c0.6 > pass", [(0.02, 0.6), (0.3, 0.6)]),
                   ("r0.02 c0.6 > r0.1 c0.6 > r0.3 c0.6 > pass", [(0.02, 0.6), (0.1, 0.6), (0.3, 0.6)])]
        for name, steps in schemes:
            k_ = f"{ds}/handover/{name}"
            if k_ not in res:
                Y = S0.copy()
                for r_, c in steps: Y = contract(Y, labs[r_], c)
                Y -= Y.mean(0); Y *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y[:, 0].std()), 1e-12)
                t = time.time(); Y = optimize_tsne(P, np.ascontiguousarray(Y, dtype=np.float32), n_iter=300, early_iter=0, n_threads=T); dt = time.time() - t
                res[k_] = score(X, Y, y, key, rf); res[k_]["t"] = dt; json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"{name:<40}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}{r['t']:>6.1f}", flush=True)
