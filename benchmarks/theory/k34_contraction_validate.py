"""k34: the community contraction of k32 as a candidate feature, on the four composite data sets and three MAGNET
layouts (full-batch, quality, hybrid). Steps, all label-free: (1) coarse level = Leiden r=0.02 on P, contraction factor
chosen by a line search on the 2000-point Spearman (c in 1 .. 0.4; 1 = off); (2) fine level = Leiden r=0.3, contraction
c_f in {0.8, 0.6}; hard (rigid per community) or soft (Z = c Y + (1 - c) W M, W = row-normalised P: a point near a
border lands between the two maps). Scores on the three axes + composite (2 axes on letter / covertype as in the
tables); figure benchmarks/plots/contraction_{ds}.png (identity / hard / soft).
Usage: python benchmarks/theory/k34_contraction_validate.py [datasets=mnist,fmnist,letter,covertype] [plot=1]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate, high_dim_pdist
from magnet import MAGNET
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(","); PLOT = kw.get("plot", "1") == "1"
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k34_contraction_validate.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
AXES = {"mnist": ("knn15", "sp", "ari"), "fmnist": ("knn15", "sp", "ari"), "letter": ("knn15", "sp"), "covertype": ("knn15", "sp")}

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    return load27(name, 20000)

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

def contract_hard(Y, lab, c):
    if c == 1.0: return Y.copy()
    Z = Y.copy()
    for cl in np.unique(lab):
        m = lab == cl; mu = Y[m].mean(0); Z[m] = mu + c * (Y[m] - mu)
    return Z

def contract_soft(Y, lab, c, W):
    if c == 1.0: return Y.copy()
    k = lab.max() + 1; M = np.stack([Y[lab == cl].mean(0) for cl in range(k)])
    H = sp.csr_matrix((np.ones(len(lab), np.float32), (np.arange(len(lab)), lab)), shape=(len(lab), k))
    return c * Y + (1 - c) * ((W @ H) @ M)

def leiden(P, resolution, seed=0):
    import igraph as ig, leidenalg as la
    C = sp.triu(P, 1).tocoo(); g = ig.Graph(n=P.shape[0], edges=np.c_[C.row, C.col].tolist()); g.es["weight"] = C.data.astype(float).tolist()
    return np.asarray(la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed, n_iterations=2).membership)

for ds in DS:
    X, y, key = load(ds); y = np.asarray(y).astype(int); rf = ref(ds); n = len(X); axes = AXES[ds]
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    if os.path.exists(pc): P = sp.load_npz(pc)
    else: P = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32); sp.save_npz(pc, P)
    W = sp.diags(1.0 / np.maximum(np.asarray(P.sum(1)).ravel(), 1e-12)) @ P
    g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1); dh = Dh[iu]
    def sp_fast(Y):
        Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); return float(spearmanr(dh, Dl[iu]).correlation)
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key); s = _idx(n, 10000, 2); kcl = len(np.unique(y))
        r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))))
        sh = [r[a] / rf[a] for a in axes]; r["geo"] = float(np.prod(sh) ** (1 / len(sh))); r["min"] = float(min(sh)); return r
    t0 = time.time(); lab_c, lab_f = leiden(P, 0.02), leiden(P, 0.3); t_leiden = time.time() - t0
    print(f"\n{ds} (n={n}, {len(np.unique(y))} classes): coarse {lab_c.max() + 1} comm (ARI {adjusted_rand_score(y, lab_c):.2f}), fine {lab_f.max() + 1} comm "
          f"(ARI {adjusted_rand_score(y, lab_f):.2f}), leiden {t_leiden:.1f} s; panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}; axes {axes}", flush=True)
    print(f"{'layout':<11}{'scheme':<34}{'knn15':>7}{'sp':>7}{'ari':>7}{'geo':>7}{'min':>6}")
    layouts = {"full-batch": os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_full-batch.npz"), "quality": os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_full-batch_quality.npz"),
               "hybrid": os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_hybrid.npz")}
    slug = {"full-batch": "magnetlosstsne", "hybrid": "magnetlosshybrid"}
    keep = {}
    for lay, p in layouts.items():
        fig = os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_{slug.get(lay, '')}.npz")
        Y0 = np.load(fig if os.path.exists(fig) else p)["Y"].astype(np.float32)
        k_ = f"{ds}/{lay}/coarse search"
        if k_ not in res: res[k_] = {str(c): sp_fast(contract_hard(Y0, lab_c, c)) for c in (1.0, 0.8, 0.6, 0.5, 0.4)}; json.dump(res, open(OUT, "w"), indent=1)
        c_best = float(max(res[k_], key=lambda c: res[k_][c]))
        print(f"  {lay}: coarse line search " + "  ".join(f"c{c}: {v:.3f}" for c, v in res[k_].items()) + f" -> c={c_best}", flush=True)
        Yc = contract_hard(Y0, lab_c, c_best)
        schemes = [("identity", lambda: Y0), (f"coarse c{c_best}", lambda: Yc)]
        for cf in (0.8, 0.6):
            schemes += [(f"coarse c{c_best} > fine hard c{cf}", (lambda cf=cf: contract_hard(Yc, lab_f, cf))), (f"coarse c{c_best} > fine soft c{cf}", (lambda cf=cf: contract_soft(Yc, lab_f, cf, W)))]
        for name, f in schemes:
            k_ = f"{ds}/{lay}/{name}"
            Z = np.ascontiguousarray(f(), dtype=np.float32)
            if k_ not in res: res[k_] = score(Z); json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"{lay:<11}{name:<34}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
            if lay == "full-batch" and name in ("identity", f"coarse c{c_best} > fine hard c0.6", f"coarse c{c_best} > fine soft c0.6"): keep[name] = Z
    if PLOT and keep:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(keep), figsize=(6 * len(keep), 6)); s = _idx(n, 20000, 5)
        for a, (name, Z) in zip(np.atleast_1d(ax), keep.items()):
            a.scatter(Z[s, 0], Z[s, 1], c=y[s], cmap="tab10" if len(np.unique(y)) <= 10 else "tab20", s=1, alpha=0.6); a.set_title(f"{ds} full-batch: {name}"); a.set_aspect("equal"); a.axis("off")
        os.makedirs(os.path.join(B, "plots"), exist_ok=True); fig.savefig(os.path.join(B, "plots", f"contraction_{ds}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
