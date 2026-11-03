"""k35: three checks on the hierarchical community contraction (k32/k34). (1) The coarse gains are chosen by a line
search on a 2000-point Spearman sample DISJOINT from the benchmark's (seed 7 vs seed 1): held-out effect. (2) Generality:
the same label-free procedure (levels at Leiden r = 0.02 / 0.1 / 0.3; coarse gains by line search; finest level at
the dial c_f) applied to the cached BH t-SNE, openTSNE, UMAP, PaCMAP layouts of MNIST / Fashion. (3) Community counts
of the three levels on MNIST 70k vs a 20k subset and on the other sets (size dependence of the resolutions).
Usage: python benchmarks/theory/k35_contraction_checks.py [datasets=mnist,fmnist] [cf=0.7]"""
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
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist").split(","); CF = float(kw.get("cf", 0.7))
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k35_contraction_checks.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
LEVELS = (0.02, 0.1, 0.3); CS = (1.0, 0.8, 0.6, 0.5, 0.4)

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

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

def graph(X, ds):
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    if os.path.exists(pc): return sp.load_npz(pc)
    P = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32); sp.save_npz(pc, P); return P

# ---- (3) community counts and size dependence
print("community counts per level (Leiden RBConfiguration on the fuzzy graph):")
for ds, n_ in (("mnist", 70000), ("mnist", 20000), ("fmnist", 70000), ("letter", 20000), ("covertype", 20000)):
    k_ = f"counts/{ds}{n_}"
    if k_ not in res:
        if ds in ("mnist", "fmnist"):
            X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000)
            if n_ < 70000: X, y = X[:n_], y[:n_]; P = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32)
            else: P = graph(X, ds)
        else:
            from k27_transfer import load as load27
            X, y, key = load27(ds, n_); P = graph(X, ds)
        y = np.asarray(y).astype(int); ncomp = sp.csgraph.connected_components(P, directed=False)[0]
        res[k_] = {"components": int(ncomp)}
        for r_ in LEVELS + (1.0,):
            lab = leiden(P, r_); res[k_][str(r_)] = [int(lab.max() + 1), float(adjusted_rand_score(y, lab))]
        json.dump(res, open(OUT, "w"), indent=1)
    r = res[k_]; print(f"  {ds} n={n_}: {r['components']} components; " + ", ".join(f"r={k}: {v[0]} comm (ARI {v[1]:.2f})" for k, v in r.items() if k != "components"), flush=True)

# ---- (1) held-out line search and (2) generality
for ds in DS:
    X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = ref(ds); n = len(X); P = graph(X, ds)
    labs = {r_: leiden(P, r_) for r_ in LEVELS}
    g7 = _idx(n, 2000, 7); Dh7 = high_dim_pdist(X, g7, key + "_s7"); iu = np.triu_indices(len(g7), 1); dh7 = Dh7[iu]
    def sp_heldout(Y):
        Dl = np.sqrt(((Y[g7, None, :] - Y[None, g7, :]) ** 2).sum(-1)); return float(spearmanr(dh7, Dl[iu]).correlation)
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key); s = _idx(n, 10000, 2); kcl = len(np.unique(y))
        r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))))
        sh = [r[a] / rf[a] for a in ("knn15", "sp", "ari")]; r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh)); return r
    print(f"\n{ds}: levels " + ", ".join(f"r={r_}: {l.max() + 1} comm" for r_, l in labs.items()) + f"; panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}; finest level at c_f={CF}")
    print(f"{'layout':<20}{'step':<34}{'knn15':>7}{'sp':>7}{'ari':>7}{'geo':>7}{'min':>6}")
    layouts = {"MAGNET full-batch": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosstsne.npz"),
               "MAGNET hybrid": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosshybrid.npz"),
               "MAGNET quality": os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_full-batch_quality.npz"),
               "sklearn t-SNE (BH)": os.path.join(B, "cache", "layouts", f"{ds}_sklearn_t-SNE_BH.npz"),
               "openTSNE": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_opentsnefitsne.npz"),
               "UMAP": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_umap0512.npz"),
               "PaCMAP": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_pacmap.npz")}
    for lay, p in layouts.items():
        Y = np.load(p)["Y"].astype(np.float32); k0 = f"{ds}/{lay}"
        if k0 + "/identity" not in res: res[k0 + "/identity"] = score(Y); json.dump(res, open(OUT, "w"), indent=1)
        r = res[k0 + "/identity"]; print(f"{lay:<20}{'identity':<34}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
        chosen = []
        for r_ in LEVELS[:-1]:     # coarse levels: held-out line search, coordinate ascent
            k_ = f"{k0}/search r{r_}"
            if k_ not in res: res[k_] = {str(c): sp_heldout(contract(Y, labs[r_], c)) for c in CS}; json.dump(res, open(OUT, "w"), indent=1)
            c = float(max(res[k_], key=lambda c: res[k_][c])); chosen.append((r_, c)); Y = contract(Y, labs[r_], c)
            k2 = f"{k0}/after r{r_} c{c}"
            if k2 not in res: res[k2] = score(Y); json.dump(res, open(OUT, "w"), indent=1)
            r = res[k2]; print(f"{lay:<20}{f'r{r_} c{c} (held-out search)':<34}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
        Y = contract(Y, labs[LEVELS[-1]], CF); k3 = f"{k0}/final cf{CF}"
        if k3 not in res: res[k3] = score(Y); json.dump(res, open(OUT, "w"), indent=1)
        r = res[k3]; print(f"{lay:<20}{f'+ r{LEVELS[-1]} c{CF} (dial)':<34}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
