"""k39: the 2 x 2 component ablation asked by the round-5 review. identity / placement only / separation only /
placement then separation, on matched layouts: MAGNET full-batch (three layout seeds) and Barnes-Hut t-SNE (the cached
sklearn layout; three Leiden + search seeds), MNIST and Fashion-MNIST, resolution 0.3, g = 0.7, communities from MAGNET's
graph (seed 0) for both. The search sample EXCLUDES the benchmark's evaluation points (the 2000-point Spearman sample,
the 5000 15-NN queries and the 10k ARI sample). All five raw metrics: 15-NN, Spearman, triplet accuracy, class-centroid
Spearman, k-means ARI; paired changes, mean +- sd over seeds. Writes paper/tables/placement.tex.
Usage: python benchmarks/theory/k39_placement_ablation.py [datasets=mnist,fmnist]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import hierarchical_contraction
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist").split(",")
OUT = os.path.join(HERE, "k39_placement_ablation.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
VARIANTS = {"identity": dict(placement=False, separation=1.0), "placement only": dict(placement=True, separation=1.0),
            "separation only": dict(placement=False, separation=0.7), "placement + separation": dict(placement=True, separation=0.7)}
AX = ("knn15", "sp", "trip", "cent", "ari")

for ds in DS:
    X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); n = len(y)
    P = sp.load_npz(os.path.join(B, "cache", f"P_{ds}.npz"))
    excl = np.unique(np.concatenate([_idx(n, 2000, 1), _idx(n, 5000, 0), _idx(n, 10000, 2)]))
    M = np.stack([np.asarray(X[y == c].mean(0)).ravel() for c in range(10)]); DM = np.sqrt(((M[:, None] - M[None]) ** 2).sum(-1)); iu = np.triu_indices(10, 1)
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key); s = _idx(n, 10000, 2)
        C = np.stack([Y[y == c].mean(0) for c in range(10)]); cent = float(spearmanr(DM[iu], np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu]).correlation)
        return dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], trip=e["triplet_acc"], cent=cent,
                    ari=float(adjusted_rand_score(y[s], KMeans(10, n_init=5, random_state=0).fit_predict(Y[s]))))
    layouts = []
    for seed in (0, 1, 2):
        p = os.path.join(B, "cache", "layouts", f"{ds}_fb_seed{seed}.npz")
        if not os.path.exists(p):
            Y = MAGNET(loss="tsne", random_state=seed, n_threads=T).fit_transform(X); np.savez(p, Y=Y)
        layouts.append((f"MAGNET full-batch", seed, np.load(p)["Y"].astype(np.float32)))
    bh = np.load(os.path.join(B, "cache", "layouts", f"{ds}_sklearn_t-SNE_BH.npz"))["Y"].astype(np.float32)
    for seed in (0, 1, 2): layouts.append(("Barnes-Hut t-SNE", seed, bh))
    print(f"\n{ds}: {len(excl)} evaluation points excluded from the search sample", flush=True)
    print(f"{'layout':<20}{'seed':>5}{'variant':<26}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'ari':>7}{'comm':>6}{'t':>6}")
    for lay, seed, Y in layouts:
        for vname, opts in VARIANTS.items():
            k_ = f"{ds}/{lay}/{seed}/{vname}"
            if k_ not in res:
                t = time.time(); Z, log = hierarchical_contraction(Y, P, X=X, random_state=seed, exclude=excl, **opts); dt = time.time() - t
                r = score(Z); r["t"] = dt; r["comm"] = int(log["communities"]); res[k_] = r; json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"{lay:<20}{seed:>5}{vname:<26}" + "".join(f"{r[a]:>7.3f}" for a in AX) + f"{r['comm']:>6}{r['t']:>6.1f}", flush=True)

# ---- summary and table
NAMES = {"mnist": "MNIST", "fmnist": "Fashion-MNIST"}
L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"data set, layout & variant & 15-NN & Spearman & triplet & centroid Sp. & $k$-means ARI \\", r"\midrule"]
print()
for ds in DS:
    for lay in ("MAGNET full-batch", "Barnes-Hut t-SNE"):
        for i, vname in enumerate(VARIANTS):
            rows = [res[f"{ds}/{lay}/{s}/{vname}"] for s in (0, 1, 2) if f"{ds}/{lay}/{s}/{vname}" in res]
            if not rows: continue
            m = {a: np.mean([r[a] for r in rows]) for a in AX}; sd = {a: np.std([r[a] for r in rows]) for a in AX}
            cells = " & ".join(f"{m[a]:.3f} $\\pm$ {sd[a]:.3f}" if sd[a] >= 0.0005 else f"{m[a]:.3f}" for a in AX)
            L.append((f"{NAMES[ds]}, {lay}" if i == 0 else "") + f" & {vname} & {cells} \\\\")
            print(f"{ds:<7}{lay:<20}{vname:<26}" + "  ".join(f"{a} {m[a]:.3f}+-{sd[a]:.3f}" for a in AX))
        L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(B, "..", "paper", "tables", "placement.tex"), "w").write("\n".join(L) + "\n"); print("table written")
