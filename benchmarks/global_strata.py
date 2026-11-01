"""Stratified global metrics on the six cached MNIST 70k layouts plus PCA-2, same 2000-point global sample
and cached high-dim distances as the benchmark table. Strata: same label / different label; terciles of
the high-dim distance (short / intermediate / long); and the arrangement of the classes themselves
(Spearman of the 45 inter-centroid distances, centroids of the full 70k). Writes paper/tables/strata.tex.
Usage: python benchmarks/global_strata.py"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from datasets import mnist
from metrics import _idx, high_dim_pdist

X, y, key = mnist(70000)
n = len(X); g = _idx(n, 2000, 1)
Dh = high_dim_pdist(X, g, key)
iu = np.triu_indices(len(g), 1)
dh = Dh[iu]; same = (y[g][iu[0]] == y[g][iu[1]])
t1, t2 = np.quantile(dh, [1 / 3, 2 / 3])
strata = {"same": same, "diff": ~same, "short": dh <= t1, "mid": (dh > t1) & (dh <= t2), "long": dh > t2}
Ch = np.array([X[y == c].mean(0) for c in range(10)])
Dch = np.sqrt(((Ch[:, None] - Ch[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]

layouts = {"PCA-2": PCA(2, svd_solver="randomized", random_state=0).fit_transform(X)}
for slug, label in [("umap0512", "UMAP 0.5.12"), ("opentsnefitsne", "openTSNE (FIt-SNE)"), ("pacmap", "PaCMAP"),
                    ("magnetsgd", "MAGNET SGD"), ("magnetlosstsne", "MAGNET full-batch"), ("magnetlosshybrid", "MAGNET hybrid")]:
    layouts[label] = np.load(os.path.join(HERE, "..", "paper", "figures", "cache", f"mnist_{slug}.npz"))["Y"]

cols = ["all", "same", "diff", "short", "mid", "long", "centroids"]
out = {}
print(f"{'method':<20}" + "".join(f"{c:>10}" for c in cols) + f"{'n pairs':>26}")
print(f"{'':<20}" + "".join(f"{'':>10}" for c in cols) + "  same/diff " + f"{same.sum()}/{(~same).sum()}" + f"  terciles at {t1:.2f}/{t2:.2f}")
for name, Y in layouts.items():
    Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); dl = Dl[iu]
    r = {"all": spearmanr(dh, dl).correlation}
    for k, m in strata.items():
        r[k] = spearmanr(dh[m], dl[m]).correlation
    Cl = np.array([Y[y == c].mean(0) for c in range(10)])
    Dcl = np.sqrt(((Cl[:, None] - Cl[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]
    r["centroids"] = spearmanr(Dch, Dcl).correlation
    out[name] = {k: float(v) for k, v in r.items()}
    print(f"{name:<20}" + "".join(f"{r[c]:>10.3f}" for c in cols), flush=True)
json.dump(out, open(os.path.join(HERE, "cache", "global_strata_mnist.json"), "w"), indent=1)

best = {c: max(v[c] for v in out.values()) for c in cols}
L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r"MNIST 70k, distance Spearman & all & same class & diff.\ class & short & intermediate & long & centroids \\", r"\midrule"]
for name, r in out.items():
    cells = [(r"\textbf{" + f"{r[c]:.3f}" + "}") if r[c] == best[c] else f"{r[c]:.3f}" for c in cols]
    L.append(name.replace("MAGNET", r"\magnet{}") + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "..", "paper", "tables", "strata.tex"), "w").write("\n".join(L) + "\n")
print("table written")
