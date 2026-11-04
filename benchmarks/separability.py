"""Fourth block of the benchmark: cluster recovery on the 2-D layout (labels used only to score). On 10 000 points of
each cached layout (paper/figures/cache) plus PCA-2 and PCA-2 refined by the pass: silhouette (compactness), ARI of
k-means with k = number of classes, ARI and noise share of HDBSCAN (min_cluster_size 200, no k); and a composite:
each axis (15-NN, distance Spearman, ARI k-means) divided by the panel's best, then geometric mean and minimum.
-> benchmarks/cache/separability.json, paper/tables/separability.tex. Usage: python benchmarks/separability.py [8 threads]"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp, hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne

LAYOUTS = [("umap0512", "UMAP 0.5.12"), ("opentsnefitsne", "openTSNE (FIt-SNE)"), ("pacmap", "PaCMAP"), ("magnetsgd", "MAGNET SGD"), ("magnetlosstsne", "MAGNET full-batch"), ("magnetlosshybrid", "MAGNET hybrid")]
if sys.argv[1:] == ["table"]:   # rewrite the LaTeX table from the cached results only
    out = json.load(open(os.path.join(HERE, "cache", "separability.json")))
else:
    out = {}
    for loader, ds, label in ((mnist, "mnist", "MNIST"), (fmnist, "fmnist", "Fashion-MNIST")):
        X, y, key = loader(70000); y = np.asarray(y).astype(int); n = len(X); s = _idx(n, 10000, 2); k = len(np.unique(y))
        lay = {}
        P = sp.csr_matrix(MAGNET(random_state=0, n_threads=8).fit(X).graph_, dtype=np.float32)
        Y0 = PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32); lay["PCA-2"] = Y0
        Y = Y0 - Y0.mean(0); Y = (Y / Y[:, 0].std() * 31.0).astype(np.float32); lay["PCA-2 + pass"] = optimize_tsne(P, Y, n_iter=300, early_iter=0, n_threads=8)
        for slug, lab in LAYOUTS: lay[lab] = np.load(os.path.join(HERE, "..", "paper", "figures", "cache", f"{ds}_{slug}.npz"))["Y"]
        rows = {}
        for lab, Z in lay.items():
            r = evaluate(X, Z, y, cache_key=key); Zs = Z[s]
            cl = hdbscan.HDBSCAN(min_cluster_size=200).fit_predict(Zs); m = cl >= 0
            rows[lab] = dict(knn15=r["knn15_preservation"], sp=r["dist_spearman"], acc=r["knn_acc"], silhouette=float(silhouette_score(Zs, y[s])),
                             ari_kmeans=float(adjusted_rand_score(y[s], KMeans(k, n_init=5, random_state=0).fit_predict(Zs))),
                             ari_hdbscan=float(adjusted_rand_score(y[s][m], cl[m])) if m.sum() else float("nan"), hdbscan_clusters=int(len(set(cl[m]))), hdbscan_noise=float(1 - m.mean()))
        A = np.array([[rows[l]["knn15"], rows[l]["sp"], rows[l]["ari_kmeans"]] for l in rows]); share = A / A.max(0)
        for l, sh in zip(rows, share): rows[l]["composite_geo"] = float(sh.prod() ** (1 / 3)); rows[l]["composite_min"] = float(sh.min())
        out[label] = rows
        print(f"\n{label:<14}{'layout':<20}{'15-NN':>7}{'Spear':>7}{'silh':>7}{'ARI-km':>8}{'ARI-hdb':>9}{'clust':>6}{'noise':>7}{'geo':>7}{'min':>6}")
        for l, r in sorted(rows.items(), key=lambda t: -t[1]["composite_geo"]):
            print(f"{'':<14}{l:<20}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['silhouette']:>7.3f}{r['ari_kmeans']:>8.3f}{r['ari_hdbscan']:>9.3f}{r['hdbscan_clusters']:>6}{r['hdbscan_noise']:>7.2f}{r['composite_geo']:>7.3f}{r['composite_min']:>6.2f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "cache", "separability.json"), "w"), indent=1)
L = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r"data set & layout & 15-NN & Spearman & silhouette & ARI $k$-means & ARI HDBSCAN & clusters / noise & geo.\ mean & min \\", r"\midrule"]
for label, rows in out.items():
    from rankmarks import marks, latex
    order = sorted(rows.items(), key=lambda t: -t[1]["composite_geo"]); names = [l for l, _ in order]
    mk = {c: dict(zip(names, marks([r[c] if r[c] == r[c] else None for _, r in order], True, 2 if c == "composite_min" else 3))) for c in ("knn15", "sp", "silhouette", "ari_kmeans", "ari_hdbscan", "composite_geo", "composite_min")}
    for l, r in order:
        f = lambda c, d=3: latex(f"{r[c]:.{d}f}", mk[c][l])
        L.append(f"{label} & {l.replace('MAGNET', chr(92) + 'magnet{}')} & {f('knn15')} & {f('sp')} & {f('silhouette')} & {f('ari_kmeans')} & {f('ari_hdbscan')} & {r['hdbscan_clusters']} / {r['hdbscan_noise']:.2f} & {f('composite_geo')} & {f('composite_min', 2)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(HERE, "..", "paper", "tables", "separability.tex"), "w").write("\n".join(L) + "\n"); print("table written")
