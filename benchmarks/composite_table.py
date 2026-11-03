"""Method x data set tables of the three-axis composite (geometric mean, and minimum, of the shares of the panel best on
15-NN preservation, distance Spearman and k-means ARI; ARI dropped where the classes are not clusters: letter, covertype).
Layouts cached in benchmarks/cache/layouts/<dataset>_<method>.npz (MNIST / Fashion figure layouts reused). 1 run, 8 threads.
-> benchmarks/cache/composite_table.json, paper/tables/composite_geo.tex, composite_min.tex, and a markdown print.
Usage: python benchmarks/composite_table.py [methods=core|all] [datasets=mnist,fmnist,letter,covertype]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "theory"))
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(","); SET = kw.get("methods", "core")
CACHE = os.path.join(HERE, "cache", "layouts"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(HERE, "cache", "composite_table.json"); out = json.load(open(OUT)) if os.path.exists(OUT) else {}
NAMES = {"mnist": "MNIST 70k", "fmnist": "Fashion 70k", "letter": "letter 20k", "covertype": "covertype 20k"}

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    return load27(name, 20000)

def methods(X, P_getter):
    import umap, pacmap
    M = {"MAGNET SGD": lambda: MAGNET(random_state=0, n_threads=T).fit_transform(X),
         "MAGNET full-batch": lambda: MAGNET(loss="tsne", random_state=0, n_threads=T).fit_transform(X),
         "MAGNET hybrid": lambda: MAGNET(loss="hybrid", random_state=0, n_threads=T).fit_transform(X),
         "MAGNET full-batch (quality)": lambda: MAGNET(loss="tsne", n_epochs=900, hybrid_tail=(100, 1.5), random_state=0, n_threads=T).fit_transform(X),
         "MAGNET full-batch + contraction": lambda: MAGNET(loss="tsne", contraction=0.7, random_state=0, n_threads=T).fit_transform(X),
         "MAGNET full-batch (quality) + contraction": lambda: MAGNET(loss="tsne", n_epochs=900, hybrid_tail=(100, 1.5), contraction=0.7, random_state=0, n_threads=T).fit_transform(X),
         "MAGNET hybrid + contraction": lambda: MAGNET(loss="hybrid", contraction=0.7, random_state=0, n_threads=T).fit_transform(X),
         "UMAP": lambda: umap.UMAP().fit_transform(X),
         "PaCMAP": lambda: pacmap.PaCMAP(random_state=0).fit_transform(X),
         "openTSNE": lambda: np.asarray(__import__("openTSNE").TSNE(n_jobs=T, random_state=0).fit(X)),
         "PCA-2": lambda: PCA(2, svd_solver="randomized", random_state=0).fit_transform(X)}
    def pca_pass():
        Y0 = PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32); Y = Y0 - Y0.mean(0); Y = (Y / Y[:, 0].std() * 31.0 * np.sqrt(len(X) / 70000.0)).astype(np.float32)
        return optimize_tsne(P_getter(), Y, n_iter=300, early_iter=0, n_threads=T)
    M["PCA-2 + pass"] = pca_pass
    if SET == "all":
        M["TriMap"] = lambda: __import__("trimap").TRIMAP().fit_transform(np.asarray(X, dtype=np.float32))
        M["h-NNE"] = lambda: __import__("hnne").HNNE().fit_transform(np.asarray(X, dtype=np.float32))
        M["densMAP"] = lambda: umap.UMAP(densmap=True).fit_transform(X)
        M["sklearn t-SNE (BH)"] = lambda: __import__("sklearn.manifold", fromlist=["TSNE"]).TSNE(n_jobs=T, init="pca", random_state=0).fit_transform(np.asarray(X, dtype=np.float32))
    return M

for ds in DS:
    X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(X); s = _idx(n, 10000, 2); kcl = len(np.unique(y))
    _P = {}
    def P_getter():
        if "P" not in _P: _P["P"] = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32)
        return _P["P"]
    import umap; umap.UMAP(n_epochs=20).fit_transform(np.random.rand(6000, X.shape[1]).astype(np.float32))
    for name, f in methods(X, P_getter).items():
        k_ = f"{ds}/{name}"
        if k_ in out: continue
        slug = {"MAGNET SGD": "magnetsgd", "MAGNET full-batch": "magnetlosstsne", "MAGNET hybrid": "magnetlosshybrid", "UMAP": "umap0512", "PaCMAP": "pacmap", "openTSNE": "opentsnefitsne"}.get(name)
        fig = os.path.join(HERE, "..", "paper", "figures", "cache", f"{ds}_{slug}.npz") if slug else None
        cache = os.path.join(CACHE, f"{ds}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz")
        if fig and os.path.exists(fig): Y = np.load(fig)["Y"]; dt = float(np.load(fig)["dt"]) if "dt" in np.load(fig) else float("nan")
        elif os.path.exists(cache): z = np.load(cache); Y, dt = z["Y"], float(z["dt"])
        else:
            try:
                t = time.time(); Y = np.asarray(f(), dtype=np.float32); dt = time.time() - t; np.savez(cache, Y=Y, dt=dt)
            except Exception as e:
                print(f"{ds} {name} failed: {e}", flush=True); continue
        e = evaluate(X, Y, y, cache_key=key)
        out[k_] = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], acc=e["knn_acc"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))), t=dt)
        json.dump(out, open(OUT, "w"), indent=1); r = out[k_]
        print(f"{ds:<10}{name:<20}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{dt:>8.1f}", flush=True)

# ---- tables
per = {}
for k_, r in out.items():
    ds, m = k_.split("/", 1); per.setdefault(ds, {})[m] = r
dss = [d for d in ("mnist", "fmnist", "letter", "covertype") if d in per]
meths = []
for d in dss:
    for m in per[d]:
        if m not in meths: meths.append(m)
ref = {d: {a: max(r[a] for r in per[d].values()) for a in ("knn15", "sp", "ari")} for d in dss}
axes = {d: (("knn15", "sp", "ari") if ref[d]["ari"] >= 0.3 else ("knn15", "sp")) for d in dss}
def comp(d, m, kind):
    if m not in per[d]: return float("nan")
    sh = [per[d][m][a] / ref[d][a] for a in axes[d]]; return float(np.prod(sh) ** (1 / len(sh))) if kind == "geo" else float(min(sh))
for kind, title in (("geo", "composite, geometric mean"), ("min", "composite, minimum")):
    rows = sorted(meths, key=lambda m: -np.nanmean([comp(d, m, kind) for d in dss]))
    print(f"\n{title}  (axes: " + ", ".join(f"{NAMES[d]} {len(axes[d])}" for d in dss) + ")")
    print("| method | " + " | ".join(NAMES[d] for d in dss) + " | mean |"); print("|---|" + "---|" * (len(dss) + 1))
    L = [r"\begin{tabular}{l" + "r" * (len(dss) + 1) + "}", r"\toprule", "method & " + " & ".join(NAMES[d] for d in dss) + r" & mean \\", r"\midrule"]
    for m in rows:
        vals = [comp(d, m, kind) for d in dss]; mean = np.nanmean(vals)
        cells = []
        for d, v in zip(dss, vals):
            col = [comp(d, mm, kind) for mm in meths]; rank = sum(1 for c in col if c > v + 1e-12)
            txt = "--" if v != v else f"{v:.3f}"; cells.append(("**" + txt + "**") if rank == 0 and v == v else (("_" + txt + "_") if rank == 1 and v == v else txt))
        print(f"| {m} | " + " | ".join(cells) + f" | {mean:.3f} |")
        L.append(m.replace("MAGNET", r"\magnet{}") + " & " + " & ".join(("--" if v != v else f"{v:.3f}") for v in vals) + f" & {mean:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "..", "paper", "tables", f"composite_{kind}.tex"), "w").write("\n".join(L) + "\n")
