"""k41: the matched post-process control asked by the round-6 review (paper/reviews/..._round6_panel.md). Every
configuration of the composite panel, MAGNET's three modes and nine comparators, is run end to end with a replicate
seed, then post-processed by the identical label-free placement + separation (MAGNET's graph of that replicate, same
communities, same search sample, evaluation points excluded, all settings frozen), and scored before and after on
15-NN / Spearman / k-means ARI (+ triplet accuracy and class-centroid Spearman, unfitted) against FROZEN unprocessed
references (round-6 calibration panel). Composites (geometric mean and minimum of the ratios) are computed per run,
then mean +- sd over the replicates; the summary prints the paired post-process gain of every configuration and the
paired difference between each post-processed comparator and MAGNET hybrid + post-process. Times are reported in
blocks: embedding, graph + communities, placement, evaluation excluded.
Usage: python benchmarks/theory/k41_matched_postprocess.py [datasets=mnist,fmnist,letter,covertype] [seeds=5] [methods=...] [part=run,summary]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from magnet.contraction import hierarchical_contraction
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:])
DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(","); SEEDS = int(kw.get("seeds", 5)); PART = kw.get("part", "run,summary").split(",")
OUT = os.path.join(HERE, "k41_matched_postprocess.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
# frozen references: maxima of the unprocessed calibration panel (composite_table.json, 2026-11-04), round-6 review
REF = {"mnist": dict(knn15=0.356, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.330, sp=0.873, ari=0.483),
       "letter": dict(knn15=0.642, sp=0.716), "covertype": dict(knn15=0.632, sp=0.755)}
NAMES = {"mnist": "MNIST 70k", "fmnist": "Fashion 70k", "letter": "letter 20k", "covertype": "covertype 20k"}
ALL = ["MAGNET SGD", "MAGNET full-batch", "MAGNET hybrid", "UMAP", "PaCMAP", "openTSNE", "sklearn t-SNE (BH)", "TriMap", "h-NNE", "densMAP", "PCA-2", "PCA-2 + pass"]
METHODS = kw.get("methods", ",".join(ALL)).split(",")

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    return load27(name, 20000)

def fit(method, X, s, P_getter):
    X32 = np.asarray(X, dtype=np.float32)
    if method == "MAGNET SGD": return MAGNET(random_state=s, n_threads=T).fit_transform(X)
    if method == "MAGNET full-batch": return MAGNET(loss="tsne", random_state=s, n_threads=T).fit_transform(X)
    if method == "MAGNET hybrid": return MAGNET(loss="hybrid", random_state=s, n_threads=T).fit_transform(X)
    if method == "UMAP": import umap; return umap.UMAP().fit_transform(X)            # unseeded (seeding serialises its layout); numba's RNG differs per run
    if method == "densMAP": import umap; return umap.UMAP(densmap=True).fit_transform(X)
    if method == "PaCMAP": import pacmap; return pacmap.PaCMAP(random_state=s).fit_transform(X)
    if method == "openTSNE": from openTSNE import TSNE; return np.asarray(TSNE(n_jobs=T, random_state=s).fit(X))
    if method == "sklearn t-SNE (BH)": from sklearn.manifold import TSNE; return TSNE(n_jobs=T, init="pca", random_state=s).fit_transform(X32)
    if method == "TriMap": import trimap; return trimap.TRIMAP().fit_transform(X32)
    if method == "h-NNE": import hnne; return hnne.HNNE().fit_transform(X32)
    if method == "PCA-2": return PCA(2, svd_solver="randomized", random_state=s).fit_transform(X)
    if method == "PCA-2 + pass":
        Y0 = PCA(2, svd_solver="randomized", random_state=s).fit_transform(X).astype(np.float32); Y = Y0 - Y0.mean(0)
        Y = (Y / Y[:, 0].std() * 31.0 * np.sqrt(len(X) / 70000.0)).astype(np.float32)
        return optimize_tsne(P_getter(), Y, n_iter=300, early_iter=0, n_threads=T)
    raise ValueError(method)

def composite(r, ds):
    sh = [r[a] / REF[ds][a] for a in REF[ds]]; return float(np.prod(sh) ** (1 / len(sh))), float(min(sh))

if "run" in PART:
    for ds in DS:
        X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); s10 = _idx(n, 10000, 2)
        excl = np.unique(np.concatenate([_idx(n, 2000, 1), _idx(n, 5000, 0), _idx(n, 10000, 2)]))
        M = np.stack([np.asarray(X[y == c].mean(0)).ravel() for c in range(kcl)]); DM = np.sqrt(((M[:, None] - M[None]) ** 2).sum(-1)); iu = np.triu_indices(kcl, 1)
        def score(Y):
            e = evaluate(X, Y, y, cache_key=key); C = np.stack([Y[y == c].mean(0) for c in range(kcl)])
            r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], trip=e["triplet_acc"],
                     cent=float(spearmanr(DM[iu], np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu]).correlation),
                     ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
            r["geo"], r["min"] = composite(r, ds); return r
        print(f"\n=== {ds}: {n} points, {kcl} classes, refs {REF[ds]}, {len(excl)} evaluation points excluded from the search sample", flush=True)
        for s in range(SEEDS):
            graphs = {}
            def P_getter():
                if "P" not in graphs:
                    t = time.time(); graphs["P"] = sp.csr_matrix(MAGNET(random_state=s, n_threads=T).fit(X).graph_, dtype=np.float32); graphs["t"] = time.time() - t
                return graphs["P"]
            for m in METHODS:
                k_ = f"{ds}/{m}/{s}"
                if k_ in res: continue
                try:
                    t = time.time(); Y = np.ascontiguousarray(np.asarray(fit(m, X, s, P_getter), dtype=np.float32)); t_fit = time.time() - t
                except Exception as e:
                    print(f"  {m} seed {s} failed: {e}", flush=True); continue
                P = P_getter(); t_graph = graphs["t"]
                r0 = score(Y)
                t = time.time(); Z, log = hierarchical_contraction(Y, P, X=X, random_state=s, exclude=excl, n_threads=T); t_post = time.time() - t
                r1 = score(np.ascontiguousarray(Z, dtype=np.float32))
                res[k_] = dict(plain=r0, post=r1, t_fit=t_fit, t_graph=t_graph, t_post=t_post, communities=int(log["communities"]))
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<20} seed {s}: {r0['knn15']:.3f}/{r0['sp']:.3f}/{r0['ari']:.3f} geo {r0['geo']:.3f} min {r0['min']:.3f} -> {r1['knn15']:.3f}/{r1['sp']:.3f}/{r1['ari']:.3f} geo {r1['geo']:.3f} min {r1['min']:.3f} | fit {t_fit:.1f} s, graph {t_graph:.1f}, post {t_post:.1f}, {log['communities']} comm", flush=True)

if "summary" in PART:
    print("\n==== summary: mean +- sd over replicates, composites per run; ratios to the frozen unprocessed references ====")
    for ds in DS:
        rows = {}
        for m in ALL:
            R = [res[f"{ds}/{m}/{s}"] for s in range(SEEDS) if f"{ds}/{m}/{s}" in res]
            if R: rows[m] = R
        if not rows: continue
        ref_runs = rows.get("MAGNET hybrid", [])
        print(f"\n{NAMES[ds]} ({', '.join(f'{m}: {len(R)} runs' for m, R in rows.items() if len(R) != SEEDS) or 'all ' + str(SEEDS) + ' runs'})")
        print(f"{'configuration':<20}{'plain 15-NN/Sp/ARI':>24}{'geo':>14}{'min':>14}   {'post 15-NN/Sp/ARI':>24}{'geo':>14}{'min':>14}{'d geo':>9}{'vs hybrid+post':>18}{'t fit':>7}{'t post':>7}")
        for m, R in rows.items():
            def ms(key, a): v = [r[key][a] for r in R]; return np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0
            p = {a: ms("plain", a) for a in ("knn15", "sp", "ari", "geo", "min")}; q = {a: ms("post", a) for a in ("knn15", "sp", "ari", "geo", "min")}
            dgeo = [r["post"]["geo"] - r["plain"]["geo"] for r in R]
            diff = ""
            if ref_runs and m != "MAGNET hybrid":
                k = min(len(R), len(ref_runs)); dd = [R[i]["post"]["geo"] - ref_runs[i]["post"]["geo"] for i in range(k)]
                diff = f"{np.mean(dd):+.3f} +- {np.std(dd, ddof=1) if k > 1 else 0:.3f}"
            print(f"{m:<20}{p['knn15'][0]:.3f}/{p['sp'][0]:.3f}/{p['ari'][0]:.3f}".ljust(44) + f"{p['geo'][0]:.3f} +- {p['geo'][1]:.3f}".rjust(14) + f"{p['min'][0]:.3f} +- {p['min'][1]:.3f}".rjust(14)
                  + "   " + f"{q['knn15'][0]:.3f}/{q['sp'][0]:.3f}/{q['ari'][0]:.3f}".rjust(24) + f"{q['geo'][0]:.3f} +- {q['geo'][1]:.3f}".rjust(14) + f"{q['min'][0]:.3f} +- {q['min'][1]:.3f}".rjust(14)
                  + f"{np.mean(dgeo):+.3f}".rjust(9) + diff.rjust(18) + f"{np.mean([r['t_fit'] for r in R]):7.1f}{np.mean([r['t_post'] for r in R]):7.1f}")
