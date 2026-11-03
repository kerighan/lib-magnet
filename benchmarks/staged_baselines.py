"""Staged baselines: is the hybrid's gain the staging (any global layout + a full-batch t-SNE pass) or the SGD layout
it starts from? The same full-batch pass (300 iterations, no early exaggeration, no mid-range spring, hand-over scale
31 sqrt(n/70000)) applied to PCA-2, UMAP, PaCMAP, openTSNE, a random layout and MAGNET's SGD layout; and the
converse, MAGNET's SGD layout handed to openTSNE's optimiser for the same 300 iterations. MNIST and Fashion-MNIST 70k,
4 threads. Metrics before/after: 15-NN preservation, distance Spearman, centroid Spearman, time of the pass.
Usage: python benchmarks/staged_baselines.py [mnist,fmnist]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("NUMBA_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne
T = 4

def centroid_spearman(X, Y, y):
    C = np.array([X[y == c].mean(0) for c in np.unique(y)]); L = np.array([Y[y == c].mean(0) for c in np.unique(y)])
    iu = np.triu_indices(len(C), 1)
    return float(spearmanr(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu], np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1))[iu]).correlation)

out = {}
for ds in (sys.argv[1] if len(sys.argv) > 1 else "mnist,fmnist").split(","):
    X, y, name = LOADERS[ds](70000); y = np.asarray(y).astype(int); n = len(X)
    cache = os.path.join(HERE, "..", "paper", "figures", "cache")
    inits = {}
    def cached(slug, f):
        p = os.path.join(cache, f"{ds}_{slug}.npz")
        if os.path.exists(p):
            return np.load(p)["Y"].astype(np.float32)
        t = time.time(); Y = np.asarray(f(), dtype=np.float32); print(f"  {slug} computed in {time.time() - t:.0f} s", flush=True)
        return Y
    inits["PCA-2"] = PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32)
    inits["random"] = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)
    import umap; umap.UMAP(n_epochs=20).fit_transform(np.random.rand(6000, 10).astype(X.dtype))
    inits["UMAP"] = cached("umap0512", lambda: umap.UMAP().fit_transform(X))
    import pacmap; inits["PaCMAP"] = cached("pacmap", lambda: pacmap.PaCMAP(random_state=0).fit_transform(X))
    from openTSNE import TSNE
    inits["openTSNE"] = cached("opentsnefitsne", lambda: TSNE(n_jobs=T, random_state=0).fit(X))
    m = MAGNET(loss="umap", mid_weight=0.5, random_state=0, n_threads=T).fit(X)
    inits["MAGNET SGD (mid 0.5)"] = m.embedding_.astype(np.float32); P = m.graph_
    inits["MAGNET SGD (default)"] = cached("magnetsgd", lambda: MAGNET(random_state=0, n_threads=T).fit_transform(X))

    def metr(Y):
        r = evaluate(X, Y, y, cache_key=name); r["centroid_spearman"] = centroid_spearman(X, Y, y); return r
    print(f"\n{name}: {'init':<24}{'15-NN':>7}{'Spear':>7}{'cent':>7}   ->  {'15-NN':>7}{'Spear':>7}{'cent':>7}{'t(s)':>7}", flush=True)
    for k, Y0 in inits.items():
        r0 = metr(Y0)
        Y = Y0.copy(); Y -= Y.mean(0); Y *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y[:, 0].std()), 1e-12)
        t = time.time(); Y = optimize_tsne(P, Y, n_iter=300, early_iter=0, n_threads=T); dt = time.time() - t
        r1 = metr(Y)
        out[f"{name}/{k} -> full-batch pass"] = dict(before=r0, after=r1, time=dt)
        print(f"{'':<{len(name) + 2}}{k:<24}{r0['knn15_preservation']:>7.3f}{r0['dist_spearman']:>7.3f}{r0['centroid_spearman']:>7.3f}   ->  "
              f"{r1['knn15_preservation']:>7.3f}{r1['dist_spearman']:>7.3f}{r1['centroid_spearman']:>7.3f}{dt:>7.1f}", flush=True)
    # converse: MAGNET's SGD layout handed to openTSNE (its affinities, its optimiser), 300 iterations without exaggeration
    from openTSNE import affinity, TSNEEmbedding
    aff = affinity.PerplexityBasedNN(X, perplexity=30, n_jobs=T, random_state=0)
    for k in ("MAGNET SGD (mid 0.5)", "PCA-2"):
        Y0 = inits[k].copy(); Y0 -= Y0.mean(0); Y0 *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y0[:, 0].std()), 1e-12)
        t = time.time()
        emb = TSNEEmbedding(Y0, aff, negative_gradient_method="fft", n_jobs=T, random_state=0).optimize(n_iter=300, momentum=0.8)
        dt = time.time() - t; r1 = metr(np.asarray(emb, dtype=np.float32))
        out[f"{name}/{k} -> openTSNE pass"] = dict(before=metr(inits[k]), after=r1, time=dt)
        print(f"{'':<{len(name) + 2}}{k + ' -> openTSNE':<24}{'':>21}   ->  {r1['knn15_preservation']:>7.3f}{r1['dist_spearman']:>7.3f}{r1['centroid_spearman']:>7.3f}{dt:>7.1f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "cache", "staged_baselines.json"), "w"), indent=1)
