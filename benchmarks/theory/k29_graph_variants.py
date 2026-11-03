"""k29: altering the kNN graph fed to the full-batch mode (hubness, symmetrisation, reweighting, affinities), same
optimiser, same PCA init, same 600 iterations. Variants of P from the same kNN (k = 15 unless said): fuzzy union
(default), fuzzy intersection, mix 0.5, mutual kNN, shared-nearest-neighbour weights (k = 15 and k = 30 neighbourhoods),
local scaling (Zelnik-Manor, sigma = 7th neighbour), Sinkhorn doubly-stochastic scaling of the union graph, hub-capped
union (incoming edges of hubs cut to 3k), symmetric degree normalisation, uniform weights on the kNN edges, and
t-SNE's perplexity-30 affinities (k = 90, openTSNE). MNIST 70k, Fashion 70k, 20 newsgroups (SVD-300, cosine).
Scores: 15-NN, Spearman, k-means ARI, composite with the references of the full-batch search. 8 threads.
-> k29_graph_variants.json / .log. Usage: python benchmarks/theory/k29_graph_variants.py"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
from datasets import mnist, fmnist, news20
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.knn import knn_nndescent, fuzzy_graph, smooth_knn
T = 8
REF = json.load(open(os.path.join(B, "cache", "fullbatch_search_summary.json")))["ref"]

def variants(ind, dist, ind30, dist30, k, metric_X):
    n = ind.shape[0]
    if ind.shape[1] > k:          # the search keeps more candidates than n_neighbors
        ind, dist = ind[:, :k], dist[:, :k]
    rows = np.repeat(np.arange(n), ind.shape[1])
    V = {}
    V["fuzzy union (default)"] = fuzzy_graph(ind, dist, k)
    V["fuzzy intersection"] = fuzzy_graph(ind, dist, k, set_op_mix_ratio=0.0)
    V["fuzzy mix 0.5"] = fuzzy_graph(ind, dist, k, set_op_mix_ratio=0.5)
    A = sp.csr_matrix((np.ones(ind.size, np.float32), (rows, ind.ravel())), shape=(n, n)); A.setdiag(0); A.eliminate_zeros()
    U = fuzzy_graph(ind, dist, k); M = A.multiply(A.T)           # mutual edges mask
    V["mutual kNN (fuzzy weights)"] = U.multiply(M > 0).tocsr()
    S = (A @ A.T).tocsr(); S.setdiag(0); S.eliminate_zeros()      # shared neighbours
    Un = ((A + A.T) > 0).astype(np.float32).tocsr()
    V["SNN weights, k 15"] = (S.multiply(Un) / k).tocsr()
    k30 = ind30.shape[1]
    A30 = sp.csr_matrix((np.ones(n * k30, np.float32), (np.repeat(np.arange(n), k30), ind30.ravel())), shape=(n, n)); A30.setdiag(0); A30.eliminate_zeros()
    S30 = (A30 @ A30.T).tocsr(); S30.setdiag(0); V["SNN weights, k 30 on k 15 edges"] = (S30.multiply(Un) / k30).tocsr()
    sig = dist[:, 6].astype(np.float64) + 1e-9
    w = np.exp(-(dist.astype(np.float64) ** 2) / (sig[:, None] * sig[ind]))
    L = sp.csr_matrix((w.ravel().astype(np.float32), (rows, ind.ravel())), shape=(n, n)); L.setdiag(0); L.eliminate_zeros(); V["local scaling (sigma 7th nn)"] = L.maximum(L.T).tocsr()
    # Sinkhorn: symmetric scaling d so that diag(d) U diag(d) has unit row sums
    d = np.ones(n); Uc = U.tocsr()
    for _ in range(50):
        r = Uc @ d; d = d / np.maximum(r, 1e-12); d = np.sqrt(d * (1.0 / np.maximum(Uc @ (1.0 / np.maximum(Uc @ d, 1e-12)), 1e-12))) if False else d
        d = np.sqrt(d * (1.0 / np.maximum(Uc @ d, 1e-12))) if False else d
    # simple symmetric Sinkhorn (Knight): d <- sqrt(d / (U d))
    d = np.ones(n)
    for _ in range(100): d = np.sqrt(d / np.maximum(Uc @ d, 1e-12))
    V["Sinkhorn (doubly stochastic)"] = (sp.diags(d) @ Uc @ sp.diags(d)).tocsr().astype(np.float32)
    indeg = np.asarray(A.sum(0)).ravel(); hub = indeg > 3 * k
    Ac = A.tolil(); At = A.T.tocsr()
    for j in np.where(hub)[0]:
        src = At.indices[At.indptr[j]:At.indptr[j + 1]]; dd = np.array([dist[i, list(ind[i]).index(j)] for i in src])
        drop = src[np.argsort(dd)[3 * k:]]
        for i in drop: Ac[i, j] = 0
    Ac = Ac.tocsr(); Ac.eliminate_zeros()
    Uw = U.multiply((Ac + Ac.T) > 0).tocsr(); V[f"hub-capped union ({hub.sum()} hubs)"] = Uw
    deg = np.asarray(U.sum(1)).ravel(); inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12)); V["sym degree-normalised union"] = (sp.diags(inv) @ U @ sp.diags(inv)).tocsr().astype(np.float32)
    V["uniform weights on kNN edges"] = Un
    return V

out = json.load(open(os.path.join(HERE, "k29_graph_variants.json"))) if os.path.exists(os.path.join(HERE, "k29_graph_variants.json")) else {}
for loader, label, metric in ((mnist, "mnist", "euclidean"), (fmnist, "fmnist", "euclidean"), (news20, "news20", "cosine")):
    X, y, key = loader(70000) if label != "news20" else loader(); y = np.asarray(y).astype(int); n = len(X); s = _idx(n, 10000, 2); kcl = len(np.unique(y))
    m = MAGNET(loss="tsne", random_state=0, n_threads=T, metric=metric).fit(X); ind, dist = m.knn_indices_, m.knn_dists_
    ind30, dist30 = knn_nndescent(m._prepared_, 30, metric, n_threads=T, random_state=0, prepared=True)
    init = PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32)
    V = variants(ind, dist, ind30, dist30, m.n_neighbors, X)
    try:
        from openTSNE.affinity import PerplexityBasedNN
        aff = PerplexityBasedNN(X, perplexity=30, metric=metric, n_jobs=T, random_state=0); V["t-SNE perplexity-30 affinities (k 90)"] = sp.csr_matrix(aff.P, dtype=np.float32)
    except Exception as e:
        print("openTSNE affinities skipped:", e)
    ref = REF.get(key, None)
    print(f"\n=== {key}: {'variant':<40}{'edges/n':>8}{'15-NN':>7}{'Spear':>7}{'ARI':>7}{'geo':>7}{'min':>6}{'t(s)':>7}", flush=True)
    for name, P in V.items():
        k_ = f"{key}/{name}"
        if k_ in out: r = out[k_]
        else:
            P = sp.csr_matrix(P, dtype=np.float32); P.setdiag(0); P.eliminate_zeros()
            t = time.time(); Y = MAGNET(loss="tsne", random_state=0, n_threads=T, graph_norm=None, init=init).fit_transform(P).astype(np.float32); dt = time.time() - t
            e = evaluate(X, Y, y, cache_key=key); r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], acc=e["knn_acc"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))), edges=float(P.nnz / n), t=dt)
            out[k_] = r; json.dump(out, open(os.path.join(HERE, "k29_graph_variants.json"), "w"), indent=1)
        if ref:
            axes = ("knn15", "sp", "ari") if ref["ari"] >= 0.3 else ("knn15", "sp"); sh = [r[a] / ref[a] for a in axes]; g, mn = np.prod(sh) ** (1 / len(sh)), min(sh)
        else: g = mn = float("nan")
        print(f"{'':<5}{name:<40}{r['edges']:>8.1f}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{g:>7.3f}{mn:>6.2f}{r['t']:>7.1f}", flush=True)
