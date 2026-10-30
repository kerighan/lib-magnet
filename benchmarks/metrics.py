"""Embedding quality metrics (all computed on fixed random subsamples so runs are comparable)."""
import os
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def _idx(n, size, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(size, n), replace=False))


def high_dim_neighbors(X, idx, k, cache_key):
    """Exact k nearest neighbours (in X, self excluded) of the rows idx. Cached on disk."""
    path = os.path.join(CACHE, f"hnn_{cache_key}_k{k}_q{len(idx)}.npy")
    if cache_key and os.path.exists(path):
        return np.load(path)
    metric = "cosine" if sp.issparse(X) else "euclidean"
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric, n_jobs=-1).fit(X)
    ind = nn.kneighbors(X[idx], return_distance=False)
    # drop self (usually first column, but be safe with duplicates)
    out = np.empty((len(idx), k), dtype=np.int64)
    for r, (i, row) in enumerate(zip(idx, ind)):
        row = row[row != i]
        out[r] = row[:k]
    if cache_key:
        np.save(path, out)
    return out


def high_dim_pdist(X, idx, cache_key):
    path = os.path.join(CACHE, f"hpd_{cache_key}_q{len(idx)}.npy")
    if cache_key and os.path.exists(path):
        return np.load(path)
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X[idx], metric="cosine" if sp.issparse(X) else "euclidean", n_jobs=-1).astype(np.float32)
    if cache_key:
        np.save(path, D)
    return D


def evaluate(X, Y, labels=None, cache_key=None, k=15, n_query=5000, n_global=2000, seed=0):
    """X: high-dim data (n, d) or None for graphs ; Y: embedding (n, 2)."""
    n = len(Y)
    res = {}
    if X is not None:
        # --- local structure: kNN preservation and trustworthiness
        q = _idx(n, n_query, seed)
        hnn = high_dim_neighbors(X, q, k, cache_key)
        lnn = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y[q], return_distance=False)
        lnn = np.array([row[row != i][:k] for i, row in zip(q, lnn)])
        inter = [len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]
        res[f"knn{k}_preservation"] = float(np.mean(inter) / k)
        res["trustworthiness"] = float(trustworthiness(X[q], Y[q], n_neighbors=k, metric="cosine" if sp.issparse(X) else "euclidean"))
        # --- global structure: distance correlation + random triplet accuracy on a sample
        g = _idx(n, n_global, seed + 1)
        Dh = high_dim_pdist(X, g, cache_key)
        Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)).astype(np.float32)
        iu = np.triu_indices(len(g), 1)
        res["dist_spearman"] = float(spearmanr(Dh[iu], Dl[iu]).correlation)
        rng = np.random.default_rng(seed)
        m = len(g)
        i, j, l = rng.integers(0, m, (3, 200000))
        ok = (i != j) & (i != l) & (j != l)
        i, j, l = i[ok], j[ok], l[ok]
        res["triplet_acc"] = float(np.mean(np.sign(Dh[i, j] - Dh[i, l]) == np.sign(Dl[i, j] - Dl[i, l])))
    if labels is not None:
        s = _idx(n, 10000, seed + 2)
        clf = KNeighborsClassifier(n_neighbors=10)
        res["knn_acc"] = float(cross_val_score(clf, Y[s], labels[s], cv=5).mean())
        res["silhouette"] = float(silhouette_score(Y[s], labels[s]))
    return res
