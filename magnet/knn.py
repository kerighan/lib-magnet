"""Approximate nearest neighbours (hnswlib) and conversion to a fuzzy neighbourhood graph."""
import os
import time
import numpy as np
import scipy.sparse as sp

_SPACES = {"euclidean": "l2", "l2": "l2", "cosine": "cosine", "ip": "ip", "inner_product": "ip"}


def knn_hnsw(X, n_neighbors, metric="euclidean", ef_construction=80, M=16, ef=None,
             n_threads=-1, random_state=0, verbose=False):
    """Return (indices, distances) of the n_neighbors - 1 nearest neighbours of each row of X,
    self excluded, sorted by increasing distance (same convention as UMAP: n_neighbors counts self)."""
    import hnswlib

    t = time.time()
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    if metric not in _SPACES:
        raise ValueError(f"metric must be one of {sorted(_SPACES)}, got {metric!r}")
    space = _SPACES[metric]
    k = min(n_neighbors, n)
    index = hnswlib.Index(space=space, dim=d)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M,
                     random_seed=int(random_state) if random_state is not None else 100)
    threads = n_threads if n_threads and n_threads > 0 else (os.cpu_count() or 4)
    index.set_num_threads(threads)
    index.add_items(X, np.arange(n))
    index.set_ef(ef if ef else max(2 * k + 10, 40))
    ind, dist = index.knn_query(X, k=k)
    if space == "l2":
        dist = np.sqrt(np.maximum(dist, 0.0))
    ind = ind.astype(np.int64)
    dist = dist.astype(np.float32)

    # remove self: the column holding i (usually the first one), else the last column
    is_self = ind == np.arange(n)[:, None]
    has_self = is_self.any(1)
    first = np.argmax(is_self, axis=1)
    first[~has_self] = k - 1
    keep = np.ones((n, k), dtype=bool)
    keep[np.arange(n), first] = False
    ind = ind[keep].reshape(n, k - 1)
    dist = dist[keep].reshape(n, k - 1)
    if verbose:
        print(f"kNN graph (hnswlib, k={k}, d={d}) - T={time.time() - t:.2f}s")
    return ind, dist


def prepare_data(X, metric):
    """Puts data in the form used by the native kernels: float32 (dense C-contiguous array or CSR with
    int32 indices). For cosine / inner product the rows are normalised and a column is appended that is 1
    for zero rows (they have no direction: this puts them at cosine distance 1 from every real row).
    Squared euclidean distances on the prepared rows equal 2 * cosine distance."""
    if metric not in _SPACES:
        raise ValueError(f"metric must be one of {sorted(_SPACES)}, got {metric!r}")
    space = _SPACES[metric]
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
        if space != "l2":
            norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
            X = sp.diags((1.0 / np.maximum(norms, 1e-12)).astype(np.float32)) @ X
            zero = sp.csr_matrix((norms == 0).astype(np.float32)[:, None])
            X = sp.hstack([X, zero], format="csr", dtype=np.float32)
        X = sp.csr_matrix(X, dtype=np.float32)
        X.sort_indices()
        if X.nnz >= 2 ** 31 - 1:
            raise ValueError("sparse inputs with more than 2^31 non-zeros are not supported")
        return X
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("data must be a 2-d array")
    if space != "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
        X = np.ascontiguousarray(np.hstack([X, (norms == 0).astype(np.float32)]))
    return X


def _convert_dists(dist, metric):
    if _SPACES[metric] == "l2":
        return np.sqrt(np.maximum(dist, 0.0)).astype(np.float32)
    return np.clip(dist / 2.0, 0.0, 2.0).astype(np.float32)


def _csr_parts(X):
    return X.indptr.astype(np.int32), X.indices.astype(np.int32), X.data.astype(np.float32)


def knn_nndescent(X, n_neighbors, metric="euclidean", k_internal=None, n_trees=None, leaf_size=30,
                  n_iters=10, delta=0.001, n_threads=-1, random_state=0, verbose=False, prepared=False):
    """Same output as knn_hnsw (n, n_neighbors - 1), computed by the native NN-descent kernel on a dense
    array or a scipy sparse matrix. k_internal (default max(k, 20)) neighbours are searched internally
    for a better recall. prepared=True means X already went through prepare_data."""
    from _magnet import nn_descent, nn_descent_sparse

    t = time.time()
    if not prepared:
        X = prepare_data(X, metric)
    n, d = X.shape
    k = min(n_neighbors, n) - 1
    if k < 1:
        raise ValueError("n_neighbors must be >= 2")
    if k_internal is None:
        k_internal = 30 if sp.issparse(X) else 20  # sparse / text data has harder neighbourhoods
    kin = min(max(k, int(k_internal)), n - 1)
    if n_trees is None:
        n_trees = int(max(2, min(8, round(np.sqrt(n) / 40))))
    threads = n_threads if n_threads and n_threads > 0 else -1
    seed = int(random_state) if random_state is not None else 0
    if sp.issparse(X):
        indptr, indices, data = _csr_parts(X)
        ind, dist = nn_descent_sparse(indptr, indices, data, int(d), int(kin), 50, int(n_iters), float(delta),
                                      int(n_trees), int(leaf_size), seed, int(threads), bool(verbose))
    else:
        ind, dist = nn_descent(X, int(kin), 50, int(n_iters), float(delta), int(n_trees), int(leaf_size),
                               seed, int(threads), bool(verbose))
    ind = np.asarray(ind)[:, :k].astype(np.int64)
    dist = _convert_dists(np.asarray(dist)[:, :k], metric)
    if verbose:
        print(f"kNN graph (nn-descent, k={n_neighbors}, d={d}, {n_trees} trees) - T={time.time() - t:.2f}s")
    return ind, dist


def choose_landmarks(n, random_state=0):
    """Random landmark rows used as entry points of the graph search (about 4 sqrt(n), at least 100)."""
    m = int(min(n, max(100, 4 * np.sqrt(n))))
    return np.sort(np.random.default_rng(random_state).choice(n, m, replace=False)).astype(np.int32)


def knn_search(X, graph, Q, k, metric="euclidean", landmarks=None, ef=None, n_entry=5, n_threads=-1,
               random_state=0):
    """k nearest rows of X (prepared with prepare_data) for each prepared query row of Q, by beam search
    over the symmetric neighbour graph `graph` (CSR over the rows of X), started from the n_entry closest
    landmarks. Returns (indices, distances)."""
    from _magnet import knn_search as _search, knn_search_sparse as _search_sparse

    graph = sp.csr_matrix(graph)
    gptr, gidx = graph.indptr.astype(np.int32), graph.indices.astype(np.int32)
    if landmarks is None:
        landmarks = choose_landmarks(X.shape[0], random_state)
    landmarks = np.asarray(landmarks, dtype=np.int32)
    ef = int(ef if ef else max(64, 2 * k + 10))
    threads = n_threads if n_threads and n_threads > 0 else -1
    seed = int(random_state) if random_state is not None else 0
    if sp.issparse(X):
        ip, ix, dt = _csr_parts(X)
        qp, qx, qd = _csr_parts(sp.csr_matrix(Q))
        ind, dist = _search_sparse(ip, ix, dt, int(X.shape[1]), gptr, gidx, landmarks, qp, qx, qd, int(k), ef, int(n_entry), seed, int(threads))
    else:
        ind, dist = _search(X, gptr, gidx, landmarks, np.ascontiguousarray(Q, dtype=np.float32), int(k), ef, int(n_entry), seed, int(threads))
    return np.asarray(ind).astype(np.int64), _convert_dists(np.asarray(dist), metric)


def smooth_knn(dist, n_neighbors, bandwidth=1.0, n_iter=64):
    """UMAP's smooth kNN kernel, vectorised over rows.
    dist: (n, k-1) sorted distances to the neighbours (self excluded).
    Returns membership strengths (n, k-1) in (0, 1]."""
    n, kk = dist.shape
    target = np.log2(n_neighbors) * bandwidth
    nonzero = dist > 0
    has = nonzero.any(1)
    first = np.argmax(nonzero, axis=1)
    rho = np.where(has, dist[np.arange(n), first], 0.0).astype(np.float32)
    dmr = np.maximum(dist - rho[:, None], 0.0)

    lo = np.zeros(n, dtype=np.float32)
    hi = np.full(n, np.inf, dtype=np.float32)
    mid = np.ones(n, dtype=np.float32)
    sigma = mid.copy()
    done = np.zeros(n, dtype=bool)
    for _ in range(n_iter):
        psum = np.exp(-dmr / mid[:, None]).sum(1)
        newly = (~done) & (np.abs(psum - target) < 1e-5)
        sigma[newly] = mid[newly]
        done |= newly
        if done.all():
            break
        gt = psum > target
        hi = np.where(gt, mid, hi)
        lo = np.where(gt, lo, mid)
        mid = np.where(gt, (lo + hi) / 2, np.where(np.isinf(hi), mid * 2, (lo + hi) / 2))
    sigma[~done] = mid[~done]

    mean_all = float(dist.mean())
    mean_row = dist.mean(1)
    sigma = np.where(rho > 0, np.maximum(sigma, 1e-3 * mean_row), np.maximum(sigma, 1e-3 * mean_all))
    vals = np.exp(-dmr / sigma[:, None]).astype(np.float32)
    vals[dmr <= 0] = 1.0
    return vals


def fuzzy_graph(ind, dist, n_neighbors, n=None, set_op_mix_ratio=1.0):
    """Symmetric fuzzy neighbourhood graph (CSR, float32) from kNN indices and distances."""
    n = ind.shape[0] if n is None else n
    vals = smooth_knn(dist, n_neighbors)
    rows = np.repeat(np.arange(n, dtype=np.int64), ind.shape[1])
    A = sp.csr_matrix((vals.ravel(), (rows, ind.ravel())), shape=(n, n), dtype=np.float32)
    A.eliminate_zeros()
    T = A.T.tocsr()
    prod = A.multiply(T)
    P = set_op_mix_ratio * (A + T - prod) + (1.0 - set_op_mix_ratio) * prod
    P = P.tocsr()
    P.eliminate_zeros()
    P.sort_indices()
    return P
