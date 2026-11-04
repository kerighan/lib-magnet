"""Structural information for the layout: per-node diffusion scale for the mid-range attraction (effective
size of the 2-step transition distribution), mid-near pairs, local intrinsic dimension and exponents."""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def transition_matrix(P):
    return normalize(sp.csr_matrix(P, dtype=np.float32), norm="l1", axis=1)


def two_step_effective_size(P, max_nnz=200_000_000):
    """1 / sum_j T2_ij^2 with T2 = T @ T: the number of nodes a 2-step walk effectively reaches."""
    T = transition_matrix(P)
    if T.nnz ** 2 / max(1, T.shape[0]) > max_nnz:  # T2 would be too dense: fall back to 1 step
        T2 = T
    else:
        T2 = (T @ T).tocsr()
    T2.data = T2.data ** 2
    ipr = np.asarray(T2.sum(1)).ravel()
    return 1.0 / np.maximum(ipr, 1e-12)


def adaptive_walk_lengths(P, mode="spread", base=4, low=2, high=6):
    """Per-node length of the mid-range random walks ("spread"): nodes whose 2-step walks already reach many
    nodes get short walks, nodes in sparse or chain-like regions get long ones (ranks mapped linearly to
    [low, high])."""
    if mode != "spread":
        raise ValueError("mode must be 'spread'")
    n = P.shape[0]
    eff = two_step_effective_size(P)
    ranks = np.argsort(np.argsort(-eff)) / max(1, n - 1)       # 0 = widest spread
    return np.rint(low + (high - low) * ranks).astype(np.int32)


def mid_near_pairs(X, n_pairs=5, n_candidates=6, random_state=0, n_threads=-1):
    """PaCMAP-style mid-near partners: for every row, the second closest (in the data) of
    n_candidates random rows, repeated n_pairs times. Dense array or CSR matrix (prepared with
    prepare_data for the cosine metric). Returns (n, n_pairs) int32."""
    from _magnet import mid_near_pairs_dense, mid_near_pairs_sparse

    seed = int(random_state) if random_state is not None else 0
    threads = n_threads if n_threads and n_threads > 0 else -1
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
        X.sort_indices()
        return np.asarray(mid_near_pairs_sparse(X.indptr.astype(np.int32), X.indices.astype(np.int32),
                                                X.data.astype(np.float32), int(X.shape[1]), int(n_pairs),
                                                int(n_candidates), seed, int(threads)))
    return np.asarray(mid_near_pairs_dense(np.ascontiguousarray(X, dtype=np.float32), int(n_pairs),
                                           int(n_candidates), seed, int(threads)))


def local_intrinsic_dimension(X, k=15, random_state=0):
    """Per-point intrinsic dimension: dimprint's calibrated estimator when importable, otherwise the
    Levina-Bickel MLE on the k nearest neighbours (uncalibrated, biased low above ~5 dimensions)."""
    try:
        from dimprint import LocalIntrinsicDimension
        return np.asarray(LocalIntrinsicDimension(k=k).fit(X).dimension_pw_, dtype=np.float64)
    except Exception:
        from .knn import knn_nndescent
        ind, dist = knn_nndescent(X, k + 1, random_state=random_state)
        d = np.maximum(dist, 1e-12)
        logs = np.log(d[:, -1:] / d[:, :-1])
        return (k - 1) / np.maximum(logs.sum(1), 1e-12)


def local_kernel_exponent(X, k=15, random_state=0, m=2, law=None, low=0.3, high=1.0, smooth=True, graph=None):
    """Per-point attraction exponent from the local intrinsic dimension q_i (docs/kernel_exponents.md).
    The attraction optimum is a step at q = m: 1 (L2, the efficient estimator) when the point's
    neighbourhood fits in the target dimension, 1/2 (L1, the geometric median) when it is crowded.
    q_i comes from dimprint's calibrated estimator when it is importable (2.1 / 3.2 / 5.6 / 11 for
    Gaussian clusters of dimension 2 / 3 / 5 / 10), otherwise from the raw Levina-Bickel MLE (which
    reads 30% high); it is then averaged in log scale over the neighbourhood (`graph`, or a fresh
    kNN) and the step is placed at m + 0.5: conservative, because a crowded point filed as L2 loses
    more than an uncrowded point filed as L1 gains. `law` overrides the step (a callable q -> b)."""
    ind = None
    try:
        from dimprint import LocalIntrinsicDimension
        q = np.asarray(LocalIntrinsicDimension(k=k).fit(X).dimension_pw_, dtype=np.float64)
    except Exception:
        from .knn import knn_nndescent
        ind, dist = knn_nndescent(X, k + 1, random_state=random_state)
        d = np.maximum(dist[:, 1:], 1e-12)
        q = (k - 1) / np.maximum(np.log(d[:, -1:] / d[:, :-1]).sum(1), 1e-12)
        ind = ind[:, 1:]
    q = np.maximum(q, 1e-3)
    if smooth:
        lq = np.log(q)
        if graph is not None:
            G = graph.tocsr()
            deg = np.diff(G.indptr)
            s = np.bincount(np.repeat(np.arange(G.shape[0]), deg), weights=lq[G.indices], minlength=G.shape[0])
            q = np.exp((lq + s) / (1 + deg))
        else:
            if ind is None:
                from .knn import knn_nndescent
                ind = knn_nndescent(X, k + 1, random_state=random_state)[0][:, 1:]
            q = np.exp((lq + lq[ind].sum(1)) / (k + 1))
    if law is None:
        law = lambda q: np.where(q <= m + 0.5, 1.0, 0.5)
    return np.clip(law(np.asarray(q, dtype=np.float64)), low, high).astype(np.float32)


def select_kernel_exponent(X, candidates=(0.4, 0.5, 0.6, 0.7), n_sub=4000, n_epochs=150, metric="euclidean",
                           random_state=0, n_threads=-1, **magnet_kwargs):
    """Empirical choice of the kernel exponent: fit each candidate on a random subsample and keep the one
    whose layout preserves the most 15-nearest-neighbour sets (exact neighbours on the subsample)."""
    from sklearn.neighbors import NearestNeighbors
    from . import MAGNET

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.choice(n, min(n_sub, n), replace=False)
    Xs = X[idx]
    hnn = NearestNeighbors(n_neighbors=16, metric=metric, algorithm="brute").fit(Xs).kneighbors(Xs, return_distance=False)[:, 1:]
    best, best_score = candidates[0], -1.0
    scores = {}
    for b in candidates:
        Y = MAGNET(kernel_b=b, n_epochs=n_epochs, metric=metric, random_state=random_state, n_threads=n_threads,
                   **magnet_kwargs).fit_transform(Xs)
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        score = float(np.mean([len(np.intersect1d(a, c)) for a, c in zip(hnn, lnn)]) / 15)
        scores[b] = score
        if score > best_score:
            best, best_score = b, score
    return best, scores
