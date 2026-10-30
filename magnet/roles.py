"""Structural information for the layout: per-node diffusion scale for the mid-range attraction and
role signatures for the placement of disconnected components. Two sources: a cheap graph proxy
(effective size of the 2-step transition distribution) or RoleWalk embeddings (the author's
structural-role library, imported lazily)."""
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


def rolewalk_embedding(P, walk_len=3, n_samples=10):
    """RoleWalk signature of every node of the (symmetric) graph P, computed on the sparse matrix
    directly (same construction as RoleWalk.transform: lazy walk on I + A)."""
    from rolewalk import RoleWalk, compute_embedding

    rw = RoleWalk(walk_len=walk_len, n_samples=n_samples)
    A = sp.csr_matrix(P, dtype=np.float32)
    H = normalize(sp.identity(A.shape[0], format="csr", dtype=np.float32) + A, norm="l1").tocsr()
    return np.asarray(compute_embedding(H, rw.theta, rw.walk_len, rw.aggregate, rw.tol, None), dtype=np.float32)


def adaptive_walk_lengths(P, mode="spread", base=4, low=2, high=6, n_roles=None):
    """Per-node length of the mid-range random walks.
    "spread": nodes whose 2-step walks already reach many nodes get short walks, nodes in sparse or
    chain-like regions get long ones (ranks mapped linearly to [low, high]).
    "rolewalk": same target, but assigned per RoleWalk role (median spread of the role), so that
    structurally equivalent nodes share a scale."""
    n = P.shape[0]
    eff = two_step_effective_size(P)
    if mode == "spread":
        ranks = np.argsort(np.argsort(-eff)) / max(1, n - 1)       # 0 = widest spread
        return np.rint(low + (high - low) * ranks).astype(np.int32)
    if mode == "rolewalk":
        from rolewalk import RoleWalk
        X = rolewalk_embedding(P)
        rw = RoleWalk()
        labels = rw.fit_predict(X, n_roles=n_roles) if n_roles else rw.fit_predict(X)
        out = np.full(n, base, dtype=np.int32)
        role_eff = np.array([np.median(eff[labels == r]) for r in range(labels.max() + 1)])
        ranks = np.argsort(np.argsort(-role_eff)) / max(1, len(role_eff) - 1)
        role_len = np.rint(low + (high - low) * ranks).astype(np.int32)
        out[:] = role_len[labels]
        return out
    raise ValueError("mode must be 'spread' or 'rolewalk'")
