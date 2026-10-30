"""Fast spectral initialisation: block power iteration on the normalised adjacency."""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def spectral_init(P, n_components=2, random_state=None, n_iter=25, tol=0.05, n_threads=-1,
                  data=None, method="chebyshev", verbose=False):
    """Spectral layout of the graph P. When the graph has several connected components they are
    laid out separately and placed according to the centroids of `data` (if given)."""
    from _magnet import subspace_iteration, chebyshev_eigenvectors

    t = time.time()
    P = sp.csr_matrix(P, dtype=np.float32)
    n = P.shape[0]
    deg = np.asarray(P.sum(1)).ravel().astype(np.float64)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0).astype(np.float32)
    n_comp, comp = connected_components(P, directed=False)
    rng = np.random.default_rng(random_state)
    indptr, indices = P.indptr.astype(np.int32), P.indices.astype(np.int32)
    data_ = P.data.astype(np.float32)
    if method == "chebyshev":
        extra = max(2, n_components)
        X = rng.standard_normal((n, n_components + extra)).astype(np.float32)
        # cap the work on very large graphs: about 3e9 "nnz x columns x matvecs", at least 3 filter
        # passes (a few seconds at 1M nodes; the init only needs the leading eigenspace, not exact vectors)
        degree = 30
        budget = int(3e9 / max(1, P.nnz * (n_components + extra) * degree))
        max_iter = int(max(3, min(n_iter, budget)))
        X = chebyshev_eigenvectors(indptr, indices, data_, X, inv_sqrt, comp.astype(np.int32),
                                   int(n_components), degree, max_iter, float(tol), int(n_threads))
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32)[:, :n_components])
    else:
        X = rng.standard_normal((n, n_components)).astype(np.float32)
        X = subspace_iteration(indptr, indices, data_, X, inv_sqrt, comp.astype(np.int32),
                               300, 1e-6, int(n_threads))
        X = np.asarray(X, dtype=np.float32)
    # eigenvectors of the normalised Laplacian are D^{1/2}-scaled: undo it for a smoother layout
    X *= inv_sqrt[:, None]
    if n_comp > 1:
        X = _place_components(X, comp, n_comp, rng, data)
    if verbose:
        print(f"Spectral init ({method}, {n_comp} component{'s' if n_comp > 1 else ''})"
              f" - T={time.time() - t:.2f}s")
    return X


def _place_components(X, comp, n_comp, rng, data=None):
    """Centre and rescale each component (radius ~ sqrt(size)), then place the components either on
    the PCA projection of their centroids in data space, or on a spiral when no data is given."""
    n, k = X.shape
    sizes = np.bincount(comp, minlength=n_comp)
    order = np.argsort(-sizes)
    radii = np.sqrt(sizes / sizes.max()).astype(np.float32)
    for c in range(n_comp):
        mask = comp == c
        Xc = X[mask] - X[mask].mean(0)
        rms = float(np.sqrt((Xc ** 2).sum(1).mean()))
        if rms < 1e-6 or sizes[c] < 3:
            Xc = rng.standard_normal(Xc.shape).astype(np.float32) * radii[c] * 0.5
        else:
            Xc *= radii[c] / rms
        X[mask] = Xc

    centers = np.zeros((n_comp, k), dtype=np.float32)
    if data is not None and n_comp > 2:
        d = np.asarray(data, dtype=np.float32)
        centroids = np.zeros((n_comp, d.shape[1]), dtype=np.float64)
        np.add.at(centroids, comp, d)
        centroids /= sizes[:, None]
        centroids -= np.average(centroids, axis=0, weights=sizes)
        if d.shape[1] > k:
            U, S, Vt = np.linalg.svd(centroids, full_matrices=False)
            centers = (U[:, :k] * S[:k]).astype(np.float32)
        else:
            centers[:, : d.shape[1]] = centroids
        # scale so that the large components do not overlap (nearest-neighbour criterion)
        big = order[: min(len(order), 200)]
        big = big[sizes[big] >= max(3, 0.005 * n)]
        if len(big) >= 2:
            D = np.sqrt(((centers[big, None, :] - centers[None, big, :]) ** 2).sum(-1))
            np.fill_diagonal(D, np.inf)
            j = D.argmin(1)
            need = 2.5 * (radii[big] + radii[big[j]]) / np.maximum(D[np.arange(len(big)), j], 1e-12)
            centers *= float(np.percentile(need, 75))
        else:
            centers *= 1.0 / max(np.abs(centers).max(), 1e-12) * 4.0
    else:
        golden = np.pi * (3 - np.sqrt(5))
        main = order[0]
        for rank, c in enumerate(order[1:], start=1):
            angle = rank * golden
            radius = 1.5 * radii[main] + 2.0 * radii[c] + 0.05 * np.sqrt(rank)
            centers[c, 0] = radius * np.cos(angle)
            if k > 1:
                centers[c, 1] = radius * np.sin(angle)
    X += centers[comp]
    return X
