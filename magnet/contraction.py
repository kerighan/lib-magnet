"""Community placement and contraction of a layout (benchmarks/theory/k30-k37).

The communities of the fuzzy graph P (Leiden) are treated as rigid bodies. (1) Placement: each community is
translated, by coordinate ascent of the Spearman correlation between the input distances and the layout distances
of a random sample (label-free), under an overlap guard (the fraction of sample points whose nearest neighbour lies
in another community may not rise). t-SNE-like layouts arrange the clusters far from rank-optimally: on MNIST 70k
the full-batch layout goes from distance Spearman 0.342 to 0.46 (hybrid 0.46, PCA-2 0.49) at unchanged 15-NN.
(2) Separation: every community is contracted towards its centroid, Z = mu + g (Y - mu): a similarity within a
community keeps the neighbourhoods (15-NN unchanged) and k-means recovers the communities (ARI 0.80 -> 0.90).
"""

import time
import numpy as np
import scipy.sparse as sp

def _igraph(P):
    try:
        import igraph as ig
    except ImportError as e:   # pragma: no cover
        raise ImportError("the contraction step needs python-igraph (pip install igraph)") from e
    C = sp.triu(sp.csr_matrix(P), 0).tocoo()
    g = ig.Graph(n=P.shape[0], edges=np.c_[C.row, C.col].tolist())
    g.es["weight"] = C.data.astype(float).tolist()
    return g


def _leiden(g, resolution, seed=0, n_iterations=2):
    import random
    random.seed(int(seed))   # igraph draws from Python's random module
    part = g.community_leiden(objective_function="modularity", weights="weight", resolution=float(resolution),
                              n_iterations=int(n_iterations))
    return np.asarray(part.membership, dtype=np.int32)


def communities(P, resolution, seed=0, n_iterations=2, backend="louvain"):
    """Communities of the symmetric weighted graph P for the generalised modularity at `resolution`: MAGNET's own
    Louvain (C++, ~0.2 s on 70k nodes) or python-igraph's Leiden (backend="leiden", ~2 s with the conversion)."""
    if backend == "leiden":
        return _leiden(_igraph(P), resolution, seed, n_iterations)
    from _magnet import louvain
    P = sp.csr_matrix(P, dtype=np.float32)
    P.sort_indices()
    return np.asarray(louvain(P.indptr.astype(np.int32), P.indices.astype(np.int32), P.data.astype(np.float32), float(resolution), int(seed)))


def contract(Y, labels, gain):
    """Rigid contraction of every community towards its centroid: Z = mu_c + gain (Y - mu_c)."""
    if gain == 1.0:
        return Y
    Z = Y.copy()
    k = int(labels.max()) + 1
    sums = np.zeros((k, Y.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, Y)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    mu = (sums / np.maximum(counts, 1)[:, None]).astype(Y.dtype)
    Z -= (1.0 - gain) * (Y - mu[labels])
    return Z


def _sample_distances(X, idx, metric):
    from sklearn.metrics import pairwise_distances
    Xs = X[idx]
    m = "cosine" if (sp.issparse(X) or metric == "cosine") else ("euclidean" if metric in ("euclidean", "l2") else metric)
    return pairwise_distances(Xs, metric=m, n_jobs=1)


def _ranks(x):
    """ranks 1..m without tie averaging (ties are measure-zero for layout distances)."""
    r = np.empty(len(x), dtype=np.float32)
    r[np.argsort(x, kind="quicksort")] = np.arange(1, len(x) + 1, dtype=np.float32)
    return r


class _SampleSpearman:
    """Spearman correlation between the input distances and the layout distances of a fixed random sample.
    Works on the sample coordinates only: shift(mask, d) scores the sample with the points in `mask` moved by d."""
    def __init__(self, X, idx, metric):
        self.idx = idx
        self.iu = np.triu_indices(len(idx), 1)
        rh = _ranks(_sample_distances(X, idx, metric)[self.iu])
        self.rh = (rh - rh.mean()) / rh.std()

    def score_coords(self, Zs):
        D = np.sqrt(((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1))[self.iu]
        rl = _ranks(D)
        rl = (rl - rl.mean()) / rl.std()
        return float(np.mean(self.rh * rl))

    def __call__(self, Y):
        return self.score_coords(Y[self.idx])


class _Mixing:
    """fraction of a sample whose nearest sample point belongs to another community (overlap guard)."""
    def __init__(self, labels, idx):
        self.idx = idx
        self.lab = labels[idx]

    def score_coords(self, Zm):
        from scipy.spatial import cKDTree
        nn = cKDTree(Zm).query(Zm, k=2)[1][:, 1]
        return float(np.mean(self.lab[nn] != self.lab))

    def __call__(self, Y):
        return self.score_coords(Y[self.idx])


def place_communities(Y, labels, spearman, mixing=None, tol=0.002, sweeps=2, min_frac=0.01, radial=(0.7, 0.85, 1.15, 1.3, 1.6),
                      step=0.15, n_threads=8, verbose=False):
    """Rigid translation of every community (largest first, those above min_frac of the points), coordinate ascent of
    the sample Spearman: candidates = radial scalings of the centroid offset and an 8-direction step grid
    (step x RMS radius), scored in parallel on the sample coordinates only; a move is kept only if the Spearman
    rises and the mixing does not exceed its start + tol."""
    from concurrent.futures import ThreadPoolExecutor
    Y = Y.copy()
    k = int(labels.max()) + 1
    centre = Y.mean(0)
    R = float(np.sqrt(((Y - centre) ** 2).sum(1).mean()))
    sizes = np.bincount(labels, minlength=k)
    Ys = Y[spearman.idx].copy()
    lab_s = labels[spearman.idx]
    cur = spearman.score_coords(Ys)
    if mixing is not None:
        Ym = Y[mixing.idx].copy()
        lab_m = labels[mixing.idx]
        mix0 = mixing.score_coords(Ym)
    log = []
    grid = [np.array([dx, dy], dtype=np.float64) * step * R for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    pool = ThreadPoolExecutor(max(1, int(n_threads)))

    def shifted(Zs, mask, d):
        Z = Zs.copy()
        Z[mask] += d.astype(np.float32)
        return Z

    for sw in range(sweeps):
        moved = 0
        for c in np.argsort(-sizes):
            if sizes[c] < min_frac * len(Y):
                break
            ms = lab_s == c
            if not ms.any():
                continue
            mu = Y[labels == c].mean(0)
            cands = [(mu - centre) * (s - 1.0) for s in radial] + grid
            vals = list(pool.map(lambda d: spearman.score_coords(shifted(Ys, ms, d)), cands))
            order = np.argsort(vals)[::-1]
            for j in order:
                if vals[j] <= cur + 1e-4:
                    break
                d = cands[j]
                if mixing is not None:
                    mm = lab_m == c
                    if mixing.score_coords(shifted(Ym, mm, d)) > mix0 + tol:
                        continue
                    Ym[mm] += d.astype(np.float32)
                Ys[ms] += d.astype(np.float32)
                Y[labels == c] += d.astype(np.float32)
                cur = float(vals[j])
                moved += 1
                break
        log.append({"sweep": sw, "moved": int(moved), "spearman": cur})
        if verbose:
            print(f"Contraction: placement sweep {sw}: {moved} communities moved, sample Spearman {cur:.3f}")
        if moved == 0:
            break
    pool.shutdown()
    return Y, log


def hierarchical_contraction(Y, P, X=None, metric="euclidean", resolution=0.3, separation=0.7, placement=True,
                             n_sample=500, n_mixing=5000, tol=0.002, sweeps=2, n_threads=8, random_state=0, exclude=None,
                             backend="louvain", verbose=False):
    """Returns (Z, log). Communities of P (Leiden at `resolution`); (1) placement: rigid translations of the communities
    chosen by coordinate ascent of the Spearman correlation between input and layout distances on a sample of
    n_sample points (label-free; X required, skipped without it), with an overlap guard (mixing of n_mixing sample
    points may not rise by more than tol); (2) separation: every community contracted rigidly towards its centroid
    by the gain `separation` (1 = off). exclude: indices never drawn into the search sample (held-out evaluation
    points)."""
    t0 = time.time()
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    n = len(Y)
    rng = np.random.default_rng(random_state)
    pool_idx = np.arange(n) if exclude is None else np.setdiff1d(np.arange(n), np.asarray(exclude))
    labels = communities(P, resolution, seed=random_state, backend=backend)
    k = int(labels.max()) + 1
    log = {"resolution": resolution, "communities": k, "labels": labels}
    Z = Y
    if placement and X is not None and k >= 2:
        spearman = _SampleSpearman(X, rng.choice(pool_idx, size=min(n_sample, len(pool_idx)), replace=False), metric)
        mixing = _Mixing(labels, rng.choice(pool_idx, size=min(n_mixing, len(pool_idx)), replace=False))
        Z, log["placement"] = place_communities(Z, labels, spearman, mixing, tol=tol, sweeps=sweeps, n_threads=n_threads, verbose=verbose)
    if separation is not None and separation != 1.0:
        Z = contract(Z, labels, float(separation))
        log["separation"] = float(separation)
        if verbose:
            print(f"Contraction: {k} communities contracted by {separation}")
    log["time"] = time.time() - t0
    return np.ascontiguousarray(Z, dtype=np.float32), log
