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


def _leiden(g, resolution, seed=0, n_iterations=2, objective="modularity"):
    import random
    random.seed(int(seed))   # igraph draws from Python's random module
    part = g.community_leiden(objective_function=objective, weights="weight", resolution=float(resolution),
                              n_iterations=int(n_iterations))
    return np.asarray(part.membership, dtype=np.int32)


ORDERS = {"random": 0, "strength": 1, "strength_asc": 2, "index": 3}


def communities(P, resolution, seed=0, n_iterations=2, backend="louvain", order="strength", consensus=1):
    """Communities of the symmetric weighted graph P for the generalised modularity at `resolution`: MAGNET's own
    Louvain (C++, ~0.2 s on 70k nodes) or python-igraph's Leiden (backend="leiden", ~2 s with the conversion).
    `order`: node visiting order of the Louvain local moving: "strength" (default; strength descending, ties by
    index: deterministic, and on MNIST it finds the ten classes on every graph where the random order splits one
    in some runs, k42), "random" (seeded), "strength_asc", "index". `consensus` > 1: run that many seeds and return
    the medoid partition (highest mean ARI to the others); as robust as "strength" on MNIST at 6x the cost."""
    if backend == "leiden":
        return _leiden(_igraph(P), resolution, seed, n_iterations)
    from _magnet import louvain
    P = sp.csr_matrix(P, dtype=np.float32)
    P.sort_indices()
    ip, ix, w = P.indptr.astype(np.int32), P.indices.astype(np.int32), P.data.astype(np.float32)
    if consensus <= 1:
        return np.asarray(louvain(ip, ix, w, float(resolution), int(seed), 20, 50, ORDERS[order]))
    from sklearn.metrics import adjusted_rand_score
    runs = [np.asarray(louvain(ip, ix, w, float(resolution), int(seed) + r, 20, 50, ORDERS[order])) for r in range(consensus)]
    A = np.zeros((consensus, consensus))
    for i in range(consensus):
        for j in range(i + 1, consensus):
            A[i, j] = A[j, i] = adjusted_rand_score(runs[i], runs[j])
    return runs[int(np.argmax(A.sum(1)))]


def interface_merge(P, labels, tau=0.10, min_boundary=20):
    """Merge adjacent communities whose interface shows no density contrast (k43). The boundary nodes of A towards B
    (nodes of A with an edge into B) send on average a share f_AB of their strength into B. A cut through a cloud
    of uniform density is symmetric and both shares are large (about 0.2 for this graph construction, whatever the
    size); a true cluster boundary is a density dip and at least one side reaches little across (<= 0.09 on MNIST).
    Modularity cannot see the difference: on a kNN graph every node has about the same strength, so its null model
    is a size^2 penalty that splits large geometric communities (the field-of-view limit) and swallows small ones
    (the resolution limit). Louvain at a high resolution avoids the second, this merge undoes the first. Pairs are
    merged greedily, largest min(f_AB, f_BA) first, while it is >= tau (0.10 = half the uniform reference, the
    midpoint between no contrast and a full gap); interfaces with fewer than `min_boundary` boundary nodes on
    either side are ignored. Sparse throughout: O(nnz(P)) per merge."""
    labels = np.asarray(labels, dtype=np.int32).copy(); n = P.shape[0]
    P = sp.csr_matrix(P, dtype=np.float64)
    strength = np.asarray(P.sum(1)).ravel(); strength[strength == 0] = 1.0
    Frac = sp.diags(1.0 / strength) @ P                       # Frac[i, j] = share of i's strength on the edge to j
    while True:
        k = int(labels.max()) + 1
        C = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, k))
        F = sp.csr_matrix(Frac @ C)                            # F[i, c] = share of i's strength going into community c
        S = np.asarray((C.T @ F).todense())                    # S[a, b] = sum over i in a of F[i, b]
        Fb = F.copy(); Fb.data[:] = 1.0
        N = np.asarray((C.T @ Fb).todense())                   # N[a, b] = number of boundary nodes of a towards b
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(N > 0, S / np.maximum(N, 1), 0.0)     # f[a, b] = mean crossing share of a's boundary nodes towards b
        np.fill_diagonal(f, 0.0)
        m = np.minimum(f, f.T); m[(N < min_boundary) | (N.T < min_boundary)] = 0.0
        a, b = np.unravel_index(int(np.argmax(m)), m.shape)
        if m[a, b] < tau:
            return labels
        labels[labels == max(a, b)] = min(a, b)
        labels = np.unique(labels, return_inverse=True)[1].astype(np.int32)


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


def hierarchical_contraction(Y, P, X=None, metric="euclidean", resolution=0.5, separation=0.7, placement=True,
                             n_sample=500, n_mixing=5000, tol=0.002, sweeps=2, n_threads=8, random_state=0, exclude=None,
                             backend="louvain", order="strength", consensus=1, merge_tau=0.10, verbose=False):
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
    labels = communities(P, resolution, seed=random_state, backend=backend, order=order, consensus=consensus)
    if merge_tau is not None:
        labels = interface_merge(P, labels, tau=float(merge_tau))
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
