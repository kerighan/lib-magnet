"""MAGNET - fast two-pass neighbour embeddings and graph layout.

One fuzzy kNN graph (NN-descent), then a sampled SGD pass (UMAP-style negative sampling with a decoupled,
heavy-tailed attraction and a transient mid-range attraction), a full-batch pass (t-SNE objective with
FFT-interpolated repulsion), or the two in sequence; optionally a label-free community post-process.
Accepts a data matrix (n, d), a sparse CSR matrix, a networkx graph or a scipy sparse adjacency matrix.
The estimator follows the scikit-learn API (fit / transform / fit_transform, get_params / set_params).
"""
import time
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin

from .knn import knn_hnsw, knn_nndescent, knn_search, fuzzy_graph, prepare_data, smooth_knn, choose_landmarks
from .spectral import spectral_init
from .contraction import hierarchical_contraction

__version__ = "3.0.0"
__all__ = ["MAGNET", "hierarchical_contraction", "find_ab_params", "__version__"]

_LOSSES = ("umap", "tsne", "hybrid", "infonce", "umap_exact")
_KERNEL_B_MODES = ("size", "consistent", "auto", "auto_dim", "local", "adaptive")


def find_ab_params(spread=1.0, min_dist=0.1, b=None, kernel="power"):
    """Fit the low-dim kernel to the (min_dist, spread) curve: 1 / (1 + a d^(2b)) ("power", a and b or
    only a when b is given), exp(-a d^2) ("gaussian") or 1 / cosh(a d) ("sech")."""
    from scipy.optimize import curve_fit

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    if kernel == "gaussian":
        params, _ = curve_fit(lambda x, a: np.exp(-a * x ** 2), xv, yv, p0=[1.0])
        return float(params[0]), 1.0
    if kernel == "sech":
        params, _ = curve_fit(lambda x, a: 1.0 / np.cosh(a * x), xv, yv, p0=[1.0])
        return float(params[0]), 1.0
    if kernel == "tanh":
        params, _ = curve_fit(lambda x, a: 1.0 - np.tanh(a * x ** 2), xv, yv, p0=[1.0])
        return float(params[0]), 1.0
    if b is None:
        params, _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), xv, yv)
        return float(params[0]), float(params[1])
    params, _ = curve_fit(lambda x, a: 1.0 / (1.0 + a * x ** (2 * b)), xv, yv, p0=[1.0])
    return float(params[0]), float(b)


def _sampling_frequencies(w, target):
    """freq_i = min(1, s * w_i) with s solved exactly so that sum(freq) == target (O(n log n)):
    with the t largest weights saturated, total(s) = t + s * (sum of the others) is piecewise linear."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    if n == 0 or n <= target:
        return np.ones_like(w)
    ws = np.sort(w)[::-1]
    tail = np.cumsum(ws[::-1])[::-1]                      # tail[t] = sum_{j >= t} ws[j]
    t = np.arange(1, n + 1)
    s_bp = 1.0 / np.maximum(ws[t - 1], 1e-300)            # s at which the t-th weight saturates
    tail_t = np.append(tail[1:], 0.0)
    total = t + s_bp * tail_t                             # non-decreasing in t
    k = int(np.searchsorted(total, target))
    s = (target - k) / tail[k] if k < n and tail[k] > 0 else s_bp[-1]
    return np.minimum(1.0, s * w)


class MAGNET(BaseEstimator, TransformerMixin):
    """Two-pass neighbour embedding (scikit-learn estimator).

    Main parameters (everything else is a research knob documented below):
    loss : "umap" (default, sampled pass), "tsne" (full-batch pass), "hybrid" (the two in sequence).
    n_components, n_neighbors, metric ("euclidean" or "cosine"), min_dist, spread : as in UMAP.
    n_epochs : epochs of the sampled pass, iterations of the full-batch pass.
    kernel_b : attraction exponent ("size" by default: 0.6 at n >= 20k, 0.8 at n <= 10k).
    contraction : None, or a float g in (0, 1] (0.7 recommended) for the community post-process.
    init, pca : initialisation and optional PCA front end.
    n_threads, random_state, verbose.
    Fitted attributes: embedding_, graph_, knn_indices_, knn_dists_, kernel_b_, and contraction_log_ when
    the post-process ran.

    Parameters
    ----------
    n_components : output dimension (2 or 3 for visualisation; any value <= 64 works).
    n_neighbors : size of the local neighbourhood (counts the point itself, like UMAP).
    metric : "euclidean" or "cosine" (hnswlib spaces).
    min_dist, spread : shape of the low-dimensional kernel (same meaning as in UMAP; 0.05 preserves
        neighbourhoods better than UMAP's 0.1 at every other metric equal).
    fft_interp, fft_box : interpolation points per box and box width for loss="tsne" (3 and 0.7;
        FIt-SNE uses 1.0, which costs 0.004 of 15-NN on MNIST (0.346 against 0.350) for 15 % less
        time: the interpolation error at unit width is 1.2 % RMS of the force, 0.4 % at 0.7, and 0.5
        buys nothing more; fft_box=1.5 halves the FFT cost again for about 3% less 15-NN).
    tsne_early : early-exaggeration iterations for loss="tsne" (default min(250, n_epochs // 3));
        with loss="tsne", n_epochs is the number of iterations (default 600 = 200 + 400; 450 with
        fft_box=1.5 is a fast preset, 750 the FIt-SNE schedule).
    kernel : low-dimensional similarity family of the SGD mode: "power" (1 / (1 + a d^2b), heavy tail),
        "gaussian" (exp(-a d^2)), "sech" (1 / cosh(a d)) or "tanh" (1 - tanh(a d^2)); the last three
        have exponential or lighter tails (kept for comparison).
    kernel_b : exponent b of the attraction kernel 1 / (1 + a d^(2b)); a is fitted to the (min_dist, spread) curve.
        The attraction exponent sets local fidelity: the force on a neighbour is ~ d^(2b-1), constant at b = 1/2,
        and 15-NN preservation peaks at 0.5 (images) to 0.7 (text) in every target dimension. Default "size":
        0.6 for n >= 20 000 points, 0.8 for n <= 10 000, log-linear in between (0.6 for graph inputs): the
        heavy tail that maximises 15-NN at large n tears small data sets into micro-clusters (MNIST
        subsamples: 56-78 HDBSCAN clusters at n <= 10k against 8 at 0.8; USPS 7k: 60 against 8), at a 15-NN
        cost of 0.02-0.03 there (k45). The exponent used is in `kernel_b_`.
        None = UMAP's fit (~0.8 to 0.9); an array = one exponent per point (the repulsion stays at repulsion_b);
        "local" = 1 where the neighbourhood-smoothed local intrinsic dimension fits in n_components, 1/2 elsewhere
        (data mixing manifolds of different dimensions; dimprint when installed; docs/kernel_exponents.md);
        "adaptive" = the same split read off the layout itself (see `adaptive`; works on graph inputs too);
        "consistent" = the heaviest tail (0.6, the best local fidelity) whose layout does not tear the graph's
        communities into density clusters they do not have (HDBSCAN on a 10k sample against the Louvain +
        density-contrast communities of the post-process, ARI >= 0.7; raised by 0.1 while that improves by
        >= 0.1, k46): the principled version of "size" for clustering pipelines (USPS 7k -> 0.7, MNIST 5k ->
        0.8, MNIST 70k -> 0.6, Fashion 70k -> 0.7 at a 15-NN cost of 0.02, letter / text stay at 0.6 since no
        exponent makes their communities density clusters); costs one extra fit per step tried and ~1 s of
        diagnostics;
        "auto" = subsample sweep; "auto_dim" = the former coupled defaults (0.6 in 2-D, 0.8 beyond).
    adaptive : (base, high, at, threshold, reference, min_degree) for kernel_b="adaptive", default
        (0.5, 1.0, 0.75, 0.0, 2.0, 2): every point runs at `base` until `at` of the epochs; the points whose
        share of graph edges longer than `reference` times the median edge length is at most `threshold`,
        and whose degree is at least `min_degree`, finish at `high` (L2, the uncrowded regime), the others at
        `base` (L1, the crowded regime). Neutral on homogeneous data (MNIST: 0.293 against 0.292 15-NN kept),
        +0.02 to +0.03 on data mixing 1-D / 2-D and 10-D manifolds, +0.01 to +0.06 on citation and airport
        graphs, 0.57 against 0.45 edge preservation on blocks with chains attached. Results in `crowding_`.
    repulsion_b : exponent of the repulsion kernel (default 1, the Cauchy tail of t-SNE; None: same as kernel_b).
        It sets the tail felt by far points: heavier (0.5) gives UMAP's compact blobs, lighter (1 to m/2 in m
        dimensions) gives a little more neighbourhood preservation and global structure, and b = 1 removes the pow()
        from the repulsion loop (15-20% faster). Only the sampled (`loss="umap"`) optimiser uses it.
    n_epochs : int -> that many epochs with a linear learning-rate decay (default None: 500 for
        n <= 10k, 300 for n <= 100k, else 200); (constant, anneal) tuple -> constant learning rate for
        the first count then linear decay over the second; "auto" -> constant learning rate until the
        layout stops moving, then anneal.
    learning_rate : initial step (SGD) or Adam step size.
    loss : "umap" (negative-sampling cross entropy, as UMAP), "infonce" (contrastive softmax over the
        positive and its negatives), "umap_exact" (research mode: the UMAP cross entropy with its
        repulsion computed exactly over all pairs by FFT, scaled by repulsion_strength; 2-D only) or
        "hybrid" (the SGD layout, then `hybrid_iter` iterations of the exact repulsion refining it at the
        exact mode's scale, without early exaggeration or mid-range term: the local structure of "tsne" with the
        global structure of the SGD mode, 2-D only; MNIST 70k: 15-NN kept 0.355, distance Spearman 0.41, 12 s),
        "tsne" (exact repulsion over all pairs through FFT interpolation,
        full-batch gradient descent as FIt-SNE / openTSNE; 2-D only; `exaggeration` is then the
        attraction multiplier of the final phase: 1 = t-SNE, ~4 looks like UMAP).
    exaggeration : multiplier on the attractive gradient (UMAP ~ t-SNE with exaggeration 4).
    hybrid_iter : iterations of the exact pass of loss="hybrid" (default 300; 150 keeps most of the gain).
    tsne_early_exaggeration : attraction multiplier of the early phase of loss="tsne" (default 12, t-SNE's).
    hybrid_tail : (iterations, exaggeration) of an optional exaggerated tail at the end of the full-batch pass
        (loss="hybrid" and loss="tsne")
        (default (0, 1.0): none). (100, 2.0) was the "clusters" preset before `contraction` (now
        the preset: contraction=0.7 costs no 15-NN); on MNIST 70k the tail trades 0.05 of 15-NN
        and 0.02 of distance Spearman for cluster recovery (HDBSCAN ARI 0.10 -> 0.76, k-means ARI 0.62 ->
        0.71; benchmarks/theory/k28_exaggeration_tail.py).
    contraction : community placement and contraction of the final layout (magnet/contraction.py). None (default):
        off. A float g in (0, 1]: the communities of the fuzzy graph (Louvain visited by decreasing strength, so
        deterministic, at resolution 0.5, then the interfaces without density contrast merged: a cut through a
        uniform cloud sends >= 0.10 of its boundary nodes' strength across on both sides, a true cluster boundary
        does not; k42-k43) are first translated as
        rigid bodies by a label-free coordinate ascent of the distance Spearman between the input and the layout on
        a 500-point sample (overlap guard: communities may not mix), then each contracted rigidly towards its
        centroid by g ("separation" dial: 0.7 = compact clusters). A similarity within a community keeps the
        neighbourhoods: MNIST 70k full-batch 15-NN 0.348 -> 0.346, Spearman 0.342 -> 0.46, k-means ARI 0.80 ->
        0.90 at g=0.7, 3 s (benchmarks/theory/k30-k38). A dict passes the arguments of hierarchical_contraction
        (resolution, separation, placement, n_sample, tol, sweeps, order, consensus, merge_tau; backend="leiden" needs python-igraph).
    tsne_init_scale, tsne_learning_rate, tsne_far, tsne_mid : research knobs of the exact mode (warm start
        scale and learning rate, constant far-field repulsion, bounded mid-range springs; docs/kernel_exponents.md).
    tsne_dof : degrees of freedom of the Student-t kernel of loss="tsne" (1 = Cauchy, the t-SNE kernel;
        smaller values give heavier tails, as in Kobak et al. 2019).
    late_exaggeration, late_phase : attraction multiplier applied during the last late_phase fraction
        of the SGD epochs (FIt-SNE's late exaggeration): tightens the clusters once the layout is set.
        late_exaggeration=1.5 gives UMAP-like compact clusters (silhouette 0.54 against 0.52 for UMAP
        on MNIST) while staying ahead of UMAP on every fidelity metric; the cost is about 4% of 15-NN
        preservation and 0.03 of distance correlation against the default.
    optimizer : "sgd" or "adam" (per-node lazy Adam); clip : "coord" (each coordinate step capped at
        4 * lr, as UMAP) or "norm" (step norm capped).
    negative_sample_rate : negatives drawn per positive sample.
    repulsion_strength : multiplier on the repulsive gradient (default 2: +8% neighbourhood preservation
        over UMAP's 1 on MNIST, same label accuracy and cluster compactness).
    walk_len, mid_weight, mid_rate, mid_phase : mid-range attraction. With probability mid_rate a
        positive sample (u, v) is followed by a random walk of walk_len steps from v, ending at w,
        and u is pulled towards w with a spring of stiffness mid_weight (annealed to zero over the
        first mid_phase fraction of the epochs). Set mid_weight=0 to disable.  None (default) means 0.1,
        or 0.5 under loss="hybrid": there the SGD pass only has to place the clusters, and the stiffer
        spring buys +0.02 to +0.04 distance Spearman at unchanged 15-NN preservation. The value used
        is stored in mid_weight_.
    mid_source : where the mid-range partners come from: "data" (PaCMAP-style mid-near pairs: the
        second closest of 6 random rows in the data space, n_mid_pairs per point; +30% distance
        correlation on MNIST for -3% 15-NN preservation), "walk" (short random walks on the graph)
        "spectral" (for graphs: mid-near pairs measured in an 8-dimensional spectral embedding of the
        graph, a diffusion-distance proxy), "hybrid" (random walks of walk_len steps on the two-layer graph made of the kNN edges and the
        mid-near edges, each step jumping along a mid-near edge with probability mid_jump: one
        mechanism for every scale) or "auto" (default: data when a data matrix is given, walks for
        graphs). With loss="tsne" the
        pairs act as a spring of stiffness 3 * mid_weight / n on the normalised gradient (+0.03
        distance correlation on MNIST at unchanged local quality).
    mid_scale : "fixed" (walk_len steps for every node) or "spread" (per-node length from the 2-step
        diffusion spread of the graph).
    init : "auto" (default: PCA for loss="tsne" on a data matrix, spectral otherwise), "spectral",
        "pca" (data only), "random" or an (n, n_components) array. PCA gives the exact-repulsion mode
        a better global structure (distance correlation 0.34 against 0.30 on MNIST 70k) at equal local
        quality; for the SGD mode it trades a little cluster compactness for global structure.
    pca : if set and the data has more columns, project it to that many PCA components before the
        kNN search (much faster on wide data such as images; neighbours change very little).
    knn : "nndescent" (native NN-descent kernel, default) or "hnsw" (hnswlib).
    ef_construction, M, ef : hnswlib parameters (higher = more exact, slower).
    neighbor_power, neighbor_rank_exponent : reshape the neighbour-distance profile before the
        affinities (d_k <- d_1 (d_k/d_1)^power, then d_k <- d_k k^rank_exponent): an input-side way
        of compensating the dimension mismatch between the data manifold and the plane.
    samples_per_node : budget of attractive samples per node and per epoch. Edge i is sampled
        with frequency min(1, s * w_i) per epoch, s being chosen so that the total number of samples
        is samples_per_node * n. The cost of an epoch is therefore the same whatever the input graph.
    input : "auto" (a sparse square matrix is an adjacency, anything else is data), "data" or "graph".
    transform_epochs : epochs used by transform() (default 50).
    parallel : how the SGD threads share the layout. "hogwild" (default): every thread writes any node, lock-free,
        as UMAP does. "owner": the nodes are split into chunks that the threads claim one at a time; a chunk's
        nodes are written by its thread only, and every other node is read from a snapshot of the layout frozen
        for the epoch, so no cache line is written by two threads and the per-epoch barrier is the only
        synchronisation (hogwild spent half its wall-clock time in cache-line transfers and in waiting for the
        slowest thread). The attraction moves the head of the edge only, scaled by 1.5 (the mirrored edge moves
        the other end); the pull of the mid-range spring on the partner is delivered when the partner is the head,
        through the reverse pairs. MNIST 70k: the SGD runs 2x faster, the fit 1.4x; 15-NN kept 0.189 against
        0.193, trustworthiness 0.948 against 0.957, distance Spearman 0.451 against 0.415 (Fashion-MNIST 0.782
        against 0.731, 20 newsgroups 0.090 against 0.046). The layout is identical for any n_threads given the
        graph and the seed (the kNN search itself is multithreaded and is not). Not available with loss="infonce".
    graph_norm : for graph inputs, "sym" rescales w_ij by 1/sqrt(deg_i deg_j) so that every node
        receives a comparable number of updates (hubs do not monopolise the budget); "none" keeps
        the raw weights.
    n_threads : threads for the native kernels (-1 = all cores).
    random_state : seed. The SGD is lock-free and multithreaded, so runs are reproducible only
        approximately unless n_threads=1.
    """

    def __init__(self, n_components=2, n_neighbors=15, metric="euclidean", min_dist=0.05, spread=1.0,
                 n_epochs=None, learning_rate=1.0, negative_sample_rate=5, repulsion_strength=2.0,
                 kernel="power", kernel_b="size", repulsion_b=1.0, fft_interp=3, fft_box=0.7, tsne_early=None, loss="umap", exaggeration=1.0,
                 late_exaggeration=1.0, late_phase=0.2, tsne_dof=1.0, optimizer="sgd", clip="coord", beta1=0.9, beta2=0.999,
                 walk_len=4, mid_weight=None, mid_rate=0.25, mid_phase=0.5, mid_scale="fixed", mid_source="auto",
                 n_mid_pairs=5, mid_jump=0.5,
                 init="auto", pca=None, knn="nndescent", ef_construction=80, M=16, ef=None,
                 set_op_mix_ratio=1.0, neighbor_power=1.0, neighbor_rank_exponent=0.0,
                 samples_per_node=5.0, graph_norm="sym", input="auto",
                 adaptive=(0.5, 1.0, 0.75, 0.0, 2.0, 2), tsne_init_scale=None, tsne_learning_rate=None, tsne_far=(0.0, 0.1), tsne_mid=("mean", 0.3), hybrid_iter=300, hybrid_tail=(0, 1.0), tsne_early_exaggeration=12.0,
                 contraction=None, transform_epochs=None, parallel="hogwild", n_threads=-1, random_state=None, verbose=False):
        self.n_components = n_components
        self.parallel = parallel
        self.adaptive = adaptive
        self.tsne_init_scale = tsne_init_scale
        self.tsne_learning_rate = tsne_learning_rate
        self.tsne_far = tsne_far
        self.tsne_mid = tsne_mid
        self.hybrid_iter = hybrid_iter
        self.hybrid_tail = hybrid_tail
        self.tsne_early_exaggeration = tsne_early_exaggeration
        self.contraction = contraction
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength
        self.kernel = kernel
        self.kernel_b = kernel_b
        self.repulsion_b = repulsion_b
        self.fft_interp = fft_interp
        self.fft_box = fft_box
        self.tsne_early = tsne_early
        self.loss = loss
        self.exaggeration = exaggeration
        self.late_exaggeration = late_exaggeration
        self.late_phase = late_phase
        self.tsne_dof = tsne_dof
        self.optimizer = optimizer
        self.clip = clip
        self.beta1 = beta1
        self.beta2 = beta2
        self.walk_len = walk_len
        self.mid_weight = mid_weight
        self.mid_rate = mid_rate
        self.mid_phase = mid_phase
        self.mid_scale = mid_scale
        self.mid_source = mid_source
        self.n_mid_pairs = n_mid_pairs
        self.mid_jump = mid_jump
        self.init = init
        self.pca = pca
        self.knn = knn
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        self.set_op_mix_ratio = set_op_mix_ratio
        self.neighbor_power = neighbor_power
        self.neighbor_rank_exponent = neighbor_rank_exponent
        self.samples_per_node = samples_per_node
        self.graph_norm = graph_norm
        self.input = input
        self.transform_epochs = transform_epochs
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ graph construction
    def build_graph(self, X):
        """Return the symmetric weighted graph (CSR) the layout is computed from.
        X : array (n, d), networkx graph (rows follow list(G.nodes)) or scipy sparse adjacency."""
        is_data = False
        self._is_data_ = False
        is_graph_input = (self.input == "graph") or (
            self.input == "auto" and sp.issparse(X) and X.shape[0] == X.shape[1])
        if sp.issparse(X) and is_graph_input:
            A = sp.csr_matrix(X, dtype=np.float32)
            if A.shape[0] != A.shape[1]:
                raise ValueError("a sparse adjacency matrix must be square")
            P = A.maximum(A.T).tocsr()
        elif hasattr(X, "nodes") and hasattr(X, "edges"):  # networkx graph
            import networkx as nx
            A = nx.to_scipy_sparse_array(X, nodelist=list(X.nodes), weight="weight", dtype=np.float32,
                                         format="csr")
            P = sp.csr_matrix(A).maximum(sp.csr_matrix(A).T).tocsr()
        else:
            is_data = True
            self._is_data_ = True
            if not sp.issparse(X):
                X = np.asarray(X)
                if X.ndim != 2:
                    raise ValueError("data must be a 2-d array")
            Z = X
            self.pca_ = None
            if self.pca is not None and X.shape[1] > self.pca:
                t = time.time()
                if sp.issparse(X):
                    from sklearn.decomposition import TruncatedSVD as _PCA
                else:
                    from sklearn.decomposition import PCA as _PCA
                self.pca_ = _PCA(self.pca, random_state=self.random_state)
                Z = self.pca_.fit_transform(X)
                if self.verbose:
                    print(f"PCA to {self.pca} dims - T={time.time() - t:.2f}s")
            self._prepared_ = prepare_data(Z, self.metric)
            if self.knn == "hnsw" and not sp.issparse(Z):
                ind, dist = knn_hnsw(Z, self.n_neighbors, self.metric, self.ef_construction, self.M, self.ef,
                                     self.n_threads, self.random_state or 0, self.verbose)
            elif self.knn in ("nndescent", "hnsw"):
                ind, dist = knn_nndescent(self._prepared_, self.n_neighbors, self.metric, n_threads=self.n_threads,
                                          random_state=self.random_state or 0, verbose=self.verbose, prepared=True)
            else:
                raise ValueError("knn must be 'nndescent' or 'hnsw'")
            self.knn_indices_, self.knn_dists_ = ind, dist
            P = fuzzy_graph(ind, dist, self.n_neighbors, set_op_mix_ratio=self.set_op_mix_ratio,
                            power=self.neighbor_power, rank_exponent=self.neighbor_rank_exponent)
        P.setdiag(0)
        P.eliminate_zeros()
        if not is_data and self.graph_norm == "sym":
            deg = np.asarray(P.sum(1)).ravel()
            inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0).astype(np.float32)
            P = (sp.diags(inv) @ P @ sp.diags(inv)).tocsr()
        P.sort_indices()
        return P

    # ------------------------------------------------------------------ initialisation
    def _initial_embedding(self, P, X, rng):
        n = P.shape[0]
        init = self.init
        if isinstance(init, str) and init == "auto":
            init = "pca" if (self.loss == "tsne" and isinstance(X, np.ndarray)) else "spectral"
        if isinstance(init, np.ndarray):
            Y = np.array(init, dtype=np.float32)
            if Y.shape != (n, self.n_components):
                raise ValueError("init array has the wrong shape")
        elif init == "random":
            Y = rng.uniform(-10, 10, (n, self.n_components)).astype(np.float32)
            return Y
        elif init == "pca":
            if X is None or sp.issparse(X) or not isinstance(X, np.ndarray):
                raise ValueError("init='pca' requires a data matrix")
            from sklearn.decomposition import PCA
            Y = PCA(self.n_components, random_state=self.random_state).fit_transform(X).astype(np.float32)
        elif init == "spectral":
            data_for_components = X if isinstance(X, np.ndarray) else None
            Y = spectral_init(P, self.n_components, self.random_state, n_threads=self.n_threads,
                              data=data_for_components, verbose=self.verbose)
        else:
            raise ValueError(f"unknown init {init!r}")
        Y -= Y.mean(0)
        scale = np.abs(Y).max()
        if scale > 0:
            Y *= 10.0 / scale
        Y += rng.normal(0, 1e-4, Y.shape).astype(np.float32)
        return np.ascontiguousarray(Y, dtype=np.float32)

    # ------------------------------------------------------------------ fit
    def _layout_consistent(self, Y, P, threshold=0.7, n_sample=10000):
        """Graph-layout consistency (k46): the density clusters of the layout (HDBSCAN on a sample, min cluster size
        max(15, m / 200)) must agree with the graph's own communities (Louvain + density-contrast merge, the ones the
        post-process uses) at an adjusted Rand index >= threshold. A layout torn into micro-clusters by a too stiff
        attraction fails (ARI 0.1-0.6), a healthy one passes (0.7-0.9). Returns (consistent, ari, n_clusters)."""
        from .contraction import communities, interface_merge
        from sklearn.metrics import adjusted_rand_score
        n = len(Y)
        if not hasattr(self, "_communities_"):
            self._communities_ = interface_merge(P, communities(P, 0.5))
        lab = self._communities_
        idx = np.sort(np.random.default_rng(self.random_state or 0).choice(n, min(n, n_sample), replace=False)) if n > n_sample else np.arange(n)
        m = len(idx)
        try:
            from sklearn.cluster import HDBSCAN
            cl = HDBSCAN(min_cluster_size=max(15, m // 200)).fit_predict(Y[idx])
        except ImportError:
            import hdbscan
            cl = hdbscan.HDBSCAN(min_cluster_size=max(15, m // 200)).fit_predict(Y[idx])
        noise = cl < 0; la = cl.copy(); la[noise] = cl.max() + 1
        ari = float(adjusted_rand_score(lab[idx], la))
        return ari >= threshold, ari, int(len(set(cl[~noise])))

    def _validate(self):
        """Reject invalid parameter values with an explicit message (called at the start of fit)."""
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {self.loss!r}")
        kb = self.kernel_b
        if isinstance(kb, str):
            if kb not in _KERNEL_B_MODES:
                raise ValueError(f"kernel_b must be a number in (0, 8], an array of per-point exponents, None, or one of {_KERNEL_B_MODES}; got {kb!r}")
        elif kb is not None and np.isscalar(kb) and not (0.0 < float(kb) <= 8.0):
            raise ValueError(f"kernel_b must be in (0, 8], got {kb!r}")
        c = self.contraction
        if c is not None and not isinstance(c, dict):
            if not np.isscalar(c) or not (0.0 < float(c) <= 1.0):
                raise ValueError(f"contraction must be None, a float in (0, 1] (the separation g; 0.7 recommended) or a dict of hierarchical_contraction arguments; got {c!r}")
        if self.mid_scale not in ("fixed", "spread"):
            raise ValueError(f"mid_scale must be 'fixed' or 'spread', got {self.mid_scale!r}")
        if not (isinstance(self.n_components, (int, np.integer)) and self.n_components >= 1):
            raise ValueError(f"n_components must be a positive integer, got {self.n_components!r}")
        if self.loss in ("tsne", "hybrid", "umap_exact") and self.n_components != 2:
            raise ValueError(f"loss={self.loss!r} (FFT repulsion) is implemented for n_components=2 only")

    def fit(self, X, y=None):
        """Fit the layout of X (array (n, d), sparse CSR, networkx graph or sparse adjacency); y is ignored."""
        from _magnet import optimize_layout

        self._validate()
        if isinstance(self.kernel_b, str) and self.kernel_b == "consistent" and self.loss in ("umap", "hybrid"):
            # the attraction exponent chosen by graph-layout consistency (k45-k46): the heaviest tail (0.6, the best
            # local fidelity) whose SGD layout does not tear the graph's communities into density clusters they do
            # not have; small data sets lack the graph redundancy that holds the clumps of a stiff short-range
            # attraction together and get 0.7-0.8. The graph is built once; the contraction is applied at the end.
            P = self.build_graph(X)
            self._reuse_graph_ = P
            contraction = self.contraction
            self.contraction = None
            self.kernel_b_search_ = []
            best = None
            try:
                for b in (0.6, 0.7, 0.8, 0.9):
                    self.kernel_b = b
                    self.fit(X)
                    ok, ari, k = self._layout_consistent(self.embedding_, P)
                    self.kernel_b_search_.append((b, ari, k))
                    if self.verbose:
                        print(f"kernel_b='consistent': b={b} graph-layout ARI {ari:.3f} ({k} density clusters) -> {'accepted' if ok else 'torn'}")
                    if ok:
                        best = (b, self.embedding_)
                        break
                    if best is not None and ari < best[2] + 0.1:
                        break   # raising the exponent no longer buys consistency (communities that are not density clusters, e.g. letter, text): keep the previous one
                    best = (b, self.embedding_, ari)
                    if b == 0.9:
                        break
            finally:
                self.kernel_b = "consistent"
                self.contraction = contraction
                del self._reuse_graph_
            b, self.embedding_ = best[0], best[1]
            self.kernel_b_ = b
            if contraction is not None:
                from .contraction import hierarchical_contraction
                opts = dict(contraction) if isinstance(contraction, dict) else {"separation": float(contraction)}
                self.embedding_, self.contraction_log_ = hierarchical_contraction(self.embedding_, P, X=X if getattr(self, "_is_data_", False) else None,
                                                                                  metric=self.metric, random_state=self.random_state or 0, verbose=self.verbose,
                                                                                  n_threads=(self.n_threads if self.n_threads and self.n_threads > 0 else 8), **opts)
            return self

        t0 = time.time()
        rng = np.random.default_rng(self.random_state)
        seed = int(rng.integers(0, 2 ** 62))
        P = self._reuse_graph_ if getattr(self, "_reuse_graph_", None) is not None else self.build_graph(X)
        n = P.shape[0]
        self.graph_ = P

        kb = self.kernel_b
        self.node_b_ = None
        if isinstance(kb, str) and kb == "size":
            # the heavy-tailed attraction (b = 0.6) maximises 15-NN at large n but tears small data sets into
            # micro-clusters (MNIST subsamples: 56-78 HDBSCAN clusters at n <= 10k, 8 at n >= 20k; USPS 7k: 60;
            # k45); b = 0.8 holds them together there at a 15-NN cost of 0.02-0.03. Graph inputs keep 0.6.
            kb = 0.6 if not getattr(self, "_is_data_", False) else float(np.clip(0.6 + 0.2 * (np.log(20000.0) - np.log(max(n, 1))) / np.log(2.0), 0.6, 0.8))
        self.kernel_b_ = kb if np.isscalar(kb) else None
        if isinstance(kb, str) and kb == "auto_dim":
            kb = 0.6 if self.n_components <= 2 else 0.8   # less crowding in 3-D: lighter tail
        if isinstance(kb, str) and kb == "auto":
            if not hasattr(self, "_prepared_"):
                raise ValueError("kernel_b='auto' needs a data matrix")
            from .roles import select_kernel_exponent
            t_auto = time.time()
            kb, self.kernel_b_scores_ = select_kernel_exponent(
                X if isinstance(X, np.ndarray) else self._prepared_, metric=self.metric, random_state=self.random_state or 0,
                n_threads=self.n_threads, mid_source="walk" if sp.issparse(X) else "auto")
            if self.verbose:
                print(f"kernel_b='auto' -> {kb} {self.kernel_b_scores_} - T={time.time() - t_auto:.2f}s")
        if isinstance(kb, str) and kb == "local":
            if not isinstance(X, np.ndarray):
                raise ValueError("kernel_b='local' needs a dense data matrix (local intrinsic dimension)")
            from .roles import local_kernel_exponent
            self.node_b_ = local_kernel_exponent(X, self.n_neighbors, random_state=self.random_state or 0, m=self.n_components, graph=P)
            kb = float(np.median(self.node_b_))
        if isinstance(kb, str) and kb == "adaptive":
            # per-point exponent read off the layout itself: every point starts at the crowded value and the
            # points whose edges all fit inside the kernel scale after `adaptive[2]` of the epochs switch to L2
            self.node_b_ = np.full(n, float(self.adaptive[0]), dtype=np.float32)
            kb = float(self.adaptive[0])
        elif kb is not None and not np.isscalar(kb):
            self.node_b_ = np.asarray(kb, dtype=np.float32)
            if self.node_b_.shape != (n,):
                raise ValueError("kernel_b array must have one entry per point")
            kb = float(np.median(self.node_b_))
        a, b = find_ab_params(self.spread, self.min_dist, kb, self.kernel)
        self.a_, self.b_ = a, b
        if self.repulsion_b is not None and not (0.0 < float(self.repulsion_b) <= 8.0):
            raise ValueError("repulsion_b must be in (0, 8] or None")
        self.rep_a_, self.rep_b_ = find_ab_params(self.spread, self.min_dist, float(self.repulsion_b), self.kernel) if self.repulsion_b is not None else (a, b)

        data_matrix = X if isinstance(X, np.ndarray) and X.ndim == 2 else None
        Y = self._initial_embedding(P, data_matrix, rng)

        t = time.time()
        # mid-range spring: 0.1 by default; 0.5 under loss="hybrid", whose SGD pass only has to place the
        # clusters (the exact pass rewrites the local structure): +0.04 distance Spearman on MNIST, +0.03 on
        # Fashion-MNIST, +0.02 on news20 and blobs at unchanged 15-NN preservation (benchmarks/theory/k8_mid_sweep.py)
        mid_w = float(self.mid_weight) if self.mid_weight is not None else (0.5 if self.loss == "hybrid" else 0.1)
        self.mid_weight_ = mid_w
        if self.loss == "umap_exact":
            if self.n_components != 2:
                raise ValueError("loss='umap_exact' (FFT repulsion) is implemented for n_components=2 only")
            from .tsne import optimize_umap_exact
            n_iter = self.n_epochs if isinstance(self.n_epochs, (int, np.integer)) else 300
            pairs = getattr(self, "_mid_pairs_", None)
            if self.mid_source in ("data", "auto") and hasattr(self, "_prepared_") and mid_w > 0 and pairs is None:
                from .roles import mid_near_pairs
                pairs = mid_near_pairs(self._prepared_, self.n_mid_pairs, random_state=self.random_state or 0, n_threads=self.n_threads)
            Y = optimize_umap_exact(P, Y, self.a_, self.b_, n_iter=int(n_iter), rep_weight=float(self.repulsion_strength),
                                    mid_pairs=pairs, mid_weight=float(mid_w), mid_phase=float(self.mid_phase),
                                    n_threads=self.n_threads, verbose=self.verbose)
            self.n_epochs_ = int(n_iter)
            head = np.zeros(0, dtype=np.int32)
        elif self.loss == "tsne":
            if self.n_components != 2:
                raise ValueError("loss='tsne' (FFT repulsion) is implemented for n_components=2 only")
            from .tsne import optimize_tsne
            n_iter = self.n_epochs if isinstance(self.n_epochs, (int, np.integer)) else 600
            if self.tsne_init_scale is None:
                Y = Y / max(float(Y[:, 0].std()), 1e-12) * 1e-4   # t-SNE convention: tiny initial spread
            else:   # warm start: keep the given layout, rescaled to the requested standard deviation
                Y = Y / max(float(Y[:, 0].std()), 1e-12) * float(self.tsne_init_scale)
            early = self.tsne_early if self.tsne_early is not None else min(250, int(n_iter) // 3)
            pairs = None
            if self.mid_source in ("data", "auto") and hasattr(self, "_prepared_") and mid_w > 0:
                from .roles import mid_near_pairs
                pairs = mid_near_pairs(self._prepared_, self.n_mid_pairs, random_state=self.random_state or 0,
                                       n_threads=self.n_threads)
            Y = optimize_tsne(P, Y, n_iter=int(n_iter), early_iter=int(early), early_exaggeration=float(self.tsne_early_exaggeration),
                              exaggeration=float(self.exaggeration), learning_rate=self.tsne_learning_rate,
                              n_interp=int(self.fft_interp), box_width=float(self.fft_box),
                              n_threads=self.n_threads, mid_pairs=pairs, mid_weight=3.0 * float(mid_w),
                              mid_phase=float(self.mid_phase), dof=float(self.tsne_dof), verbose=self.verbose,
                              far_weight=float(self.tsne_far[0]), far_r0=float(self.tsne_far[1]),
                              mid_form=str(self.tsne_mid[0]), mid_c=float(self.tsne_mid[1]))
            tail_iter, tail_exag = int(self.hybrid_tail[0]), float(self.hybrid_tail[1])
            if tail_iter > 0 and tail_exag != 1.0:
                Y = optimize_tsne(P, Y, n_iter=tail_iter, early_iter=0, exaggeration=tail_exag, learning_rate=self.tsne_learning_rate,
                                  n_interp=int(self.fft_interp), box_width=float(self.fft_box), n_threads=self.n_threads,
                                  dof=float(self.tsne_dof), verbose=self.verbose)
            self.n_epochs_ = int(n_iter)
            head = np.zeros(0, dtype=np.int32)
        else:
            n_epochs = self._resolve_epochs(n)
            head, tail, epochs_per_sample = self._edge_schedule(P, n_epochs)
            self._mid_pairs_ = None
            self._jump_graph_ = None
            if self.mid_source == "spectral" and mid_w > 0:
                # graphs: mid-near pairs measured in a spectral (diffusion) embedding of the graph
                from .roles import mid_near_pairs
                S = spectral_init(P, 8, self.random_state, n_threads=self.n_threads)
                self._mid_pairs_ = mid_near_pairs(np.ascontiguousarray(S, dtype=np.float32), self.n_mid_pairs,
                                                  random_state=self.random_state or 0, n_threads=self.n_threads)
            if self.mid_source in ("data", "auto", "hybrid") and hasattr(self, "_prepared_") and mid_w > 0:
                from .roles import mid_near_pairs
                self._mid_pairs_ = mid_near_pairs(self._prepared_, self.n_mid_pairs, random_state=self.random_state or 0,
                                                  n_threads=self.n_threads)
                if self.mid_source == "hybrid":
                    rows = np.repeat(np.arange(n), self.n_mid_pairs)
                    J = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, self._mid_pairs_.ravel())), shape=(n, n))
                    J = (J + J.T).tocsr()
                    J.sort_indices()
                    self._jump_graph_ = J
            self._optimize(Y, head, tail, epochs_per_sample, P, n_epochs, n_fixed=0, seed=seed,
                           mid_weight=mid_w)
            if self.loss == "hybrid":
                # second pass: the exact (FFT) repulsion refines the local structure of the SGD layout. The layout
                # is first brought to the exact mode's own scale (31 x sqrt(n / 70000)): started from its natural
                # scale the normalised repulsion collapses it, started from t-SNE's tiny init it re-expands and
                # loses the arrangement. No early exaggeration and no mid-range spring (it would contract the
                # layout at this scale): the global arrangement is the SGD pass's job, kept through the init.
                if self.n_components != 2:
                    raise ValueError("loss='hybrid' (FFT repulsion) is implemented for n_components=2 only")
                from .tsne import optimize_tsne
                t_h = time.time()
                Y -= Y.mean(0)
                Y *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y[:, 0].std()), 1e-12)
                Y = optimize_tsne(P, Y, n_iter=int(self.hybrid_iter), early_iter=0, exaggeration=float(self.exaggeration),
                                  learning_rate=self.tsne_learning_rate, n_interp=int(self.fft_interp),
                                  box_width=float(self.fft_box), n_threads=self.n_threads, dof=float(self.tsne_dof),
                                  verbose=self.verbose)
                tail_iter, tail_exag = int(self.hybrid_tail[0]), float(self.hybrid_tail[1])
                if tail_iter > 0 and tail_exag != 1.0:   # exaggerated tail: compacts the clusters at a local cost (superseded by contraction=0.7)
                    Y = optimize_tsne(P, Y, n_iter=tail_iter, early_iter=0, exaggeration=tail_exag,
                                      learning_rate=self.tsne_learning_rate, n_interp=int(self.fft_interp),
                                      box_width=float(self.fft_box), n_threads=self.n_threads, dof=float(self.tsne_dof),
                                      verbose=self.verbose)
                if self.verbose:
                    print(f"Hybrid: exact pass ({self.hybrid_iter} + {tail_iter} iterations) - T={time.time() - t_h:.2f}s")
        if self.verbose:
            print(f"Layout ({self.n_epochs_} epochs, {len(head)} edges) - T={time.time() - t:.2f}s"
                  f" | total {time.time() - t0:.2f}s")
        if self.contraction is not None:
            from .contraction import hierarchical_contraction
            opts = dict(self.contraction) if isinstance(self.contraction, dict) else {"separation": float(self.contraction)}
            Y, self.contraction_log_ = hierarchical_contraction(Y, P, X=X if getattr(self, "_is_data_", False) else None,
                                                                metric=self.metric, random_state=self.random_state or 0, verbose=self.verbose,
                                                                n_threads=(self.n_threads if self.n_threads and self.n_threads > 0 else 8), **opts)
        self.embedding_ = Y
        return self

    # ------------------------------------------------------------------ optimisation helpers
    def _resolve_epochs(self, n):
        if self.n_epochs is None:
            return 500 if n <= 10000 else (300 if n <= 100000 else 200)
        return self.n_epochs

    def _edge_schedule(self, P, n_epochs):
        """Edge i is visited every 1 / min(1, s w_i) epochs, s chosen so that the number of attractive
        samples per epoch is samples_per_node * n. Edges visited less than once are dropped."""
        C = P.tocoo()
        w = C.data.astype(np.float64)
        freq = _sampling_frequencies(w, self.samples_per_node * P.shape[0])
        budget = n_epochs if isinstance(n_epochs, (int, np.integer)) else 300
        keep = freq >= 1.0 / budget
        head = C.row[keep].astype(np.int32)
        tail = C.col[keep].astype(np.int32)
        return head, tail, (1.0 / freq[keep]).astype(np.float32)

    def _crowding(self, Y, P, ref=None):
        """Share of each point's graph edges whose embedded length exceeds a reference length: the kernel
        scale a^(-1/2b) when `ref` is None (a converged layout), or `ref` times the median edge length (a
        layout still contracting, mid-run). This is the contamination of the attraction
        (docs/kernel_exponents.md, section 2)."""
        P = P.tocsr()
        deg = np.diff(P.indptr)
        i = np.repeat(np.arange(P.shape[0]), deg)
        d = np.sqrt(((Y[i] - Y[P.indices]) ** 2).sum(1))
        far_len = float(self.a_) ** (-1.0 / (2.0 * float(self.b_))) if ref is None else float(ref) * float(np.median(d))
        return np.bincount(i, weights=(d > far_len).astype(np.float64), minlength=P.shape[0]) / np.maximum(deg, 1)

    def _optimize(self, Y, head, tail, epochs_per_sample, P, n_epochs, n_fixed, seed, mid_weight):
        """Runs the native SGD on Y (in place) with the requested schedule. Sets self.n_epochs_."""
        from _magnet import optimize_layout

        n = Y.shape[0]
        lr = float(self.learning_rate)
        loss = {"umap": 0, "infonce": 1, "hybrid": 0}[self.loss]
        if self.parallel not in ("hogwild", "owner"):
            raise ValueError("parallel must be 'hogwild' or 'owner'")
        if self.parallel == "owner" and loss == 1:
            raise ValueError("parallel='owner' is not available with loss='infonce' (the negatives are moved)")
        kernel = {"power": 0, "gaussian": 1, "sech": 2, "tanh": 3}[self.kernel]
        node_b = getattr(self, "node_b_", None)
        node_b = np.zeros(0, dtype=np.float32) if node_b is None else np.ascontiguousarray(node_b, dtype=np.float32)
        optimizer = {"sgd": 0, "adam": 1}[self.optimizer]
        clip = {"coord": 0, "norm": 1}[self.clip]
        if optimizer == 1:
            adam_m = np.zeros_like(Y); adam_v = np.zeros_like(Y); adam_t = np.zeros(n, dtype=np.int32)
        else:
            adam_m = adam_v = np.zeros((0, Y.shape[1]), dtype=np.float32); adam_t = np.zeros(0, dtype=np.int32)
        eons = epochs_per_sample.copy()
        eonns = (epochs_per_sample / self.negative_sample_rate).astype(np.float32)
        mid_pairs = getattr(self, "_mid_pairs_", None)
        mid_pairs = np.zeros((0, 1), dtype=np.int32) if mid_pairs is None else np.ascontiguousarray(mid_pairs, dtype=np.int32)
        J = getattr(self, "_jump_graph_", None)
        jump_indptr = J.indptr.astype(np.int32) if J is not None else np.zeros(0, dtype=np.int32)
        jump_indices = J.indices.astype(np.int32) if J is not None else np.zeros(0, dtype=np.int32)
        use_mid = P is not None and mid_weight > 0 and (self.walk_len > 0 or mid_pairs.size > 0) and self.mid_rate > 0
        walk_lens = np.zeros(0, dtype=np.int32)
        if use_mid:
            indptr = P.indptr.astype(np.int32)
            indices = P.indices.astype(np.int32)
            if self.mid_scale != "fixed":
                from .roles import adaptive_walk_lengths
                walk_lens = adaptive_walk_lengths(P, self.mid_scale, base=self.walk_len)
                self.walk_lens_ = walk_lens
        else:
            indptr = indices = np.zeros(0, dtype=np.int32)

        rev_indptr = rev_indices = np.zeros(0, dtype=np.int32)
        if self.parallel == "owner" and mid_pairs.size > 0 and J is None:
            # who chose each node as a mid-range partner: the owner mode delivers the partner's pull from this side
            R = sp.csr_matrix((np.ones(mid_pairs.size, np.float32), (mid_pairs.ravel(), np.repeat(np.arange(n), mid_pairs.shape[1]))), shape=(n, n))
            R.sort_indices()
            rev_indptr, rev_indices = R.indptr.astype(np.int32), R.indices.astype(np.int32)
        def run(e0, e1, a0, a1, m0, m1, exag=None):
            optimize_layout(Y, head, tail, epochs_per_sample, eons, eonns, indptr, indices,
                            int(e0), int(e1), float(a0), float(a1), float(m0), float(m1),
                            float(self.a_), float(self.b_), float(self.rep_a_), float(self.rep_b_), float(self.negative_sample_rate), loss,
                            float(self.exaggeration if exag is None else exag), float(self.repulsion_strength), int(self.walk_len),
                            float(self.mid_rate), optimizer, clip, int(n_fixed),
                            adam_m, adam_v, adam_t, float(self.beta1), float(self.beta2), walk_lens, mid_pairs, jump_indptr, jump_indices,
                            float(self.mid_jump) if J is not None else 0.0, kernel, node_b, seed, int(self.n_threads),
                            {"hogwild": 0, "owner": 1}[self.parallel], rev_indptr, rev_indices)

        # ---- schedule: list of (epoch_start, epoch_end, lr_start, lr_end), plus the mid-range weight
        if isinstance(n_epochs, (int, np.integer)):
            total = int(n_epochs)
            mid_len = max(1.0, self.mid_phase * total)
            alpha = lambda e: lr * (1.0 - e / total)
            mid = (lambda e: mid_weight * max(0.0, 1.0 - e / mid_len)) if use_mid else (lambda e: 0.0)
            late_start = total - int(round(self.late_phase * total)) if self.late_exaggeration != 1.0 else total
            chunk = max(1, total // 20) if self.verbose else total
            cb = getattr(self, "epoch_callback", None)          # research hook: callable(Y, epoch, total) between segments
            if cb is not None:
                chunk = max(1, min(chunk, int(getattr(self, "callback_every", 10))))
            adapt_at = total + 1
            if isinstance(self.kernel_b, str) and self.kernel_b == "adaptive" and P is not None and n_fixed == 0:
                adapt_at = int(round(float(self.adaptive[2]) * total))
            bounds = sorted(set([0, total, late_start, min(adapt_at, total)] + list(range(0, total, chunk))))
            segments = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            iterator = segments
            if self.verbose:
                import tqdm
                iterator = tqdm.tqdm(segments, desc="Layout")
            for e0, e1 in iterator:
                if e0 == adapt_at:
                    self.crowding_ = self._crowding(Y, P, ref=float(self.adaptive[4]) if len(self.adaptive) > 4 else 2.0)
                    min_deg = int(self.adaptive[5]) if len(self.adaptive) > 5 else 5
                    uncrowded = (self.crowding_ <= float(self.adaptive[3])) & (np.diff(P.tocsr().indptr) >= min_deg)
                    node_b[:] = np.where(uncrowded, float(self.adaptive[1]), float(self.adaptive[0]))
                    self.node_b_ = node_b
                exag = self.exaggeration * self.late_exaggeration if e0 >= late_start else None
                run(e0, e1, alpha(e0), alpha(e1), mid(e0), mid(e1), exag)
                if cb is not None:
                    cb(Y, e1, total)
            self.n_epochs_ = total
            return
        if isinstance(n_epochs, (tuple, list)):
            const, anneal = int(n_epochs[0]), int(n_epochs[1])
        elif n_epochs == "auto":
            const, anneal = None, None
        else:
            raise ValueError("n_epochs must be an int, a (constant, anneal) tuple or 'auto'")
        # constant phase (mid-range weight annealed over its own window)
        chunk = 10
        mid_len = max(1.0, self.mid_phase * (const if const is not None else 200))
        mid = (lambda e: mid_weight * max(0.0, 1.0 - e / mid_len)) if use_mid else (lambda e: 0.0)
        e = 0
        history = []
        rms = lambda A: float(np.sqrt((A ** 2).sum(1).mean()))
        max_const = const if const is not None else 400
        while e < max_const:
            Y_prev = Y.copy()
            run(e, e + chunk, lr, lr, mid(e), mid(e + chunk))
            e += chunk
            if const is None:
                disp = rms(Y - Y_prev) / max(rms(Y - Y.mean(0)), 1e-9) / chunk
                history.append(disp)
                if e >= 50 and len(history) >= 3 and history[-1] > 0.97 * min(history[:-1]) \
                        and history[-2] > 0.97 * min(history[:-2]):
                    break
        if anneal is None:
            anneal = max(30, e // 2)
        for e0 in range(e, e + anneal, chunk):
            e1 = min(e + anneal, e0 + chunk)
            run(e0, e1, lr * (1.0 - (e0 - e) / anneal), lr * (1.0 - (e1 - e) / anneal), 0.0, 0.0)
        self.n_epochs_ = e + anneal

    def fit_transform(self, X, y=None):
        """Fit and return the layout (n, n_components)."""
        return self.fit(X).embedding_

    # ------------------------------------------------------------------ out-of-sample
    def transform(self, X):
        """Embed new rows into the fitted layout (data inputs only). Each new point is attached to its
        nearest training points (beam search over the training neighbour graph), initialised at the
        weighted mean of their positions and optimised for transform_epochs while the training points
        stay fixed."""
        if not hasattr(self, "_prepared_"):
            raise ValueError("transform() needs a model fitted on a data matrix")
        t0 = time.time()
        Z = X
        if self.pca_ is not None:
            Z = self.pca_.transform(X)
        Q = prepare_data(Z, self.metric)
        if Q.shape[1] != self._prepared_.shape[1]:
            raise ValueError("new data must have the same number of features as the training data")
        n_train = self.embedding_.shape[0]
        k = min(self.n_neighbors, n_train) - 1
        if not hasattr(self, "landmarks_"):
            self.landmarks_ = choose_landmarks(n_train, self.random_state or 0)
        ind, dist = knn_search(self._prepared_, self.graph_, Q, k, self.metric, landmarks=self.landmarks_,
                               n_threads=self.n_threads, random_state=self.random_state or 0)
        vals = smooth_knn(dist, self.n_neighbors)
        W = vals / np.maximum(vals.sum(1, keepdims=True), 1e-12)
        Y_new = np.einsum("mk,mkd->md", W, self.embedding_[ind]).astype(np.float32)
        m = Q.shape[0]
        Y_all = np.ascontiguousarray(np.vstack([self.embedding_, Y_new]), dtype=np.float32)
        n_epochs = self.transform_epochs or 50
        w = vals.ravel().astype(np.float64)
        freq = _sampling_frequencies(w, self.samples_per_node * m)
        keep = freq >= 1.0 / n_epochs
        head = (n_train + np.repeat(np.arange(m), k)).astype(np.int32)[keep]
        tail = ind.ravel().astype(np.int32)[keep]
        eps = (1.0 / freq[keep]).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        saved_pairs, self._mid_pairs_ = getattr(self, "_mid_pairs_", None), None
        saved_jumps, self._jump_graph_ = getattr(self, "_jump_graph_", None), None
        self._optimize(Y_all, head, tail, eps, None, int(n_epochs), n_fixed=n_train,
                       seed=int(rng.integers(0, 2 ** 62)), mid_weight=0.0)
        self._mid_pairs_ = saved_pairs
        self._jump_graph_ = saved_jumps
        if self.verbose:
            print(f"transform ({m} points, {n_epochs} epochs) - T={time.time() - t0:.2f}s")
        return Y_all[n_train:]
