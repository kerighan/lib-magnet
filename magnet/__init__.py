"""MAGNET v3 - fast graph-based dimensionality reduction.

Pipeline: kNN graph (hnswlib) -> fuzzy neighbourhood graph -> spectral init (power iteration)
-> multithreaded Hogwild SGD with negative sampling and random-walk mid-range attraction.
Accepts a data matrix (n, d), a networkx graph or a scipy sparse adjacency matrix.
"""
import time
import numpy as np
import scipy.sparse as sp

from .knn import knn_hnsw, knn_nndescent, knn_search, fuzzy_graph, prepare_data, smooth_knn, choose_landmarks
from .spectral import spectral_init

__version__ = "3.0.0"
__all__ = ["MAGNET", "find_ab_params"]


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


class MAGNET:
    """
    Parameters
    ----------
    n_components : output dimension (2 or 3 for visualisation; any value <= 64 works).
    n_neighbors : size of the local neighbourhood (counts the point itself, like UMAP).
    metric : "euclidean" or "cosine" (hnswlib spaces).
    min_dist, spread : shape of the low-dimensional kernel (same meaning as in UMAP; 0.05 preserves
        neighbourhoods better than UMAP's 0.1 at every other metric equal).
    fft_interp, fft_box : interpolation points per box and box width for loss="tsne" (3 and 1.0, as
        FIt-SNE; fft_box=1.5 halves the FFT cost for about 3% less 15-NN preservation).
    tsne_early : early-exaggeration iterations for loss="tsne" (default min(250, n_epochs // 3));
        with loss="tsne", n_epochs is the number of iterations (default 600 = 200 + 400; 450 with
        fft_box=1.5 is a fast preset at 7 s instead of 11 s on MNIST 70k, 750 the FIt-SNE schedule).
    kernel : low-dimensional similarity family of the SGD mode: "power" (1 / (1 + a d^2b), heavy tail),
        "gaussian" (exp(-a d^2)), "sech" (1 / cosh(a d)) or "tanh" (1 - tanh(a d^2)); the last three
        have exponential or lighter tails (kept for comparison).
    kernel_b : exponent b of the kernel 1 / (1 + a d^(2b)); a is fitted to the (min_dist, spread) curve.
        None reproduces UMAP's fit (b ~ 0.8). Heavier tails preserve neighbourhoods much better:
        on MNIST 70k, 15-NN preservation 0.13 (b 0.8) -> 0.19 (0.6) -> 0.21 (0.5) at equal global
        structure and label accuracy; text data peaks at 0.6-0.7, images at 0.5. Default 0.6.
        kernel_b=1 (Cauchy) removes the pow() from the inner loop (faster, but the worst local quality).
        "auto" picks it among {0.4, 0.5, 0.6, 0.7} by 15-NN preservation on a 4k subsample (a few
        seconds; the optimum drifts with the sample size, so the fixed default is usually as good);
        an array gives one exponent per point; "local" derives it from the local intrinsic dimension
        (dimprint if installed, Levina-Bickel MLE on the kNN distances otherwise).
    n_epochs : int -> that many epochs with a linear learning-rate decay (default None: 500 for
        n <= 10k, 300 for n <= 100k, else 200); (constant, anneal) tuple -> constant learning rate for
        the first count then linear decay over the second; "auto" -> constant learning rate until the
        layout stops moving, then anneal.
    learning_rate : initial step (SGD) or Adam step size.
    loss : "umap" (negative-sampling cross entropy, as UMAP), "infonce" (contrastive softmax over the
        positive and its negatives) or "tsne" (exact repulsion over all pairs through FFT interpolation,
        full-batch gradient descent as FIt-SNE / openTSNE; 2-D only; `exaggeration` is then the
        attraction multiplier of the final phase: 1 = t-SNE, ~4 looks like UMAP).
    exaggeration : multiplier on the attractive gradient (UMAP ~ t-SNE with exaggeration 4).
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
        first mid_phase fraction of the epochs). Set mid_weight=0 to disable.
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
    mid_scale : "fixed" (walk_len steps for every node), "spread" (per-node length from the 2-step
        diffusion spread of the graph) or "rolewalk" (per-role length from RoleWalk signatures).
    component_placement : "centroid" (data centroids, spiral for graphs) or "rolewalk" (components
        of a graph placed by their mean RoleWalk signature).
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
    graph_norm : for graph inputs, "sym" rescales w_ij by 1/sqrt(deg_i deg_j) so that every node
        receives a comparable number of updates (hubs do not monopolise the budget); "none" keeps
        the raw weights.
    n_threads : threads for the native kernels (-1 = all cores).
    random_state : seed. The SGD is lock-free and multithreaded, so runs are reproducible only
        approximately unless n_threads=1.
    """

    def __init__(self, n_components=2, n_neighbors=15, metric="euclidean", min_dist=0.05, spread=1.0,
                 n_epochs=None, learning_rate=1.0, negative_sample_rate=5, repulsion_strength=2.0,
                 kernel="power", kernel_b=0.6, fft_interp=3, fft_box=1.0, tsne_early=None, loss="umap", exaggeration=1.0,
                 late_exaggeration=1.0, late_phase=0.2, tsne_dof=1.0, optimizer="sgd", clip="coord", beta1=0.9, beta2=0.999,
                 walk_len=4, mid_weight=0.1, mid_rate=0.25, mid_phase=0.5, mid_scale="fixed", mid_source="auto",
                 n_mid_pairs=5, mid_jump=0.5,
                 component_placement="centroid",
                 init="auto", pca=None, knn="nndescent", ef_construction=80, M=16, ef=None,
                 set_op_mix_ratio=1.0, neighbor_power=1.0, neighbor_rank_exponent=0.0,
                 samples_per_node=5.0, graph_norm="sym", input="auto",
                 transform_epochs=None, n_threads=-1, random_state=None, verbose=False):
        self.n_components = n_components
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
        self.component_placement = component_placement
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
            if data_for_components is None and self.component_placement == "rolewalk":
                from .roles import rolewalk_embedding
                data_for_components = rolewalk_embedding(P)
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
    def fit(self, X, y=None):
        from _magnet import optimize_layout

        t0 = time.time()
        rng = np.random.default_rng(self.random_state)
        seed = int(rng.integers(0, 2 ** 62))
        P = self.build_graph(X)
        n = P.shape[0]
        self.graph_ = P

        kb = self.kernel_b
        self.node_b_ = None
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
            self.node_b_ = local_kernel_exponent(X, self.n_neighbors, random_state=self.random_state or 0)
            kb = float(np.median(self.node_b_))
        elif kb is not None and not np.isscalar(kb):
            self.node_b_ = np.asarray(kb, dtype=np.float32)
            if self.node_b_.shape != (n,):
                raise ValueError("kernel_b array must have one entry per point")
            kb = float(np.median(self.node_b_))
        a, b = find_ab_params(self.spread, self.min_dist, kb, self.kernel)
        self.a_, self.b_ = a, b

        data_matrix = X if isinstance(X, np.ndarray) and X.ndim == 2 else None
        Y = self._initial_embedding(P, data_matrix, rng)

        t = time.time()
        if self.loss == "tsne":
            if self.n_components != 2:
                raise ValueError("loss='tsne' (FFT repulsion) is implemented for n_components=2 only")
            from .tsne import optimize_tsne
            n_iter = self.n_epochs if isinstance(self.n_epochs, (int, np.integer)) else 600
            Y = Y / max(float(Y[:, 0].std()), 1e-12) * 1e-4   # t-SNE convention: tiny initial spread
            early = self.tsne_early if self.tsne_early is not None else min(250, int(n_iter) // 3)
            pairs = None
            if self.mid_source in ("data", "auto") and hasattr(self, "_prepared_") and self.mid_weight > 0:
                from .roles import mid_near_pairs
                pairs = mid_near_pairs(self._prepared_, self.n_mid_pairs, random_state=self.random_state or 0,
                                       n_threads=self.n_threads)
            Y = optimize_tsne(P, Y, n_iter=int(n_iter), early_iter=int(early),
                              exaggeration=float(self.exaggeration), learning_rate=None,
                              n_interp=int(self.fft_interp), box_width=float(self.fft_box),
                              n_threads=self.n_threads, mid_pairs=pairs, mid_weight=3.0 * float(self.mid_weight),
                              mid_phase=float(self.mid_phase), dof=float(self.tsne_dof), verbose=self.verbose)
            self.n_epochs_ = int(n_iter)
            head = np.zeros(0, dtype=np.int32)
        else:
            n_epochs = self._resolve_epochs(n)
            head, tail, epochs_per_sample = self._edge_schedule(P, n_epochs)
            self._mid_pairs_ = None
            self._jump_graph_ = None
            if self.mid_source == "spectral" and self.mid_weight > 0:
                # graphs: mid-near pairs measured in a spectral (diffusion) embedding of the graph
                from .roles import mid_near_pairs
                S = spectral_init(P, 8, self.random_state, n_threads=self.n_threads)
                self._mid_pairs_ = mid_near_pairs(np.ascontiguousarray(S, dtype=np.float32), self.n_mid_pairs,
                                                  random_state=self.random_state or 0, n_threads=self.n_threads)
            if self.mid_source in ("data", "auto", "hybrid") and hasattr(self, "_prepared_") and self.mid_weight > 0:
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
                           mid_weight=self.mid_weight)
        if self.verbose:
            print(f"Layout ({self.n_epochs_} epochs, {len(head)} edges) - T={time.time() - t:.2f}s"
                  f" | total {time.time() - t0:.2f}s")
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

    def _optimize(self, Y, head, tail, epochs_per_sample, P, n_epochs, n_fixed, seed, mid_weight):
        """Runs the native SGD on Y (in place) with the requested schedule. Sets self.n_epochs_."""
        from _magnet import optimize_layout

        n = Y.shape[0]
        lr = float(self.learning_rate)
        loss = {"umap": 0, "infonce": 1}[self.loss]
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

        def run(e0, e1, a0, a1, m0, m1, exag=None):
            optimize_layout(Y, head, tail, epochs_per_sample, eons, eonns, indptr, indices,
                            int(e0), int(e1), float(a0), float(a1), float(m0), float(m1),
                            float(self.a_), float(self.b_), float(self.negative_sample_rate), loss,
                            float(self.exaggeration if exag is None else exag), float(self.repulsion_strength), int(self.walk_len),
                            float(self.mid_rate), optimizer, clip, int(n_fixed),
                            adam_m, adam_v, adam_t, float(self.beta1), float(self.beta2), walk_lens, mid_pairs, jump_indptr, jump_indices,
                            float(self.mid_jump) if J is not None else 0.0, kernel, node_b, seed, int(self.n_threads))

        # ---- schedule: list of (epoch_start, epoch_end, lr_start, lr_end), plus the mid-range weight
        if isinstance(n_epochs, (int, np.integer)):
            total = int(n_epochs)
            mid_len = max(1.0, self.mid_phase * total)
            alpha = lambda e: lr * (1.0 - e / total)
            mid = (lambda e: mid_weight * max(0.0, 1.0 - e / mid_len)) if use_mid else (lambda e: 0.0)
            late_start = total - int(round(self.late_phase * total)) if self.late_exaggeration != 1.0 else total
            chunk = max(1, total // 20) if self.verbose else total
            bounds = sorted(set([0, total, late_start] + list(range(0, total, chunk))))
            segments = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            iterator = segments
            if self.verbose:
                import tqdm
                iterator = tqdm.tqdm(segments, desc="Layout")
            for e0, e1 in iterator:
                exag = self.exaggeration * self.late_exaggeration if e0 >= late_start else None
                run(e0, e1, alpha(e0), alpha(e1), mid(e0), mid(e1), exag)
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
