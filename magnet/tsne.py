"""Exact-repulsion optimiser (t-SNE objective with FFT-accelerated interpolation, after FIt-SNE), used
by MAGNET(loss="tsne"). Attraction over the kNN graph, repulsion over all pairs through a convolution
on an equispaced grid, delta-bar-delta gains and momentum as in openTSNE / FIt-SNE.
`exaggeration` is the attraction multiplier of the final phase: 1 is t-SNE, about 4 looks like UMAP
(Boehm, Berens & Kobak 2022)."""
import time
import numpy as np
import scipy.sparse as sp


class FFTRepulsion2D:
    """Potentials sum_j K(|y_i - y_j|) c_j for the two t-SNE kernels K1 = 1/(1+r^2), K2 = K1^2,
    by Lagrange interpolation on boxes of fixed width (unit by default, as FIt-SNE) and FFT
    convolution on the equispaced nodes. Kernel spectra are cached per grid size."""

    def __init__(self, n_interp=3, box_width=1.0, min_boxes=20, max_boxes=1000, n_threads=-1):
        self.n_interp = n_interp
        self.box_width = box_width
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.n_threads = n_threads
        self._cache = {}

    def _grid(self, Y):
        lo = Y.min(0) - 1e-3
        hi = Y.max(0) + 1e-3
        span = float((hi - lo).max())
        n_boxes = int(np.clip(np.ceil(span / self.box_width), self.min_boxes, self.max_boxes))
        width = max(span / n_boxes, 1e-12)
        return float(lo[0]), float(lo[1]), width, n_boxes

    def _kernels(self, n_boxes, width):
        """Spectra of K1 and K2 on the padded torus (size M >= 2N - 1, chosen FFT-friendly)."""
        import scipy.fft as fft
        key = (n_boxes, round(width, 6))
        if key not in self._cache:
            N = n_boxes * self.n_interp
            M = fft.next_fast_len(2 * N, real=True)
            delta = width / self.n_interp
            off = np.arange(M)
            off = np.where(off <= M // 2, off, off - M).astype(np.float64) * delta
            r2 = off[:, None] ** 2 + off[None, :] ** 2
            K1 = (1.0 / (1.0 + r2)).astype(np.float32)
            workers = self.n_threads if self.n_threads and self.n_threads > 0 else -1
            self._cache = {key: (M, fft.rfft2(K1, workers=workers), fft.rfft2(K1 * K1, workers=workers))}
        return self._cache[key]

    def __call__(self, Y):
        """Returns (sumQ, repulsive forces (n, 2)) with q_ij = 1 / (1 + d_ij^2), self terms removed."""
        import scipy.fft as fft
        from _magnet import interp_scatter, interp_gather

        n = Y.shape[0]
        xmin, ymin, width, n_boxes = self._grid(Y)
        N = n_boxes * self.n_interp
        workers = self.n_threads if self.n_threads and self.n_threads > 0 else -1
        M, K1h, K2h = self._kernels(n_boxes, width)
        charges = np.ascontiguousarray(np.column_stack([np.ones(n, np.float32), Y[:, 0], Y[:, 1]]), dtype=np.float32)
        W = np.asarray(interp_scatter(Y, charges, xmin, ymin, width, n_boxes, self.n_interp, self.n_threads))
        Wh = fft.rfft2(W, s=(M, M), axes=(1, 2), workers=workers)   # zero-padded to the torus
        phi = np.empty((4, N, N), dtype=np.float32)
        phi[0] = fft.irfft2(K1h * Wh[0], s=(M, M), workers=workers)[:N, :N]
        phi[1:] = fft.irfft2(K2h[None] * Wh, s=(M, M), axes=(1, 2), workers=workers)[:, :N, :N]
        pot = np.asarray(interp_gather(Y, np.ascontiguousarray(phi), xmin, ymin, width, n_boxes, self.n_interp, self.n_threads))
        sumQ = float(pot[:, 0].sum() - n)
        F = pot[:, 1:2] * Y - pot[:, 2:4]
        return sumQ, F


def optimize_tsne(P, Y, n_iter=750, early_iter=250, early_exaggeration=12.0, exaggeration=1.0,
                  learning_rate=None, momentum=(0.5, 0.8), n_interp=3, box_width=1.0, n_threads=-1,
                  mid_pairs=None, mid_weight=0.0, mid_phase=0.5, verbose=False, callback=None):
    """Full-batch t-SNE gradient descent. P: symmetric CSR affinities (will be normalised to sum 1),
    Y: (n, 2) float32 initial layout (modified in place and returned).
    mid_pairs: optional (n, m) partners (PaCMAP-style mid-near pairs); a spring of stiffness
    mid_weight / n towards their mean position is added, annealed to zero over mid_phase * n_iter."""
    from _magnet import tsne_attraction, tsne_update, mid_spring

    n = P.shape[0]
    P = sp.csr_matrix(P, dtype=np.float32)
    P = P / P.sum()
    indptr, indices, data = P.indptr.astype(np.int32), P.indices.astype(np.int32), P.data.astype(np.float32)
    if learning_rate is None:
        learning_rate = max(200.0, n / early_exaggeration)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    if mid_pairs is not None:
        mid_pairs = np.ascontiguousarray(mid_pairs, dtype=np.int32)
    update = np.zeros_like(Y)
    gains = np.ones_like(Y)
    rep = FFTRepulsion2D(n_interp=n_interp, box_width=box_width, n_threads=n_threads)
    t0 = time.time()
    timers = {"attraction": 0.0, "repulsion": 0.0, "update": 0.0}
    for it in range(n_iter):
        early = it < early_iter
        rho = early_exaggeration if early else exaggeration
        mom = momentum[0] if early else momentum[1]
        t1 = time.time()
        attr = np.asarray(tsne_attraction(indptr, indices, data, Y, n_threads))
        t2 = time.time()
        sumQ, F = rep(Y)
        t3 = time.time()
        grad = rho * attr
        grad -= F * (1.0 / max(sumQ, 1e-12))
        if mid_pairs is not None and mid_weight > 0:
            w_mid = mid_weight * max(0.0, 1.0 - it / max(1.0, mid_phase * n_iter)) / n
            if w_mid > 0:
                grad = np.ascontiguousarray(grad, dtype=np.float32)
                mid_spring(Y, mid_pairs, grad, float(w_mid), n_threads)
        tsne_update(Y, update, gains, np.ascontiguousarray(grad, dtype=np.float32), float(learning_rate), float(mom), n_threads)
        t4 = time.time()
        timers["attraction"] += t2 - t1; timers["repulsion"] += t3 - t2; timers["update"] += t4 - t3
        if callback is not None and (it % 50 == 0 or it == n_iter - 1):
            callback(it, Y)
        if verbose and (it % 100 == 0 or it == n_iter - 1):
            print(f"  t-SNE iteration {it:4d}  exaggeration={rho:g}  T={time.time() - t0:.1f}s")
    if verbose:
        xmin, ymin, width, n_boxes = rep._grid(Y)
        print(f"  t-SNE timers: {', '.join(f'{k} {v:.2f}s' for k, v in timers.items())} | final grid {n_boxes} boxes")
    return Y
