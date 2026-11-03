"""k17: two small theorems for the hand-over.
(a) Virial identity. d/ds KL(sY) = (2/s) [ sum_ij p_ij u_ij - (1 - sum q~^2 / Z) ], u = s^2 d^2 / (1 + s^2 d^2): at a stationary
    point the p-weighted mean of d^2/(1+d^2) equals its q-weighted mean. The hand-over scale is s* = argmin_s KL(s Y0), a 1-D
    search with two FFT evaluations per s. Scanned on MNIST 70k for the stiff SGD start, the default start, UMAP and PCA, and
    on MNIST 20k and Fashion 70k (does s* follow 31 sqrt(n / 70000)?).
(b) Projected refinement. U = orthonormal basis of (constant + K leading eigenvectors of the graph's normalised Laplacian);
    after every update Y <- Y - U U^T (Y - Y0): U^T Y is invariant, the coarse coordinates of the start are kept exactly.
    Pass of 300 and 1000 iterations for K in {0, 8, 32, 128, 512} from the stiff SGD start and from UMAP's layout; 15-NN,
    Spearman, triplets, centroids, time. -> k17_virial_projected.json / .log. Usage: python benchmarks/theory/k17_virial_projected.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist, fmnist
from metrics import _idx, high_dim_pdist, high_dim_neighbors
from magnet import MAGNET
from magnet.tsne import FFTRepulsion2D
from magnet.spectral import spectral_init
from _magnet import tsne_attraction, tsne_update

def prep(X, y, key):
    n = len(X); q = _idx(n, 5000, 0); hnn = high_dim_neighbors(X, q, 15, key)
    g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1)
    rng = np.random.default_rng(0); i, j, l = rng.integers(0, len(g), (3, 200000)); ok = (i != j) & (i != l) & (j != l)
    Ch = np.array([X[y == c].mean(0) for c in range(10)]); Dch = np.sqrt(((Ch[:, None] - Ch[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]
    def metrics(Y):
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y[q], return_distance=False)
        lnn = np.array([row[row != a][:15] for a, row in zip(q, lnn)])
        knn = float(np.mean([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15)
        Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); dl = Dl[iu]
        Cl = np.array([Y[y == c].mean(0) for c in range(10)]); Dcl = np.sqrt(((Cl[:, None] - Cl[None]) ** 2).sum(-1))[np.triu_indices(10, 1)]
        return dict(knn15=knn, sp=float(spearmanr(Dh[iu], dl).correlation), trip=float(np.mean(np.sign(Dh[i[ok], j[ok]] - Dh[i[ok], l[ok]]) == np.sign(Dl[i[ok], j[ok]] - Dl[i[ok], l[ok]]))),
                    cent=float(spearmanr(Dch, Dcl).correlation), std=float(Y[:, 0].std()))
    return metrics

def sum_q2(rep, Y):
    """sum_{i != j} (1 + d^2)^-2 through the FFT (unit charges on the K2 kernel), numpy path of FFTRepulsion2D."""
    import scipy.fft as fft
    from _magnet import interp_scatter, interp_gather
    n = Y.shape[0]; xmin, ymin, width, n_boxes = rep._grid(Y); N = n_boxes * rep.n_interp
    M, K1h, K2h = rep._kernels(n_boxes, width)
    charges = np.ascontiguousarray(np.ones((n, 1), np.float32))
    W = np.asarray(interp_scatter(Y, charges, xmin, ymin, width, n_boxes, rep.n_interp, rep.n_threads))
    Wh = fft.rfft2(W, s=(M, M), axes=(1, 2), workers=-1)
    phi = np.empty((2, N, N), dtype=np.float32)
    phi[0] = fft.irfft2(K1h * Wh[0], s=(M, M), workers=-1)[:N, :N]; phi[1] = fft.irfft2(K2h * Wh[0], s=(M, M), workers=-1)[:N, :N]
    pot = np.asarray(interp_gather(Y, np.ascontiguousarray(phi), xmin, ymin, width, n_boxes, rep.n_interp, rep.n_threads))
    return float(pot[:, 0].sum() - n), float(pot[:, 1].sum() - n)

def kl_and_virial(P, Y, rep):
    C = P.tocoo(); d2 = ((Y[C.row] - Y[C.col]) ** 2).sum(1).astype(np.float64)
    Z, Q2 = sum_q2(rep, Y)
    kl = float((C.data * (np.log(C.data) + np.log1p(d2))).sum() + np.log(Z))
    V = float((C.data * d2 / (1.0 + d2)).sum() - (1.0 - Q2 / Z))
    return kl, V

def scan(P, Y0, label, scales):
    rep = FFTRepulsion2D(n_interp=3, box_width=1.0, n_threads=-1)
    Yc = Y0 - Y0.mean(0); Yc = Yc / max(float(Yc[:, 0].std()), 1e-12)
    rows = []
    for s in scales:
        kl, V = kl_and_virial(P, np.ascontiguousarray(s * Yc, dtype=np.float32), rep); rows.append(dict(s=float(s), kl=kl, V=V))
    best = min(rows, key=lambda r: r["kl"])
    # refine the minimum by a parabola through the three points around it
    k = rows.index(best)
    if 0 < k < len(rows) - 1:
        xs = np.log([rows[k - 1]["s"], rows[k]["s"], rows[k + 1]["s"]]); ys = [rows[k - 1]["kl"], rows[k]["kl"], rows[k + 1]["kl"]]
        a, b, c = np.polyfit(xs, ys, 2); s_star = float(np.exp(-b / (2 * a))) if a > 0 else best["s"]
    else:
        s_star = best["s"]
    print(f"{label:<34} s* = {s_star:6.1f}   " + "  ".join(f"{r['s']:.0f}:{r['kl']:.3f}/{r['V']:+.3f}" for r in rows), flush=True)
    return dict(s_star=s_star, rows=rows)

def coarse_basis(P, K, seed=0):
    n = P.shape[0]
    cols = [np.ones((n, 1), np.float32)]
    if K > 0:
        U = spectral_init(P, K, random_state=seed, n_threads=-1)
        cols.append(np.asarray(U, dtype=np.float32))
    U = np.concatenate(cols, 1).astype(np.float64)
    Q, _ = np.linalg.qr(U)
    return np.ascontiguousarray(Q.astype(np.float32))

def projected_pass(P, Y0, U, n_iter, lr=None, momentum=0.8, n_threads=-1, callback=None):
    n = P.shape[0]; Pn = sp.csr_matrix(P, dtype=np.float32); Pn = Pn / Pn.sum()
    indptr, indices, data = Pn.indptr.astype(np.int32), Pn.indices.astype(np.int32), Pn.data.astype(np.float32)
    lr = lr or max(200.0, n / 12.0)
    Y = np.ascontiguousarray(Y0, dtype=np.float32).copy(); update = np.zeros_like(Y); gains = np.ones_like(Y)
    rep = FFTRepulsion2D(n_interp=3, box_width=1.0, n_threads=n_threads)
    coarse0 = U.T @ Y0 if U is not None else None
    for it in range(n_iter):
        attr = np.asarray(tsne_attraction(indptr, indices, data, Y, 1.0, n_threads))
        sumQ, F = rep(Y)
        grad = attr - F * (1.0 / max(sumQ, 1e-12))
        if U is not None:
            grad -= U @ (U.T @ grad)           # gradient in the complement (keeps the gains' statistics sane)
        tsne_update(Y, update, gains, np.ascontiguousarray(grad, dtype=np.float32), float(lr), float(momentum), n_threads)
        if U is not None:
            Y -= U @ (U.T @ Y - coarse0)       # exact re-projection onto U^T Y = U^T Y0 (gains and recentring break it)
        if callback is not None and (it + 1) in (300, 1000):
            callback(it + 1, Y)
    return Y

if __name__ == "__main__":
    out = {"virial": {}, "projected": {}}
    scales = [4, 8, 12, 16, 22, 31, 44, 62, 88, 124]
    # ---- (a) virial scans
    for loader, n, label in ((mnist, 70000, "MNIST 70k"), (mnist, 20000, "MNIST 20k"), (fmnist, 70000, "Fashion 70k")):
        X, y, key = loader(n); y = np.asarray(y).astype(int)
        m = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float32); P = P / P.sum()
        starts = {"SGD spring 0.5": m.embedding_.astype(np.float32)}
        if n == 70000:
            starts["SGD spring 0.1"] = MAGNET(random_state=0).fit_transform(X).astype(np.float32)
            starts["PCA-2"] = PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32)
            cache = os.path.join(HERE, "..", "..", "paper", "figures", "cache", f"{'mnist' if loader is mnist else 'fmnist'}_umap0512.npz")
            starts["UMAP"] = np.load(cache)["Y"].astype(np.float32)
            starts["full-batch layout"] = MAGNET(loss="tsne", random_state=0).fit_transform(X).astype(np.float32)
        print(f"\n[{label}] 31 sqrt(n/70000) = {31 * np.sqrt(n / 70000):.1f}")
        for k, Y0 in starts.items():
            out["virial"][f"{label}/{k}"] = scan(P, Y0, k, scales)
            out["virial"][f"{label}/{k}"]["start_std"] = float(Y0[:, 0].std())
        json.dump(out, open(os.path.join(HERE, "k17_virial_projected.json"), "w"), indent=1)
        if n == 70000 and loader is mnist:
            X_m, y_m, P_m, starts_m = X, y, P, starts
    # ---- (b) projected refinement on MNIST 70k
    metrics = prep(X_m, y_m, "mnist70000"); n = len(X_m)
    print(f"\n{'start':<16}{'K':>5}{'it':>6}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'std':>7}{'t(s)':>7}")
    for sname in ("SGD spring 0.5", "UMAP"):
        Y0 = starts_m[sname]; Y0 = Y0 - Y0.mean(0); Y0 = (Y0 / max(float(Y0[:, 0].std()), 1e-12) * 31.0).astype(np.float32)
        r0 = metrics(Y0); print(f"{sname:<16}{'':>5}{0:>6}{r0['knn15']:>7.3f}{r0['sp']:>7.3f}{r0['trip']:>7.3f}{r0['cent']:>7.3f}{r0['std']:>7.1f}", flush=True)
        for K in (0, 8, 32, 128, 512):
            U = coarse_basis(P_m, K) if K >= 0 else None
            if K == 0: U = None
            t0 = time.time(); res = {}
            def cb(it, Y, t0=t0, K=K, sname=sname):
                r = metrics(Y); r["t"] = time.time() - t0; res[it] = r
                print(f"{sname:<16}{K:>5}{it:>6}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{r['std']:>7.1f}{r['t']:>7.1f}", flush=True)
            projected_pass(P_m, Y0, U, 1000, callback=cb)
            out["projected"][f"{sname}/K={K}"] = dict(start=r0, **{str(k): v for k, v in res.items()})
            json.dump(out, open(os.path.join(HERE, "k17_virial_projected.json"), "w"), indent=1)
