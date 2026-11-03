"""k20: a rest-length stress on cell barycentres during the full-batch pass (idea 1 of the GPT-6 note).
Label-free cells: k-means with B centres on an 8-d spectral embedding of the graph. Targets: either the input-space
distances between the cells' mean vectors (D_ab = |mu_a - mu_b|, scaled by alpha so that the coarse RMS distance of the start
is matched), or the start layout's own barycentre distances. S = mean_ab ((|c_a - c_b| - alpha D_ab) / (alpha D_ab + delta))^2,
lifted to the points as (1/n_a) dS/dc_a; weight lambda set at iteration 0 by the ratio of the lifted stress gradient to the
cell-averaged KL gradient (0.1 / 0.3 / 1). MNIST 70k, starts = SGD layouts (spring 0.5 and 0.1) of 3 seeds, 300 iterations,
all cores. -> k20_coarse_stress.json / .log. Usage: python benchmarks/theory/k20_coarse_stress.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from magnet import MAGNET
from magnet.tsne import FFTRepulsion2D
from magnet.spectral import spectral_init
from _magnet import tsne_attraction, tsne_update
from k17_virial_projected import prep

def cell_stress(Y, lab, B, target, delta):
    """Returns S and its gradient lifted to the points. target: (B, B) rest lengths."""
    cnt = np.bincount(lab, minlength=B).astype(np.float64)
    C = np.stack([np.bincount(lab, Y[:, 0], B), np.bincount(lab, Y[:, 1], B)], 1) / cnt[:, None]
    D = C[:, None, :] - C[None, :, :]; dist = np.sqrt((D ** 2).sum(-1)) + 1e-9
    iu = np.triu_indices(B, 1)
    den = target + delta; resid = (dist - target) / den
    S = float((resid[iu] ** 2).mean())
    # dS/dC_a = (2 / npairs) sum_b resid_ab / den_ab * (C_a - C_b) / dist_ab
    coef = 2.0 / len(iu[0]) * resid / den / dist; np.fill_diagonal(coef, 0.0)
    G = (coef[:, :, None] * D).sum(1)            # (B, 2)
    lifted = G[lab] / cnt[lab][:, None]
    return S, lifted.astype(np.float32), C

def run_pass(Y0, lab, B, target, lam_ratio, n_iter=300, delta_frac=0.05, log=None, P=None):
    global n, indptr, indices, data
    if P is not None:
        Pn = sp.csr_matrix(P, dtype=np.float32); Pn = Pn / Pn.sum(); n = Pn.shape[0]
        indptr, indices, data = Pn.indptr.astype(np.int32), Pn.indices.astype(np.int32), Pn.data.astype(np.float32)
    Y = np.ascontiguousarray(Y0, dtype=np.float32).copy(); update = np.zeros_like(Y); gains = np.ones_like(Y)
    rep = FFTRepulsion2D(n_interp=3, box_width=1.0, n_threads=-1); lr = max(200.0, n / 12.0)
    delta = delta_frac * float(target[np.triu_indices(B, 1)].mean()); lam = None
    for it in range(n_iter):
        attr = np.asarray(tsne_attraction(indptr, indices, data, Y, 1.0, -1)); sumQ, F = rep(Y)
        grad = attr - F * (1.0 / max(sumQ, 1e-12))
        if lam_ratio > 0:
            S, gS, _ = cell_stress(Y.astype(np.float64), lab, B, target, delta)
            if lam is None:
                cnt = np.bincount(lab, minlength=B).astype(np.float64)
                cellavg = np.stack([np.bincount(lab, grad[:, 0], B), np.bincount(lab, grad[:, 1], B)], 1) / cnt[:, None]
                pcell = np.sqrt(((cellavg[lab]) ** 2).sum())
                # calibrate on a 10 % dilation of the start (a 'start' target has zero gradient at the start itself)
                _, gS_cal, _ = cell_stress(1.1 * Y.astype(np.float64), lab, B, target, delta)
                lam = lam_ratio * pcell / max(float(np.sqrt((gS_cal ** 2).sum())), 1e-12)
                if log is not None: log(f"    lambda = {lam:.3g} (|P_cell grad KL| {pcell:.3g}, |grad S| at 1.1 Y0 {np.sqrt((gS_cal ** 2).sum()):.3g}, S0 = {S:.4f})")
            grad = grad + lam * gS
        tsne_update(Y, update, gains, np.ascontiguousarray(grad, dtype=np.float32), float(lr), 0.8, -1)
    S_end = cell_stress(Y.astype(np.float64), lab, B, target, delta)[0] if lam_ratio > 0 else float("nan")
    return Y, S_end

if __name__ == "__main__":
    X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key)
    fits = {s: MAGNET(loss="umap", mid_weight=0.5, random_state=s).fit(X) for s in range(3)}
    P = sp.csr_matrix(fits[0].graph_, dtype=np.float32); Pn = P / P.sum()
    indptr, indices, data = Pn.indptr.astype(np.int32), Pn.indices.astype(np.int32), Pn.data.astype(np.float32)
    starts = {f"spring 0.5 / seed {s}": fits[s].embedding_.astype(np.float32) for s in range(3)}
    starts.update({f"spring 0.1 / seed {s}": MAGNET(random_state=s).fit_transform(X).astype(np.float32) for s in range(3)})

    S8 = np.asarray(spectral_init(P, 8, random_state=0, n_threads=-1), dtype=np.float64)
    cells = {B: MiniBatchKMeans(B, random_state=0, n_init=3, batch_size=4096).fit_predict(S8) for B in (64, 128)}

    out = json.load(open(os.path.join(HERE, "k20_coarse_stress.json"))) if os.path.exists(os.path.join(HERE, "k20_coarse_stress.json")) else {}
    print(f"{'start':<22}{'B':>5}{'target':>7}{'ratio':>6}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'S_end':>8}{'t(s)':>7}")
    for sname, Y0 in starts.items():
        Y0 = Y0 - Y0.mean(0); Y0 = (Y0 / max(float(Y0[:, 0].std()), 1e-12) * 31.0).astype(np.float32)
        r0 = metrics(Y0); print(f"{sname:<22}{'':>5}{'start':>7}{'':>6}{r0['knn15']:>7.3f}{r0['sp']:>7.3f}{r0['trip']:>7.3f}{r0['cent']:>7.3f}", flush=True)
        configs = [(64, "none", 0.0)] + [(B, tg, ratio) for B in (64, 128) for tg in ("input", "start", "mds") for ratio in (0.1, 0.3, 1.0, 3.0)]
        if "seed 0" not in sname:
            configs = [(64, "none", 0.0), (64, "input", 0.3), (64, "input", 1.0), (128, "input", 0.3), (64, "start", 0.3), (64, "mds", 1.0), (64, "mds", 3.0), (128, "mds", 1.0)]
        for B, tg, ratio in configs:
            if f"{sname}/B={B}/{tg}/{ratio}" in out:
                r = out[f"{sname}/B={B}/{tg}/{ratio}"]; print(f"{sname:<22}{B:>5}{tg:>7}{ratio:>6}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{r['S_end']:>8.4f}{r['t']:>7.1f}  (cached)", flush=True); continue
            lab = cells[B]; cnt = np.bincount(lab, minlength=B).astype(np.float64)
            if tg in ("input", "mds"):
                mu = np.stack([np.bincount(lab, X[:, d], B) for d in range(X.shape[1])], 1) / cnt[:, None]
                D = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(-1))
                if tg == "mds":   # a 2-D-feasible target: classical MDS of the B x B input distances
                    J = np.eye(B) - 1.0 / B; G = -0.5 * J @ (D ** 2) @ J; w, V = np.linalg.eigh(G)
                    Z2 = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0)); D = np.sqrt(((Z2[:, None] - Z2[None]) ** 2).sum(-1))
            else:
                C0 = np.stack([np.bincount(lab, Y0[:, 0].astype(np.float64), B), np.bincount(lab, Y0[:, 1].astype(np.float64), B)], 1) / cnt[:, None]
                D = np.sqrt(((C0[:, None] - C0[None]) ** 2).sum(-1))
            if tg != "none":
                C0 = np.stack([np.bincount(lab, Y0[:, 0].astype(np.float64), B), np.bincount(lab, Y0[:, 1].astype(np.float64), B)], 1) / cnt[:, None]
                rms0 = np.sqrt((((C0[:, None] - C0[None]) ** 2).sum(-1)[np.triu_indices(B, 1)]).mean()); alpha = rms0 / np.sqrt((D[np.triu_indices(B, 1)] ** 2).mean())
                target = alpha * D
            else:
                target = np.ones((B, B))
            t = time.time(); Y, S_end = run_pass(Y0, lab, B, target, ratio, log=lambda m: print(m, flush=True)); dt = time.time() - t
            r = metrics(Y); r.update(S_end=S_end, t=dt); out[f"{sname}/B={B}/{tg}/{ratio}"] = r
            print(f"{sname:<22}{B:>5}{tg:>7}{ratio:>6}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{S_end:>8.4f}{dt:>7.1f}", flush=True)
            json.dump(out, open(os.path.join(HERE, "k20_coarse_stress.json"), "w"), indent=1)
