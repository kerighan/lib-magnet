"""k23: a coarse-metric budget instead of frozen coordinates (idea 4 of the GPT-6 note). Constraint G(Y) <= eps^2 with
G = mean over cell pairs of ((|c_a - c_b| - l_ab) / (l_ab + delta))^2, l_ab the start's own coarse distances (B = 64
label-free cells). Each iteration: KL gradient step with momentum and gains; if G > eps^2 afterwards, a normal correction
along -grad G (bisection on the step so that G = eps^2, at most 6 evaluations, O(n + B^2) each). Budgets eps = 2 / 5 / 10 %
RMS; 300 and 1000 iterations; against the unconstrained pass, the penalty (k20, ratio 1) and the projection (k17).
MNIST 70k, stiff start, all cores. -> k23_metric_budget.json / .log. Usage: python benchmarks/theory/k23_metric_budget.py"""
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

X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key)
m = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float32); Pn = P / P.sum()
indptr, indices, data = Pn.indptr.astype(np.int32), Pn.indices.astype(np.int32), Pn.data.astype(np.float32)
Y0 = m.embedding_.astype(np.float32); Y0 -= Y0.mean(0); Y0 *= 31.0 / max(float(Y0[:, 0].std()), 1e-12)
B = 64; S8 = np.asarray(spectral_init(P, 8, random_state=0, n_threads=-1), dtype=np.float64)
lab = MiniBatchKMeans(B, random_state=0, n_init=3, batch_size=4096).fit_predict(S8); cnt = np.bincount(lab, minlength=B).astype(np.float64); iu = np.triu_indices(B, 1)
def coarse(Y):
    C = np.stack([np.bincount(lab, Y[:, 0].astype(np.float64), B), np.bincount(lab, Y[:, 1].astype(np.float64), B)], 1) / cnt[:, None]
    return C, np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1)) + 1e-9
C0, L = coarse(Y0); delta = 0.05 * L[iu].mean(); den = L + delta
def G_and_grad(Y):
    C, dist = coarse(Y); resid = (dist - L) / den; G = float((resid[iu] ** 2).mean())
    coef = 2.0 / len(iu[0]) * resid / den / dist; np.fill_diagonal(coef, 0.0)
    Gc = (coef[:, :, None] * (C[:, None, :] - C[None, :, :])).sum(1)
    return G, (Gc[lab] / cnt[lab][:, None]).astype(np.float32)

def run(eps, n_iter, log):
    Y = Y0.copy(); upd = np.zeros_like(Y); gains = np.ones_like(Y); rep = FFTRepulsion2D(n_interp=3, box_width=1.0, n_threads=-1); lr = max(200.0, n / 12.0)
    corrections = 0; t0 = time.time(); trace = {}
    for it in range(n_iter):
        attr = np.asarray(tsne_attraction(indptr, indices, data, Y, 1.0, -1)); sumQ, F = rep(Y)
        g = attr - F * (1.0 / max(sumQ, 1e-12))
        if eps is not None:
            G, gG = G_and_grad(Y)
            if G >= 0.8 * eps ** 2:     # near the boundary: remove the outward component of the step (tangent projection)
                dot = float((g * gG).sum())
                if dot < 0: g = g - (dot / max(float((gG ** 2).sum()), 1e-30)) * gG   # descent direction -g must not increase G
        tsne_update(Y, upd, gains, np.ascontiguousarray(g, dtype=np.float32), float(lr), 0.8, -1)
        if eps is not None:
            G, gG = G_and_grad(Y)
            if G > eps ** 2:            # normal correction: Y <- Y - t gG with G(Y) = eps^2 (bisection on t)
                corrections += 1; nrm = float((gG ** 2).sum()); t_hi = (G - eps ** 2) / max(nrm, 1e-30) * 2.0; lo, hi = 0.0, t_hi
                for _ in range(6):
                    if G_and_grad(Y - hi * gG)[0] > eps ** 2: lo, hi = hi, hi * 2
                    else: break
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    if G_and_grad(Y - mid * gG)[0] > eps ** 2: lo = mid
                    else: hi = mid
                Y -= hi * gG
        if (it + 1) in (300, 1000):
            r = metrics(Y); r.update(t=time.time() - t0, corrections=corrections, G=G_and_grad(Y)[0]); trace[it + 1] = r
            log(f"{'none' if eps is None else f'{eps:.2f}':>6}{it + 1:>6}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{np.sqrt(r['G']):>8.3f}{corrections:>6}{r['t']:>7.1f}")
    return trace

r0 = metrics(Y0); print(f"start: {r0['knn15']:.3f}/{r0['sp']:.3f}/{r0['trip']:.3f}/{r0['cent']:.3f}")
print(f"{'eps':>6}{'it':>6}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'rmsG':>8}{'corr':>6}{'t(s)':>7}")
out = {"start": r0}
for eps in (None, 0.02, 0.05, 0.10, 0.20):
    out[str(eps)] = run(eps, 1000, lambda s: print(s, flush=True))
    json.dump(out, open(os.path.join(HERE, "k23_metric_budget.json"), "w"), indent=1)
