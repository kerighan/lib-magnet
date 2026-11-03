"""k22: conditional KL by blocks (idea 8 of the GPT-6 note), exact pilot on a 5k MNIST subset.
Chain rule: KL(P||Q) = KL(P_b||Q_b) + sum_b P_b KL(P(.|b)||Q(.|b)) over blocks b = (cell(i), cell(j)). The conditional
objective keeps the second term (neighbourhood organisation inside and between cells) and replaces the block-mass term
by a coarse rest-length stress on the cell barycentres (target: 2-D classical MDS of the cells' input means). Gradient:
the repulsion coefficient of pair (i, j) becomes P_b / Z_b (Z_b = sum of the kernel over the block) instead of 1 / Z.
Compared, from the same start (stiff SGD layout, scale 31 sqrt(n/70000)), 600 exact iterations with t-SNE's momentum and
gains: (1) KL; (2) KL + lambda S; (3) conditional KL + lambda S; (4) conditional KL with floored block masses + lambda S.
B = 16 label-free cells. Metrics every 100 iterations. -> k22_conditional_kl.json / .log. Usage: python benchmarks/theory/k22_conditional_kl.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from magnet import MAGNET
from magnet.spectral import spectral_init
from k17_virial_projected import prep

n = 5000
X, y, key = mnist(n); y = np.asarray(y).astype(int); metrics = prep(X, y, key)
m = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float64); P = P / P.sum()
Pd = P.toarray(); np.fill_diagonal(Pd, 0.0)
Y0 = m.embedding_.astype(np.float64); Y0 -= Y0.mean(0); Y0 *= 31.0 * np.sqrt(n / 70000.0) / Y0[:, 0].std()
B = 16
S8 = np.asarray(spectral_init(sp.csr_matrix(m.graph_, dtype=np.float32), 8, random_state=0, n_threads=-1), dtype=np.float64)
lab = KMeans(B, random_state=0, n_init=5).fit_predict(S8); cnt = np.bincount(lab, minlength=B).astype(np.float64)
blk = lab[:, None] * B + lab[None, :]                       # block index of every pair
Pb = np.bincount(blk.ravel(), Pd.ravel(), B * B).reshape(B, B)   # block masses
print("block masses: min %.2e, zeros %d of %d, diagonal share %.2f" % (Pb.min(), (Pb == 0).sum(), B * B, np.trace(Pb)))
mu = np.stack([np.bincount(lab, X[:, d], B) for d in range(X.shape[1])], 1) / cnt[:, None]
D = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(-1)); J = np.eye(B) - 1.0 / B; G = -0.5 * J @ (D ** 2) @ J; w, V = np.linalg.eigh(G)
Z2 = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0)); Dmds = np.sqrt(((Z2[:, None] - Z2[None]) ** 2).sum(-1)); iu = np.triu_indices(B, 1)

def coarse(Y):
    C = np.stack([np.bincount(lab, Y[:, 0], B), np.bincount(lab, Y[:, 1], B)], 1) / cnt[:, None]
    return C, np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1)) + 1e-9
def stress_grad(Y, target):
    C, dist = coarse(Y); delta = 0.05 * target[iu].mean(); den = target + delta; resid = (dist - target) / den
    S = float((resid[iu] ** 2).mean()); coef = 2.0 / len(iu[0]) * resid / den / dist; np.fill_diagonal(coef, 0.0)
    Gc = (coef[:, :, None] * (C[:, None, :] - C[None, :, :])).sum(1)
    return S, Gc[lab] / cnt[lab][:, None]

def kl_grad(Y, mode, Pb_eff):
    Dif = Y[:, None, :] - Y[None, :, :]; k = 1.0 / (1.0 + (Dif ** 2).sum(-1)); np.fill_diagonal(k, 0.0)
    if mode == "kl":
        coef_rep = k / k.sum()
        obj = float((Pd[Pd > 0] * np.log(Pd[Pd > 0])).sum() - (Pd * np.log(k + np.eye(n))).sum() + np.log(k.sum()))
    else:
        Zb = np.bincount(blk.ravel(), k.ravel(), B * B).reshape(B, B) + 1e-300
        coef_rep = (Pb_eff / Zb)[lab[:, None], lab[None, :]] * k
        obj = float((Pd[Pd > 0] * np.log(Pd[Pd > 0])).sum() - (Pd * np.log(k + np.eye(n))).sum() + (Pb_eff * np.log(Zb)).sum() - (Pb_eff[Pb_eff > 0] * np.log(Pb_eff[Pb_eff > 0])).sum())
    g = 4.0 * (((Pd - coef_rep) * k)[:, :, None] * Dif).sum(1)
    return obj, g

def run(mode, ratio, floor=0.0, n_iter=600, log=None):
    Pb_eff = Pb.copy()
    if floor > 0:
        Pb_eff = np.maximum(Pb_eff, floor / (B * B)); Pb_eff /= Pb_eff.sum()
    Y = Y0.copy(); upd = np.zeros_like(Y); gains = np.ones_like(Y); lr = max(200.0, n / 12.0)
    C0, dist0 = coarse(Y0); alpha = np.sqrt((dist0[iu] ** 2).mean()) / np.sqrt((Dmds[iu] ** 2).mean()); target = alpha * Dmds
    lam = None; trace = []
    for it in range(n_iter):
        obj, g = kl_grad(Y, mode, Pb_eff)
        if ratio > 0:
            S, gS = stress_grad(Y, target)
            if lam is None:
                cellavg = np.stack([np.bincount(lab, g[:, 0], B), np.bincount(lab, g[:, 1], B)], 1) / cnt[:, None]
                _, gS_cal = stress_grad(1.1 * Y, target)
                lam = ratio * np.sqrt((cellavg[lab] ** 2).sum()) / max(np.sqrt((gS_cal ** 2).sum()), 1e-12)
            g = g + lam * gS
        else:
            S = float("nan")
        gains = np.where(np.sign(g) != np.sign(upd), gains + 0.2, gains * 0.8); gains = np.maximum(gains, 0.01)
        upd = 0.8 * upd - lr * gains * g; Y += upd; Y -= Y.mean(0)
        if (it + 1) % 100 == 0:
            r = metrics(Y.astype(np.float32)); r.update(it=it + 1, obj=obj, S=S); trace.append(r)
            if log: log(f"{mode + ('+floor' if floor else ''):<10}{ratio:>5}{it + 1:>5}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{obj:>9.4f}{S:>8.3f}{r['std']:>7.1f}")
    return trace

out = {"start": metrics(Y0.astype(np.float32))}
r0 = out["start"]; print(f"start: {r0['knn15']:.3f}/{r0['sp']:.3f}/{r0['trip']:.3f}/{r0['cent']:.3f}")
print(f"{'objective':<10}{'ratio':>5}{'it':>5}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'obj':>9}{'S':>8}{'std':>7}")
for mode, ratio, floor in (("kl", 0.0, 0.0), ("kl", 1.0, 0.0), ("kl", 3.0, 0.0), ("cond", 1.0, 0.0), ("cond", 3.0, 0.0), ("cond", 1.0, 0.1), ("cond", 3.0, 0.1), ("cond", 0.0, 0.1)):
    t = time.time(); out[f"{mode}/{ratio}/{floor}"] = run(mode, ratio, floor, log=lambda s: print(s, flush=True)); print(f"   ({time.time() - t:.0f} s)", flush=True)
    json.dump(out, open(os.path.join(HERE, "k22_conditional_kl.json"), "w"), indent=1)
