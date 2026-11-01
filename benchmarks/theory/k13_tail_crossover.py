"""k13: per-octave repulsive force on a FIXED layout, binned by the kernel crossover d_kr = a^(-1/2 b_r)
instead of the median 1-NN distance (e1 / tail_force.log used d / d1 and fitted octaves 1-5, which sit
below the crossover for every b_r: there the force is ~ 2 b_r / d whatever b_r, so the slope cannot
depend on b_r).

Protocol (reviewer's): 256 anchors, distances to every point; per octave of d / d_kr: pair count N,
total scalar force S, mean scalar force per pair, norm of the vector sum; every b_r evaluated on the
same coordinates; fit window a d^{2 b_r} > 100 and d < R / 2 with at least 20 pairs per anchor per
octave; theory slope m - 1 - 2 b_r; shell-count-adjusted prediction
alpha(d) - 1 - 2 b_r a d^{2b_r} / (1 + a d^{2b_r}) with alpha = d log N / d log d measured in the window.
The layout is rescaled by s (equivalent to shrinking a) to widen the asymptotic window.
Also reports the old-protocol slope (octaves 1-5 of d / d1) for reference."""
import sys, os, json, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET, find_ab_params
from datasets import mnist
from sklearn.neighbors import NearestNeighbors

OUT = os.path.join(REPO, "benchmarks", "theory", "k13_tail_crossover.json")
rows = []


def spectrum(Y, label, br_list=(0.5, 1.0, 1.5, 2.0), scales=(1, 4, 16, 64), n_anchor=256, seed=0):
    n, m = Y.shape
    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, n_anchor, replace=False)
    diff = Y[anchors][:, None, :] - Y[None, :, :]                  # (A, n, m)
    D = np.sqrt((diff ** 2).sum(-1))                                # (A, n)
    self_mask = np.zeros_like(D, dtype=bool); self_mask[np.arange(n_anchor), anchors] = True
    D[self_mask] = np.inf
    d1 = np.median(NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y[:5000])[0][:, 1])
    R = np.sqrt(((Y - Y.mean(0)) ** 2).sum(1)).max()
    print(f"== {label}  m={m} n={n}  R/d1=2^{np.log2(R/d1):.1f}  (window: a d^2br > 100, d < R/2, >= 20 pairs/anchor/octave)")
    print(f"{'s':>3} {'b_r':>4} {'a':>5} {'window log2(d/dkr)':>20} {'oct':>3} {'theory':>7} {'adj.pred':>8} {'meas.mag':>8} {'meas.net':>8} {'alpha':>6} {'old(d/d1 oct1-5)':>16}")
    for s in scales:
        Ds = D * s; R_s = R * s; d1_s = d1 * s
        for br in br_list:
            a, _ = find_ab_params(1.0, 0.05, br, "power")
            dkr = a ** (-1.0 / (2 * br))
            f = 2 * br / (Ds * (1 + a * Ds ** (2 * br)))             # scalar force, (A, n), 0 at self (inf)
            x = np.log2(Ds / dkr)
            lo = -12; nb = 32
            o = np.clip(np.floor(x).astype(int) - lo, 0, nb - 1); o[self_mask] = 0
            fo = f.copy(); fo[self_mask] = 0
            cnt = np.bincount(o.ravel(), minlength=nb).astype(float); cnt[0] -= n_anchor
            S = np.bincount(o.ravel(), weights=fo.ravel(), minlength=nb) / n_anchor
            # vector sum per anchor per octave
            unit = (fo / np.where(np.isfinite(Ds), Ds, 1.0))[..., None] * np.where(np.isfinite(Ds)[..., None], diff * s, 0.0)
            flat = (np.arange(n_anchor)[:, None] * nb + o).ravel()
            net = np.zeros((n_anchor, nb))
            for k in range(m):
                net += np.bincount(flat, weights=unit[..., k].ravel(), minlength=n_anchor * nb).reshape(n_anchor, nb) ** 2
            net = np.sqrt(net).mean(0)
            centers = lo + np.arange(nb) + 0.5                        # log2(d / dkr)
            d_c = dkr * 2.0 ** centers
            d_lo = dkr * 2.0 ** (lo + np.arange(nb))
            win = (a * d_lo ** (2 * br) > 100) & (dkr * 2.0 ** (lo + np.arange(nb) + 1) <= R_s / 2) & (cnt / n_anchor >= 20)
            ks = np.where(win)[0]
            def slope(v):
                return float(np.polyfit(centers[ks], np.log2(v[ks] + 1e-30), 1)[0]) if len(ks) >= 2 else float("nan")
            sl_mag, sl_net, alpha = slope(S), slope(net), slope(cnt)
            adj = alpha - 1 - np.mean(2 * br * a * d_c[ks] ** (2 * br) / (1 + a * d_c[ks] ** (2 * br))) if len(ks) >= 2 else float("nan")
            # old protocol: octaves 1-5 of d / d1, magnitude slope
            xo = np.log2(Ds / d1_s); oo = np.clip(np.floor(xo).astype(int) + 6, 0, 39); oo[self_mask] = 0
            So = np.bincount(oo.ravel(), weights=fo.ravel(), minlength=40) / n_anchor
            ko = np.arange(7, 12)
            sl_old = float(np.polyfit(ko, np.log2(So[ko] + 1e-30), 1)[0]) if (So[ko] > 0).all() else float("nan")
            wtxt = f"[{centers[ks[0]]-0.5:+.0f},{centers[ks[-1]]+0.5:+.0f}]" if len(ks) else "none"
            print(f"{s:>3} {br:>4} {a:>5.2f} {wtxt:>20} {len(ks):>3} {m-1-2*br:>+7.2f} {adj:>+8.2f} {sl_mag:>+8.2f} {sl_net:>+8.2f} {alpha:>+6.2f} {sl_old:>+16.2f}")
            rows.append(dict(layout=label, m=m, s=s, br=br, a=a, dkr=dkr, window=[int(centers[ks[0]]-0.5), int(centers[ks[-1]]+0.5)] if len(ks) else None,
                             n_oct=int(len(ks)), theory=m-1-2*br, adjusted=adj, slope_mag=sl_mag, slope_net=sl_net, alpha=alpha, slope_old=sl_old,
                             centers=centers.tolist(), S=S.tolist(), net=net.tolist(), cnt=cnt.tolist()))


X, y, key = mnist(20000)
for m in (2, 3):
    Y = MAGNET(n_components=m, n_threads=4, random_state=0).fit_transform(X)
    spectrum(Y, f"MNIST 20k {m}-D (defaults)")
rng = np.random.default_rng(0)
for m in (2, 3):
    spectrum((rng.normal(size=(20000, m)) * 3).astype(np.float32), f"Gaussian cloud {m}-D")
json.dump(rows, open(OUT, "w"))
print("saved", OUT)
