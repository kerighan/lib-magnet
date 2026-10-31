"""E1b: pure far-field regime (cloud scaled so that the kernel scale is far below the 1-NN distance):
per-octave NET force vs octave for the power-law repulsion f = 2b/(a d^(2b+1)); theory A predicts slope m-2b, theory B m-1-2b."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng(0)
def run(Y, label, bs=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    n, m = Y.shape
    d1 = np.median(NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y[:5000])[0][:, 1])
    R = np.sqrt(((Y - Y.mean(0)) ** 2).sum(1)).max(); K = int(np.log2(R / d1)) + 2
    idx = rng.choice(n, 1500, replace=False)
    print(f"== {label} m={m} octaves available {np.log2(R/d1):.1f}")
    for b in bs:
        net2 = np.zeros(K + 8); mag = np.zeros(K + 8); cnt = np.zeros(K + 8)
        for i in idx:
            diff = Y[i] - Y; d = np.sqrt((diff ** 2).sum(1)); d[i] = np.inf
            f = 2 * b / d ** (2 * b + 1); f[i] = 0
            o = np.clip(np.floor(np.log2(d / d1)).astype(int) + 3, 0, K + 7); o[i] = 0
            v = (f / np.maximum(d, 1e-12))[:, None] * diff
            s = np.zeros(K + 8)
            for k in range(m): s += np.bincount(o, weights=v[:, k], minlength=K + 8) ** 2
            net2 += s; mag += np.bincount(o, weights=f, minlength=K + 8); cnt += np.bincount(o, minlength=K + 8)
        net = np.sqrt(net2 / len(idx)); mag /= len(idx)
        ks = np.array([k for k in range(K + 8) if cnt[k] > 50 and 2.0 ** (k - 3) >= 4 and 2.0 ** (k - 3) * d1 <= R / 4])   # 4 d1 .. R/4
        sn = np.polyfit(ks, np.log2(net[ks]), 1)[0]; sm = np.polyfit(ks, np.log2(mag[ks]), 1)[0]
        print(f"b={b:<4} net slope {sn:+.2f} (A: m-2b={m-2*b:+.1f})   magnitude slope {sm:+.2f} (B: m-1-2b={m-1-2*b:+.1f})   octaves used {len(ks)}")
for m in (2, 3):
    run(rng.normal(size=(40000, m)), f"Gaussian cloud {m}-D")
    run(rng.uniform(-1, 1, size=(40000, m)), f"uniform cube {m}-D")
