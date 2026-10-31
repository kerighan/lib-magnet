"""E1: net repulsive force per octave on a FIXED embedding, as a function of the repulsion exponent.
Theory A (density gradient): per-octave net force ~ d^(m - 2b)  -> flat at b = m/2.
Theory B (magnitude):        per-octave magnitude ~ d^(m - 1 - 2b) -> flat at b = (m-1)/2."""
import sys, os, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET, find_ab_params
from datasets import mnist
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng(0)

def spectrum(Y, bs, n_sample=1500, label=""):
    n, m = Y.shape
    d1 = np.median(NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y[:5000])[0][:, 1])
    R = np.sqrt(((Y - Y.mean(0)) ** 2).sum(1)).max()
    idx = rng.choice(n, n_sample, replace=False)
    print(f"== {label}  m={m} n={n}  R/d1=2^{np.log2(R/d1):.1f}")
    for b in bs:
        a, _ = find_ab_params(1.0, 0.05, b, "power")
        net = np.zeros(40); mag = np.zeros(40); cnt = np.zeros(40)
        for i in idx:
            diff = Y[i] - Y                                  # push away from j: direction (y_i - y_j)
            d = np.sqrt((diff ** 2).sum(1)); d[i] = np.inf
            f = 2 * b / (d * (1 + a * d ** (2 * b)))          # force magnitude of the power kernel's repulsion
            o = np.clip(np.floor(np.log2(d / d1)).astype(int) + 6, 0, 39); o[i] = 0
            f[i] = 0
            v = (f / np.maximum(d, 1e-12))[:, None] * diff   # force vectors
            for k in range(m):
                net_k = np.bincount(o, weights=v[:, k], minlength=40)
                net += net_k ** 2 if k == 0 else net_k ** 2   # accumulate squares per component, sqrt below
            mag += np.bincount(o, weights=f, minlength=40); cnt += np.bincount(o, minlength=40)
        # net: we summed squared components per octave for each i -> this is sum_i |net_i,o|^2; take sqrt of mean
        net = np.sqrt(net / n_sample); mag /= n_sample
        far = [k for k in range(40) if cnt[k] > 0 and (2.0 ** (k - 6)) * d1 >= 2.0 and (2.0 ** (k - 6)) * d1 <= R / 2]
        ks = np.array(far); sl_net = np.polyfit(ks, np.log2(net[ks] + 1e-30), 1)[0] if len(ks) > 2 else np.nan
        sl_mag = np.polyfit(ks, np.log2(mag[ks] + 1e-30), 1)[0] if len(ks) > 2 else np.nan
        shares = net / net.sum()
        occ = [k for k in range(40) if cnt[k] > 0]
        print(f"b={b:<4} a={a:.2f} far-field octaves d in [2, R/2]: net slope {sl_net:+.2f} (theory A m-2b = {m-2*b:+.1f}); magnitude slope {sl_mag:+.2f} (theory B m-1-2b = {m-1-2*b:+.1f})")
        print("   net share by octave (d/d1 = 2^k, k from", occ[0]-6, "):", " ".join(f"{shares[k]:.3f}" for k in occ))

bs = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
X, y, key = mnist(20000)
for m in (2, 3):
    Y = MAGNET(n_components=m, n_threads=4, random_state=0).fit_transform(X)
    spectrum(Y, bs, label=f"MNIST 20k embedding {m}-D (defaults)")
for m in (2, 3):
    G = rng.normal(size=(20000, m)).astype(np.float32) * 3
    spectrum(G, bs, label=f"Gaussian cloud {m}-D")
