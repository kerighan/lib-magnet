"""Mechanism diagnostic: where does the expected repulsive force on a point come from?
Negative samples are uniform over the n points, so the expected repulsion on point i is
sum_j f(d_ij) with f(d) = 2b / (d (1 + a d^2b)) (UMAP gradient magnitude, far field ~ d^-(2b+1)).
We bin the n-1 partners by octave of distance (relative to the median 1-NN distance of the embedding)
and report the share of the force magnitude per octave and the same for the net vector (after cancellation).
A uniform cloud of dimension m gives a per-octave mass ~ 2^(m-1-2b): flat (scale-free) at b = (m-1)/2."""
import sys, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from datasets import mnist
from magnet import MAGNET
from sklearn.neighbors import NearestNeighbors

X, y, key = mnist(20000)
rng = np.random.default_rng(0)
S = rng.choice(len(X), 1500, replace=False)
for m in (2, 3):
    print(f"== m={m}  (per-octave share of the repulsive force on a point, octaves of d / median 1-NN distance)")
    for b in (0.4, 0.5, 0.6, 0.8, 1.0, 1.5):
        mod = MAGNET(n_components=m, kernel_b=b, random_state=0, n_threads=8).fit(X)
        Y, a = mod.embedding_.astype(np.float64), mod.a_
        d1 = np.median(NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y[S])[0][:, 1])
        D = Y[S, None, :] - Y[None, :, :]
        d = np.sqrt((D ** 2).sum(-1)); d[np.arange(len(S)), S] = np.inf
        f = 2 * b / ((1e-3 + d ** 2) * (1 + a * d ** (2 * b)))     # grad coefficient; force vector = f * (yi - yj)
        oct_ = np.floor(np.log2(d / d1)).astype(int)
        lo, hi = -1, int(oct_[np.isfinite(d)].max())
        mag = np.zeros(hi - lo + 1); net = np.zeros(hi - lo + 1)
        for o in range(lo, hi + 1):
            mask = (oct_ == o) & np.isfinite(d)
            mag[o - lo] = np.where(mask, f * d, 0.0).sum()
            v = (np.where(mask, f, 0.0)[..., None] * np.where(mask[..., None], D, 0.0)).sum(1)              # net vector per sample point from this octave
            net[o - lo] = np.sqrt((v ** 2).sum(-1)).sum()
        mag /= mag.sum(); net /= net.sum()
        sl = lambda arr, k0=2, k1=6: np.polyfit(np.arange(lo + k0, lo + k1 + 1), np.log2(arr[k0:k1 + 1] + 1e-12), 1)[0]
        print(f"b={b:<4} a={a:5.2f}  R/d1=2^{np.log2(np.nanmax(np.where(np.isfinite(d), d, np.nan)) / d1):4.1f}  theory slope={m-1-2*b:+.2f}  measured slope (octaves 1-5): magnitude {sl(mag):+.2f} net {sl(net):+.2f}")
        print("   magnitude share by octave: " + " ".join(f"{v:.3f}" for v in mag))
        print("   net-vector share by octave: " + " ".join(f"{v:.3f}" for v in net))
