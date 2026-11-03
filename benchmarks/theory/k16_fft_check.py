"""k16: accuracy of the FFT-interpolated repulsion against the exact all-pairs sum, on the hybrid's own layouts
(MNIST 70k hybrid layout of k14, and subsamples): relative error of sum_Q and of the per-point force (RMS and max,
relative to the RMS force), at the default grid (3 interpolation points per box, box width 1) and at box width 1.5.
Exact sums in float64, chunked. Usage: python benchmarks/theory/k16_fft_check.py"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from magnet.tsne import FFTRepulsion2D

def exact(Y, chunk=500):
    Y = Y.astype(np.float64); n = len(Y); F = np.zeros_like(Y); sumQ = 0.0
    for s in range(0, n, chunk):
        D = Y[s:s + chunk, None, :] - Y[None, :, :]; d2 = (D ** 2).sum(-1); q = 1.0 / (1.0 + d2)
        q[np.arange(min(chunk, n - s)), np.arange(s, min(s + chunk, n))] = 0.0
        sumQ += q.sum(); F[s:s + chunk] = (q[:, :, None] ** 2 * D).sum(1)
    return sumQ, F

Yfull = np.load(os.path.join(HERE, "..", "cache", "k14_hybrid_default.npz"))["Y"].astype(np.float32)
out = {}
for n in (5000, 20000, 70000):
    Y = Yfull[np.random.default_rng(0).choice(len(Yfull), n, replace=False)] if n < len(Yfull) else Yfull
    t = time.time(); sQ, F = exact(Y); te = time.time() - t
    for bw, ni in ((1.0, 3), (1.5, 3), (1.0, 5)):
        rep = FFTRepulsion2D(n_interp=ni, box_width=bw, n_threads=-1)
        t = time.time(); sQf, Ff = rep(Y); tf = time.time() - t
        err = np.sqrt(((Ff - F) ** 2).sum(1)); rms = np.sqrt((F ** 2).sum(1).mean())
        r = dict(n=n, box_width=bw, n_interp=ni, sumQ_rel_err=abs(sQf - sQ) / sQ, force_rms_rel_err=float(np.sqrt((err ** 2).mean()) / rms),
                 force_max_rel_err=float(err.max() / rms), grad_cos=float((Ff * F).sum() / np.sqrt((Ff ** 2).sum() * (F ** 2).sum())), t_exact=te, t_fft=tf, boxes=rep._grid(Y)[3])
        out[f"{n}/{bw}/{ni}"] = r; print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(HERE, "k16_fft_check.json"), "w"), indent=1)
