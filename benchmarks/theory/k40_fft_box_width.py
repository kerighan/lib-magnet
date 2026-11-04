"""k40: the FFT interpolation error is the accuracy floor of the full-batch pass. Found while re-timing after the
optimisation round: the fixed-width box (commit 1a319f7, FIt-SNE's convention) had cost 15-NN on the blobs
(0.126 -> 0.118) and 0.001 on MNIST; the old grid (width = extent / n_boxes, changing every iteration) was an
accidental middle ground. (a) force error of the FFT repulsion against the exact O(n^2) sum on a fixed layout for
several (box width, interpolation points); (b) 15-NN / Spearman of the full-batch mode along the box width on the
blobs (3 seeds) and MNIST (2 seeds), with time at 8 threads; (c) the same for n_interp 4 at width 1 and 1.3.
Result: width 0.7 is the knee (MNIST 0.346 -> 0.350 = Barnes-Hut's, blobs 0.118 -> 0.133; 0.5 adds nothing; the
finer grid matters over the whole late phase, not only its end), for ~15 % more time; it is the default since.
Usage: python benchmarks/theory/k40_fft_box_width.py [part=error,width,interp]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from datasets import LOADERS
from metrics import evaluate
import magnet.tsne as T
from magnet import MAGNET
T8 = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); PART = kw.get("part", "error,width,interp").split(",")
OUT = os.path.join(HERE, "k40_fft_box_width.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def exact(Y):
    n = len(Y); F = np.zeros((n, 2)); sumQ = 0.0
    for i0 in range(0, n, 1000):
        D = Y[i0:i0 + 1000, None, :] - Y[None, :, :]; r2 = (D ** 2).sum(-1); K1 = 1.0 / (1.0 + r2)
        K1[np.arange(i0, min(i0 + 1000, n)) - i0, np.arange(i0, min(i0 + 1000, n))] = 0.0
        sumQ += K1.sum(); F[i0:i0 + 1000] = ((K1 ** 2)[..., None] * D).sum(1)
    return sumQ, F

if "error" in PART:
    print("(a) force error of the FFT repulsion on a fixed layout (MNIST full-batch layout, first 20k points)")
    Y = np.load(os.path.join(B, "cache", "layouts", "mnist_fb_seed0.npz"))["Y"].astype(np.float32)[:20000]
    sumQ, F = exact(Y); Frms = np.sqrt((F ** 2).sum(1).mean())
    for width, ni in ((1.0, 3), (0.8, 3), (0.7, 3), (0.5, 3), (1.0, 4), (1.3, 4), (1.0, 5)):
        rep = T.FFTRepulsion2D(n_interp=ni, box_width=width, n_threads=T8); t = time.time()
        for _ in range(10): sq, Ff = rep(Y)
        dt = (time.time() - t) / 10; Ff = np.asarray(Ff)
        r = dict(sumQ_err=abs(sq - sumQ) / sumQ, F_rms_err=float(np.sqrt(((Ff - F) ** 2).sum(1).mean()) / Frms), ms=dt * 1e3)
        r = {k: float(v) for k, v in r.items()}; res[f"error/{width}/{ni}"] = r; print(f"  width {width} n_interp {ni}: sumQ err {r['sumQ_err']:.1e}  force rms err {r['F_rms_err']:.1e}  {r['ms']:.1f} ms/call", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

def run(ds, X, y, name, seeds, width, ni, label):
    rows = []
    for s in seeds:
        k_ = f"{label}/{ds}/{width}/{ni}/{s}"
        if k_ not in res:
            t = time.time(); Y = MAGNET(loss="tsne", random_state=s, n_threads=T8, fft_box=width, fft_interp=ni).fit_transform(X); dt = time.time() - t
            e = evaluate(X, Y, y, cache_key=name); res[k_] = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], t=dt); json.dump(res, open(OUT, "w"), indent=1)
        rows.append(res[k_])
    print(f"  {ds:<6} width {width} n_interp {ni}: knn15 {np.mean([r['knn15'] for r in rows]):.4f} +- {np.std([r['knn15'] for r in rows]):.4f}  sp {np.mean([r['sp'] for r in rows]):.3f}  {np.mean([r['t'] for r in rows]):.1f} s", flush=True)

if "width" in PART:
    print("(b) full-batch mode along the box width (8 threads)")
    for ds, seeds in (("blobs", (0, 1, 2)), ("mnist", (0, 1))):
        X, y, name = LOADERS[ds](70000) if ds == "mnist" else LOADERS[ds]()
        for width in (1.0, 0.8, 0.7, 0.6, 0.5): run(ds, X, y, name, seeds, width, 3, "width")
if "interp" in PART:
    print("(c) more interpolation points instead of narrower boxes")
    for ds, seeds in (("blobs", (0, 1, 2)), ("mnist", (0, 1))):
        X, y, name = LOADERS[ds](70000) if ds == "mnist" else LOADERS[ds]()
        for width, ni in ((1.0, 4), (1.3, 4), (1.0, 5)): run(ds, X, y, name, seeds, width, ni, "interp")
