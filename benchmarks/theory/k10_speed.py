"""k10: where the time goes and what buys speed. (1) phase profile of the three modes on MNIST 70k (verbose=True);
(2) hybrid: wider FFT boxes, fewer exact iterations at a larger step; (3) exact mode: box width 1.5 for the whole run or
for the late phase only; (4) SGD mode: thread count, attraction exponent (pow), negative rate, mid term, epochs.
Usage: python benchmarks/theory/k10_speed.py [mnist|fmnist|news20|blobs] ; results in k10_speed.log."""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET
ds = sys.argv[1] if len(sys.argv) > 1 else "mnist"
data, labels, name = LOADERS[ds](70000) if ds in ("mnist", "fmnist") else LOADERS[ds]()
n = len(data); kw0 = {"metric": "cosine"} if ds == "news20" else {}

def run(nm, seeds=(0,), **kw):
    for seed in seeds:
        t = time.time(); Y = MAGNET(random_state=seed, **kw0, **kw).fit_transform(data); dt = time.time() - t
        r = evaluate(data, Y, labels, cache_key=name)
        print(f"{ds:7s} {nm:26s} s{seed} {dt:5.1f}s  knn {r['knn15_preservation']:.3f}  trust {r['trustworthiness']:.3f}  sp {r['dist_spearman']:.3f}", flush=True)

print("== profile"); [MAGNET(verbose=True, random_state=0, **kw0, **kw).fit(data) for kw in ({}, {"loss": "tsne"}, {"loss": "hybrid"})]
print("== hybrid: box width, iterations x learning rate")
run("hybrid", (0, 1), loss="hybrid")
for box in (1.25, 1.5, 2.0, 3.0): run(f"hybrid box {box}", (0, 1), loss="hybrid", fft_box=box)
run("hybrid it200 lr n/8", (0, 1), loss="hybrid", hybrid_iter=200, tsne_learning_rate=n / 8)
run("hybrid it150 lr n/6", (0, 1), loss="hybrid", hybrid_iter=150, tsne_learning_rate=n / 6)
run("hybrid it150 lr n/6 box 1.5", (0, 1), loss="hybrid", hybrid_iter=150, tsne_learning_rate=n / 6, fft_box=1.5)
print("== exact mode: box width")
for box in (1.0, 1.5, 2.0): run(f"tsne box {box}", loss="tsne", fft_box=box)
print("== SGD mode: timings only")
for nm, kw in [("b0.6 t20", {}), ("b0.6 t10", {"n_threads": 10}), ("b1.0 (no pow)", {"kernel_b": 1.0}), ("b0.5", {"kernel_b": 0.5}),
               ("neg 3", {"negative_sample_rate": 3}), ("mid 0", {"mid_weight": 0.0}), ("200 epochs", {"n_epochs": 200})]:
    ts = []
    for _ in range(2):
        t = time.time(); MAGNET(random_state=0, **kw0, **kw).fit(data); ts.append(time.time() - t)
    print(f"{ds:7s} SGD {nm:16s} {min(ts):.2f}s", flush=True)
