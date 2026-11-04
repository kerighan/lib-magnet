"""k45: the SGD mode at small n (after k44 showed it fragmenting USPS and phonemes into 30-60 HDBSCAN blobs while UMAP
gives 8). Cause: the attraction exponent kernel_b = 0.6 (force ~ d^(2b-1): steep at short range, so neighbours glue
into tight clumps that only the graph's redundancy knits together, which small data sets lack). (a) MNIST subsamples
3k-35k x kernel_b 0.6-0.9 with UMAP as the reference: HDBSCAN ARI / clusters / noise, 15-NN, Spearman; (b) the
shipped rule kernel_b="size" (0.6 at n >= 20k, 0.8 at n <= 10k, log-linear between) on USPS, phonemes, MNIST 5k /
10k / 20k; neither "adaptive" nor "local" nor n_epochs, learning rate, n_neighbors, threads or the coupled kernel
fixes it (tested, see the log). Usage: python benchmarks/theory/k45_small_n.py [part=scan,rule]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
PYB = "/home/maixent/Dropbox (Reputation Squad)/DATAPROJECTS/lib/pybench"; sys.path.insert(0, PYB)
import numpy as np, hdbscan, h5py
from sklearn.metrics import adjusted_rand_score
from metrics import evaluate
from datasets import mnist
from magnet import MAGNET
kw = dict(a.split("=", 1) for a in sys.argv[1:]); PART = kw.get("part", "scan,rule").split(",")
OUT = os.path.join(HERE, "k45_small_n.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
Xm, ym, _ = mnist(70000); ym = np.asarray(ym).astype(int); rng = np.random.default_rng(0)
def sets():
    with h5py.File(os.path.join(PYB, "pybench", "clustering", "datasets", "usps.h5"), "r") as hf: yield "usps 7k", np.asarray(hf["train"]["data"][:], dtype=np.float32), np.asarray(hf["train"]["target"][:]).astype(int)
    from pybench.clustering import datasets as D
    Xp, yp = D.load("phonemes")[:2]; yield "phonemes 4.5k", np.ascontiguousarray(np.asarray(Xp, dtype=np.float32)), np.asarray(yp).astype(int)
    for n in (5000, 10000, 20000):
        idx = np.sort(np.random.default_rng(0).choice(70000, n, replace=False)); yield f"mnist {n}", np.ascontiguousarray(np.asarray(Xm[idx], dtype=np.float32)), ym[idx]   # same draw as the scan (same evaluate cache key)
def score(X, y, Y, key):
    mcs = max(15, len(y) // 200); cl = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(Y); noise = cl < 0; la = cl.copy(); la[noise] = cl.max() + 1
    e = evaluate(X, Y, y, cache_key=key)
    return dict(ari=float(adjusted_rand_score(y, la)), clusters=int(len(set(cl[~noise]))), noise=float(noise.mean()), knn15=e["knn15_preservation"], sp=e["dist_spearman"])
def show(label, r, dt): print(f"  {label:<16} HDBSCAN ARI {r['ari']:.3f} ({r['clusters']:>2} clusters, noise {r['noise']:.2f})  knn15 {r['knn15']:.3f}  sp {r['sp']:.3f}  {dt:.1f} s", flush=True)
if "scan" in PART:
    import umap
    for n in (3000, 5000, 10000, 20000, 35000):
        idx = np.sort(np.random.default_rng(0).choice(70000, n, replace=False)); X = np.ascontiguousarray(np.asarray(Xm[idx], dtype=np.float32)); y = ym[idx]; key = f"k45_mnist_{n}"
        print(f"\nMNIST subsample n={n}", flush=True)
        for label, f in [("UMAP", lambda: umap.UMAP(random_state=0).fit_transform(X))] + [(f"kernel_b={b}", (lambda b=b: MAGNET(kernel_b=b, random_state=0, n_threads=8).fit_transform(X))) for b in (0.6, 0.7, 0.8, 0.9)]:
            k_ = f"scan/{n}/{label}"
            if k_ not in res: t = time.time(); Y = f(); dt = time.time() - t; res[k_] = score(X, y, Y, key); res[k_]["t"] = dt; json.dump(res, open(OUT, "w"), indent=1)
            show(label, res[k_], res[k_]["t"])
if "rule" in PART:
    print('\n(b) the shipped default kernel_b="size" against 0.6, 3 seeds', flush=True)
    for name, X, y in sets():
        key = f"k45_{name.replace(' ', '_')}"; print(f"\n{name}")
        for label, kw_ in (("size (default)", {}), ("0.6", dict(kernel_b=0.6))):
            rows = []
            for s in range(3):
                k_ = f"rule/{name}/{label}/{s}"
                if k_ not in res: t = time.time(); m = MAGNET(random_state=s, n_threads=8, **kw_); Y = m.fit_transform(X); dt = time.time() - t; res[k_] = score(X, y, Y, key); res[k_]["t"] = dt; res[k_]["b"] = float(m.kernel_b_); json.dump(res, open(OUT, "w"), indent=1)
                rows.append(res[k_])
            r = {a: float(np.mean([x[a] for x in rows])) for a in ("ari", "clusters", "noise", "knn15", "sp")}; show(f"{label} (b={rows[0]['b']:.2f})", r, np.mean([x["t"] for x in rows]))
