"""Scaling study on synthetic data: python benchmarks/scaling.py METHOD N [d=50] [clusters=50]
Prints one JSON line with per-phase timings, peak RSS and quality metrics (subsampled)."""
import sys, os, time, json, resource, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np


def synthetic(n, d=50, clusters=50, intrinsic=8, seed=0):
    """Anisotropic Gaussian clusters living on random `intrinsic`-dimensional subspaces of R^d."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, clusters, n)
    centers = rng.normal(0, 6, (clusters, d)).astype(np.float32)
    X = np.empty((n, d), dtype=np.float32)
    for c in range(clusters):
        idx = np.where(y == c)[0]
        basis = rng.normal(size=(intrinsic, d)).astype(np.float32)
        scales = (rng.uniform(0.5, 2.0, intrinsic) * np.exp(-np.arange(intrinsic) / 4)).astype(np.float32)
        Z = rng.normal(size=(len(idx), intrinsic)).astype(np.float32) * scales
        X[idx] = centers[c] + Z @ basis + rng.normal(0, 0.05, (len(idx), d)).astype(np.float32)
    return X, y


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    method, n = sys.argv[1], int(sys.argv[2])
    kw = dict(a.split("=") for a in sys.argv[3:])
    d, clusters = int(kw.get("d", 50)), int(kw.get("clusters", 50))
    X, y = synthetic(n, d, clusters)
    base_rss = rss_gb()
    phases = {}
    t0 = time.time()
    if method == "magnet_tsne":
        from magnet import MAGNET
        m = MAGNET(loss="tsne", random_state=0, verbose=True)
        t = time.time(); Y = m.fit_transform(X); phases["fit_total"] = time.time() - t
    elif method == "magnet":
        # the full default pipeline (an earlier version replayed the fit's internals by hand and drifted from it)
        from magnet import MAGNET
        m = MAGNET(random_state=0)
        t = time.time(); Y = m.fit_transform(X); phases["fit_total"] = time.time() - t
        phases["epochs"] = m.n_epochs_
    elif method == "umap":
        import umap
        umap.UMAP(n_epochs=20, random_state=0).fit_transform(np.random.rand(6000, 10).astype(X.dtype))   # numba warm-up on the data's dtype, large enough for the NN-descent path
        t = time.time(); Y = umap.UMAP().fit_transform(X); phases["umap_total"] = time.time() - t
    else:
        raise ValueError(method)
    total = time.time() - t0
    peak = rss_gb()
    from metrics import evaluate
    res = evaluate(X, Y, y, cache_key=None, n_query=3000, n_global=2000)
    res.update(method=method, n=n, d=d, total_time=round(total, 2), phases={k: round(v, 2) for k, v in phases.items()},
               peak_rss_gb=round(peak, 2), data_rss_gb=round(base_rss, 2))
    print(json.dumps(res), flush=True)
    with open(os.path.join(HERE, "results_scaling.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
