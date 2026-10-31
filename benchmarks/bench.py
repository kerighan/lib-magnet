"""Usage: python benchmarks/bench.py METHOD DATASET [n=70000] [key=value ...]
METHOD in {umap, magnet, opentsne, pacmap, trimap, sktsne, densmap, phate, hnne, tsnecuda, ncvis} ; results appended to benchmarks/results.jsonl"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from datasets import LOADERS
from metrics import evaluate


def parse(argv):
    kw = {}
    for a in argv:
        k, v = a.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        kw[k] = v
    return kw


def run(method, dataset, n=70000, tag="", plot=True, **kw):
    load = LOADERS[dataset]
    data, labels, name = load(n) if dataset in ("mnist", "fmnist") else load()
    is_graph = dataset == "sbm"
    if method == "umap":
        import umap
        if not is_graph:
            umap.UMAP(n_neighbors=5, n_epochs=20).fit_transform(np.random.rand(500, 10))  # numba warm-up
        t = time.time()
        Y = umap.UMAP(**kw).fit_transform(data if not is_graph else _graph_to_dist(data))
    elif method == "magnet":
        from magnet import MAGNET
        mk = dict(kw)
        if isinstance(mk.get("init"), str) and mk["init"].endswith(".npy"):
            mk["init"] = np.load(mk["init"])
        t = time.time()
        Y = MAGNET(**mk).fit_transform(data)
        kw.pop("verbose", None)
    elif method == "opentsne":
        from openTSNE import TSNE
        t = time.time()
        Y = np.asarray(TSNE(n_jobs=-1, **kw).fit(data))
    elif method == "pacmap":
        import pacmap
        t = time.time()
        Y = pacmap.PaCMAP(**kw).fit_transform(np.asarray(data, dtype=np.float32))
    elif method == "trimap":
        import trimap
        t = time.time()
        Y = trimap.TRIMAP(**kw).fit_transform(np.asarray(data, dtype=np.float32))
    elif method == "sktsne":
        from sklearn.manifold import TSNE
        t = time.time()
        Y = TSNE(n_jobs=-1, init="pca", **kw).fit_transform(np.asarray(data, dtype=np.float32))
    elif method == "densmap":
        import umap
        umap.UMAP(n_neighbors=5, n_epochs=20, densmap=True).fit_transform(np.random.rand(500, 10))  # numba warm-up
        t = time.time()
        Y = umap.UMAP(densmap=True, **kw).fit_transform(data)
    elif method == "phate":
        import phate
        t = time.time()
        Y = phate.PHATE(n_jobs=-1, verbose=0, **kw).fit_transform(np.asarray(data, dtype=np.float32))
    elif method == "hnne":
        import hnne
        t = time.time()
        Y = hnne.HNNE(**kw).fit_transform(np.asarray(data, dtype=np.float32))
    elif method == "tsnecuda":   # GPU; needs the CUDA runtime on LD_LIBRARY_PATH (the nvidia pip wheels do)
        from tsnecuda import TSNE
        Xc = np.asarray(data, dtype=np.float32)
        if kw.pop("metric", None) == "cosine":   # no cosine option: L2-normalise, then euclidean
            Xc = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
        TSNE(n_components=2, perplexity=30, random_seed=0).fit_transform(np.random.rand(2000, 10).astype(np.float32))  # context warm-up
        t = time.time()
        Y = TSNE(n_components=2, **kw).fit_transform(Xc)
    elif method == "ncvis":
        import ncvis
        t = time.time()
        Y = ncvis.NCVis(**kw).fit_transform(np.asarray(data, dtype=np.float32))
    else:
        raise ValueError(method)
    elapsed = time.time() - t
    Y = np.asarray(Y, dtype=np.float32)
    res = evaluate(None if is_graph else data, Y, labels, cache_key=name)
    res.update(method=method, dataset=name, time=round(elapsed, 2), params=kw, tag=tag)
    print(json.dumps(res))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.scatter(Y[:, 0], Y[:, 1], s=0.5, c=labels, cmap="tab10")
        plt.title(f"{method} {name} {kw} T={elapsed:.1f}s")
        plt.axis("off")
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", f"{method}_{name}{('_' + tag) if tag else ''}.png"), dpi=110, bbox_inches="tight")
    return res


def _graph_to_dist(G):
    import networkx as nx, scipy.sparse as sp
    A = sp.csr_matrix(nx.adjacency_matrix(G).astype(np.float32))
    A.data = 1.0 / A.data  # similarity -> distance
    return A


if __name__ == "__main__":
    method, dataset = sys.argv[1], sys.argv[2]
    kw = parse(sys.argv[3:])
    n = int(kw.pop("n", 70000))
    run(method, dataset, n=n, **kw)
