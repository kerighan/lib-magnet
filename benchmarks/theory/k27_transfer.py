"""k27: frozen-rule, held-out transfer study (the addition both reviewers asked for). Nothing here was tuned on these
data: kappa = 0.5, hand-over scale 31 sqrt(n / 70000), 300 iterations, no exaggeration, box width 1; PaCMAP stiff =
MN_ratio 4 (frozen on MNIST); UMAP and PaCMAP defaults; PCA-2. Every start goes through the same pass on MAGNET's 15-NN
graph. Data sets never used in development: covertype (UCI, 54 features z-scored, 7 classes) at 20k / 70k / 250k
points, letter (UCI, 16 features z-scored, 26 classes, 20k), RCV1 (TF-IDF -> SVD-300, rows L2-normalised, 70k, no
labels used). 3 runs at 20k / 70k, 1 at 250k; openTSNE and the standalone full-batch mode as references (1 run,
not at 250k). 8 threads (shared machine), RAM kept under ~2 GB. -> k27_transfer.json / .log, paper/tables/transfer.tex.
Usage: python benchmarks/theory/k27_transfer.py"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("NUMBA_NUM_THREADS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from metrics import _idx, high_dim_pdist, high_dim_neighbors
from magnet import MAGNET
from magnet.tsne import optimize_tsne
T = 8

def load(name, n):
    from sklearn.datasets import fetch_openml, fetch_rcv1
    rng = np.random.default_rng(0)
    if name == "covertype":
        d = fetch_openml("covertype", version=3, as_frame=False); X = np.asarray(d.data, dtype=np.float32); y = np.asarray(d.target)
        _, y = np.unique(y, return_inverse=True)
    elif name == "letter":
        d = fetch_openml("letter", version=1, as_frame=False); X = np.asarray(d.data, dtype=np.float32); _, y = np.unique(np.asarray(d.target), return_inverse=True)
    elif name == "rcv1":
        d = fetch_rcv1(); idx = rng.choice(d.data.shape[0], n, replace=False)
        Z = TruncatedSVD(300, random_state=0).fit_transform(d.data[idx]).astype(np.float32)
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z, None, f"rcv1_{n}"
    if n < len(X):
        idx = rng.choice(len(X), n, replace=False); X, y = X[idx], y[idx]
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
    return np.ascontiguousarray(X, dtype=np.float32), y.astype(int), f"{name}_{len(X)}"

def make_metrics(X, y, key):
    n = len(X); q = _idx(n, 5000, 0); hnn = high_dim_neighbors(X, q, 15, key)
    g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1)
    rng = np.random.default_rng(0); i, j, l = rng.integers(0, len(g), (3, 200000)); ok = (i != j) & (i != l) & (j != l); i, j, l = i[ok], j[ok], l[ok]
    if y is not None:
        cls = np.unique(y); Ch = np.array([X[y == c].mean(0) for c in cls]); Dch = np.sqrt(((Ch[:, None] - Ch[None]) ** 2).sum(-1))[np.triu_indices(len(cls), 1)]
        s = _idx(n, 10000, 2)
    def metrics(Y):
        lnn = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y[q], return_distance=False)
        lnn = np.array([row[row != a][:15] for a, row in zip(q, lnn)])
        r = dict(knn15=float(np.mean([len(np.intersect1d(a, b)) for a, b in zip(hnn, lnn)]) / 15))
        Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); dl = Dl[iu]
        r["sp"] = float(spearmanr(Dh[iu], dl).correlation); r["trip"] = float(np.mean(np.sign(Dh[i, j] - Dh[i, l]) == np.sign(Dl[i, j] - Dl[i, l])))
        if y is not None:
            Cl = np.array([Y[y == c].mean(0) for c in cls]); Dcl = np.sqrt(((Cl[:, None] - Cl[None]) ** 2).sum(-1))[np.triu_indices(len(cls), 1)]
            r["cent"] = float(spearmanr(Dch, Dcl).correlation); r["acc"] = float(cross_val_score(KNeighborsClassifier(10), Y[s], y[s], cv=5).mean())
        else:
            r["cent"] = float("nan"); r["acc"] = float("nan")
        return r
    return metrics

out = json.load(open(os.path.join(HERE, "k27_transfer.json"))) if os.path.exists(os.path.join(HERE, "k27_transfer.json")) else {}
def save(): json.dump(out, open(os.path.join(HERE, "k27_transfer.json"), "w"), indent=1)
def show(key, label, r0, r1, t0, t1):
    f = lambda r: f"{r['knn15']:.3f}/{r['sp']:.3f}/{r['trip']:.3f}/{r['cent']:.3f}/{r['acc']:.3f}"
    print(f"{key:<18}{label:<26}{f(r0) if r0 else '':>34} -> {f(r1):>34}  ({t0:.1f} + {t1:.1f} s)", flush=True)

PLAN = [("covertype", 20000, 3), ("letter", 20000, 3), ("covertype", 70000, 3), ("rcv1", 70000, 2), ("covertype", 250000, 1)]
for name, n, runs in PLAN:
    X, y, key = load(name, n); n = len(X); metrics = make_metrics(X, y, key); metric = "cosine" if name == "rcv1" else "euclidean"
    scale = 31.0 * np.sqrt(n / 70000.0)
    print(f"\n=== {key} ({X.shape[1]} features, metric {metric}, hand-over scale {scale:.1f})", flush=True)
    import umap, pacmap
    umap.UMAP(n_epochs=20, metric=metric).fit_transform(np.random.rand(6000, X.shape[1]).astype(np.float32)); pacmap.PaCMAP(num_iters=(5, 5, 5)).fit_transform(X[:6000])
    graph = None
    for run in range(runs):
        starts = [("MAGNET kappa 0.5 (frozen)", lambda s: MAGNET(loss="umap", mid_weight=0.5, random_state=s, n_threads=T, metric=metric).fit(X)),
                  ("MAGNET kappa 0.1", lambda s: MAGNET(loss="umap", mid_weight=0.1, random_state=s, n_threads=T, metric=metric).fit(X)),
                  ("PaCMAP stiff (MN 4)", lambda s: pacmap.PaCMAP(MN_ratio=4.0, random_state=s, distance="angular" if metric == "cosine" else "euclidean").fit_transform(X)),
                  ("PaCMAP default", lambda s: pacmap.PaCMAP(random_state=s, distance="angular" if metric == "cosine" else "euclidean").fit_transform(X)),
                  ("UMAP default", lambda s: umap.UMAP(metric=metric).fit_transform(X)),
                  ("PCA-2", lambda s: PCA(2, svd_solver="randomized", random_state=0).fit_transform(X))]
        for label, f in starts:
            k_ = f"{key}/{label}/run {run}"
            if k_ in out: continue
            t = time.time(); obj = f(run); t_start = time.time() - t
            if isinstance(obj, MAGNET):
                Y0 = obj.embedding_.astype(np.float32); graph = graph if graph is not None else sp.csr_matrix(obj.graph_, dtype=np.float32)
            else:
                Y0 = np.asarray(obj, dtype=np.float32)
            if graph is None:
                graph = sp.csr_matrix(MAGNET(random_state=0, n_threads=T, metric=metric).fit(X).graph_, dtype=np.float32)
            r0 = metrics(Y0)
            Y = Y0 - Y0.mean(0); Y = (Y / max(float(Y[:, 0].std()), 1e-12) * scale).astype(np.float32)
            t = time.time(); Y = optimize_tsne(graph, Y, n_iter=300, early_iter=0, n_threads=T); t_pass = time.time() - t
            r1 = metrics(Y); out[k_] = dict(start=r0, after=r1, t_start=t_start, t_pass=t_pass); save(); show(key, label, r0, r1, t_start, t_pass)
    # references, one run each
    for label, f in [("MAGNET full-batch", lambda: MAGNET(loss="tsne", random_state=0, n_threads=T, metric=metric).fit_transform(X)),
                     ("MAGNET hybrid (shipped)", lambda: MAGNET(loss="hybrid", random_state=0, n_threads=T, metric=metric).fit_transform(X)),
                     ("openTSNE default", lambda: np.asarray(__import__("openTSNE").TSNE(n_jobs=T, random_state=0, metric=metric).fit(X)))]:
        k_ = f"{key}/{label}/run 0"
        if k_ in out or (n > 100000 and label == "openTSNE default"): continue
        t = time.time(); Y = np.asarray(f(), dtype=np.float32); dt = time.time() - t
        r = metrics(Y); out[k_] = dict(start=None, after=r, t_start=dt, t_pass=0.0); save(); show(key, label, None, r, dt, 0.0)
