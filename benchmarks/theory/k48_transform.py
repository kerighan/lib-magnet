"""k48: out-of-sample transform. MNIST 70k split at random into 60k fitted / 10k held out (seed 0); each method is
fitted on the 60k and places the 10k with its transform. Axes on the held-out points, all against the fitted points only:
  knn_acc : 10-NN classification of the held-out points in the layout from the fitted points' labels
  nn15    : overlap of the 15 input-space nearest fitted points with the 15 layout-space nearest fitted points
  self    : the first 2000 fitted points re-transformed, median distance to their own fitted position over the median
            1-NN distance of the fitted layout (1 = one neighbour spacing away)
plus, as the in-sample reference, nn15 of fitting all 70k at once (held-out rows, neighbours restricted to the 60k).
Rows marked "(v3.0.0 ...)" are the released transform (unit step, 50 epochs); the others use the step scaled to the
layout spacing and 200 epochs.
Usage: python benchmarks/theory/k48_transform.py [methods=...]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from datasets import mnist
from magnet import MAGNET
T = 8
OUT = os.path.join(HERE, "k48_transform.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
X, y, _ = mnist(70000); X = np.asarray(X, dtype=np.float32)
perm = np.random.default_rng(0).permutation(len(X)); tr, te = perm[:60000], perm[60000:]
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
NN_IN = NearestNeighbors(n_neighbors=15).fit(Xtr).kneighbors(Xte, return_distance=False)

def axes(Ytr, Yte, Yself):
    acc = float((KNeighborsClassifier(10).fit(Ytr, ytr).predict(Yte) == yte).mean())
    nn_out = NearestNeighbors(n_neighbors=15).fit(Ytr).kneighbors(Yte, return_distance=False)
    nn15 = float(np.mean([len(set(a) & set(b)) / 15 for a, b in zip(NN_IN, nn_out)]))
    d1 = NearestNeighbors(n_neighbors=2).fit(Ytr).kneighbors(Ytr[:5000])[0][:, 1]
    selfd = float(np.median(np.linalg.norm(Yself - Ytr[:2000], axis=1)) / np.median(d1)) if Yself is not None else None
    return dict(knn_acc=acc, nn15=nn15, self=selfd)

def magnet(**k):
    def run():
        m = MAGNET(random_state=0, n_threads=T, **k); t0 = time.time(); m.fit(Xtr); tf = time.time() - t0
        t0 = time.time(); Yte = m.transform(Xte); tt = time.time() - t0
        return m.embedding_, Yte, m.transform(Xtr[:2000]), tf, tt
    return run
def umap_():
    import umap
    m = umap.UMAP(random_state=0, n_jobs=T); t0 = time.time(); Ytr = m.fit_transform(Xtr); tf = time.time() - t0
    t0 = time.time(); Yte = m.transform(Xte); tt = time.time() - t0
    return Ytr, Yte, m.transform(Xtr[:2000]), tf, tt
def otsne():
    from openTSNE import TSNE
    t0 = time.time(); E = TSNE(n_jobs=T, random_state=0).fit(Xtr); tf = time.time() - t0
    t0 = time.time(); Yte = np.asarray(E.transform(Xte)); tt = time.time() - t0
    return np.asarray(E), Yte, np.asarray(E.transform(Xtr[:2000])), tf, tt
M = {"MAGNET SGD": magnet(), "MAGNET full-batch": magnet(loss="tsne"), "MAGNET hybrid": magnet(loss="hybrid"),
     "MAGNET hybrid + contraction": magnet(loss="hybrid", contraction=0.7), "UMAP": umap_, "openTSNE": otsne}
sel = dict(a.split("=", 1) for a in sys.argv[1:]).get("methods"); sel = sel.split(",") if sel else list(M)
for name in sel:
    Ytr, Yte, Ys, tf, tt = M[name]()
    r = axes(Ytr, Yte, Ys); r.update(fit_s=round(tf, 1), transform_s=round(tt, 2)); res[name] = r
    print(name, r, flush=True); json.dump(res, open(OUT, "w"), indent=1)
# in-sample reference: fit all 70k, read the held-out rows against the 60k
if "ref" in sel or not any(s.startswith("ref") for s in res):   # skipped once present
    for name, k in [("MAGNET SGD", {}), ("MAGNET hybrid", dict(loss="hybrid"))]:
        Y = MAGNET(random_state=0, n_threads=T, **k).fit_transform(X)
        r = axes(Y[tr], Y[te], None); res["ref (all 70k) " + name] = r; print("ref", name, r, flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
