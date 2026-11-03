"""k26: the two controls asked by the third review.
(a) Common-affinity backend control: the same affinity matrix (MAGNET's normalised fuzzy graph P), the same start (the
    stiff SGD layout at scale 31), the same iteration budget (300, exaggeration 1, momentum 0.8, learning rate n / 12),
    refined by our FFT pass and by openTSNE's optimiser (PrecomputedAffinities, FFT gradient, same thread count); paired
    changes of 15-NN, Spearman, centroids. 5 starts x 2 data sets.
(b) Scale x exaggeration from one cached start: {tiny 1e-4, 31} x {exaggeration 1, 12 for the first 100 of 300 iterations}.
-> k26_backend_control.json / .log. Usage: python benchmarks/theory/k26_backend_control.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist, fmnist
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from k17_virial_projected import prep
from openTSNE import TSNEEmbedding
from openTSNE.affinity import PrecomputedAffinities

out = {"backend": {}, "scale_x_exag": {}}
T = 20
for loader, label in ((mnist, "mnist"), (fmnist, "fmnist")):
    X, y, key = loader(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key)
    print(f"\n[{label}] {'start':<10}{'backend':>10}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'t(s)':>7}", flush=True)
    for seed in range(5):
        m = MAGNET(loss="umap", mid_weight=0.5, random_state=seed, n_threads=T).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float64)
        Y0 = m.embedding_.astype(np.float32); Y0 -= Y0.mean(0); Y0 *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y0[:, 0].std()), 1e-12)
        r0 = metrics(Y0); rows = {"start": r0}
        t = time.time(); Y = optimize_tsne(sp.csr_matrix(P, dtype=np.float32), Y0.copy(), n_iter=300, early_iter=0, n_threads=T); dt = time.time() - t
        r = metrics(Y); r["t"] = dt; rows["magnet"] = r
        print(f"{'':<9}{'seed ' + str(seed):<10}{'MAGNET':>10}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{dt:>7.1f}", flush=True)
        aff = PrecomputedAffinities(P / P.sum(), normalize=True)
        t = time.time()
        emb = TSNEEmbedding(Y0.astype(np.float64), aff, negative_gradient_method="fft", n_jobs=T, learning_rate=max(200.0, n / 12.0), random_state=0)
        emb = emb.optimize(n_iter=300, exaggeration=1.0, momentum=0.8); dt = time.time() - t
        r = metrics(np.asarray(emb, dtype=np.float32)); r["t"] = dt; rows["opentsne"] = r
        print(f"{'':<9}{'':<10}{'openTSNE':>10}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{dt:>7.1f}", flush=True)
        out["backend"][f"{label}/{seed}"] = rows
        json.dump(out, open(os.path.join(HERE, "k26_backend_control.json"), "w"), indent=1)
    if label == "mnist":
        print(f"\n[scale x exaggeration, {label}, seed 0 start] {'scale':>8}{'exag':>6}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}", flush=True)
        m = MAGNET(loss="umap", mid_weight=0.5, random_state=0, n_threads=T).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float32)
        Yb = m.embedding_.astype(np.float32); Yb -= Yb.mean(0); Yb /= max(float(Yb[:, 0].std()), 1e-12)
        for scale in (1e-4, 31.0):
            for exag in (1, 12):
                Y = (Yb * scale).astype(np.float32)
                Y = optimize_tsne(P, Y, n_iter=300, early_iter=100 if exag == 12 else 0, early_exaggeration=12.0, n_threads=T)
                r = metrics(Y); out["scale_x_exag"][f"{scale}/{exag}"] = r
                print(f"{'':<42}{scale:>8g}{exag:>6}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}", flush=True)
        json.dump(out, open(os.path.join(HERE, "k26_backend_control.json"), "w"), indent=1)
