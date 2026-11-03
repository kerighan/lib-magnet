"""k19: hand-over scale x grid x iterations for the hybrid's pass, from the stiff SGD start (MNIST 70k, all cores):
scale {31, 62} x box width {1.0, 1.5} x iterations {150, 300}; 15-NN / Spearman / triplets / centroids / time.
Usage: python benchmarks/theory/k19_scale_preset.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from k17_virial_projected import prep
X, y, key = mnist(70000); y = np.asarray(y).astype(int); metrics = prep(X, y, key)
m = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float32); Y0 = m.embedding_.astype(np.float32)
r0 = metrics(Y0); print(f"start: {r0['knn15']:.3f}/{r0['sp']:.3f}/{r0['cent']:.3f}")
out = {}
print(f"{'scale':>6}{'box':>5}{'it':>5}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'t(s)':>7}")
for s in (31.0, 62.0):
    for box in (1.0, 1.5):
        for it in (150, 300):
            Y = Y0 - Y0.mean(0); Y = (Y / max(float(Y[:, 0].std()), 1e-12) * s).astype(np.float32)
            t = time.time(); Y = optimize_tsne(P, Y, n_iter=it, early_iter=0, box_width=box, n_threads=-1); dt = time.time() - t
            r = metrics(Y); r["t"] = dt; out[f"{s}/{box}/{it}"] = r
            print(f"{s:>6.0f}{box:>5.1f}{it:>5}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{dt:>7.1f}", flush=True)
json.dump(out, open(os.path.join(HERE, "k19_scale_preset.json"), "w"), indent=1)
