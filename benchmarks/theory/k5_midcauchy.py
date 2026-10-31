"""Exact mode with a bounded-influence mid-range term (one spring per partner, force w d / (1 + d^2/c^2), c a
fraction of the layout radius) instead of the linear spring to the partners' mean. Sweep weight, c and phase.
MNIST 70k, 1 seed. Target: the SGD mode's global structure (Spearman 0.41-0.43) with the exact mode's local (0.35)."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    print(f"  {label:50s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r['knn_acc']:.3f} sil {r['silhouette']:.3f}", flush=True)
X, y, key = mnist(70000)
for mw, c, mp in ((0.3, 0.3, 0.5), (1.0, 0.3, 0.5), (3.0, 0.3, 0.5), (1.0, 0.3, 1.0), (3.0, 0.3, 1.0), (10.0, 0.3, 1.0), (3.0, 0.1, 1.0), (3.0, 1.0, 1.0), (10.0, 1.0, 1.0)):
    t = time.time(); Y = MAGNET(loss="tsne", mid_weight=mw, mid_phase=mp, tsne_mid=("cauchy", c), random_state=0).fit_transform(X)
    show(f"tsne, cauchy mid w {mw}, c {c} rms, phase {mp}", X, y, key, Y, time.time() - t)
