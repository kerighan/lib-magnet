"""Where does the SGD mode's global structure come from, and can the exact mode have it? (1) The mid-range term
of the exact mode is annealed to zero at half the iterations, after which t-SNE relaxes the global arrangement
freely: keep it on longer / stronger. (2) Fine sweep of the constant far-field repulsion below the breaking point.
MNIST 70k, 1 seed."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    print(f"  {label:46s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r['knn_acc']:.3f} sil {r['silhouette']:.3f}", flush=True)
X, y, key = mnist(70000)
for mw, mp in ((0.1, 0.5), (0.1, 1.0), (0.3, 0.5), (0.3, 1.0), (1.0, 1.0), (3.0, 1.0), (0.0, 0.5)):
    t = time.time(); Y = MAGNET(loss="tsne", mid_weight=mw, mid_phase=mp, random_state=0).fit_transform(X); show(f"tsne, mid_weight {mw}, mid_phase {mp}", X, y, key, Y, time.time() - t)
for mw in (0.0, 0.3, 1.0):
    t = time.time(); Y = MAGNET(mid_weight=mw, random_state=0).fit_transform(X); show(f"SGD, mid_weight {mw}", X, y, key, Y, time.time() - t)
for w in (3e-8, 2e-7, 4e-7):
    t = time.time(); Y = MAGNET(loss="tsne", tsne_far=(w, 0.1), random_state=0).fit_transform(X); show(f"tsne + far field w={w:g}", X, y, key, Y, time.time() - t)
