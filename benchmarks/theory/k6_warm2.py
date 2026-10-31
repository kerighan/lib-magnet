"""Warm start of the exact mode from the SGD layout, again, without the linear mid-range spring that collapsed the
layout in k2 (a spring to the partners' mean at the exact mode's scale is a strong contraction). SGD builds the
global arrangement, the exact mode refines locally at the exact mode's scale with a small learning rate."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    print(f"  {label:52s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r['knn_acc']:.3f} sil {r['silhouette']:.3f}  std {Y[:,0].std():.1f}", flush=True)
X, y, key = mnist(70000)
t = time.time(); Y0 = MAGNET(random_state=0).fit_transform(X); t_sgd = time.time() - t; show("A  SGD default", X, y, key, Y0, t_sgd)
for scale, lr, n_it in ((31.0, 200, 150), (31.0, 200, 300), (31.0, 1000, 300), (31.0, 5800, 300), (15.0, 1000, 300), (31.0, 1000, 600)):
    t = time.time(); Y = MAGNET(loss="tsne", init=Y0, tsne_init_scale=scale, tsne_learning_rate=lr, n_epochs=n_it, tsne_early=0, mid_weight=0.0, random_state=0).fit_transform(X)
    show(f"C' SGD -> tsne, scale {scale}, lr {lr}, {n_it} it, no mid", X, y, key, Y, time.time() - t + t_sgd)
# the SGD side of the fade, for reference: more global at the cost of local
for mw in (0.3, 0.5):
    t = time.time(); Y = MAGNET(mid_weight=mw, random_state=0).fit_transform(X); show(f"SGD, mid_weight {mw}", X, y, key, Y, time.time() - t)
