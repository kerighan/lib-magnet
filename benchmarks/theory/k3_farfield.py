"""One loss, two regimes by scale: t-SNE's normalised repulsion in the near field, a constant-strength Cauchy
repulsion (the sampled optimiser's far field) beyond r0 = 0.1 x RMS radius, plus the mid-range pairs. Sweep of
the far weight on MNIST 70k; the global metrics should rise towards the SGD mode's, the local ones should stay."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    print(f"  {label:40s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r['knn_acc']:.3f} sil {r['silhouette']:.3f}  std {Y[:,0].std():.1f}", flush=True)
X, y, key = mnist(70000)
for w in (1e-7, 1e-6, 1e-5, 1e-4):
    t = time.time(); Y = MAGNET(loss="tsne", tsne_far=(w, 0.1), random_state=0).fit_transform(X); show(f"tsne + far field w={w:g}, r0=0.1 rms", X, y, key, Y, time.time() - t)
for w, r0 in ((1e-4, 0.3), (1e-4, 0.03)):
    t = time.time(); Y = MAGNET(loss="tsne", tsne_far=(w, r0), random_state=0).fit_transform(X); show(f"tsne + far field w={w:g}, r0={r0} rms", X, y, key, Y, time.time() - t)
