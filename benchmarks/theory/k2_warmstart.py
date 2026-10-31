"""Warm start of the exact mode from the SGD layout at the exact mode's own scale and with a small learning rate
(the first attempt rescaled the init to 1e-4 and used lr = n/12, which re-expanded the layout and lost the global
arrangement); and the reverse order (exact first, SGD to finish). MNIST 70k."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    print(f"  {label:50s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r['knn_acc']:.3f} sil {r['silhouette']:.3f}  std {Y[:,0].std():.1f}", flush=True)
X, y, key = mnist(70000)
t = time.time(); Y0 = MAGNET(random_state=0).fit_transform(X); t_sgd = time.time() - t; show("A  SGD default", X, y, key, Y0, t_sgd)
t = time.time(); Yt = MAGNET(loss="tsne", random_state=0).fit_transform(X); t_tsne = time.time() - t; show("B  tsne default", X, y, key, Yt, t_tsne)
std_t = float(Yt[:, 0].std()); print(f"  exact-mode layout std {std_t:.1f}, SGD layout std {Y0[:,0].std():.1f}")
for scale_f, lr, n_it in ((1.0, 200, 150), (1.0, 200, 300), (1.0, 1000, 300), (0.5, 200, 300), (2.0, 200, 300), (1.0, 50, 300)):
    t = time.time(); Y = MAGNET(loss="tsne", init=Y0, tsne_init_scale=std_t * scale_f, tsne_learning_rate=lr, n_epochs=n_it, tsne_early=0, random_state=0).fit_transform(X)
    show(f"C  SGD -> tsne at {scale_f:.1f}x scale, lr {lr}, {n_it} it", X, y, key, Y, time.time() - t + t_sgd)
for n_ep in (60, 150):
    t = time.time(); Y = MAGNET(init=Yt / std_t * Y0[:, 0].std(), n_epochs=n_ep, random_state=0).fit_transform(X)
    show(f"R  tsne -> SGD {n_ep} epochs", X, y, key, Y, time.time() - t + t_tsne)
