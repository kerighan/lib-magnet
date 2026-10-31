"""Two objectives, one layout: (C) the sampled SGD mode (best global structure) builds the layout, the exact
t-SNE mode (best local structure) finishes it from that init without early exaggeration; (D) the attraction-
repulsion spectrum inside the exact mode (final-phase exaggeration 1 = t-SNE, 4 ~ UMAP), i.e. the continuous
fade between the two regimes in one loss. MNIST 70k, Fashion 70k."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist, fmnist
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    print(f"  {label:44s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r['knn_acc']:.3f} sil {r['silhouette']:.3f}", flush=True)
for name, (X, y, key) in (("mnist70k", mnist(70000)), ("fmnist70k", fmnist(70000))):
    print(f"== {name}", flush=True)
    t = time.time(); M = MAGNET(random_state=0).fit(X); t_sgd = time.time() - t; Y0 = M.embedding_.copy(); show("A  SGD default", X, y, key, Y0, t_sgd)
    t = time.time(); Y = MAGNET(loss="tsne", random_state=0).fit_transform(X); show("B  tsne default (600 it, early 200)", X, y, key, Y, time.time() - t)
    for n_it, early in ((150, 0), (300, 0), (300, 50)):
        t = time.time(); Y = MAGNET(loss="tsne", init=Y0, n_epochs=n_it, tsne_early=early, random_state=0).fit_transform(X)
        show(f"C  SGD init -> tsne {n_it} it, early {early}", X, y, key, Y, time.time() - t + t_sgd)
    for ex in (1.5, 2.0, 4.0):
        t = time.time(); Y = MAGNET(loss="tsne", exaggeration=ex, random_state=0).fit_transform(X); show(f"D  tsne, final exaggeration {ex}", X, y, key, Y, time.time() - t)
    for ex in (1.5, 2.0):
        t = time.time(); Y = MAGNET(loss="tsne", init=Y0, n_epochs=300, tsne_early=0, exaggeration=ex, random_state=0).fit_transform(X)
        show(f"F  SGD init -> tsne 300 it, exaggeration {ex}", X, y, key, Y, time.time() - t + t_sgd)
