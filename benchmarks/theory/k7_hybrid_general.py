"""The hybrid (SGD layout, then the exact mode refining it without the mid spring) on the four benchmark data sets:
(a) exact pass from the SGD layout at a scale set by a sqrt(n) rule (31 at 70k), lr 1000 and n/12, 300 iterations;
(b) exact pass with t-SNE's usual tiny init (1e-4) from the SGD layout, i.e. the standard expansion from an organised
start. 2 seeds on MNIST, 1 elsewhere."""
import sys, os, time, numpy as np
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist, fmnist, news20, blobs
from metrics import evaluate
def show(label, X, y, key, Y, dt):
    r = evaluate(X, Y, y, cache_key=key)
    sil = r.get("silhouette"); sil = f"{sil:.3f}" if sil is not None else "  -  "
    print(f"  {label:48s} T={dt:5.1f}s knn15 {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spear {r['dist_spearman']:.3f} trip {r['triplet_acc']:.3f} acc {r.get('knn_acc', float('nan')):.3f} sil {sil}  std {Y[:,0].std():.1f}", flush=True)
for name, (X, y, key), kw, seeds in (("mnist70k", mnist(70000), {}, (0, 1)), ("fmnist70k", fmnist(70000), {}, (0,)), ("news20", news20(), {"metric": "cosine"}, (0,)), ("blobs10k", blobs(), {}, (0,))):
    n = len(X); scale = 31.0 * np.sqrt(n / 70000.0)
    print(f"== {name} n={n} scale rule {scale:.1f}", flush=True)
    for seed in seeds:
        t = time.time(); Y0 = MAGNET(random_state=seed, **kw).fit_transform(X); t_sgd = time.time() - t; show(f"SGD default s={seed}", X, y, key, Y0, t_sgd)
        t = time.time(); Y = MAGNET(loss="tsne", random_state=seed, **kw).fit_transform(X); show(f"tsne default s={seed}", X, y, key, Y, time.time() - t)
        for lr in (1000, None):
            t = time.time(); Y = MAGNET(loss="tsne", init=Y0, tsne_init_scale=scale, tsne_learning_rate=lr, n_epochs=300, tsne_early=0, mid_weight=0.0, random_state=seed, **kw).fit_transform(X)
            show(f"(a) SGD -> tsne scale {scale:.0f}, lr {lr or 'n/12'}, 300 it s={seed}", X, y, key, Y, time.time() - t + t_sgd)
        t = time.time(); Y = MAGNET(loss="tsne", init=Y0, n_epochs=300, tsne_early=0, mid_weight=0.0, random_state=seed, **kw).fit_transform(X)
        show(f"(b) SGD -> tsne tiny init, 300 it s={seed}", X, y, key, Y, time.time() - t + t_sgd)
