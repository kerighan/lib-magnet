"""The complete pipeline next to the two reference methods: UMAP, Barnes-Hut t-SNE (cached layouts of the long version)
and MAGNET sequence + post-process (fit here, seed 0), MNIST and Fashion-MNIST 70k, 20k points shown, coloured by
class (labels unused by every method). Usage: python paper/human/make_fig_pipeline.py"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets import mnist, fmnist
from metrics import evaluate
from magnet import MAGNET
fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
for row, (ds, loader, name) in enumerate((("mnist", mnist, "MNIST"), ("fmnist", fmnist, "Fashion-MNIST"))):
    X, y, key = loader(70000); y = np.asarray(y).astype(int); n = len(y); show = np.random.default_rng(0).choice(n, 20000, replace=False)
    cols = [("UMAP", np.load(os.path.join(ROOT, "paper", "figures", "cache", f"{ds}_umap0512.npz"))["Y"], None),
            ("Barnes-Hut t-SNE", np.load(os.path.join(ROOT, "benchmarks", "cache", "layouts", f"{ds}_sklearn_t-SNE_BH.npz"))["Y"], None)]
    t = time.time(); Y = MAGNET(loss="hybrid", contraction=0.7, random_state=0, n_threads=8).fit_transform(X); dt = time.time() - t
    cols.append(("MAGNET complete", Y, dt))
    for col, (label, lay, dt) in enumerate(cols):
        lay = np.asarray(lay, dtype=np.float32); e = evaluate(X, lay, y, cache_key=key); ax = axes[row, col]
        ax.scatter(lay[show, 0], lay[show, 1], c=y[show], cmap="tab10", s=1.0, linewidths=0, alpha=0.8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(f"{name}, {label}" + (f" ({dt:.0f} s)" if dt else ""), fontsize=9, loc="left"); ax.text(0.0, -0.02, f"15-NN {e['knn15_preservation']:.3f}   Spearman {e['dist_spearman']:.3f}", transform=ax.transAxes, fontsize=8, va="top", color="#444444")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_pipeline.png"), dpi=200); print("written")
