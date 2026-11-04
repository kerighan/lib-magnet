"""MNIST full-batch layout before / after the final post-process (strength Louvain 0.5 + density-contrast merge,
placement, separation 0.7), 20k points coloured by class. Usage: python paper/human/make_fig_placement.py"""
import os, sys, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import numpy as np, scipy.sparse as sp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets import mnist, fmnist
from metrics import _idx
from magnet.contraction import hierarchical_contraction
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
for row, (ds, loader) in enumerate((("mnist", mnist), ("fmnist", fmnist))):
    X, y, key = loader(70000); y = np.asarray(y).astype(int); n = len(y)
    Y = np.load(os.path.join(ROOT, "benchmarks", "cache", "layouts", f"{ds}_fb_seed0.npz"))["Y"].astype(np.float32)
    P = sp.load_npz(os.path.join(ROOT, "benchmarks", "cache", f"P_{ds}.npz"))
    excl = np.unique(np.concatenate([_idx(n, 2000, 1), _idx(n, 5000, 0), _idx(n, 10000, 2)]))
    Z, log = hierarchical_contraction(Y, P, X=X, random_state=0, exclude=excl, n_threads=8)
    show = np.random.default_rng(0).choice(n, 20000, replace=False)
    for col, (lay, title) in enumerate(((Y, "full-batch layout"), (Z, f"after placement and separation ({log['communities']} communities)"))):
        ax = axes[row, col]; ax.scatter(lay[show, 0], lay[show, 1], c=y[show], cmap="tab10", s=1.2, linewidths=0, alpha=0.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(("MNIST, " if row == 0 else "Fashion-MNIST, ") + title, fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_placement.png"), dpi=200); print("written")
