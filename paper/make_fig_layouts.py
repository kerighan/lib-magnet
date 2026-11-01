"""Figure 1 of the paper (layouts cached in paper/figures/cache/, delete to recompute): MNIST 70k under six methods (UMAP, openTSNE, PaCMAP / MAGNET SGD, exact, hybrid).
Usage: python paper/make_fig_layouts.py [mnist|fmnist]  -> paper/figures/layouts_<dataset>.png"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET

ds = sys.argv[1] if len(sys.argv) > 1 else "mnist"
data, labels, name = LOADERS[ds](70000)
labels = np.asarray(labels).astype(int)

def umap_(X):
    import umap; return umap.UMAP(random_state=0).fit_transform(X)
def opentsne_(X):
    from openTSNE import TSNE; return np.asarray(TSNE(n_jobs=-1, random_state=0).fit(X))
def pacmap_(X):
    import pacmap; return pacmap.PaCMAP(random_state=0).fit_transform(X)
methods = [("UMAP", umap_), ("openTSNE (FIt-SNE)", opentsne_), ("PaCMAP", pacmap_),
           ("MAGNET (SGD)", lambda X: MAGNET(random_state=0).fit_transform(X)),
           ('MAGNET loss="tsne"', lambda X: MAGNET(loss="tsne", random_state=0).fit_transform(X)),
           ('MAGNET loss="hybrid"', lambda X: MAGNET(loss="hybrid", random_state=0).fit_transform(X))]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, (title, f) in zip(axes.ravel(), methods):
    slug = "".join(ch for ch in title.lower() if ch.isalnum())
    cache = os.path.join(HERE, "figures", "cache", f"{ds}_{slug}.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        z = np.load(cache); Y, dt = z["Y"], float(z["dt"])
    else:
        t = time.time(); Y = np.asarray(f(data), dtype=np.float32); dt = time.time() - t
        np.savez(cache, Y=Y, dt=dt)
    r = evaluate(data, Y, labels, cache_key=name)
    ax.scatter(Y[:, 0], Y[:, 1], s=0.25, c=labels, cmap="tab10", rasterized=True)
    ax.set_title(f"{title}  ({dt:.0f} s)\n15-NN {r['knn15_preservation']:.3f}   Spearman {r['dist_spearman']:.3f}", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim"); ax.set_box_aspect(1)
    print(title, f"{dt:.1f}s", round(r["knn15_preservation"], 3), round(r["dist_spearman"], 3), flush=True)
plt.tight_layout()
out = os.path.join(HERE, "figures", f"layouts_{ds}.png")
plt.savefig(out, dpi=120, bbox_inches="tight"); print(out)
