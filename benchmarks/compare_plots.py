"""Side-by-side 2-D plots UMAP vs MAGNET on several datasets -> benchmarks/plots/compare_<dataset>.png
Usage: python benchmarks/compare_plots.py [dataset ...]   (default: mnist fmnist news20 blobs)"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import LOADERS
from metrics import evaluate

DEFAULT = ["mnist", "fmnist", "news20", "blobs"]
LABELS = {"mnist": "MNIST (70k, 784-d)", "fmnist": "Fashion-MNIST (70k, 784-d)",
          "news20": "20 newsgroups (18.8k, SVD-300, cosine)", "blobs": "15 Gaussian blobs (10k, 200-d)"}


def main(names):
    import umap
    from magnet import MAGNET
    umap.UMAP(n_neighbors=5, n_epochs=20).fit_transform(np.random.rand(500, 10))  # numba warm-up
    for name in names:
        X, y, key = LOADERS[name](70000) if name in ("mnist", "fmnist") else LOADERS[name]()
        metric = "cosine" if name == "news20" else "euclidean"
        runs = []
        t = time.time(); Yu = umap.UMAP(metric=metric).fit_transform(X); runs.append(("UMAP", Yu, time.time() - t))
        t = time.time(); Ym = MAGNET(metric=metric, random_state=0).fit_transform(X); runs.append(("MAGNET v3", Ym, time.time() - t))
        if "--tsne" in sys.argv:
            t = time.time(); Yt = MAGNET(metric=metric, loss="tsne", random_state=0).fit_transform(X); runs.append(('MAGNET v3, loss="tsne"', Yt, time.time() - t))
        fig, axes = plt.subplots(1, len(runs), figsize=(8 * len(runs), 8))
        for ax, (label, Y, dt) in zip(axes, runs):
            m = evaluate(X, Y, y, cache_key=key)
            ax.scatter(Y[:, 0], Y[:, 1], s=0.3 if len(Y) > 30000 else 1.5, c=y, cmap="tab10" if y is not None and y.max() < 10 else "tab20")
            sub = f"15-NN pres. {m['knn15_preservation']:.3f} | trust. {m['trustworthiness']:.3f} | dist. Spearman {m['dist_spearman']:.2f}"
            if "knn_acc" in m:
                sub += f" | 10-NN acc. {m['knn_acc']:.3f}"
            ax.set_title(f"{label}  -  {dt:.1f} s\n{sub}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(LABELS.get(name, name), fontsize=14)
        plt.tight_layout()
        out = os.path.join(HERE, "plots", f"compare_{key}{'_tsne' if '--tsne' in sys.argv else ''}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("saved", out, {l: round(dt, 1) for l, _, dt in runs})


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT)
