"""Quality-versus-time maps from the full benchmark runs (tags full_*) -> benchmarks/plots/tradeoff.png"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"magnet": "MAGNET", "umap": "UMAP", "opentsne": "openTSNE", "pacmap": "PaCMAP", "trimap": "TriMap", "sktsne": "sklearn t-SNE"}
DATASETS = [("mnist70000", "MNIST 70k"), ("fmnist70000", "Fashion-MNIST 70k"), ("news20", "20 newsgroups"), ("blobs10k", "15 blobs 10k")]
METRICS = [("knn15_preservation", "15-NN preservation (local)"), ("dist_spearman", "distance Spearman (global)")]


def load():
    latest = {}   # latest run per (dataset, method, tag=seed), so refreshed rows replace the old ones
    for line in open(os.path.join(HERE, "results.jsonl")):
        r = json.loads(line)
        if not str(r.get("tag", "")).startswith("full_"):
            continue
        name = LABELS.get(r["method"], r["method"])
        if r["method"] == "magnet" and r["params"].get("loss") == "tsne":
            name = 'MAGNET loss="tsne"'
        latest[(r["dataset"], name, r["tag"])] = r
    runs = collections.defaultdict(list)
    for (d, name, _), r in latest.items():
        runs[(d, name)].append(r)
    return runs


def main():
    runs = load()
    fig, axes = plt.subplots(len(METRICS), len(DATASETS), figsize=(5 * len(DATASETS), 4.6 * len(METRICS)), squeeze=False)
    colors = {"MAGNET": "#d62728", 'MAGNET loss="tsne"': "#ff7f0e", "UMAP": "#1f77b4", "openTSNE": "#2ca02c", "PaCMAP": "#9467bd", "TriMap": "#8c564b", "sklearn t-SNE": "#7f7f7f"}
    for j, (dkey, dlabel) in enumerate(DATASETS):
        for i, (mkey, mlabel) in enumerate(METRICS):
            ax = axes[i][j]
            for (d, name), rs in runs.items():
                if d != dkey or not all(mkey in r for r in rs):
                    continue
                t = np.mean([r["time"] for r in rs]); q = np.mean([r[mkey] for r in rs])
                ours = name.startswith("MAGNET")
                ax.scatter(t, q, s=160 if ours else 70, c=colors.get(name, "k"), marker="*" if ours else "o", zorder=3, edgecolor="k", linewidth=0.5)
                ax.annotate(name, (t, q), textcoords="offset points", xytext=(6, 4), fontsize=8, fontweight="bold" if ours else "normal")
            ax.set_xscale("log")
            ax.set_xlabel("time (s, log scale)  <- faster")
            ax.set_ylabel(mlabel + "  better ->")
            ax.set_title(dlabel, fontsize=11)
            ax.grid(alpha=0.3, which="both")
    fig.suptitle("Quality against wall-clock time (defaults, 20-core CPU). Top left is best.", fontsize=13)
    plt.tight_layout()
    out = os.path.join(HERE, "plots", "tradeoff.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
