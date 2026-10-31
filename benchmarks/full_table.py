"""Full benchmark table: MAGNET (both modes) against UMAP, openTSNE, PaCMAP, TriMap and sklearn t-SNE.
Usage: python benchmarks/full_table.py [--quick]   -> benchmarks/BENCHMARK.md"""
import sys, os, json, time, traceback, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from bench import run

QUICK = "--quick" in sys.argv
METHODS = [
    ("MAGNET (SGD, default)", "magnet", {}),
    ('MAGNET (loss="tsne")', "magnet", {"loss": "tsne"}),
    ("UMAP 0.5.11", "umap", {}),
    ("openTSNE 1.0.4", "opentsne", {}),
    ("PaCMAP", "pacmap", {}),
    ("TriMap", "trimap", {}),
    ("sklearn t-SNE (Barnes-Hut)", "sktsne", {}),
    ("densMAP (umap-learn 0.5.11)", "densmap", {}),
    ("PHATE 2.0", "phate", {}),
    ("h-NNE", "hnne", {}),
    ("t-SNE-CUDA 3.0.1 (GPU, RTX 2070)", "tsnecuda", {}),
]
DATASETS = [  # (dataset key, n, seeds, metric kwargs per method family, label)
    ("mnist", 70000, 2, "MNIST (70k x 784)"),
    ("fmnist", 70000, 1, "Fashion-MNIST (70k x 784)"),
    ("news20", 0, 2, "20 newsgroups (18.8k x 300 SVD, cosine)"),
    ("blobs", 0, 2, "15 Gaussian blobs (10k x 200)"),
]
KEYS = [("time", "time (s)", False), ("knn15_preservation", "15-NN pres.", True), ("trustworthiness", "trust.", True),
        ("dist_spearman", "dist. Spearman", True), ("triplet_acc", "triplet acc.", True), ("knn_acc", "10-NN acc.", True),
        ("silhouette", "silhouette", True)]


def main():
    out = ["# Benchmark", "", "Mean over seeds; best value per column in bold. 20-core i9-9900X, 54 GB RAM. "
           "Metrics: 15-NN preservation and trustworthiness (local), Spearman correlation of pairwise distances and "
           "random-triplet accuracy on a 2000-point sample (global), 10-NN accuracy of the labels in 2-D, silhouette.", ""]
    for key, n, seeds, label in DATASETS:
        if QUICK and n > 20000:
            n = 20000
        rows = []
        for name, method, kw in METHODS:
            kw = dict(kw)
            if key == "news20":
                if method in ("umap", "opentsne", "magnet", "densmap", "tsnecuda"):
                    kw["metric"] = "cosine"
                elif method == "phate":
                    kw["knn_dist"] = "cosine"
                elif method == "pacmap":
                    kw["distance"] = "angular"
            if method == "sktsne" and n > 30000 and not QUICK:
                pass  # slow but let it run
            vals = []
            for s in range(seeds):
                k = dict(kw)
                if method == "magnet":
                    k["random_state"] = s
                try:
                    r = run(method, key, n=n, tag=f"full_{name.split()[0]}_{s}", plot=False, **k)
                    vals.append(r)
                except Exception as e:
                    print(f"[{name} on {key}] failed: {e}", file=sys.stderr)
                    traceback.print_exc()
                    break
            if vals:
                rows.append((name, {k: float(np.mean([v[k] for v in vals if k in v])) if all(k in v for v in vals) else float("nan") for k, _, _ in KEYS}))
            print(f"done {name} on {label}", flush=True)
        out.append(f"## {label}" + (f", {seeds} seed{'s' if seeds > 1 else ''}" if seeds else ""))
        out.append("")
        out.append("| method | " + " | ".join(h for _, h, _ in KEYS) + " |")
        out.append("|---|" + "---|" * len(KEYS))
        best = {}
        for k, _, higher in KEYS:
            cand = [(r[k], i) for i, (_, r) in enumerate(rows) if np.isfinite(r[k])]
            if cand:
                best[k] = (max if higher else min)(cand)[1]
        for i, (name, r) in enumerate(rows):
            cells = []
            for k, _, _ in KEYS:
                v = r[k]
                txt = "" if not np.isfinite(v) else (f"{v:.1f}" if k == "time" else f"{v:.3f}")
                cells.append(f"**{txt}**" if best.get(k) == i and txt else txt)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
        with open(os.path.join(HERE, "BENCHMARK.md"), "w") as f:
            f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
