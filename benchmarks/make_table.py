"""Build benchmarks/BENCHMARK.md from the full_* runs stored in results.jsonl (no recomputation)."""
import os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

LABELS = {"umap": "UMAP 0.5.12", "opentsne": "openTSNE 1.0.4", "pacmap": "PaCMAP", "trimap": "TriMap", "sktsne": "sklearn t-SNE (Barnes-Hut)",
          "densmap": "densMAP (umap-learn 0.5.12)", "phate": "PHATE 2.0", "hnne": "h-NNE", "tsnecuda": "t-SNE-CUDA 3.0.1 (GPU, RTX 2070)"}
DATASETS = [("mnist70000", "MNIST (70k x 784)"), ("fmnist70000", "Fashion-MNIST (70k x 784)"),
            ("news20", "20 newsgroups (18.8k x 300 SVD, cosine)"), ("blobs10k", "15 Gaussian blobs (10k x 200)")]
KEYS = [("time", "time (s)", False), ("knn15_preservation", "15-NN pres.", True), ("trustworthiness", "trust.", True),
        ("dist_spearman", "dist. Spearman", True), ("triplet_acc", "triplet acc.", True), ("knn_acc", "10-NN acc.", True),
        ("silhouette", "compactness*", True)]
ORDER = ["MAGNET (SGD, default)", 'MAGNET (loss="tsne")', 'MAGNET (loss="hybrid")', "UMAP 0.5.12", "densMAP (umap-learn 0.5.12)", "openTSNE 1.0.4", "PaCMAP", "TriMap", "PHATE 2.0", "h-NNE", "sklearn t-SNE (Barnes-Hut)", "t-SNE-CUDA 3.0.1 (GPU, RTX 2070)"]


def name_of(r):
    if r["method"] == "magnet":
        loss = r["params"].get("loss")
        return 'MAGNET (loss="tsne")' if loss == "tsne" else ('MAGNET (loss="hybrid")' if loss == "hybrid" else "MAGNET (SGD, default)")
    return LABELS.get(r["method"], r["method"])


def main():
    latest = {}   # one run per (dataset, method, tag=seed): the most recent one wins, so rows can be refreshed
    for line in open(os.path.join(HERE, "results.jsonl")):
        r = json.loads(line)
        if str(r.get("tag", "")).startswith("full_"):
            latest[(r["dataset"], name_of(r), r["tag"])] = r
    runs = collections.defaultdict(list)
    for (d, nm, _), r in latest.items():
        runs[(d, nm)].append(r)
    out = ["# Benchmark", "",
           "Mean over seeds, defaults for every method, best value per column in bold. 20-core i9-9900X, 54 GB RAM.",
           "Local structure: 15-NN preservation (fraction of each point's 15 nearest neighbours in the data that are "
           "among its 15 nearest in 2-D) and trustworthiness. Global structure: Spearman correlation of pairwise "
           "distances and random-triplet accuracy on a 2000-point sample. Labels: 10-NN accuracy in 2-D.",
           "*Compactness is the silhouette of the labels: it rewards tight, well-separated clusters and can be raised "
           "at the expense of neighbourhood quality, so it describes the look rather than the fidelity of a layout.", ""]
    for dkey, dlabel in DATASETS:
        rows = [(nm, runs[(dkey, nm)]) for nm in ORDER if runs.get((dkey, nm))]
        if not rows:
            continue
        seeds = max(len(rs) for _, rs in rows)
        agg = [(nm, {k: float(np.mean([r[k] for r in rs])) if all(k in r for r in rs) else float("nan") for k, _, _ in KEYS}) for nm, rs in rows]
        out += [f"## {dlabel}, {seeds} seed{'s' if seeds > 1 else ''}", "", "| method | " + " | ".join(h for _, h, _ in KEYS) + " |", "|---|" + "---|" * len(KEYS)]
        best = {}
        for k, _, higher in KEYS:
            cand = [(v[k], i) for i, (_, v) in enumerate(agg) if np.isfinite(v[k])]
            if cand:
                best[k] = (max if higher else min)(cand)[1]
        for i, (nm, v) in enumerate(agg):
            cells = []
            for k, _, _ in KEYS:
                x = v[k]
                txt = "" if not np.isfinite(x) else (f"{x:.1f}" if k == "time" else f"{x:.3f}")
                cells.append(f"**{txt}**" if best.get(k) == i and txt else txt)
            out.append(f"| {nm} | " + " | ".join(cells) + " |")
        out.append("")
    open(os.path.join(HERE, "BENCHMARK.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
