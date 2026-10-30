"""Render benchmarks/BENCHMARK.md as an image (benchmarks/plots/benchmark_table.png)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["method", "time (s)", "15-NN pres.", "trust.", "dist. Spearman", "triplet acc.", "10-NN acc.", "compactness*"]


def main():
    sections, current = {}, None
    for line in open(os.path.join(HERE, "BENCHMARK.md")):
        if line.startswith("## "):
            current = line[3:].strip(); sections[current] = []
        elif current and line.startswith("| ") and not line.startswith("| method") and not line.startswith("|---"):
            sections[current].append([c.strip().replace("**", "") for c in line.strip().strip("|").split("|")])
    names = list(sections)
    fig, axes = plt.subplots(len(names), 1, figsize=(14, sum(0.42 * len(sections[n]) + 1.3 for n in names)))
    for ax, name in zip(np.atleast_1d(axes), names):
        rows = sections[name]
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=COLS, loc="center", cellLoc="center", colWidths=[0.30] + [0.10] * 7)
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.45)
        for j in range(1, len(COLS)):
            vals = [float(r[j]) if r[j] else np.nan for r in rows]
            if np.all(np.isnan(vals)):
                continue
            best = int(np.nanargmin(vals)) if j == 1 else int(np.nanargmax(vals))
            tbl[best + 1, j].get_text().set_weight("bold")
        for i, r in enumerate(rows):
            if r[0].startswith("MAGNET"):
                for j in range(len(COLS)):
                    tbl[i + 1, j].set_facecolor("#fde8d8")
        for j in range(len(COLS)):
            tbl[0, j].set_facecolor("#e8e8e8"); tbl[0, j].get_text().set_weight("bold")
        ax.set_title(name, fontsize=12, loc="left", pad=4)
    fig.suptitle("MAGNET v3 against UMAP, openTSNE, PaCMAP, TriMap and sklearn t-SNE (defaults; best per column in bold; "
                 "*compactness = label silhouette, a look rather than a fidelity score)", fontsize=10.5, y=1.0)
    plt.tight_layout()
    out = os.path.join(HERE, "plots", "benchmark_table.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
