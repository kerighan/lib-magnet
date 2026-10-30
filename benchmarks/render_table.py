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
    underline = []
    fig, axes = plt.subplots(len(names), 1, figsize=(14, sum(0.42 * len(sections[n]) + 1.3 for n in names)))
    for ax, name in zip(np.atleast_1d(axes), names):
        rows = sections[name]
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=COLS, loc="center", cellLoc="center", colWidths=[0.30] + [0.10] * 7)
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.45)
        for j in range(1, len(COLS)):
            vals = np.array([float(r[j]) if r[j] else np.nan for r in rows])
            if np.all(np.isnan(vals)):
                continue
            order = np.argsort(vals if j == 1 else -vals)   # best first (time: lower is better)
            order = [i for i in order if not np.isnan(vals[i])]
            tbl[order[0] + 1, j].get_text().set_weight("bold")
            if len(order) > 1:  # second best: underlined (drawn after layout, see below)
                underline.append((ax, tbl[order[1] + 1, j].get_text()))
        for i, r in enumerate(rows):
            if r[0].startswith("MAGNET"):
                for j in range(len(COLS)):
                    tbl[i + 1, j].set_facecolor("#fde8d8")
        for j in range(len(COLS)):
            tbl[0, j].set_facecolor("#e8e8e8"); tbl[0, j].get_text().set_weight("bold")
        ax.set_title(name, fontsize=12, loc="left", pad=4)
    fig.suptitle("MAGNET v3 against UMAP, openTSNE, PaCMAP, TriMap and sklearn t-SNE (defaults; best per column in bold, "
                 "second best underlined; *compactness = label silhouette, a look rather than a fidelity score)", fontsize=10.5, y=1.0)
    plt.tight_layout()
    fig.canvas.draw()  # positions are known only after the layout: draw the underlines by hand
    renderer = fig.canvas.get_renderer()
    for ax, text in underline:
        ext = text.get_window_extent(renderer)
        inv = ax.transAxes.inverted()
        (x0, y0), (x1, _) = inv.transform([(ext.x0, ext.y0), (ext.x1, ext.y0)])
        ax.plot([x0, x1], [y0 - 0.004, y0 - 0.004], transform=ax.transAxes, color="black", lw=0.9, clip_on=False)
    out = os.path.join(HERE, "plots", "benchmark_table.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
