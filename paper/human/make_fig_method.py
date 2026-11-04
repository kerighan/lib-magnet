"""Method schematic of the short paper: shared graph -> sampled pass (decoupled kernels, transient mid-range
attraction) -> rescaling -> full-batch refinement -> optional community placement and separation, with the note that
the post-process applies to any layout. Usage: python paper/human/make_fig_method.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(figsize=(9.6, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"
def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", fontsize=8.5, color=color, weight="bold")
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h - 5.6 - 3.0 * i, l, ha="center", va="center", fontsize=6.4, color="#222222")
def arrow(x0, x1, y, label=None, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))
    if label: ax.text((x0 + x1) / 2, y + 2.2, label, ha="center", va="bottom", fontsize=7, color=color)
box(1, 11, 15, 20, "fuzzy kNN graph", ["k = 15, NN-descent", "UMAP bandwidths", "symmetrised", "1.2 s on 70k"], GREY)
arrow(16.6, 21.4, 21)
box(22, 11, 23, 20, "1. sampled pass", ["negative-sampling SGD, 100 epochs", "attraction 1 / (1 + a d^2b), b by size", "repulsion Cauchy (b_r = 1)", "transient mid-range attraction", "4 s on 70k: the arrangement"], BLUE)
arrow(45.6, 50.4, 21, "scale\n31 sqrt(n/70k)")
box(51, 11, 23, 20, "2. full-batch pass", ["t-SNE objective, same graph", "FFT repulsion, box width 0.7", "300 iterations, no exaggeration", "momentum and gains", "+4 s: the neighbourhoods"], BLUE)
arrow(74.6, 79.4, 21, "optional")
box(80, 11, 19, 20, "3. post-process", ["communities of the graph", "(Louvain + density merge)", "placement: sampled Spearman", "under a mixing guard", "contraction g = 0.7; +1-2 s"], ORANGE)
ax.text(89.5, 6.5, "label-free, applies to any layout:\nthe control gives it to every method", ha="center", va="center", fontsize=6.4, color=ORANGE)
ax.text(33.5, 6.5, "standalone: loss=\"umap\"", ha="center", fontsize=6.4, color=BLUE); ax.text(62.5, 6.5, "standalone: loss=\"tsne\"\n(600 iterations from PCA)", ha="center", va="center", fontsize=6.4, color=BLUE)
ax.text(48, 34.2, "sequence: loss=\"hybrid\"", ha="center", fontsize=8, color=BLUE)
ax.add_patch(FancyBboxPatch((21.2, 10.2), 53.6, 22.2, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec=BLUE, lw=0.8, ls=(0, (3, 3))))
fig.savefig(os.path.join(HERE, "figures", "fig_method.png"), dpi=220, bbox_inches="tight"); print("written")
