"""Preliminary figures of the paper drawn from the recorded runs (docs/kernel_exponents.md, benchmarks/theory/*.log,
README scaling table): paper/figures/exponents.png (theory), hybrid.png (hybrid parametrisation), scaling.png."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00", "grey": "#777777"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "legend.frameon": False})

# ---------- Figure: the two exponents ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
# (a) attraction exponent optimum against the intrinsic dimension: a step at q = m (e5_law.py)
ax[0].plot([2, 3, 5, 10, 20], [1.0, 0.5, 0.5, 0.4, 0.4], "o-", color=C["blue"], lw=2, ms=7, label="target $m=2$")
ax[0].plot([3, 5, 10], [1.0, 0.5, 0.4], "s-", color=C["verm"], lw=2, ms=7, label="target $m=3$")
ax[0].axhline(0.5, color=C["grey"], lw=1, ls="--"); ax[0].text(20.5, 0.52, "L1 point", ha="right", color=C["grey"])
ax[0].axhline(1.0, color=C["grey"], lw=1, ls="--"); ax[0].text(20.5, 1.02, "L2 point", ha="right", color=C["grey"])
ax[0].set_xscale("log"); ax[0].set_xticks([2, 3, 5, 10, 20]); ax[0].set_xticklabels([2, 3, 5, 10, 20])
ax[0].set_xlabel("intrinsic dimension $q$ of the clusters"); ax[0].set_ylabel("best attraction exponent $b$")
ax[0].set_ylim(0.3, 1.1); ax[0].legend(loc="center right"); ax[0].set_title("(a) attraction: a step at $q = m$")
# (b) repulsive force per octave on a uniform square, slope against b_r (e1b_powerlaw.py)
b = np.array([0.5, 1.0, 1.5, 2.0]); pred = 2 - 1 - 2 * b; meas = np.array([0.07, -0.97, -2.01, -3.06])
ax[1].plot(b, pred, "-", color=C["grey"], lw=2, label="predicted $m-1-2b_r$")
ax[1].plot(b, meas, "o", color=C["blue"], ms=8, label="measured (40k points, 2-D)")
ax[1].axvline(1.0, color=C["verm"], lw=1, ls="--"); ax[1].text(1.03, -2.8, "Cauchy\n$b_r = m/2$", color=C["verm"])
ax[1].set_xlabel("repulsion exponent $b_r$"); ax[1].set_ylabel("slope of log$_2$(force per octave)")
ax[1].legend(loc="upper right"); ax[1].set_title("(b) repulsion: loss per octave $\\propto d^{\\,m-2b_r}$")
# (c) contamination against 15-NN preservation / optimum (e2)
q = ["q=2", "q=3", "q=5", "MNIST", "Fashion", "q=10", "q=20", "news20"]
share = [0.06, 0.14, 0.23, 0.24, 0.24, 0.36, 0.46, 0.34]; opt = [1.0, 0.5, 0.5, 0.5, 0.55, 0.4, 0.45, 0.65]
cols = [C["blue"] if o >= 0.9 else (C["orange"] if o > 0.55 else C["verm"]) for o in opt]
ax[2].scatter(share, opt, c=cols, s=60, zorder=3)
offs = {"q=5": (-26, 6), "MNIST": (6, -12), "Fashion": (6, 4)}
for x, y, t in zip(share, opt, q):
    ax[2].annotate(t, (x, y), xytext=offs.get(t, (5, 4)), textcoords="offset points", fontsize=9)
ax[2].set_xlabel("share of kNN edges beyond the kernel scale $d_k$ (crowding)"); ax[2].set_ylabel("measured attraction optimum")
ax[2].set_ylim(0.3, 1.1); ax[2].set_title("(c) crowding sets the exponent; text is the open case")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "exponents.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------- Figure: local / global fade and the hybrid ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mw = [0, 0.1, 0.3, 1.0]; knn = [0.206, 0.193, 0.187, 0.168]; sp = [0.285, 0.413, 0.427, 0.478]
ax[0].plot(sp, knn, "o-", color=C["blue"], lw=2, ms=7, label="SGD mode, mid_weight 0 → 1")
for x, y, t in zip(sp, knn, mw):
    ax[0].annotate(f"{t}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=9)
ax[0].scatter([0.348], [0.349], marker="s", color=C["verm"], s=60, zorder=3, label='exact mode (loss="tsne")')
ax[0].scatter([0.338], [0.350], marker="^", color=C["grey"], s=60, zorder=3, label="scikit-learn t-SNE")
ax[0].scatter([0.444], [0.353], marker="*", color=C["green"], s=160, zorder=4, label='hybrid (SGD → exact)')
ax[0].set_xlabel("distance Spearman (global)"); ax[0].set_ylabel("15-NN preservation (local)")
ax[0].legend(loc="center left", fontsize=9); ax[0].set_title("(a) MNIST 70k: fade inside the SGD mode, then the sequence", fontsize=11)
mw2 = [0.1, 0.3, 0.5, 1.0, 2.0]
k_sgd = [0.193, 0.187, 0.176, 0.171, 0.154]; k_hyb = [0.356, 0.354, 0.355, 0.351, 0.350]
s_sgd = [0.413, 0.430, 0.455, 0.474, 0.487]; s_hyb = [0.410, 0.423, 0.449, 0.452, 0.453]
ax[1].plot(mw2, s_sgd, "o--", color=C["blue"], lw=1.5, label="Spearman, SGD pass alone")
ax[1].plot(mw2, s_hyb, "o-", color=C["blue"], lw=2.5, label="Spearman, after the exact pass")
ax[1].plot(mw2, k_sgd, "s--", color=C["verm"], lw=1.5, label="15-NN, SGD pass alone")
ax[1].plot(mw2, k_hyb, "s-", color=C["verm"], lw=2.5, label="15-NN, after the exact pass")
ax[1].set_xscale("log"); ax[1].set_xticks(mw2); ax[1].set_xticklabels(mw2)
ax[1].set_xlabel("mid_weight of the SGD pass"); ax[1].set_ylabel("score"); ax[1].legend(fontsize=9, loc="center", bbox_to_anchor=(0.5, 0.44))
ax[1].set_title("(b) hybrid: the exact pass erases the local cost, keeps the global gain", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "hybrid.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------- Figure: scaling ----------
fig, ax = plt.subplots(figsize=(5.5, 4.2))
n = [1e5, 2.5e5, 5e5, 1e6]
ax.plot(n, [6.5, 12.0, 23.1, 43.9], "o-", color=C["blue"], lw=2, ms=7, label="MAGNET (SGD)")
ax.plot([2.5e5, 1e6], [17, 56], "s-", color=C["verm"], lw=2, ms=7, label='MAGNET loss="tsne"')
ax.plot(n[:3], [9.5, 23.4, 61.7], "^-", color=C["orange"], lw=2, ms=7, label="UMAP 0.5.12")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("points (50 clusters in 50-D)"); ax.set_ylabel("wall-clock time (s), 20 cores")
ax.set_yticks([10, 20, 50, 100]); ax.set_yticklabels([10, 20, 50, 100]); ax.legend(); ax.set_title("Scaling (MAGNET peak RSS 3.3 GB at 1M)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "scaling.png"), dpi=150, bbox_inches="tight"); plt.close()
print("ok")
