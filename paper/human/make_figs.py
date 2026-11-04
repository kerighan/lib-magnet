"""Figures of the human-readable paper, drawn from the k* result files of the long version. One idea per figure, one
axis per panel, direct labels, a colour-blind-safe palette (Okabe-Ito). Usage: python paper/human/make_figs.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); T = os.path.join(ROOT, "benchmarks", "theory")
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#222222"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#BBBBBB", "xtick.color": "#444444", "ytick.color": "#444444", "axes.labelcolor": "#222222", "legend.frameon": False})
def load(name): return json.load(open(os.path.join(T, name)))
def mean_sd(vals): vals = np.asarray(vals, float); return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)

# ---------------------------------------------------------------- figure 1: the FFT box width (k40)
k40 = load("k40_fft_box_width.json")
widths = [1.0, 0.8, 0.7, 0.6, 0.5]
fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))
for ax, ds, label, col in ((axes[0], "mnist", "MNIST 70k", C["blue"]), (axes[1], "blobs", "15 Gaussian blobs 10k", C["orange"])):
    m = [mean_sd([k40[k]["knn15"] for k in k40 if k.startswith(f"width/{ds}/{w}/3/")]) for w in widths]
    ax.errorbar(widths, [x[0] for x in m], yerr=[x[1] for x in m], color=col, lw=1.6, marker="o", ms=4, capsize=2)
    ax.set_xlabel("box width (layout units)"); ax.set_ylabel("15-NN preservation"); ax.set_title(label, loc="left", fontsize=9, color="#222222")
    ax.invert_xaxis(); ax.axvline(0.7, color=C["grey"], lw=0.8, ls=":"); ax.text(0.7, ax.get_ylim()[0] + 0.0005, " default 0.7", color=C["grey"], fontsize=8, va="bottom")
    if ds == "mnist": ax.axhline(0.350, color=C["green"], lw=0.8, ls="--"); ax.text(1.0, 0.3502, "Barnes-Hut t-SNE 0.350", color=C["green"], fontsize=8, va="bottom", ha="left")
ax = axes[2]
t = [mean_sd([k40[k]["t"] for k in k40 if k.startswith(f"width/mnist/{w}/3/")]) for w in widths]
err = [k40[f"error/{w}/3"]["F_rms_err"] * 100 if f"error/{w}/3" in k40 else None for w in widths]
ax.plot(widths, [x[0] for x in t], color=C["blue"], lw=1.6, marker="o", ms=4); ax.set_xlabel("box width (layout units)"); ax.set_ylabel("time, MNIST 70k (s, 8 threads)")
ax.invert_xaxis(); ax.set_title("cost, and force error", loc="left", fontsize=9, color="#222222")
for w, tt, e in zip(widths, t, err):
    if e is not None: ax.annotate(f"{e:.1f} %", (w, tt[0]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=C["red"])
ax.text(1.0, ax.get_ylim()[1] * 0.98, "labels: RMS force error of the interpolation", color=C["red"], fontsize=7.5, va="top", ha="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_box_width.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- figure 2: interfaces (k43)
k43 = load("k43_interface_merge.json")
fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
ax = axes[0]
same = k43["interfaces/mnist/1.0"]["same"] + k43["interfaces/mnist/0.5"]["same"]; diff = k43["interfaces/mnist/1.0"]["diff"] + k43["interfaces/mnist/0.5"]["diff"]
bins = np.linspace(0, 0.2, 21)
ax.hist(diff, bins=bins, color=C["blue"], alpha=0.85, label=f"real class boundaries (n={len(diff)})")
ax.hist(same, bins=bins, color=C["orange"], alpha=0.9, label=f"same-class splits (n={len(same)})")
ax.axvline(0.10, color=C["black"], lw=1.0, ls="--"); ax.text(0.102, ax.get_ylim()[1] * 0.95, "merge threshold 0.10", fontsize=8, va="top")
ax.axvline(0.20, color=C["grey"], lw=0.8, ls=":"); ax.text(0.198, ax.get_ylim()[1] * 0.3, "uniform cloud\nreference 0.2", fontsize=7.5, ha="right", color=C["grey"])
ax.set_xlabel("min crossing share of the interface's boundary nodes"); ax.set_ylabel("interfaces"); ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.55))
ax.set_title("MNIST, Louvain at resolution 0.5 and 1.0", loc="left", fontsize=9)
ax = axes[1]
rs = k43["ratio/mnist/1.0"]; rs5 = k43["ratio/mnist/0.5"]
ax.scatter(rs["diff"] + rs5["diff"], np.zeros(len(rs["diff"]) + len(rs5["diff"])) + np.random.default_rng(0).uniform(-0.15, 0.15, len(rs["diff"]) + len(rs5["diff"])), color=C["blue"], s=14, alpha=0.8, label="real boundaries")
ax.scatter(rs["same"] + rs5["same"], np.ones(len(rs["same"]) + len(rs5["same"])) + np.random.default_rng(1).uniform(-0.15, 0.15, len(rs["same"]) + len(rs5["same"])), color=C["orange"], s=18, label="same-class splits")
ax.axvline(0.5, color=C["grey"], lw=0.8, ls=":"); ax.axvline(1.0, color=C["grey"], lw=0.8, ls=":")
ax.set_yticks([0, 1]); ax.set_yticklabels(["real\nboundary", "artificial\nsplit"]); ax.set_xlim(0, 1.15); ax.set_xlabel("interface share / the community's own thinnest-cut share")
ax.set_title("relative to the community's internal cut", loc="left", fontsize=9); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_interfaces.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- figure 3: small n (k45)
k45 = load("k45_small_n.json")
ns = [3000, 5000, 10000, 20000, 35000]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.7))
for ax, key, ylabel in ((axes[0], "clusters", "HDBSCAN clusters (10 classes)"), (axes[1], "knn15", "15-NN preservation")):
    ends = []
    for label, col, ls in (("kernel_b=0.6", C["red"], "-"), ("kernel_b=0.7", C["orange"], "-"), ("kernel_b=0.8", C["green"], "-"), ("UMAP", C["grey"], "--")):
        v = [k45[f"scan/{n}/{label}"][key] for n in ns]; ax.plot(ns, v, color=col, lw=1.6, ls=ls, marker="o", ms=3.5); ends.append((v[-1], label, col))
    lo, hi = ax.get_ylim(); gap = 0.07 * (hi - lo); last = None
    for y_, label, col in sorted(ends):   # direct labels, pushed apart (in axis units) when they would collide
        yy = y_ if last is None or y_ - last > gap else last + gap
        ax.annotate(label.replace("kernel_b=", "b = "), (ns[-1], y_), xytext=(ns[-1] * 1.12, yy), textcoords="data", fontsize=7.5, color=col, va="center"); last = yy
    ax.set_ylim(lo, max(hi, (last or hi) + gap))
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(["3k", "5k", "10k", "20k", "35k"]); ax.set_xlabel("n (MNIST subsample)"); ax.set_ylabel(ylabel); ax.set_xlim(2500, 60000)
axes[0].set_title("a heavy tail tears small data sets", loc="left", fontsize=9); axes[1].set_title("and buys local fidelity at large n", loc="left", fontsize=9)
axes[0].axhline(10, color=C["black"], lw=0.7, ls=":"); axes[0].text(2600, 10.5, "10 classes", fontsize=7.5, color=C["black"])
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_small_n.png"), dpi=200); plt.close(fig)
print("figures written to", OUT)
