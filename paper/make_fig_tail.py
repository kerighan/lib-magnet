"""Figure: per-octave repulsive force on fixed layouts, binned by the kernel crossover (benchmarks/theory/k13).
Also writes tables/kscale.tex from benchmarks/cache/extra_metrics_mnist.json."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
rows = json.load(open(os.path.join(ROOT, "benchmarks", "theory", "k13_tail_crossover.json")))
COL = {0.5: "#1f4e79", 1.0: "#2a9d8f", 1.5: "#e9a23b", 2.0: "#c0392b"}
panels = [("MNIST 20k 2-D (defaults)", 1, "(a) MNIST 2-D, layout scale"),
          ("MNIST 20k 2-D (defaults)", 64, "(b) MNIST 2-D, rescaled x64"),
          ("MNIST 20k 3-D (defaults)", 64, "(c) MNIST 3-D, rescaled x64"),
          ("Gaussian cloud 2-D", 64, "(d) Gaussian 2-D, rescaled x64")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=False)
for ax, (lab, s, title) in zip(axes, panels):
    for r in rows:
        if r["layout"] != lab or r["s"] != s: continue
        c = np.array(r["centers"]); S = np.array(r["S"]); cnt = np.array(r["cnt"])
        ok = cnt > 0
        ax.plot(c[ok], np.log2(S[ok] + 1e-30), color=COL[r["br"]], lw=1.6, marker="o", ms=3,
                label=f"$b_r$={r['br']:g}")
        if r["window"]:
            lo, hi = r["window"]; w = (c > lo) & (c < hi)
            x = c[w]; y = np.log2(S[w] + 1e-30)
            if len(x) >= 2:
                p = np.polyfit(x, y, 1)
                ax.plot([lo, hi], np.polyval(p, [lo, hi]), color=COL[r["br"]], lw=3, alpha=0.35, solid_capstyle="butt")
                ax.annotate(f"{r['slope_mag']:+.2f} ({r['adjusted']:+.2f})", (hi, np.polyval(p, hi)),
                            fontsize=7, color=COL[r["br"]], xytext=(3, 0), textcoords="offset points", va="center")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(r"$\log_2(d / d_{k,r})$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_box_aspect(0.9)
axes[0].set_ylabel(r"$\log_2$ total scalar force per octave")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.text(0.5, -0.04, "thick segment: fit window ($a d^{2b_r} > 100$, $d < R/2$); label: measured slope (shell-count-adjusted prediction $\\alpha - 1 - 2b_r$)",
         ha="center", fontsize=8, color="0.3")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "tail_crossover.png"), dpi=200, bbox_inches="tight")
print("figure written")

# ---- table of neighbourhood scale
em = json.load(open(os.path.join(ROOT, "benchmarks", "cache", "extra_metrics_mnist.json")))
L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r"MNIST 70k & $k=5$ & $k=15$ & $k=50$ & $k=100$ & trust$_{15}$ & cont$_{15}$ & trust$_{50}$ & cont$_{50}$ \\", r"\midrule"]
best = {k: max(v[k] for v in em.values()) for k in ("knn5", "knn15", "knn50", "knn100", "trust15", "cont15", "trust50", "cont50")}
for name, r in em.items():
    cells = []
    for k in ("knn5", "knn15", "knn50", "knn100", "trust15", "cont15", "trust50", "cont50"):
        v = f"{r[k]:.3f}"; cells.append(r"\textbf{" + v + "}" if r[k] == best[k] else v)
    L.append(name.replace("MAGNET", r"\magnet{}") + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "tables", "kscale.tex"), "w").write("\n".join(L) + "\n")
print("table written")
