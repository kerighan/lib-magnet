"""The matched post-process control (k41): every configuration with and without the identical post-process, five
end-to-end seeds, composites against the frozen unprocessed references computed per run. Produces
figures/fig_control.png (geometric composite, plain and post-processed, mean +- sd, four data sets) and prints the
paired differences against MAGNET full-batch + post-process and MAGNET hybrid + post-process, plus LaTeX tables
(tables/control_geo.tex, tables/control_min.tex). Usage: python paper/human/make_fig_control.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
res = json.load(open(os.path.join(ROOT, "benchmarks", "theory", "k41_matched_postprocess.json")))
OUT = os.path.join(HERE, "figures"); TAB = os.path.join(HERE, "tables"); os.makedirs(TAB, exist_ok=True)
DS = [("mnist", "MNIST 70k"), ("fmnist", "Fashion-MNIST 70k"), ("letter", "letter 20k"), ("covertype", "covertype 20k")]
METHODS = ["MAGNET hybrid", "MAGNET full-batch", "MAGNET SGD", "sklearn t-SNE (BH)", "openTSNE", "UMAP", "PaCMAP", "TriMap", "h-NNE", "densMAP", "PCA-2 + pass", "PCA-2"]
LABEL = {"MAGNET hybrid": "MAGNET sequence", "MAGNET full-batch": "MAGNET full-batch", "MAGNET SGD": "MAGNET sampled", "sklearn t-SNE (BH)": "Barnes-Hut t-SNE", "PCA-2 + pass": "PCA-2 + our pass"}
C = {"magnet": "#0072B2", "other": "#7F7F7F", "post": "#E69F00"}

def runs(ds, m, which, key="geo"):
    return [res[f"{ds}/{m}/{s}"][which][key] for s in range(5) if f"{ds}/{m}/{s}" in res]
def stat(v): v = np.asarray(v, float); return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, np.nan, 0)

# ---- figure: geometric composite per data set, plain (grey/blue) and post-processed (orange marker), mean +- sd
fig, axes = plt.subplots(1, 4, figsize=(11, 3.7), sharey=False)
for ax, (ds, title) in zip(axes, DS):
    rows = [(m, stat(runs(ds, m, "plain")), stat(runs(ds, m, "post"))) for m in METHODS]
    rows = [r for r in rows if r[2][2] > 0]
    rows.sort(key=lambda r: r[2][0])
    y = np.arange(len(rows))
    for i, (m, p, q) in enumerate(rows):
        col = C["magnet"] if m.startswith("MAGNET") else C["other"]
        ax.plot([p[0], q[0]], [i, i], color="#CCCCCC", lw=1.2, zorder=1)
        ax.errorbar(p[0], i, xerr=p[1], fmt="o", color=col, ms=4, capsize=2, lw=1, zorder=2); ax.scatter(runs(ds, m, "post"), [i] * len(runs(ds, m, "post")), color=C["post"], s=5, alpha=0.5, zorder=2.5)
        ax.errorbar(q[0], i, xerr=q[1], fmt="D", color=C["post"], ms=4, capsize=2, lw=1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(r[0], r[0]) for r in rows], fontsize=7.5)
    d = np.array([res[f"{ds}/MAGNET hybrid/{s}"]["post"]["geo"] - res[f"{ds}/sklearn t-SNE (BH)/{s}"]["post"]["geo"] for s in range(5) if f"{ds}/MAGNET hybrid/{s}" in res and f"{ds}/sklearn t-SNE (BH)/{s}" in res])
    if len(d): ax.set_xlabel(f"geometric composite (frozen references)\nsequence minus Barnes-Hut, both post-processed: {d.mean():+.3f} +- {d.std(ddof=1):.3f} (sd, {len(d)} paired runs)", fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=9); ax.grid(axis="y", visible=False)
axes[0].plot([], [], "o", color=C["magnet"], label="MAGNET, as shipped"); axes[0].plot([], [], "o", color=C["other"], label="comparator, as shipped"); axes[0].plot([], [], "D", color=C["post"], label="+ the same post-process")
fig.legend(loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(OUT, "fig_control.png"), dpi=200); plt.close(fig)

# ---- tables and paired differences
for kind in ("geo", "min"):
    L = [r"\begin{tabular}{l" + "rr" * len(DS) + "r}", r"\toprule", "configuration & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for _, t in DS) + r" & mean \\",
         " & " + " & ".join("plain & + post" for _ in DS) + r" & + post \\", r"\midrule"]
    print(f"\n==== composite {kind} (mean +- sd over seeds; paired difference of the post-processed configuration against MAGNET full-batch + post and MAGNET sequence + post)")
    print(f"{'configuration':<22}" + "".join(f"{t:>26}" for _, t in DS) + f"{'mean post':>11}{'d vs fb+post':>14}{'d vs seq+post':>15}")
    for m in METHODS:
        cells, means, diffs_fb, diffs_hy = [], [], [], []
        for ds, _ in DS:
            pp = stat([res[f"{ds}/{m}/{s}"]["plain"][kind] for s in range(5) if f"{ds}/{m}/{s}" in res]); qq = stat([res[f"{ds}/{m}/{s}"]["post"][kind] for s in range(5) if f"{ds}/{m}/{s}" in res])
            cells.append((pp, qq)); means.append(qq[0])
            for ref, acc in (("MAGNET full-batch", diffs_fb), ("MAGNET hybrid", diffs_hy)):
                ks = [s for s in range(5) if f"{ds}/{m}/{s}" in res and f"{ds}/{ref}/{s}" in res]
                acc.extend([res[f"{ds}/{m}/{s}"]["post"][kind] - res[f"{ds}/{ref}/{s}"]["post"][kind] for s in ks])
        if all(np.isnan(c[1][0]) for c in cells): continue
        L.append(LABEL.get(m, m).replace("MAGNET", r"\magnet{}") + " & " + " & ".join(f"{pp[0]:.3f} & {qq[0]:.3f}" if qq[2] else "-- & --" for pp, qq in cells) + f" & {np.nanmean(means):.3f} \\\\")
        print(f"{LABEL.get(m, m):<22}" + "".join(f"{pp[0]:.3f}+-{pp[1]:.2f} > {qq[0]:.3f}+-{qq[1]:.2f}".rjust(26) if qq[2] else "".rjust(26) for pp, qq in cells)
              + f"{np.nanmean(means):>11.3f}" + (f"{np.mean(diffs_fb):>+10.3f}+-{np.std(diffs_fb, ddof=1) if len(diffs_fb) > 1 else 0:.3f}" if diffs_fb else "".rjust(14)) + (f"{np.mean(diffs_hy):>+10.3f}+-{np.std(diffs_hy, ddof=1) if len(diffs_hy) > 1 else 0:.3f}" if diffs_hy else ""))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, f"control_{kind}.tex"), "w").write("\n".join(L) + "\n")
print("\nwritten:", os.path.join(OUT, "fig_control.png"), "and tables/control_{geo,min}.tex")
