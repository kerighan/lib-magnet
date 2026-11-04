"""Tables of the short paper from the result files: tables/raw_axes.tex (k41, mean over five runs, every configuration
as shipped) and tables/timing.tex (benchmarks/cache/timing_quiet.json). Best per column in bold, second underlined,
compared at the printed precision. The control tables are written by make_fig_control.py.
Usage: python paper/human/make_tables.py"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from rankmarks import marks, latex
TAB = os.path.join(HERE, "tables")
k = json.load(open(os.path.join(ROOT, "benchmarks", "theory", "k41_matched_postprocess.json")))
DS = [("mnist", "MNIST 70k", ("knn15", "sp", "ari")), ("fmnist", "Fashion-MNIST 70k", ("knn15", "sp", "ari")), ("letter", "letter 20k", ("knn15", "sp")), ("covertype", "covertype 20k", ("knn15", "sp"))]
M = [("MAGNET hybrid", r"\magnet{} sequence"), ("MAGNET full-batch", r"\magnet{} full-batch"), ("MAGNET SGD", r"\magnet{} sampled"), ("sklearn t-SNE (BH)", "Barnes--Hut $t$-SNE"), ("openTSNE", "openTSNE"), ("UMAP", "UMAP"), ("PaCMAP", "PaCMAP"), ("TriMap", "TriMap"), ("h-NNE", "h-NNE"), ("densMAP", "densMAP"), ("PCA-2 + pass", "PCA-2 + our pass"), ("PCA-2", "PCA-2")]
H = {"knn15": "15-NN", "sp": "Sp.", "ari": "ARI"}
mean = lambda ds, m, a: float(np.mean([k[f"{ds}/{m}/{s}"]["plain"][a] for s in range(5)]))
cols = [(ds, a) for ds, _, axes in DS for a in axes]
REF = {"PCA-2 + pass", "PCA-2"}   # linear baseline and our pass on it: reference rows below a rule, not ranked
ranked = [m for m, _ in M if m not in REF]
mk = {c: dict(zip(ranked, marks([mean(c[0], m, c[1]) for m in ranked], True, 3))) for c in cols}
L = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", "configuration & " + " & ".join(f"\\multicolumn{{{len(axes)}}}{{c}}{{{t}}}" for _, t, axes in DS) + r" \\",
     " & " + " & ".join(H[a] for _, _, axes in DS for a in axes) + r" \\", r"\midrule"]
for m, lab in M:
    if m == "PCA-2 + pass": L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{reference rows (linear baseline, and our pass on it), not ranked}} \\" % (len(cols) + 1)]
    L.append(lab + " & " + " & ".join(latex(f"{mean(ds, m, a):.3f}", mk[(ds, a)].get(m)) for ds, a in cols) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "raw_axes.tex"), "w").write("\n".join(L) + "\n")
t = json.load(open(os.path.join(ROOT, "benchmarks", "cache", "timing_quiet.json")))
TM = ["MAGNET sampled", "MAGNET full-batch", "MAGNET sequence", "MAGNET sequence + post", "UMAP", "PaCMAP", "openTSNE", "Barnes-Hut t-SNE"]
v = {(ds, m): [t[x]["t"] for x in t if x.startswith(f"{ds}/{m}/")] for ds in ("mnist", "fmnist") for m in TM}
tk = {ds: marks([np.mean(v[(ds, m)]) for m in TM], False, 1) for ds in ("mnist", "fmnist")}
L = [r"\begin{tabular}{lrr}", r"\toprule", r"configuration & MNIST 70k & Fashion-MNIST 70k \\", r"\midrule"]
for i, m in enumerate(TM):
    cells = [latex(f"{np.mean(v[(ds, m)]):.1f}", tk[ds][i]) + f" ({min(v[(ds, m)]):.1f}--{max(v[(ds, m)]):.1f})" for ds in ("mnist", "fmnist")]
    L.append(m.replace("MAGNET", r"\magnet{}").replace("Barnes-Hut t-SNE", "Barnes--Hut $t$-SNE") + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "timing.tex"), "w").write("\n".join(L) + "\n"); print("raw_axes.tex and timing.tex written")

# ---- held-out scRNA-seq (k47): raw axes as shipped and the geometric composite plain / + post, mean over five runs
k47 = json.load(open(os.path.join(ROOT, "benchmarks", "theory", "k47_scrna_heldout.json")))
SC = [("tasic2018", "Tasic 2018 (23.8k, 26 subclasses)"), ("pbmc68k", "PBMC 68k (11 cell types)")]
mean47 = lambda ds, m, w, a: float(np.mean([k47[f"{ds}/{m}/{s}"][w][a] for s in range(5)]))
cols47 = [(ds, w, a) for ds, _ in SC for w, a in (("plain", "knn15"), ("plain", "sp"), ("plain", "ari"), ("plain", "geo"), ("post", "geo"))]
mk47 = {c: dict(zip(ranked, marks([mean47(c[0], m, c[1], c[2]) for m in ranked], True, 3))) for c in cols47}
L = [r"\begin{tabular}{l" + "r" * len(cols47) + "}", r"\toprule", "configuration & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{t}}}" for _, t in SC) + r" \\",
     " & " + " & ".join(["15-NN", "Sp.", "ARI", "geo", "+ post"] * len(SC)) + r" \\", r"\midrule"]
for m, lab in M:
    if m == "PCA-2 + pass": L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{reference rows (linear baseline, and our pass on it), not ranked}} \\" % (len(cols47) + 1)]
    L.append(lab + " & " + " & ".join(latex(f"{mean47(*c[:1], m, c[1], c[2]):.3f}", mk47[c].get(m)) for c in cols47) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "scrna.tex"), "w").write("\n".join(L) + "\n")
pd_ = {ds: np.array([k47[f"{ds}/MAGNET hybrid/{s}"]["post"]["geo"] - k47[f"{ds}/sklearn t-SNE (BH)/{s}"]["post"]["geo"] for s in range(5)]) for ds, _ in SC}
print("scrna.tex written; sequence - BH after the step:", {ds: f"{v.mean():+.3f} +- {v.std(ddof=1):.3f}" for ds, v in pd_.items()})
