"""Summary of k27_transfer.json -> paper/tables/transfer.tex (means over runs of 15-NN / Spearman / centroids after the
pass, with the start's Spearman before it; references without a start). Usage: python benchmarks/theory/k27_table.py"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "k27_transfer.json")))
groups = {}
for k, v in d.items():
    ds, label, run = k.split("/"); groups.setdefault(ds, {}).setdefault(label, []).append(v)
ORDER = ["MAGNET kappa 0.5 (frozen)", "MAGNET kappa 0.1", "PaCMAP stiff (MN 4)", "PaCMAP default", "UMAP default", "PCA-2", "MAGNET hybrid (shipped)", "MAGNET full-batch", "openTSNE default"]
NAMES = {"covertype_20000": "covertype 20k", "covertype_70000": "covertype 70k", "covertype_250000": "covertype 250k", "letter_20000": "letter 20k", "rcv1_70000": "RCV1 70k (text)"}
L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"data set & start & $n$ & Spearman before & 15-NN after & Spearman after & centroids after \\", r"\midrule"]
def m(rows, key, w="after"):
    v = [r[w][key] for r in rows if r[w] is not None and r[w][key] == r[w][key]]; return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)
for ds in ["covertype_20000", "letter_20000", "covertype_70000", "rcv1_70000", "covertype_250000"]:
    if ds not in groups: continue
    print(f"\n{NAMES[ds]:<18}{'start':<28}{'n':>3}{'sp before':>11}{'knn15':>8}{'sp':>8}{'cent':>8}{'t(s)':>8}")
    for label in ORDER:
        if label not in groups[ds]: continue
        rows = groups[ds][label]; sb = m(rows, "sp", "start") if rows[0]["start"] else (float("nan"), 0)
        k15, sp, ce = m(rows, "knn15"), m(rows, "sp"), m(rows, "cent"); t = np.mean([r["t_start"] + r["t_pass"] for r in rows])
        print(f"{'':<18}{label:<28}{len(rows):>3}{sb[0]:>11.3f}{k15[0]:>8.3f}{sp[0]:>8.3f}{ce[0]:>8.3f}{t:>8.1f}")
        fmt = lambda x: "--" if x[0] != x[0] else f"{x[0]:.3f}" + (f" $\\pm$ {x[1]:.3f}" if len(rows) > 1 else "")
        lab = label.replace("MAGNET", r"\magnet{}").replace("kappa", r"$\kappa$ =")
        L.append(f"{NAMES[ds]} & {lab} & {len(rows)} & {fmt(sb)} & {fmt(k15)} & {fmt(sp)} & {fmt(ce)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(HERE, "..", "..", "paper", "tables", "transfer.tex"), "w").write("\n".join(L) + "\n")
