"""tables/timing.tex from benchmarks/cache/timing_quiet.json (interleaved timings, mean (min-max) over rounds)."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
res = json.load(open(os.path.join(ROOT, "benchmarks", "cache", "timing_quiet.json")))
M = ["MAGNET sampled", "MAGNET full-batch", "MAGNET sequence", "MAGNET sequence + post", "UMAP", "PaCMAP", "openTSNE", "Barnes-Hut t-SNE"]
L = [r"\begin{tabular}{lrr}", r"\toprule", r"configuration & MNIST 70k & Fashion-MNIST 70k \\", r"\midrule"]
for m in M:
    cells = []
    for ds in ("mnist", "fmnist"):
        v = [res[k]["t"] for k in res if k.startswith(f"{ds}/{m}/")]
        cells.append(f"{np.mean(v):.1f} ({min(v):.1f}--{max(v):.1f})" if v else "--")
    L.append(m.replace("MAGNET", r"\magnet{}").replace("Barnes-Hut t-SNE", "Barnes--Hut $t$-SNE") + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "tables", "timing.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[4:-2]))
