"""Mean and standard deviation over the seeds of benchmarks/seeds.py -> benchmarks/cache/seeds_summary.json and
paper/tables/seeds.tex. Usage: python benchmarks/seeds_summary.py"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl")) if '"tag": "seeds_' in l]
KEYS = [("time", "time (s)"), ("knn15_preservation", "15-NN"), ("trustworthiness", "trust."), ("dist_spearman", "Spearman"), ("triplet_acc", "triplet"),
        ("knn_acc", "10-NN acc."), ("centroid_spearman", "centroids")]
LABEL = {"magnet_sgd": r"\magnet{} SGD", "magnet_tsne": r"\magnet{} full-batch", "magnet_hybrid": r"\magnet{} hybrid",
         "umap": "UMAP 0.5.12", "opentsne": "openTSNE 1.0.4", "pacmap": "PaCMAP"}
ORDER = ["magnet_sgd", "magnet_tsne", "magnet_hybrid", "umap", "opentsne", "pacmap"]
out, L = {}, [r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r"data set & method & $n$ & " + " & ".join(h for _, h in KEYS) + r" \\", r"\midrule"]
for ds, label in (("mnist70000", "MNIST"), ("fmnist70000", "Fashion-MNIST")):
    print(f"\n{label}")
    print(f"{'method':<16}{'n':>3}" + "".join(f"{h:>18}" for _, h in KEYS))
    for m in ORDER:
        R = [r for r in rows if r["dataset"] == ds and r["method"] == m]
        if not R:
            continue
        # one row per seed: keep the last run of each seed
        by = {r["params"]["random_state"]: r for r in R}; R = list(by.values())
        st = {k: (float(np.mean([r[k] for r in R])), float(np.std([r[k] for r in R], ddof=1)) if len(R) > 1 else 0.0) for k, _ in KEYS}
        out[f"{ds}/{m}"] = dict(n=len(R), **{k: v for k, v in st.items()})
        fmt = lambda k: (f"{st[k][0]:.1f} $\\pm$ {st[k][1]:.1f}" if k == "time" else f"{st[k][0]:.3f} $\\pm$ {st[k][1]:.3f}")
        print(f"{m:<16}{len(R):>3}" + "".join((f"{st[k][0]:>11.1f} ±{st[k][1]:.1f}  " if k == "time" else f"{st[k][0]:>11.3f} ±{st[k][1]:.3f}") for k, _ in KEYS))
        L.append(f"{label} & {LABEL[m]} & {len(R)} & " + " & ".join(fmt(k) for k, _ in KEYS) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
json.dump(out, open(os.path.join(HERE, "cache", "seeds_summary.json"), "w"), indent=1)
open(os.path.join(HERE, "..", "paper", "tables", "seeds.tex"), "w").write("\n".join(L) + "\n")
