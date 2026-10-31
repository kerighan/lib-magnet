"""Summarise results_tail_law.jsonl: optimum b per (dataset, m) and per metric, candidate laws."""
import os, json, sys
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results_tail_law.jsonl"))]
METRICS = ["knn15_preservation", "trustworthiness", "dist_spearman", "triplet_acc"]


def agg(stage):
    d = defaultdict(list)
    for r in rows:
        if r["stage"] == stage:
            d[(r["dataset"], r["n"], r["m"], r["q"], r["b"])].append(r)
    return {k: {met: float(np.mean([x[met] for x in v])) for met in METRICS} for k, v in d.items()}


def interp_opt(bs, vals):
    """Argmax refined by a parabola through the best point and its neighbours (in log b)."""
    i = int(np.argmax(vals))
    if 0 < i < len(bs) - 1:
        x = np.log(bs[i - 1:i + 2]); y = np.array(vals[i - 1:i + 2])
        c = np.polyfit(x, y, 2)
        if c[0] < 0:
            return float(np.exp(-c[1] / (2 * c[0])))
    return float(bs[i])


for stage in ("A", "B", "C"):
    a = agg(stage)
    if not a:
        continue
    print(f"\n######## stage {stage}")
    groups = sorted({(k[0], k[1], k[2], k[3]) for k in a}, key=lambda g: (g[0], g[1], g[3] or 0, g[2]))
    summary = []
    for g in groups:
        bs = sorted(k[4] for k in a if k[:4] == g)
        print(f"\n-- {g[0]} n={g[1]} m={g[2]}" + (f" q={g[3]}" if g[3] else ""))
        print("b      " + "".join(f"{b:>7}" for b in bs))
        opts = {}
        for met in METRICS:
            vals = [a[(*g, b)][met] for b in bs]
            opts[met] = (bs[int(np.argmax(vals))], interp_opt(bs, vals))
            print(f"{met[:6]} " + "".join(f"{v:7.3f}" for v in vals) + f"   best b={opts[met][0]} (~{opts[met][1]:.2f})")
        summary.append((g, opts))
    print(f"\n== stage {stage} summary: best b by metric (grid / parabolic)")
    print(f"{'dataset':12s} {'n':>6} {'m':>2} {'q':>3}  " + "  ".join(f"{m[:6]:>12}" for m in METRICS))
    for g, opts in summary:
        print(f"{g[0]:12s} {g[1]:6d} {g[2]:2d} {str(g[3] or ''):>3}  " + "  ".join(f"{opts[m][0]:>5} /{opts[m][1]:5.2f}" for m in METRICS))
    if stage == "A":
        print("\n== candidate laws for the 15-NN optimum (parabolic), mean over datasets per m")
        per_m = defaultdict(list)
        for g, opts in summary:
            per_m[g[2]].append(opts["knn15_preservation"][1])
        ms = sorted(per_m)
        meas = np.array([np.mean(per_m[m]) for m in ms])
        print("m:        " + " ".join(f"{m:5d}" for m in ms))
        print("measured: " + " ".join(f"{v:5.2f}" for v in meas))
        for name, law in [("(m-1)/2 force scale-free", lambda m: (m - 1) / 2), ("m/2 kernel-mass scale-free", lambda m: m / 2), ("m/3.3", lambda m: m / 3.3)]:
            pred = np.array([max(law(m), 0.3) for m in ms])
            print(f"{name:28s}" + " ".join(f"{v:5.2f}" for v in pred) + f"   rmse={np.sqrt(np.mean((pred - meas) ** 2)):.2f}")
        c = np.polyfit(ms, meas, 1)
        print(f"linear fit: b = {c[0]:.3f} m + {c[1]:.3f}")
