"""Staged 2 x 2 design with seeds: is the hybrid's global lead the stiff mid-range spring, or MAGNET's SGD start?
Starts: MAGNET SGD at spring 0 / 0.1 (default) / 0.5 (hybrid) / 1.0; PaCMAP at its defaults and at the 'stiff'
mid-near setting chosen by pacmap_sweep.py on MNIST (frozen before Fashion-MNIST); UMAP at its defaults; openTSNE
(2 seeds, slow). Every start goes through the same full-batch pass (300 iterations, no exaggeration, no spring, hand-over
scale 31 sqrt(n/70000), MAGNET's 15-NN graph of the data). 5 seeds (UMAP: unseeded independent runs), MNIST and
Fashion-MNIST 70k, all cores. Paired before / after: 15-NN, trustworthiness, distance Spearman, triplets, centroid
Spearman, time of the start and of the pass. -> benchmarks/cache/staged_seeds.json, paper/tables/staged_seeds.tex.
Usage: python benchmarks/staged_seeds.py [mnist,fmnist] [seeds=5]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import spearmanr
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne
import scipy.sparse as sp

kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
SEEDS = int(kw.get("seeds", 5))
stiff = json.load(open(os.path.join(HERE, "cache", "pacmap_sweep.json")))["stiff"]
KEYS = ("knn15_preservation", "trustworthiness", "dist_spearman", "triplet_acc", "centroid_spearman")

def centroid_spearman(X, Y, y):
    C = np.array([X[y == c].mean(0) for c in np.unique(y)]); L = np.array([Y[y == c].mean(0) for c in np.unique(y)]); iu = np.triu_indices(len(C), 1)
    return float(spearmanr(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu], np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1))[iu]).correlation)

out = json.load(open(os.path.join(HERE, "cache", "staged_seeds.json"))) if os.path.exists(os.path.join(HERE, "cache", "staged_seeds.json")) else {}
for ds in (sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else "mnist,fmnist").split(","):
    X, y, name = LOADERS[ds](70000); y = np.asarray(y).astype(int); n = len(X)
    import umap; umap.UMAP(n_epochs=20).fit_transform(np.random.rand(6000, 10).astype(X.dtype))
    import pacmap; pacmap.PaCMAP(num_iters=(5, 5, 5)).fit_transform(X[:6000])
    P = None
    def graph():
        global P
        if P is None:
            P = sp.csr_matrix(MAGNET(random_state=0).fit(X[:0] if False else X).graph_, dtype=np.float32)
        return P
    def metr(Y):
        r = evaluate(X, Y, y, cache_key=name); r["centroid_spearman"] = centroid_spearman(X, Y, y); return {k: r[k] for k in KEYS}
    starts = [("MAGNET SGD, spring 0", lambda s: MAGNET(loss="umap", mid_weight=0.0, random_state=s).fit_transform(X), SEEDS),
              ("MAGNET SGD, spring 0.1", lambda s: MAGNET(loss="umap", mid_weight=0.1, random_state=s).fit_transform(X), SEEDS),
              ("MAGNET SGD, spring 0.5", lambda s: MAGNET(loss="umap", mid_weight=0.5, random_state=s).fit_transform(X), SEEDS),
              ("MAGNET SGD, spring 1.0", lambda s: MAGNET(loss="umap", mid_weight=1.0, random_state=s).fit_transform(X), SEEDS),
              ("PaCMAP default", lambda s: pacmap.PaCMAP(random_state=s).fit_transform(X), SEEDS),
              (f"PaCMAP stiff (MN {stiff['MN_ratio']}, iters {tuple(stiff['num_iters'])})", lambda s: pacmap.PaCMAP(MN_ratio=stiff["MN_ratio"], num_iters=tuple(stiff["num_iters"]), random_state=s).fit_transform(X), SEEDS),
              ("UMAP default", lambda s: umap.UMAP().fit_transform(X), SEEDS),
              ("openTSNE default", lambda s: np.asarray(__import__("openTSNE").TSNE(n_jobs=-1, random_state=s).fit(X)), min(2, SEEDS))]
    for label, f, ns in starts:
        key = f"{name}/{label}"
        if key in out and len(out[key]) >= ns:
            print(f"skip {key}", flush=True); continue
        rows = []
        for s in range(ns):
            t = time.time(); Y0 = np.asarray(f(s), dtype=np.float32); t_start = time.time() - t
            r0 = metr(Y0)
            Y = Y0.copy(); Y -= Y.mean(0); Y *= 31.0 * np.sqrt(n / 70000.0) / max(float(Y[:, 0].std()), 1e-12)
            t = time.time(); Y = optimize_tsne(graph(), Y, n_iter=300, early_iter=0, n_threads=-1); t_pass = time.time() - t
            r1 = metr(Y)
            rows.append(dict(seed=s, before=r0, after=r1, t_start=t_start, t_pass=t_pass))
            print(f"{name} {label:<42} seed {s}: {r0['knn15_preservation']:.3f}/{r0['dist_spearman']:.3f}/{r0['centroid_spearman']:.3f} -> "
                  f"{r1['knn15_preservation']:.3f}/{r1['dist_spearman']:.3f}/{r1['centroid_spearman']:.3f}  ({t_start:.1f} + {t_pass:.1f} s)", flush=True)
        out[key] = rows
        json.dump(out, open(os.path.join(HERE, "cache", "staged_seeds.json"), "w"), indent=1)

# summary table
L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"data & start & $n$ & 15-NN & Spearman & triplets & centroids \\", r"\midrule"]
for ds, label in (("mnist70000", "MNIST"), ("fmnist70000", "Fashion")):
    print(f"\n{label}: {'start':<42}{'n':>3}   15-NN before -> after      Spearman before -> after     centroids before -> after")
    for key, rows in out.items():
        if not key.startswith(ds + "/"): continue
        st = key.split("/", 1)[1]
        def ms(k, w): v = [r[w][k] for r in rows]; return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)
        cells = []
        for k in ("knn15_preservation", "dist_spearman", "triplet_acc", "centroid_spearman"):
            b, a = ms(k, "before"), ms(k, "after"); cells.append(f"{b[0]:.3f} $\\to$ {a[0]:.3f} $\\pm$ {a[1]:.3f}")
        print(f"{'':<{len(label) + 2}}{st:<42}{len(rows):>3}   " + "   ".join(f"{ms(k, 'before')[0]:.3f} -> {ms(k, 'after')[0]:.3f} ±{ms(k, 'after')[1]:.3f}" for k in ("knn15_preservation", "dist_spearman", "centroid_spearman")))
        L.append(f"{label} & {st.replace('MAGNET', chr(92) + 'magnet{}')} & {len(rows)} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(HERE, "..", "paper", "tables", "staged_seeds.tex"), "w").write("\n".join(L) + "\n")
