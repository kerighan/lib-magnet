"""k33: the k32 community contraction as a linear graph filter of the layout. Z = c Y + (1 - c) S_t Y with S = D^-1 P
(random-walk diffusion on the fuzzy graph): slow modes (community arrangement) kept at unit gain, fast modes (detail
within communities) attenuated to gain c. No community detection, continuous across community borders; t plays the
role of the resolution, c of the contraction. Two-scale variant Z = a0 Y + a1 S^t1 Y + a2 S^t2 Y with the coarse gain
chosen by a line search on the sample Spearman (label-free). Scores on the cached full-batch / hybrid layouts of MNIST
and Fashion. Usage: python benchmarks/theory/k33_diffusion_filter.py [datasets=mnist,fmnist]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate, high_dim_pdist
from magnet import MAGNET
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist").split(",")
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k33_diffusion_filter.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
TS = (10, 30, 100, 300, 1000); CS = (0.8, 0.6, 0.4)

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

def score(X, Y, y, key, rf):
    e = evaluate(X, Y, y, cache_key=key); s = _idx(len(Y), 10000, 2); kcl = len(np.unique(y))
    ari = float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s])))
    r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], trip=e["triplet_acc"], ari=ari)
    sh = [r[a] / rf[a] for a in ("knn15", "sp", "ari")]; r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh)); return r

def diffuse(S, Y, t):
    """S^t Y by repeated sparse products; returns all intermediate powers requested in TS."""
    out = {}; Z = Y.astype(np.float64)
    for k in range(1, max(TS) + 1):
        Z = S @ Z
        if k in TS: out[k] = Z.astype(np.float32)
    return out

for ds in DS:
    X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = ref(ds); n = len(X)
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    if os.path.exists(pc): P = sp.load_npz(pc)
    else: P = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32); sp.save_npz(pc, P)
    d = np.asarray(P.sum(1)).ravel(); S = sp.diags(1.0 / np.maximum(d, 1e-12)) @ P
    g = _idx(n, 2000, 1); Dh = high_dim_pdist(X, g, key); iu = np.triu_indices(len(g), 1); dh = Dh[iu]
    def sp_fast(Y):
        Dl = np.sqrt(((Y[g, None, :] - Y[None, g, :]) ** 2).sum(-1)); return float(spearmanr(dh, Dl[iu]).correlation)
    print(f"\n{ds}: panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}", flush=True)
    print(f"{'layout':<11}{'filter':<36}{'knn15':>7}{'sp':>7}{'trip':>7}{'ari':>7}{'geo':>7}{'min':>6}")
    layouts = {"full-batch": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosstsne.npz"),
               "hybrid": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosshybrid.npz")}
    for lay, p in layouts.items():
        Y0 = np.load(p)["Y"].astype(np.float32); t0 = time.time(); SY = diffuse(S, Y0, max(TS)); dt = time.time() - t0
        print(f"  ({lay}: diffusion up to t={max(TS)} in {dt:.1f} s)", flush=True)
        k_ = f"{ds}/{lay}/identity"
        if k_ not in res: res[k_] = score(X, Y0, y, key, rf); json.dump(res, open(OUT, "w"), indent=1)
        r = res[k_]; print(f"{lay:<11}{'identity':<36}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
        # one-scale filter
        for t in TS:
            for c in CS:
                k_ = f"{ds}/{lay}/t{t} c{c}"
                if k_ not in res:
                    Z = c * Y0 + (1 - c) * SY[t]; res[k_] = score(X, np.ascontiguousarray(Z, dtype=np.float32), y, key, rf); json.dump(res, open(OUT, "w"), indent=1)
                r = res[k_]; print(f"{lay:<11}{f't={t} c={c}':<36}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
        # two-scale: coarse gain by Spearman line search (t2 = 1000), then fine level
        k_ = f"{ds}/{lay}/coarse search"
        if k_ not in res:
            res[k_] = {str(a): sp_fast((1 - a) * Y0 + a * SY[1000]) for a in (0.0, 0.2, 0.4, 0.5, 0.6)}; json.dump(res, open(OUT, "w"), indent=1)
        print(f"  coarse line search (t=1000): " + "  ".join(f"a={a}: {v:.3f}" for a, v in res[k_].items()), flush=True)
        a_best = float(max(res[k_], key=lambda a: res[k_][a]))
        for t1 in (30, 100):
            for c in (0.8, 0.6):
                k_ = f"{ds}/{lay}/coarse a{a_best} > t{t1} c{c}"
                if k_ not in res:
                    Z = (1 - a_best) * Y0 + a_best * SY[1000]; Zs = (1 - a_best) * SY[t1] + a_best * SY[1000]   # filter applied after the coarse step: S commutes with itself
                    Z = c * Z + (1 - c) * Zs; res[k_] = score(X, np.ascontiguousarray(Z, dtype=np.float32), y, key, rf); json.dump(res, open(OUT, "w"), indent=1)
                r = res[k_]; print(f"{lay:<11}{f'coarse a={a_best} > t={t1} c={c}':<36}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
