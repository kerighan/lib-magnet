"""k36: replace the fragile coarse-split contraction (k35: the useful 2-split is '0 vs the rest' and depends on the
Leiden seed) by a community PLACEMENT: the fine communities (Leiden r=0.3) are translated as rigid bodies, coordinate
ascent of the sample Spearman (input vs layout distances, 1000 points, seed 7 = disjoint from the benchmark sample)
over (a) a radial scale of each centroid offset s in {0.7 .. 1.6}, or (b) a 2-D step grid; two sweeps. Then the fine
contraction at 0.7. Scores on the benchmark sample on the four data sets, MAGNET full-batch / hybrid layouts.
Usage: python benchmarks/theory/k36_community_placement.py [datasets=mnist,fmnist,letter,covertype]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import communities, contract
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(",")
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k36_community_placement.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
AXES = {"mnist": ("knn15", "sp", "ari"), "fmnist": ("knn15", "sp", "ari"), "letter": ("knn15", "sp"), "covertype": ("knn15", "sp")}

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    return load27(name, 20000)

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

class Spearman:
    """sample Spearman between input distances and layout distances; incremental: only the rows of moved points change."""
    def __init__(self, X, idx):
        self.idx = idx; self.iu = np.triu_indices(len(idx), 1)
        rh = rankdata(pairwise_distances(X[idx])[self.iu]); self.rh = (rh - rh.mean()) / rh.std()
    def __call__(self, Y):
        Z = Y[self.idx]; D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))[self.iu]
        rl = rankdata(D); rl = (rl - rl.mean()) / rl.std(); return float(np.mean(self.rh * rl))

def place(Y, lab, spf, mode, sweeps=2, log=None):
    """coordinate ascent over the communities' rigid translations."""
    Y = Y.copy(); k = lab.max() + 1; centre = Y.mean(0)
    cur = spf(Y)
    for sw in range(sweeps):
        moved = 0
        for c in range(k):
            m = lab == c; mu = Y[m].mean(0)
            if mode == "radial":
                cands = [(mu - centre) * (s - 1.0) for s in (0.7, 0.85, 1.15, 1.3, 1.6)]
            else:
                step = 0.15 * np.sqrt(((Y - centre) ** 2).sum(1).mean())
                cands = [np.array([dx, dy]) * step for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
            best, best_v = None, cur
            for d in cands:
                Z = Y.copy(); Z[m] += d.astype(np.float32); v = spf(Z)
                if v > best_v + 1e-4: best, best_v = d, v
            if best is not None: Y[m] += best.astype(np.float32); cur = best_v; moved += 1
        if log is not None: log.append((sw, moved, cur))
        if moved == 0: break
    return Y

for ds in DS:
    X, y, key = load(ds); y = np.asarray(y).astype(int); rf = ref(ds); n = len(X); axes = AXES[ds]
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    P = sp.load_npz(pc) if os.path.exists(pc) else sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32)
    lab = communities(P, 0.3)
    spf = Spearman(np.asarray(X, dtype=np.float32), _idx(n, 1000, 7))
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key); s = _idx(n, 10000, 2); kcl = len(np.unique(y))
        r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))))
        sh = [r[a] / rf[a] for a in axes]; r["geo"] = float(np.prod(sh) ** (1 / len(sh))); r["min"] = float(min(sh)); return r
    print(f"\n{ds}: {lab.max() + 1} communities (ARI vs labels {adjusted_rand_score(y, lab):.2f}); panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}; axes {axes}", flush=True)
    print(f"{'layout':<11}{'scheme':<30}{'knn15':>7}{'sp':>7}{'ari':>7}{'geo':>7}{'min':>6}{'t':>6}")
    slug = {"full-batch": "magnetlosstsne", "hybrid": "magnetlosshybrid"}
    for lay in ("full-batch", "hybrid"):
        fig = os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_{slug[lay]}.npz"); p = os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_{lay}.npz")
        Y0 = np.load(fig if os.path.exists(fig) else p)["Y"].astype(np.float32)
        for name, f in (("identity", lambda: (Y0, 0.0)), ("fine 0.7", lambda: (contract(Y0, lab, 0.7), 0.0)),
                        ("radial placement", lambda: (lambda t: (place(Y0, lab, spf, "radial"), time.time() - t))(time.time())),
                        ("radial placement > fine 0.7", lambda: (lambda t: (contract(place(Y0, lab, spf, "radial"), lab, 0.7), time.time() - t))(time.time())),
                        ("grid placement > fine 0.7", lambda: (lambda t: (contract(place(Y0, lab, spf, "grid"), lab, 0.7), time.time() - t))(time.time())),
                        ("radial > grid > fine 0.7", lambda: (lambda t: (contract(place(place(Y0, lab, spf, "radial"), lab, spf, "grid"), lab, 0.7), time.time() - t))(time.time()))):
            k_ = f"{ds}/{lay}/{name}"
            if k_ not in res:
                Z, dt = f(); res[k_] = score(np.ascontiguousarray(Z, dtype=np.float32)); res[k_]["t"] = dt; json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"{lay:<11}{name:<30}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}{r['t']:>6.1f}", flush=True)
