"""k37: collision-aware community placement. k36 showed rigid translations of the fine communities raise the sample
Spearman a lot (MNIST full-batch 0.342 -> 0.501) but the bodies overlap (15-NN 0.348 -> 0.313, ARI loss). Here a move
is accepted only if the Spearman improves AND the mixing of the layout (fraction of a 10k sample whose nearest sample
point belongs to another community) does not rise above its starting value + tol. Candidates: radial scales and an
8-direction step grid; communities below 1 % of the points are not moved. Orders: placement then fine contraction 0.7,
or contraction first. Usage: python benchmarks/theory/k37_placement_safe.py [datasets=mnist,fmnist,letter,covertype]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import communities, contract
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist,letter,covertype").split(",")
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k37_placement_safe.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
AXES = {"mnist": ("knn15", "sp", "ari"), "fmnist": ("knn15", "sp", "ari"), "letter": ("knn15", "sp"), "covertype": ("knn15", "sp")}
REF = {"mnist": dict(knn15=0.352, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.328, sp=0.873, ari=0.483),
       "letter": dict(knn15=0.642, sp=0.716, ari=0.246), "covertype": dict(knn15=0.632, sp=0.755, ari=0.103)}   # panel bests of the 12-method table (frozen)

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    from k27_transfer import load as load27
    return load27(name, 20000)

class Spearman:
    def __init__(self, X, idx):
        self.idx = idx; self.iu = np.triu_indices(len(idx), 1)
        rh = rankdata(pairwise_distances(X[idx])[self.iu]); self.rh = (rh - rh.mean()) / rh.std()
    def __call__(self, Y):
        Z = Y[self.idx]; D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))[self.iu]
        rl = rankdata(D); rl = (rl - rl.mean()) / rl.std(); return float(np.mean(self.rh * rl))

class Mixing:
    def __init__(self, lab, idx): self.idx = idx; self.lab = lab[idx]
    def __call__(self, Y):
        Z = Y[self.idx]; nn = cKDTree(Z).query(Z, k=2)[1][:, 1]; return float(np.mean(self.lab[nn] != self.lab))

def place(Y, lab, spf, mixf=None, tol=0.002, sweeps=2, min_frac=0.01, log=None):
    Y = Y.copy(); k = lab.max() + 1; centre = Y.mean(0); R = np.sqrt(((Y - centre) ** 2).sum(1).mean())
    sizes = np.bincount(lab, minlength=k); cur = spf(Y); mix0 = mixf(Y) if mixf else None
    for sw in range(sweeps):
        moved = 0
        for c in np.argsort(-sizes):
            if sizes[c] < min_frac * len(Y): break
            m = lab == c; mu = Y[m].mean(0)
            cands = [(mu - centre) * (s - 1.0) for s in (0.7, 0.85, 1.15, 1.3, 1.6)]
            cands += [np.array([dx, dy]) * 0.15 * R for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
            best, best_v = None, cur
            for d in cands:
                Z = Y.copy(); Z[m] += d.astype(np.float32); v = spf(Z)
                if v > best_v + 1e-4 and (mixf is None or mixf(Z) <= mix0 + tol): best, best_v = d, v
            if best is not None: Y[m] += best.astype(np.float32); cur = best_v; moved += 1
        if log is not None: log.append((sw, moved, round(cur, 4)))
        if moved == 0: break
    return Y

for ds in DS:
    X, y, key = load(ds); y = np.asarray(y).astype(int); rf = REF[ds]; n = len(X); axes = AXES[ds]
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    P = sp.load_npz(pc) if os.path.exists(pc) else sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32)
    lab = communities(P, 0.3)
    spf = Spearman(np.asarray(X, dtype=np.float32), _idx(n, 1000, 7)); mixf = Mixing(lab, _idx(n, 10000, 11))
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key); s = _idx(n, 10000, 2); kcl = len(np.unique(y))
        r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s]))))
        sh = [r[a] / rf[a] for a in axes]; r["geo"] = float(np.prod(sh) ** (1 / len(sh))); r["min"] = float(min(sh)); return r
    print(f"\n{ds}: {lab.max() + 1} communities, {int((np.bincount(lab) >= 0.01 * n).sum())} above 1 %; frozen panel best {rf}; axes {axes}", flush=True)
    print(f"{'layout':<11}{'scheme':<34}{'knn15':>7}{'sp':>7}{'ari':>7}{'geo':>7}{'min':>6}{'t':>6}  sweeps")
    slug = {"full-batch": "magnetlosstsne", "hybrid": "magnetlosshybrid"}
    for lay in ("full-batch", "hybrid"):
        fig = os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_{slug[lay]}.npz"); p = os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_{lay}.npz")
        Y0 = np.load(fig if os.path.exists(fig) else p)["Y"].astype(np.float32)
        def run(name, f):
            k_ = f"{ds}/{lay}/{name}"
            if k_ not in res:
                lg = []; t = time.time(); Z = f(lg); res[k_] = score(np.ascontiguousarray(Z, dtype=np.float32)); res[k_]["t"] = time.time() - t; res[k_]["log"] = lg; json.dump(res, open(OUT, "w"), indent=1)
            r = res[k_]; print(f"{lay:<11}{name:<34}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}{r['t']:>6.1f}  {r['log']}", flush=True)
        run("identity", lambda lg: Y0)
        run("fine 0.7", lambda lg: contract(Y0, lab, 0.7))
        run("safe place > fine 0.7", lambda lg: contract(place(Y0, lab, spf, mixf, log=lg), lab, 0.7))
        run("fine 0.7 > safe place", lambda lg: place(contract(Y0, lab, 0.7), lab, spf, mixf, log=lg))
        run("fine 0.7 > safe place tol 0.01", lambda lg: place(contract(Y0, lab, 0.7), lab, spf, mixf, tol=0.01, log=lg))
        run("fine 0.7 > free place", lambda lg: place(contract(Y0, lab, 0.7), lab, spf, None, log=lg))
