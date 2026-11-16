"""k56: onion layers of the neighbour graph in place of the communities. The onion decomposition (Hebert-Dufresne,
Grochow, Allard 2016) refines the k-core one: at each step every node whose remaining degree is at most the current
core number is removed, and the step index is the node's layer. On a kNN graph every node has at least k
neighbours, so the layers measure how deep a point sits inside a dense region, away from the points nobody picks
as a neighbour. The communities failed as a gate of the mid-range pairs when they are many (k52: letter,
covertype); a depth is defined per point and does not depend on a partition.
part=diag   : layers of the fuzzy graph (union) and of the mutual kNN graph; by decile of depth: class purity of the
              graph neighbours, class purity of the 15 nearest neighbours in the default SGD layout, share of the
              point's graph weight and of its mid-near pairs that cross a label-free community.
part=gate   : the mid-near pairs gated by the depth in the mutual kNN graph instead of the communities: a pair is kept
              when both ends are deeper than the quantile `thr` (`deep`), the others are redirected to the point's
              strongest neighbour as in k52. Controls at the same share of kept pairs, (1 - thr)^2: both ends among
              the shallowest (`shallow`), pairs drawn at random (`random`).
part=sharpen: a partition-free post-process on the default layout: every point moves towards the mean of its graph
              neighbours that are deeper than itself, Z_i = Y_i + (1 - g)(mean - Y_i), `it` times; against the
              contraction of the communities by 0.7 (separation alone and whole post-process).
part=panel  : `deep` at thr 0.3, with the stiffness of the mode and with twice it (` x2`: about half of the pairs are
              kept, so the total pull is the default's, carried by the deep points only), SGD and hybrid, five seeds;
part=summary: paired differences with the default rows of k51 (SGD) and the 0.2 rows of k53 (hybrid), same seeds.
Result: negative as a default. The depth in the mutual graph does find the blurred points (MNIST, shallowest to
deepest decile: class purity of the 15 neighbours in the layout 0.83 to 0.98, crossing graph weight 0.107 to 0.009;
the union graph has a constant core number and 8-12 layers only), and at equal numbers of pairs the deep pairs
carry more Spearman than random or shallow ones (MNIST 0.407 / 0.378 / 0.334, one seed). Over five seeds the gate
behaves like a lower stiffness: paired composite difference with the default, hybrid + post-process, MNIST -0.001
+- 0.021, Fashion +0.001 +- 0.010, letter -0.034 +- 0.010, covertype -0.024 +- 0.007, PBMC +0.018 +- 0.021, Tasic
+0.015 +- 0.010 (mean -0.004; with twice the stiffness -0.006; SGD as is -0.010). The sharpening loses 15-NN
preservation at once (MNIST 0.191 to 0.148 at g 0.7, one pass) and gains no ARI.
Scoring of k41 / k51 (15-NN, distance Spearman, k-means ARI on 10 000 points; composites against the frozen
references).
Usage: python benchmarks/theory/k56_onion.py [datasets=mnist,letter,tasic2018] [part=diag,gate,sharpen,panel,summary] [seeds=1] [modes=SGD] [out=k56_onion.json]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from datasets import mnist, fmnist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from metrics import _idx, evaluate
from magnet import MAGNET, hierarchical_contraction
from magnet.contraction import communities, interface_merge, contract
T = 8

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    if name in ("pbmc68k", "tasic2018"):
        z = np.load(os.path.join(B, "cache", "scrna", f"{name}.npz")); return z["X"], z["y"], f"scrna_{name}"
    from k27_transfer import load as load27
    return load27(name, 20000)

def onion(A):
    """Onion layers and core numbers of the simple undirected graph whose adjacency structure is A."""
    A = sp.csr_matrix((np.ones(A.nnz, dtype=np.int64), A.indices, A.indptr), shape=A.shape); A.setdiag(0); A.eliminate_zeros()
    n = A.shape[0]; deg = np.asarray(A.sum(1)).ravel(); alive = np.ones(n, bool)
    layer = np.zeros(n, np.int64); core = np.zeros(n, np.int64); L = 0; k = 0
    while alive.any():
        k = max(k, int(deg[alive].min())); rm = alive & (deg <= k); L += 1
        layer[rm] = L; core[rm] = k; alive[rm] = False; deg = deg - A @ rm.astype(np.int64)
    return layer, core

def depth_rank(layer, seed=0):
    """Rank of the layer in [0, 1), ties broken at random."""
    r = np.random.default_rng(seed).random(len(layer)); o = np.lexsort((r, layer)); q = np.empty(len(layer)); q[o] = np.arange(len(layer)) / len(layer); return q

class Capture(MAGNET):
    def _label_mid_pairs(self, pairs, P):
        self._pairs_ = None if pairs is None else np.array(pairs, copy=True); return pairs

def mutual_graph(I):
    n = I.shape[0]; D = sp.csr_matrix((np.ones(I.size), (np.repeat(np.arange(n), I.shape[1]), np.asarray(I).ravel())), shape=(n, n)); return D.multiply(D.T).tocsr()

class OnionMid(MAGNET):
    """MAGNET whose mid-near pairs are gated by the onion depth of their two ends in the mutual kNN graph."""
    gate = "deep"; thr = 0.5
    def _label_mid_pairs(self, pairs, P):
        if pairs is None: return pairs
        pairs = np.array(pairs, copy=True); n, k = pairs.shape; rows = np.repeat(np.arange(n), k).reshape(pairs.shape)
        q = depth_rank(onion(mutual_graph(np.asarray(self.knn_indices_)))[0]); self.depth_ = q
        if self.gate == "deep": bad = (q[rows] < self.thr) | (q[pairs] < self.thr)
        elif self.gate == "shallow": bad = (q[rows] >= 1 - self.thr) | (q[pairs] >= 1 - self.thr)
        else: bad = np.random.default_rng(0).random(pairs.shape) >= (1 - self.thr) ** 2
        self.kept_ = float(1 - bad.mean()); P = P.tocsr(); u = np.unique(rows[bad])
        best = np.array([P.indices[P.indptr[i] + np.argmax(P.data[P.indptr[i]:P.indptr[i + 1]])] if P.indptr[i + 1] > P.indptr[i] else pairs[i, 0] for i in u])
        lut = np.zeros(n, dtype=pairs.dtype); lut[u] = best; pairs[bad] = lut[rows[bad]]; return pairs

def sharpen(Y, P, q, g, it):
    """Each point moves towards the weighted mean of its graph neighbours deeper than itself."""
    P = P.tocsr(); rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr)); up = q[P.indices] > q[rows]
    W = sp.csr_matrix((P.data[up].astype(np.float64), (rows[up], P.indices[up])), shape=P.shape); ws = np.asarray(W.sum(1)).ravel(); has = ws > 0
    W = sp.diags(np.where(has, 1.0 / np.maximum(ws, 1e-12), 0.0)) @ W; Z = Y.astype(np.float64)
    for _ in range(it): Z = np.where(has[:, None], Z + (1 - g) * (W @ Z - Z), Z)
    return Z.astype(np.float32)

REF = {"mnist": dict(knn15=0.356, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.330, sp=0.873, ari=0.483),
       "letter": dict(knn15=0.642, sp=0.716), "covertype": dict(knn15=0.632, sp=0.755)}
REF.update(json.load(open(os.path.join(HERE, "k47_scrna_heldout.json")))["_refs"])
def composite(r, ds):
    sh = [r[a] / REF[ds][a] for a in REF[ds]]; return float(np.prod(sh) ** (1 / len(sh))), float(min(sh))

if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    DS = kw.get("datasets", "mnist,letter,tasic2018").split(","); PART = kw.get("part", "diag").split(",")
    if "diag" in PART:
        for ds in DS:
            X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y)
            m = Capture(random_state=0, n_threads=T); Y = m.fit_transform(X); P = m.graph_.tocsr(); pairs = m._pairs_
            c = interface_merge(P, communities(P, 0.5))
            I = np.asarray(m.knn_indices_); D = sp.csr_matrix((np.ones(I.size), (np.repeat(np.arange(n), I.shape[1]), I.ravel())), shape=(n, n)); M = D.multiply(D.T).tocsr()
            rows = np.repeat(np.arange(n), np.diff(P.indptr)); w = P.data.astype(np.float64); ws = np.maximum(np.bincount(rows, weights=w, minlength=n), 1e-12)
            pur_g = np.bincount(rows, weights=w * (y[rows] == y[P.indices]), minlength=n) / ws
            cross_g = np.bincount(rows, weights=w * (c[rows] != c[P.indices]), minlength=n) / ws
            nn = NearestNeighbors(n_neighbors=16, n_jobs=T).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
            pur_2d = (y[nn] == y[:, None]).mean(1)
            cross_mid = (c[pairs] != c[:, None]).mean(1) if pairs is not None else np.zeros(n)
            for gname, G in (("union", P), ("mutual", M)):
                t0 = time.time(); layer, core = onion(G); t = time.time() - t0; q = depth_rank(layer); dec = np.minimum((q * 10).astype(int), 9)
                deg = np.diff(G.indptr)
                print(f"\n=== {ds} | {gname}: {int(layer.max())} layers, core {int(core.min())}-{int(core.max())}, degree {deg.min()}-{int(np.median(deg))}-{deg.max()}, {int(c.max()) + 1} communities, onion {t:.1f} s")
                print("  Spearman of the depth with: graph purity %+.3f, 2-D purity %+.3f, graph crossing %+.3f, mid crossing %+.3f, degree %+.3f"
                      % tuple(spearmanr(layer, v)[0] for v in (pur_g, pur_2d, cross_g, cross_mid, deg)))
                print("  decile      " + " ".join(f"{d:6d}" for d in range(10)))
                for name, v in (("layer", layer), ("core", core), ("graph purity", pur_g), ("2-D purity", pur_2d), ("graph cross", cross_g), ("mid cross", cross_mid)):
                    print(f"  {name:12s}" + " ".join(f"{v[dec == d].mean():6.2f}" if name in ("layer", "core") else f"{v[dec == d].mean():6.3f}" for d in range(10)))

    OUT = os.path.join(HERE, kw.get("out", "k56_onion.json")); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    SEEDS = int(kw.get("seeds", 1)); MODES = {"SGD": "umap", "hybrid": "hybrid"}; MODES = {k: MODES[k] for k in kw.get("modes", "SGD").split(",")}
    GATES = [("deep", 0.3), ("deep", 0.5), ("random", 0.3), ("random", 0.5), ("shallow", 0.3), ("shallow", 0.5)]
    BASE_MID = {"SGD": 0.1, "hybrid": 0.2}
    if "panel" in PART:
        for ds in DS:
            X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); s10 = _idx(n, 10000, 2)
            def score(Y):
                e = evaluate(X, Y, y, cache_key=key)
                r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"],
                         ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
                r["geo"], r["min"] = composite(r, ds); return r
            for s in range(SEEDS):
                for tag, loss in MODES.items():
                    for mult in (1, 2):
                        k = f"{ds} | {tag} deep 0.3{' x2' if mult == 2 else ''} | {s}"; kp = k.replace(f" | {s}", f" + post | {s}")
                        if kp in res: continue
                        m = OnionMid(loss=loss, random_state=s, n_threads=T, mid_weight=mult * BASE_MID[tag]); m.gate, m.thr = "deep", 0.3; Y = m.fit_transform(X)
                        res[k] = score(Y); res[kp] = score(hierarchical_contraction(Y, m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=0.7)[0])
                        for kk in (k, kp): print(kk, {a: round(v, 3) for a, v in res[kk].items()}, flush=True)
                        json.dump(res, open(OUT, "w"), indent=1)
    if "summary" in PART:
        import glob
        for f in glob.glob(os.path.join(HERE, "k56_onion*.json")): res.update(json.load(open(f)))
        b51 = json.load(open(os.path.join(HERE, "k51_soft_community_panel.json"))); b53 = json.load(open(os.path.join(HERE, "k53_hybrid_kappa.json")))
        AX = ("knn15", "sp", "ari", "geo")
        for ds in DS:
            print(f"\n=== {ds}")
            for tag in MODES:
                for suf, bsuf in (("", ""), (" + post", " + contraction")):
                    bk = (lambda s: f"{ds} | SGD default{bsuf} | {s}") if tag == "SGD" else (lambda s: f"{ds} | hybrid mid 0.2{bsuf} | {s}"); base = b51 if tag == "SGD" else b53
                    v = [base[bk(s)] for s in range(SEEDS) if bk(s) in base]
                    if v: print(f"  {tag + ' default' + suf:28s} " + "  ".join(f"{a} {np.mean([x[a] for x in v]):.3f}" for a in AX) + f"  (n={len(v)})")
                    for var in ("deep 0.3", "deep 0.3 x2"):
                        vk = lambda s: f"{ds} | {tag} {var}{suf} | {s}"; ss = [s for s in range(SEEDS) if vk(s) in res and bk(s) in base]
                        if not ss: continue
                        d = np.array([[res[vk(s)][a] - base[bk(s)][a] for a in AX] for s in ss])
                        print(f"  {tag + ' ' + var + suf:28s} " + "  ".join(f"{a} {np.mean([res[vk(s)][a] for s in ss]):.3f}" for a in AX) + f"  (n={len(ss)})  diff: " + "  ".join(f"{a} {m:+.3f} +- {sd:.3f}" for a, m, sd in zip(AX, d.mean(0), d.std(0))))
    if "gate" in PART or "sharpen" in PART:
        for ds in DS:
            X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); s10 = _idx(n, 10000, 2)
            def score(Y):
                e = evaluate(X, Y, y, cache_key=key)
                r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"],
                         ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
                r["geo"], r["min"] = composite(r, ds); return r
            def put(k, r):
                res[k] = r; print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True); json.dump(res, open(OUT, "w"), indent=1)
            for s in range(SEEDS):
                for tag, loss in MODES.items():
                    post = lambda m, Y: hierarchical_contraction(Y, m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=0.7)[0]
                    if f"{ds} | {tag} default + post | {s}" not in res or "sharpen" in PART:
                        m = MAGNET(loss=loss, random_state=s, n_threads=T); Y = m.fit_transform(X)
                        put(f"{ds} | {tag} default | {s}", score(Y)); put(f"{ds} | {tag} default + post | {s}", score(post(m, Y)))
                    if "sharpen" in PART:
                        P = m.graph_; q = depth_rank(onion(mutual_graph(np.asarray(m.knn_indices_)))[0]); c = interface_merge(P, communities(P, 0.5))
                        put(f"{ds} | {tag} default > communities 0.7 | {s}", score(contract(Y, c, 0.7)))
                        for g, it in ((0.7, 1), (0.5, 1), (0.7, 3), (0.5, 3), (0.5, 10)):
                            put(f"{ds} | {tag} default > onion g{g} x{it} | {s}", score(sharpen(Y, P, q, g, it)))
                    if "gate" in PART:
                        for gate, thr in GATES:
                            k = f"{ds} | {tag} {gate} {thr} | {s}"
                            if k + " + post" in res: continue
                            m = OnionMid(loss=loss, random_state=s, n_threads=T); m.gate, m.thr = gate, thr; Y = m.fit_transform(X)
                            r = score(Y); r["kept"] = round(m.kept_, 3); put(k, r); put(k.replace(" | " + str(s), " + post | " + str(s)), score(post(m, Y)))
