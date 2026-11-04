"""k43: why modularity fails on kNN graphs, and a density-contrast merge that fixes it (after Maixent's remark that a
graph has no notion of "in the community but geometrically far"). On a kNN graph every node has about the same
strength (MNIST: mean 6.6, cv 0.33), so the configuration null model gamma k_i k_j / 2m is nearly a constant and the
modularity reduces to internal weight minus a size^2 penalty (it coincides with CPM at gamma' = gamma kbar^2 / 2m:
checked below). Two known consequences: large geometric communities are split (their internal cuts are thin, the
field-of-view limit of Schaub et al. 2012) and small ones are swallowed (the resolution limit). The fix: Louvain at a
high resolution (no swallowing), then merge the interfaces that show no density contrast. Statistic: the boundary
nodes of A towards B send on average a share f_AB of their strength into B; a cut through a uniform cloud is
symmetric and large on both sides (0.14-0.16 on the MNIST graph, whatever the sizes), a true boundary is a density
dip and at least one side reaches little across (<= 0.09). Merge while min(f_AB, f_BA) >= tau.
(a) interface statistics of the Louvain partitions; (b) CPM at the matched scale; (c) tau x resolution scan on the
four graphs and three MNIST graphs; (d) downstream through the post-process on the cached full-batch layouts.
Usage: python benchmarks/theory/k43_interface_merge.py [part=interfaces,cpm,scan,downstream]"""
import os, sys, json, time, random, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.contraction import communities, interface_merge, hierarchical_contraction, _igraph
kw = dict(a.split("=", 1) for a in sys.argv[1:]); PART = kw.get("part", "interfaces,cpm,scan,downstream").split(",")
OUT = os.path.join(HERE, "k43_interface_merge.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
REF = {"mnist": dict(knn15=0.352, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.328, sp=0.873, ari=0.483), "letter": dict(knn15=0.642, sp=0.716), "covertype": dict(knn15=0.632, sp=0.755)}
def load(ds):
    if ds in ("mnist", "fmnist"): return (mnist if ds == "mnist" else fmnist)(70000)
    from k27_transfer import load as load27; return load27(ds, 20000)
def graph(ds, gs=0):
    if gs == 0: return sp.load_npz(os.path.join(B, "cache", f"P_{ds}.npz"))
    X, y, key = load(ds); return sp.csr_matrix(MAGNET(random_state=gs, n_threads=8).fit(X).graph_, dtype=np.float32)

if "interfaces" in PART:
    for ds, rs in (("mnist", (0.3, 0.5, 1.0)), ("fmnist", (0.3, 0.5))):
        X, y, key = load(ds); y = np.asarray(y).astype(int); P = graph(ds); n = P.shape[0]; strength = np.asarray(P.sum(1)).ravel()
        for r in rs:
            lab = communities(P, r); k = lab.max() + 1
            C = sp.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, k)); W = (P @ C).toarray(); frac = W / strength[:, None]
            comp = [np.bincount(y[lab == c], minlength=y.max() + 1) for c in range(k)]; rows = []
            for a in range(k):
                for b in range(a + 1, k):
                    ia = (lab == a) & (W[:, b] > 0); ib = (lab == b) & (W[:, a] > 0)
                    if ia.sum() < 20: continue
                    rows.append((min(frac[ia, b].mean(), frac[ib, a].mean()), frac[ia, b].mean(), frac[ib, a].mean(), a, b, int(ia.sum()), int(ib.sum()), comp[a].argmax() == comp[b].argmax()))
            rows.sort(reverse=True)
            print(f"\n{ds} r={r}: {k} communities; interfaces by min crossing share (SAME = same majority class)")
            for f, fa, fb, a, b, na, nb, same in rows[:8]: print(f"  {a:>2}-{b:<3} boundary {na:>5}/{nb:<5} shares {fa:.3f}/{fb:.3f} min {f:.3f} {'SAME' if same else ''}")
            res[f"interfaces/{ds}/{r}"] = dict(same=[float(x[0]) for x in rows if x[7]], diff=[float(x[0]) for x in rows if not x[7]])
    json.dump(res, open(OUT, "w"), indent=1)

if "cpm" in PART:
    X, y, key = load("mnist"); y = np.asarray(y).astype(int); P = graph("mnist"); st = np.asarray(P.sum(1)).ravel(); kbar = st.mean(); two_m = st.sum()
    print(f"\nMNIST graph: strength mean {kbar:.2f}, cv {st.std() / kbar:.2f}; modularity 0.3 <-> CPM gamma' = {0.3 * kbar ** 2 / two_m:.2e}")
    g = _igraph(P)
    for gam in (1e-5, 2e-5, 3e-5, 5e-5):
        runs = []
        for s in range(3): random.seed(s); runs.append(np.asarray(g.community_leiden(objective_function="CPM", weights="weight", resolution=gam, n_iterations=2).membership))
        r = dict(comm=[int(x.max() + 1) for x in runs], ari=[float(adjusted_rand_score(y, x)) for x in runs]); res[f"cpm/{gam}"] = r
        print(f"  CPM gamma {gam:.0e}: communities {r['comm']}, label ARI {[round(a, 3) for a in r['ari']]}")
    json.dump(res, open(OUT, "w"), indent=1)

if "scan" in PART:
    for ds in ("mnist", "fmnist", "letter", "covertype"):
        X, y, key = load(ds); y = np.asarray(y).astype(int)
        for gs in ((0, 1, 2) if ds == "mnist" else (0,)):
            P = graph(ds, gs); print(f"\n{ds} graph {gs}: " + "  ".join(f"r={r}: {communities(P, r).max() + 1}" for r in (0.3, 0.5, 1.0, 1.5)) + " communities before the merge")
            for tau in (0.08, 0.10, 0.11, 0.12, 0.14):
                cells = []
                for r in (0.5, 1.0, 1.5):
                    lab = interface_merge(P, communities(P, r), tau); res[f"scan/{ds}/{gs}/{tau}/{r}"] = dict(comm=int(lab.max() + 1), ari=float(adjusted_rand_score(y, lab))); cells.append(f"r={r}: {lab.max() + 1:>3} ({adjusted_rand_score(y, lab):.3f})")
                print(f"   tau {tau:.2f}: " + "  ".join(cells), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

if "downstream" in PART:
    VARIANTS = (("r0.3, no merge (previous default)", dict(resolution=0.3, merge_tau=None)), ("r0.5 + merge 0.11 (default)", dict(resolution=0.5, merge_tau=0.11)),
                ("r1.0 + merge 0.11", dict(resolution=1.0, merge_tau=0.11)), ("r0.5 + merge 0.10", dict(resolution=0.5, merge_tau=0.10)), ("r0.5, no merge", dict(resolution=0.5, merge_tau=None)))
    for ds in ("mnist", "fmnist", "letter", "covertype"):
        X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); P = graph(ds)
        excl = np.unique(np.concatenate([_idx(n, 2000, 1), _idx(n, 5000, 0), _idx(n, 10000, 2)])); s10 = _idx(n, 10000, 2)
        def score(Y):
            e = evaluate(X, Y, y, cache_key=key); r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
            r["geo"] = float(np.prod([r[a] / REF[ds][a] for a in REF[ds]]) ** (1 / len(REF[ds]))); return r
        layouts = [(s, os.path.join(B, "cache", "layouts", f"{ds}_fb_seed{s}.npz")) for s in (0, 1, 2)] if ds in ("mnist", "fmnist") else [(0, os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_full-batch.npz"))]
        print(f"\n{ds} downstream ({len(layouts)} full-batch layouts; axes {list(REF[ds])})")
        for name, opts in VARIANTS:
            rows = []
            for s, p in layouts:
                k_ = f"downstream/{ds}/{name}/{s}"
                if k_ not in res:
                    Y = np.load(p)["Y"].astype(np.float32); t = time.time(); Z, log = hierarchical_contraction(Y, P, X=X, random_state=s, exclude=excl, n_threads=8, **opts); dt = time.time() - t
                    r = score(np.ascontiguousarray(Z, dtype=np.float32)); r["comm"] = int(log["communities"]); r["t"] = dt; res[k_] = r; json.dump(res, open(OUT, "w"), indent=1)
                rows.append(res[k_])
            print(f"  {name:<36} comm {[r['comm'] for r in rows]}  " + "/".join(f"{np.mean([r[a] for r in rows]):.3f}" for a in ("knn15", "sp", "ari")) + f"  composite {np.mean([r['geo'] for r in rows]):.3f} +- {np.std([r['geo'] for r in rows]):.3f}  step {np.mean([r['t'] for r in rows]):.1f} s", flush=True)
