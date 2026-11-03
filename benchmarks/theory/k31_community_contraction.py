"""k31: label-free version of the k30 oracle. Communities of the fuzzy graph P (Leiden, several resolutions) or
clusters of the layout (HDBSCAN); each community is contracted rigidly towards its centroid by a factor c, then
(optionally) a short full-batch repair at exaggeration 1. Measures 15-NN / Spearman / k-means ARI / composite on the
cached MNIST and Fashion layouts (full-batch, hybrid, quality). Also reports how well the communities match the
labels (ARI, count). Usage: python benchmarks/theory/k31_community_contraction.py [datasets=mnist,fmnist]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "mnist,fmnist").split(",")
comp = json.load(open(os.path.join(B, "cache", "composite_table.json")))
OUT = os.path.join(HERE, "k31_community_contraction.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def ref(ds):
    rows = [v for k, v in comp.items() if k.startswith(ds + "/")]
    return {a: max(r[a] for r in rows) for a in ("knn15", "sp", "ari")}

def score(X, Y, y, key, rf):
    e = evaluate(X, Y, y, cache_key=key); s = _idx(len(Y), 10000, 2); kcl = len(np.unique(y))
    ari = float(adjusted_rand_score(y[s], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s])))
    r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"], ari=ari)
    sh = [r[a] / rf[a] for a in ("knn15", "sp", "ari")]; r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh)); return r

def contract(Y, lab, c):
    Z = Y.copy()
    for cl in np.unique(lab):
        if cl < 0: continue
        m = lab == cl; mu = Y[m].mean(0); Z[m] = mu + c * (Y[m] - mu)
    return Z

def leiden(P, resolution, seed=0):
    import igraph as ig, leidenalg as la
    C = sp.triu(P, 1).tocoo(); g = ig.Graph(n=P.shape[0], edges=np.c_[C.row, C.col].tolist()); g.es["weight"] = C.data.astype(float).tolist()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed, n_iterations=2)
    return np.asarray(part.membership)

def hdb(Y, min_size):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=10, core_dist_n_jobs=T).fit_predict(Y)

for ds in DS:
    X, y, key = mnist(70000) if ds == "mnist" else fmnist(70000); y = np.asarray(y).astype(int); rf = ref(ds); n = len(X)
    pc = os.path.join(B, "cache", f"P_{ds}.npz")
    if os.path.exists(pc): P = sp.load_npz(pc)
    else: P = sp.csr_matrix(MAGNET(random_state=0, n_threads=T).fit(X).graph_, dtype=np.float32); sp.save_npz(pc, P)
    print(f"\n{ds}: panel best knn15 {rf['knn15']:.3f} sp {rf['sp']:.3f} ari {rf['ari']:.3f}", flush=True)
    labs = {}
    for r_ in (0.02, 0.05, 0.1, 0.3):
        t = time.time(); lab = leiden(P, r_); labs[f"leiden r={r_}"] = lab
        print(f"  leiden r={r_}: {lab.max() + 1} communities, ARI vs labels {adjusted_rand_score(y, lab):.3f} ({time.time() - t:.0f} s)", flush=True)
    layouts = {"full-batch": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosstsne.npz"),
               "hybrid": os.path.join(B, "..", "paper", "figures", "cache", f"{ds}_magnetlosshybrid.npz"),
               "quality": os.path.join(B, "cache", "layouts", f"{ds}_MAGNET_full-batch_quality.npz")}
    print(f"{'layout':<12}{'communities':<16}{'c':>5}{'repair':>7}{'knn15':>7}{'sp':>7}{'ari':>7}{'geo':>7}{'min':>6}")
    for lay, p in layouts.items():
        Y0 = np.load(p)["Y"].astype(np.float32)
        L = dict(labs); L["hdbscan(layout)"] = hdb(Y0, max(50, n // 300))
        for cname, lab in L.items():
            if cname.startswith("hdbscan"):
                print(f"  {lay} hdbscan: {lab.max() + 1} clusters, noise {np.mean(lab < 0):.2f}, ARI vs labels {adjusted_rand_score(y, lab):.3f}", flush=True)
            for c in (0.6, 0.4):
                for rep in (0, 30):
                    k_ = f"{ds}/{lay}/{cname}/c{c}/rep{rep}"
                    if k_ not in res:
                        Y = contract(Y0, lab, c)
                        if rep: Y = optimize_tsne(P, np.ascontiguousarray(Y), n_iter=rep, early_iter=0, n_threads=T)
                        res[k_] = score(X, Y, y, key, rf); res[k_]["n_comm"] = int(lab.max() + 1); res[k_]["ari_comm"] = float(adjusted_rand_score(y, lab))
                        json.dump(res, open(OUT, "w"), indent=1)
                    r = res[k_]; print(f"{lay:<12}{cname:<16}{c:>5}{rep:>7}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['ari']:>7.3f}{r['geo']:>7.3f}{r['min']:>6.2f}", flush=True)
