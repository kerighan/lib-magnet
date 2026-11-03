"""k28: can a short exaggeration tail in the pass give the hybrid the separability of PaCMAP / UMAP while keeping its
15-NN and Spearman? From the stiff SGD start (seed 0, MNIST 70k, scale 31): 300 - L iterations at exaggeration 1, then L
at exaggeration rho (L in {50, 100}, rho in {2, 4, 8}); the whole pass at rho = 1.5 / 2; the SGD compact preset
(late_exaggeration 1.5) followed by the plain pass; and a tail of rho = 4 followed by 50 plain iterations (relaxation).
Scores: 15-NN, Spearman, triplets, centroids, silhouette, ARI k-means (k = 10), ARI / noise of HDBSCAN, composite
(shares of the fixed references PaCMAP ARI 0.914, hybrid 15-NN 0.351, Spearman 0.461). 8 threads.
-> k28_exaggeration_tail.json / .log. Usage: python benchmarks/theory/k28_exaggeration_tail.py"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, scipy.sparse as sp, hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from metrics import _idx
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from k17_virial_projected import prep
T = 8
X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key); s = _idx(n, 10000, 2)
REF = dict(knn15=0.351, sp=0.461, ari=0.914)
def sep(Y):
    Zs = Y[s]; cl = hdbscan.HDBSCAN(min_cluster_size=200).fit_predict(Zs); m = cl >= 0
    return dict(silhouette=float(silhouette_score(Zs, y[s])), ari_kmeans=float(adjusted_rand_score(y[s], KMeans(10, n_init=5, random_state=0).fit_predict(Zs))),
                ari_hdbscan=float(adjusted_rand_score(y[s][m], cl[m])) if m.sum() else float("nan"), clusters=int(len(set(cl[m]))), noise=float(1 - m.mean()))
def score(Y):
    r = metrics(Y); r.update(sep(Y)); sh = [r["knn15"] / REF["knn15"], r["sp"] / REF["sp"], r["ari_kmeans"] / REF["ari"]]
    r["geo"] = float(np.prod(sh) ** (1 / 3)); r["min"] = float(min(sh)); return r
def show(name, r, t):
    print(f"{name:<36}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['cent']:>7.3f}{r['silhouette']:>7.3f}{r['ari_kmeans']:>8.3f}{r['ari_hdbscan']:>9.3f}{r['clusters']:>6}{r['noise']:>7.2f}{r['geo']:>7.3f}{r['min']:>6.2f}{t:>7.1f}", flush=True)

m = MAGNET(loss="umap", mid_weight=0.5, random_state=0, n_threads=T).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float32)
Y0 = m.embedding_.astype(np.float32); Y0 -= Y0.mean(0); Y0 *= 31.0 / Y0[:, 0].std()
out = {}
print(f"{'variant':<36}{'knn15':>7}{'sp':>7}{'cent':>7}{'silh':>7}{'ARI-km':>8}{'ARI-hdb':>9}{'clust':>6}{'noise':>7}{'geo':>7}{'min':>6}{'t(s)':>7}")
def run(name, stages, Ystart=None):
    Y = (Ystart if Ystart is not None else Y0).copy(); t = time.time()
    for it, rho in stages:
        Y = optimize_tsne(P, Y, n_iter=it, early_iter=0, exaggeration=rho, n_threads=T)
    dt = time.time() - t; r = score(Y); r["t"] = dt; out[name] = r; show(name, r, dt)
run("SGD start", [])
run("plain pass (hybrid)", [(300, 1.0)])
for rho in (1.5, 2.0): run(f"whole pass at rho {rho}", [(300, rho)])
for L_ in (50, 100):
    for rho in (2.0, 4.0, 8.0): run(f"tail {L_} it at rho {rho}", [(300 - L_, 1.0), (L_, rho)])
run("tail 100 at rho 4 + 50 relax", [(200, 1.0), (100, 4.0), (50, 1.0)])
mc = MAGNET(loss="umap", mid_weight=0.5, late_exaggeration=1.5, random_state=0, n_threads=T).fit(X)
Yc = mc.embedding_.astype(np.float32); Yc -= Yc.mean(0); Yc *= 31.0 / Yc[:, 0].std()
run("SGD compact preset (start)", [], Yc); run("SGD compact preset + plain pass", [(300, 1.0)], Yc); run("SGD compact + tail 100 at rho 4", [(200, 1.0), (100, 4.0)], Yc)
json.dump(out, open(os.path.join(HERE, "k28_exaggeration_tail.json"), "w"), indent=1)
