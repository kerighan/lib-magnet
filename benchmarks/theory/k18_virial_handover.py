"""k18: the hand-over at the KL-optimal dilation s* (virial identity, k17) instead of the constant 31.
(a) MNIST 70k: PCA-2 handed over at its s* (~10) against 31; the stiff SGD start at 31 / 62 / its s* (~87); UMAP at 31 and at
    its s* (bracketed from above at 124): 300 iterations, 15-NN / Spearman / triplets / centroids, time of the pass.
(b) s* against n: MNIST subsamples 5k / 20k / 70k and Fashion 70k, stiff SGD start; fit of log s* against log n.
-> k18_virial_handover.json / .log. Usage: python benchmarks/theory/k18_virial_handover.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import PCA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist, fmnist
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from k17_virial_projected import prep, scan

def s_star(P, Y0, label):
    r = scan(P, Y0, label, [4, 6, 8, 12, 16, 22, 31, 44, 62, 88, 124, 176, 250]); return r["s_star"], r

out = {"handover": {}, "s_vs_n": {}}
X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key)
m = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X); P = sp.csr_matrix(m.graph_, dtype=np.float32); Pn = P / P.sum()
starts = {"SGD spring 0.5": m.embedding_.astype(np.float32), "PCA-2": PCA(2, svd_solver="randomized", random_state=0).fit_transform(X).astype(np.float32),
          "UMAP": np.load(os.path.join(HERE, "..", "..", "paper", "figures", "cache", "mnist_umap0512.npz"))["Y"].astype(np.float32)}
print(f"{'start':<16}{'scale':>8}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'std':>7}{'t(s)':>7}")
for name, Y0 in starts.items():
    ss, r = s_star(Pn, Y0, name); out["handover"][name] = {"s_star": ss, "scan": r, "runs": {}}
    for s in sorted(set([31.0, 62.0, float(round(ss))])):
        Y = Y0 - Y0.mean(0); Y = (Y / max(float(Y[:, 0].std()), 1e-12) * s).astype(np.float32)
        t = time.time(); Y = optimize_tsne(P, Y, n_iter=300, early_iter=0, n_threads=-1); dt = time.time() - t
        res = metrics(Y); res["t"] = dt; out["handover"][name]["runs"][str(s)] = res
        print(f"{name:<16}{s:>8.0f}{res['knn15']:>7.3f}{res['sp']:>7.3f}{res['trip']:>7.3f}{res['cent']:>7.3f}{res['std']:>7.1f}{dt:>7.1f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "k18_virial_handover.json"), "w"), indent=1)
print("\ns* against n (stiff SGD start)")
for loader, nn, label in ((mnist, 5000, "MNIST 5k"), (mnist, 20000, "MNIST 20k"), (mnist, 70000, "MNIST 70k"), (fmnist, 70000, "Fashion 70k"), (fmnist, 20000, "Fashion 20k")):
    Xs, ys, ks = loader(nn)
    ms = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(Xs); Ps = sp.csr_matrix(ms.graph_, dtype=np.float32); Ps = Ps / Ps.sum()
    ss, r = s_star(Ps, ms.embedding_.astype(np.float32), label); out["s_vs_n"][label] = dict(n=nn, s_star=ss)
ns = np.array([v["n"] for k, v in out["s_vs_n"].items() if k.startswith("MNIST")]); ss = np.array([v["s_star"] for k, v in out["s_vs_n"].items() if k.startswith("MNIST")])
slope = np.polyfit(np.log(ns), np.log(ss), 1)[0]; out["s_vs_n"]["mnist_exponent"] = float(slope)
print(f"MNIST: s* ~ n^{slope:.2f}")
json.dump(out, open(os.path.join(HERE, "k18_virial_handover.json"), "w"), indent=1)
