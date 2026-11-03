"""k25: is the spring's global structure graph diffusion? (idea 2 of the GPT-6 note). The mid-near pair graph M (5 partners
per point) symmetrised, L = D - (M + M^T)/2. SGD WITHOUT the spring, and between segments of 10 epochs a reversible
heat-flow step on the layout: implicit Y <- (I + tau L)^-1 Y (CG, warm start) or explicit Y <- Y - tau L Y, tau annealed to
zero over the first half like the spring. Compared with the spring at 0.5 and with no spring, 3 seeds, before and after the
usual pass. MNIST 70k, all cores. -> k25_implicit_spring.json / .log. Usage: python benchmarks/theory/k25_implicit_spring.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import cg
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from magnet import MAGNET
from magnet.tsne import optimize_tsne
from k17_virial_projected import prep

X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key)
ref = MAGNET(loss="umap", mid_weight=0.5, random_state=0).fit(X); P = sp.csr_matrix(ref.graph_, dtype=np.float32)
pairs = ref._mid_pairs_; rows = np.repeat(np.arange(n), pairs.shape[1])
M = sp.csr_matrix((np.ones(pairs.size), (rows, pairs.ravel())), shape=(n, n)); W = 0.5 * (M + M.T)
Lap = (sp.diags(np.asarray(W.sum(1)).ravel()) - W).tocsr()

def make_hook(tau, implicit, phase=0.5):
    def hook(Y, epoch, total):
        if epoch > phase * total: return
        t = tau * max(0.0, 1.0 - epoch / (phase * total))
        if t <= 0: return
        if implicit:
            A = (sp.eye(n) + t * Lap).tocsr()
            for d in range(2):
                sol, _ = cg(A, Y[:, d].astype(np.float64), x0=Y[:, d].astype(np.float64), maxiter=20, rtol=1e-4)
                Y[:, d] = sol.astype(np.float32)
        else:
            Y -= (t * (Lap @ Y.astype(np.float64))).astype(np.float32)
    return hook

out = json.load(open(os.path.join(HERE, "k25_implicit_spring.json"))) if os.path.exists(os.path.join(HERE, "k25_implicit_spring.json")) else {}
print(f"{'config':<34}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'t(s)':>7}   ->  {'knn15':>7}{'sp':>7}{'cent':>7}")
configs = [("spring 0.5", dict(mid_weight=0.5), None), ("no spring", dict(mid_weight=0.0), None)]
configs += [(f"{'implicit' if imp else 'explicit'} tau {tau}", dict(mid_weight=0.0), (tau, imp)) for imp in (True, False) for tau in (0.3, 1.0, 3.0)]
for seed in range(3):
    for name, kw, hook in configs:
        k_ = f"{name} / seed {seed}"
        if k_ in out: continue
        m = MAGNET(loss="umap", random_state=seed, **kw)
        if hook is not None:
            m.epoch_callback = make_hook(*hook); m.callback_every = 10
        t = time.time(); Y0 = m.fit_transform(X).astype(np.float32); dt = time.time() - t
        if not np.all(np.isfinite(Y0)):
            print(f"{k_:<34} diverged", flush=True); out[k_] = {"diverged": True}; continue
        r0 = metrics(Y0); r0["t"] = dt
        Yc = Y0 - Y0.mean(0); Yc = (Yc / max(float(Yc[:, 0].std()), 1e-12) * 31.0).astype(np.float32)
        Y = optimize_tsne(P, Yc, n_iter=300, early_iter=0, n_threads=-1); r1 = metrics(Y)
        out[k_] = {"sgd": r0, "pass": r1}
        print(f"{k_:<34}{r0['knn15']:>7.3f}{r0['sp']:>7.3f}{r0['trip']:>7.3f}{r0['cent']:>7.3f}{dt:>7.1f}   ->  {r1['knn15']:>7.3f}{r1['sp']:>7.3f}{r1['cent']:>7.3f}", flush=True)
        json.dump(out, open(os.path.join(HERE, "k25_implicit_spring.json"), "w"), indent=1)
