"""k21: the coarse rest-length stress in the FIRST stage (where the arrangement forms), through the epoch hook of the SGD.
Label-free cells (k-means, B centres, on an 8-d spectral embedding), target = 2-D classical MDS of the cells' input-space
means (feasible by construction), alpha matched to the layout's coarse RMS at each application. Between SGD segments of
`every` epochs, the barycentre displacement -eta * dS/dc_a is applied to every point of cell a (a translation of the cell:
the within-cell configuration is untouched), with eta set so that the first step moves the cells by a fraction `step` of
their mean separation, and annealed over the mid-phase like the spring. Then the usual pass (300 it, scale 31), with or
without the 'start' scaffold (k20) to prevent the drift. MNIST 70k, springs 0.1 / 0.5, 3 seeds, all cores.
-> k21_scaffold_sgd.json / .log. Usage: python benchmarks/theory/k21_scaffold_sgd.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from datasets import mnist
from magnet import MAGNET
from magnet.spectral import spectral_init
from k17_virial_projected import prep
from k20_coarse_stress import cell_stress, run_pass   # noqa: k20 runs its experiment on import? no: guarded below

X, y, key = mnist(70000); y = np.asarray(y).astype(int); n = len(X); metrics = prep(X, y, key)
m0 = MAGNET(random_state=0).fit(X); P = sp.csr_matrix(m0.graph_, dtype=np.float32)
S8 = np.asarray(spectral_init(P, 8, random_state=0, n_threads=-1), dtype=np.float64)
B = 64; lab = MiniBatchKMeans(B, random_state=0, n_init=3, batch_size=4096).fit_predict(S8); cnt = np.bincount(lab, minlength=B).astype(np.float64)
mu = np.stack([np.bincount(lab, X[:, d], B) for d in range(X.shape[1])], 1) / cnt[:, None]
D = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(-1)); J = np.eye(B) - 1.0 / B; G = -0.5 * J @ (D ** 2) @ J; w, V = np.linalg.eigh(G)
Z2 = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0)); Dmds = np.sqrt(((Z2[:, None] - Z2[None]) ** 2).sum(-1)); iu = np.triu_indices(B, 1)

def coarse_dists(Y):
    C = np.stack([np.bincount(lab, Y[:, 0].astype(np.float64), B), np.bincount(lab, Y[:, 1].astype(np.float64), B)], 1) / cnt[:, None]
    return C, np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))

def make_hook(step, phase=0.5, every=10):
    state = {"eta": None}
    def hook(Y, epoch, total):
        if epoch > phase * total: return
        C, dist = coarse_dists(Y); alpha = np.sqrt((dist[iu] ** 2).mean()) / np.sqrt((Dmds[iu] ** 2).mean()); target = alpha * Dmds
        delta = 0.05 * target[iu].mean(); den = target + delta; resid = (dist - target) / den
        coef = 2.0 / len(iu[0]) * resid / den / (dist + 1e-9); np.fill_diagonal(coef, 0.0)
        Gc = (coef[:, :, None] * (C[:, None, :] - C[None, :, :])).sum(1)
        if state["eta"] is None:
            state["eta"] = step * dist[iu].mean() / max(float(np.sqrt((Gc ** 2).sum(1)).mean()), 1e-12)
        eta = state["eta"] * max(0.0, 1.0 - epoch / (phase * total))
        Y -= (eta * Gc[lab]).astype(np.float32)
    return hook

out = json.load(open(os.path.join(HERE, "k21_scaffold_sgd.json"))) if os.path.exists(os.path.join(HERE, "k21_scaffold_sgd.json")) else {}
print(f"{'config':<44}{'knn15':>7}{'sp':>7}{'trip':>7}{'cent':>7}{'t(s)':>7}")
for spring in (0.5, 0.1):
    for seed in range(3):
        for step in (0.0, 0.03, 0.1, 0.3):
            key_ = f"spring {spring} / seed {seed} / step {step}"
            if key_ in out:
                continue
            m = MAGNET(loss="umap", mid_weight=spring, random_state=seed)
            if step > 0:
                m.epoch_callback = make_hook(step); m.callback_every = 10
            t = time.time(); Y0 = m.fit_transform(X).astype(np.float32); t_sgd = time.time() - t
            r0 = metrics(Y0); r0["t"] = t_sgd
            Y0c = Y0 - Y0.mean(0); Y0c = (Y0c / max(float(Y0c[:, 0].std()), 1e-12) * 31.0).astype(np.float32)
            # pass without scaffold, and with the 'start' scaffold (k20) at ratio 1
            C0, dist0 = coarse_dists(Y0c)
            res = {"start": r0}
            for scaf, ratio in (("none", 0.0), ("start", 1.0)):
                t = time.time(); Y, _ = run_pass(Y0c, lab, B, dist0 if scaf == "start" else np.ones((B, B)), ratio, P=P); dt = time.time() - t
                r = metrics(Y); r["t"] = dt; res[f"pass/{scaf}"] = r
            out[key_] = res
            print(f"{key_:<44}{r0['knn15']:>7.3f}{r0['sp']:>7.3f}{r0['trip']:>7.3f}{r0['cent']:>7.3f}{t_sgd:>7.1f}   SGD", flush=True)
            for scaf in ("none", "start"):
                r = res[f"pass/{scaf}"]; print(f"{'   + pass, scaffold ' + scaf:<44}{r['knn15']:>7.3f}{r['sp']:>7.3f}{r['trip']:>7.3f}{r['cent']:>7.3f}{r['t']:>7.1f}", flush=True)
            json.dump(out, open(os.path.join(HERE, "k21_scaffold_sgd.json"), "w"), indent=1)
