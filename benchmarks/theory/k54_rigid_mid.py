"""k54: mid-range attraction between the label-free communities applied to whole communities. k50 / k52: the
mid-near pairs that cross a community of the post-process partition (Louvain at resolution 0.5 + interface merge)
are what blurs the classes (MNIST k-means ARI 0.76 with them, 0.93 without), and they are also what arranges the
communities (distance Spearman 0.42 with them, 0.29 without); a placement after the fact gives the arrangement
back for ten communities and not for sixty (k52: letter, covertype, Tasic lose). Here the crossing pairs keep their
pull during the SGD pass, projected on the rigid translations of the communities: after each epoch, the
displacement the crossing pairs would have given to the points (expected number of mid-range samples of the
epoch x clipped spring step, on both ends) is averaged over each community and the community is translated by
that mean (times `gain`); no point is pulled out of its community. Inside the kernel the crossing pairs are
redirected to the strongest neighbour, as in k52.
Result (seed 0, SGD): negative. At gain 1 the layout collapses (MNIST 0.205 / 0.196 / 0.439): the point-wise pull
of the default saturates against the bounded kNN attraction, the rigid one does not, so the net pull is far above
the default's (median centroid distance / within-community spread at the end: default 9.9 / 1.6, gain 1 2.6 / 7.7,
gain 0.1 11.2 / 1.5, no crossing pair 12.6 / 1.5). At gain 0.1-0.3 the Spearman stays near the no-crossing level
(MNIST 0.31 against 0.42 for the default and 0.29 without crossing pairs; letter 0.34 against 0.58 and 0.32) and
the ARI falls (MNIST 0.79 against 0.93 without crossing pairs). k55 gives the reason.
Prototype on the research hook `epoch_callback` (one native call per epoch), no kernel change.
One fit per (mode, seed), scored as is and after the post-process (contraction 0.7); baselines: the default rows of
k51 and the mid_in_comm rows of k52 (same seeds, same scoring).
Usage: python benchmarks/theory/k54_rigid_mid.py [datasets=mnist,letter,tasic2018] [seeds=5] [gain=1] [part=run,summary]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from magnet import MAGNET, hierarchical_contraction
from magnet.contraction import communities, interface_merge
T = 8

class RigidMid(MAGNET):
    """MAGNET whose mid-near pairs across the label-free communities pull the communities as rigid bodies."""
    rigid_gain = 1.0
    def _label_mid_pairs(self, pairs, P):
        if pairs is None: return pairs
        c = interface_merge(P, communities(P, 0.5)); self.n_comm_ = int(c.max() + 1); self._comm_ = c
        pairs = np.asarray(pairs); n, m = pairs.shape
        rows = np.repeat(np.arange(n), m); cols = pairs.ravel().astype(np.int64); x = c[rows] != c[cols]
        self._cross_ = (rows[x], cols[x]); self._n_mid_ = m; self.cross_share_ = float(x.mean())
        self._y_fit_ = c
        try: return super()._label_mid_pairs(pairs, P)
        finally: self._y_fit_ = None
    def _optimize(self, Y, head, tail, epochs_per_sample, P, n_epochs, n_fixed, seed, mid_weight):
        cross = getattr(self, "_cross_", None)
        if n_fixed or cross is None or mid_weight <= 0 or not isinstance(n_epochs, (int, np.integer)) or self.rigid_gain <= 0:
            return super()._optimize(Y, head, tail, epochs_per_sample, P, n_epochs, n_fixed, seed, mid_weight)
        n, total = Y.shape[0], int(n_epochs); mid_len = max(1.0, self.mid_phase * total); lr = float(self.learning_rate)
        u, w = cross; c, K = self._comm_, self.n_comm_; cu, cw = c[u], c[w]
        size = np.maximum(np.bincount(c, minlength=K), 1).astype(np.float64)
        # expected mid-range samples of u per epoch towards one given partner: mid_rate per edge sample whose head is u
        rate = (self.mid_rate * np.bincount(head, weights=1.0 / epochs_per_sample, minlength=n) / self._n_mid_)[u].astype(np.float32)[:, None]
        gain = float(self.rigid_gain)
        def cb(Y, e1, _total):
            e = e1 - 1; mw = mid_weight * max(0.0, 1.0 - e / mid_len)
            if mw <= 0: return
            g = np.clip(mw * (Y[w] - Y[u]), -4.0, 4.0) * (lr * (1.0 - e / total) * rate)   # step of u; w takes the opposite
            D = np.stack([np.bincount(cu, weights=g[:, d], minlength=K) - np.bincount(cw, weights=g[:, d], minlength=K) for d in range(Y.shape[1])], 1)
            Y += (gain * D / size[:, None]).astype(np.float32)[c]
        self.epoch_callback, self.callback_every = cb, 1
        try: return super()._optimize(Y, head, tail, epochs_per_sample, P, n_epochs, n_fixed, seed, mid_weight)
        finally: del self.epoch_callback, self.callback_every

if __name__ == "__main__":
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from datasets import mnist, fmnist
    from metrics import _idx, evaluate
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    DS = kw.get("datasets", "mnist,letter,tasic2018").split(","); SEEDS = int(kw.get("seeds", 5)); PART = kw.get("part", "run,summary").split(",")
    GAIN = float(kw.get("gain", 1.0)); NAME = "rigid" if GAIN == 1.0 else f"rigid x{GAIN:g}"
    OUT = os.path.join(HERE, "k54_rigid_mid.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    REF = {"mnist": dict(knn15=0.356, sp=0.488, ari=0.914), "fmnist": dict(knn15=0.330, sp=0.873, ari=0.483),
           "letter": dict(knn15=0.642, sp=0.716), "covertype": dict(knn15=0.632, sp=0.755)}
    REF.update(json.load(open(os.path.join(HERE, "k47_scrna_heldout.json")))["_refs"])
    MODES = {"SGD": "umap", "hybrid": "hybrid"}

    def load(name):
        if name == "mnist": return mnist(70000)
        if name == "fmnist": return fmnist(70000)
        if name in ("pbmc68k", "tasic2018"):
            z = np.load(os.path.join(B, "cache", "scrna", f"{name}.npz")); return z["X"], z["y"], f"scrna_{name}"
        from k27_transfer import load as load27
        return load27(name, 20000)

    def composite(r, ds):
        sh = [r[a] / REF[ds][a] for a in REF[ds]]; return float(np.prod(sh) ** (1 / len(sh))), float(min(sh))

    if "run" in PART:
        for ds in DS:
            X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); s10 = _idx(n, 10000, 2)
            def score(Y):
                e = evaluate(X, Y, y, cache_key=key)
                r = dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"],
                         ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
                r["geo"], r["min"] = composite(r, ds); return r
            for s in range(SEEDS):
                for tag, loss in MODES.items():
                    ka, kb = f"{ds} | {tag} {NAME} | {s}", f"{ds} | {tag} {NAME} + contraction | {s}"
                    if ka in res and kb in res: continue
                    m = RigidMid(loss=loss, random_state=s, n_threads=T); m.rigid_gain = GAIN
                    t0 = time.time(); Y = m.fit_transform(X); t = time.time() - t0
                    t1 = time.time(); Z = hierarchical_contraction(Y, m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=0.7)[0]; tc = time.time() - t1
                    for k, L, tt in ((ka, Y, t), (kb, Z, t + tc)):
                        r = score(L); r["fit_s"] = round(tt, 1); r["n_comm"] = m.n_comm_; r["cross_share"] = round(m.cross_share_, 3); res[k] = r
                        print(k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)

    if "summary" in PART:
        b51 = json.load(open(os.path.join(HERE, "k51_soft_community_panel.json"))); b52 = json.load(open(os.path.join(HERE, "k52_mid_in_comm_placement.json")))
        AX = ("knn15", "sp", "ari", "geo")
        for ds in DS:
            print(f"\n=== {ds} (refs {REF[ds]})")
            for tag in MODES:
                for suf in ("", " + contraction"):
                    rows = ((f"{tag} default{suf}", b51), (f"{tag} mid_in_comm{suf}", b52), (f"{tag} {NAME}{suf}", res))
                    for name, src in rows:
                        v = [src[f"{ds} | {name} | {s}"] for s in range(SEEDS) if f"{ds} | {name} | {s}" in src]
                        if v: print(f"  {name:34s} " + "  ".join(f"{a} {np.mean([x[a] for x in v]):.3f}" for a in AX + ("min",)) + f"  (n={len(v)})")
                    d = [[res[f"{ds} | {tag} {NAME}{suf} | {s}"][x] - b51[f"{ds} | {tag} default{suf} | {s}"][x] for x in AX]
                         for s in range(SEEDS) if f"{ds} | {tag} {NAME}{suf} | {s}" in res and f"{ds} | {tag} default{suf} | {s}" in b51]
                    if d:
                        d = np.array(d); print(f"  {tag} {NAME}{suf} - default{suf}: " + "  ".join(f"{a} {m:+.3f} +- {sd:.3f}" for a, m, sd in zip(AX, d.mean(0), d.std(0))))
