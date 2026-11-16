"""k55: where does the distance Spearman of the crossing mid-range pairs live? Oracle for k54. Two SGD layouts of the
same graph and seed: the default (mid-near pairs as they are) and `mid_in_comm` (pairs that cross a label-free
community redirected, k52). Each community of the second layout is mapped onto the same community of the first by
the least-squares translation, rigid motion (translation + rotation, Procrustes) or affine map, and the result is
scored. If the translated layout reaches the default's Spearman, the arrangement of the centroids carries it and a
force on whole communities can reproduce it; if only the affine map does, the shape of the communities carries
it; if none does, it is in the arrangement of the points inside the communities. Each layout is also scored after the separation
alone (every community contracted by 0.7 around its centroid, ` > 0.7`) and after the whole post-process (placement
and separation, ` + post`).
Result (2 seeds, SGD, 15-NN / Spearman / ARI): the oracle translation recovers about half of the Spearman gap and
lowers the ARI below the default's (MNIST default 0.192 / 0.418 / 0.747, no crossing pair 0.206 / 0.285 / 0.934,
translation 0.200 / 0.359 / 0.621, rigid motion 0.201 / 0.397 / 0.686, affine 0.192 / 0.401 / 0.729; letter
Spearman 0.577, 0.317, 0.403, 0.463, 0.524). After the post-process the order is unchanged (MNIST Spearman / ARI:
default 0.502 / 0.758, no crossing pair 0.445 / 0.883, translation 0.470 / 0.783, affine 0.489 / 0.772): the
Spearman of the crossing pairs is paid in ARI whatever the motion of the communities, so a force on whole
communities cannot give both.
Usage: python benchmarks/theory/k55_rigid_oracle.py [datasets=mnist,letter,tasic2018] [seeds=2]"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist, fmnist
from metrics import _idx, evaluate
from magnet import MAGNET, hierarchical_contraction
from magnet.contraction import communities, interface_merge, contract
from k54_rigid_mid import RigidMid
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:])
DS = kw.get("datasets", "mnist,letter,tasic2018").split(","); SEEDS = int(kw.get("seeds", 2))
OUT = os.path.join(HERE, "k55_rigid_oracle.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def load(name):
    if name == "mnist": return mnist(70000)
    if name == "fmnist": return fmnist(70000)
    if name in ("pbmc68k", "tasic2018"):
        z = np.load(os.path.join(B, "cache", "scrna", f"{name}.npz")); return z["X"], z["y"], f"scrna_{name}"
    from k27_transfer import load as load27
    return load27(name, 20000)

def fit_map(A, Bt, kind):
    """Least-squares map of the points A onto the points Bt (same rows): translation, rigid motion or affine map."""
    ca, cb = A.mean(0), Bt.mean(0); A0, B0 = A - ca, Bt - cb
    if kind == "translation" or len(A) < 4: return A0 + cb
    if kind == "rigid":
        U, _, Vt = np.linalg.svd(A0.T @ B0); R = U @ Vt
        if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
        return A0 @ R + cb
    M = np.linalg.lstsq(A0, B0, rcond=None)[0]; return A0 @ M + cb

for ds in DS:
    X, y, key = load(ds); y = np.asarray(y).astype(int); n = len(y); kcl = len(np.unique(y)); s10 = _idx(n, 10000, 2)
    def score(Y):
        e = evaluate(X, Y, y, cache_key=key)
        return dict(knn15=e["knn15_preservation"], sp=e["dist_spearman"],
                    ari=float(adjusted_rand_score(y[s10], KMeans(kcl, n_init=5, random_state=0).fit_predict(Y[s10]))))
    for s in range(SEEDS):
        if f"{ds} | affine + post | {s}" in res: continue
        Yd = MAGNET(random_state=s, n_threads=T).fit_transform(X)
        m = RigidMid(random_state=s, n_threads=T); m.rigid_gain = 0.0; Yn = m.fit_transform(X); c = m._comm_
        L = {"default": Yd, "mid_in_comm": Yn}
        for kind in ("translation", "rigid", "affine"):
            Z = np.empty_like(Yn)
            for k in range(m.n_comm_):
                i = np.flatnonzero(c == k); Z[i] = fit_map(Yn[i].astype(np.float64), Yd[i].astype(np.float64), kind)
            L[kind] = Z
        for name in list(L):
            L[name + " > 0.7"] = contract(L[name], c, 0.7)
            L[name + " + post"] = hierarchical_contraction(L[name], m.graph_, X=X, metric=m.metric, random_state=s, n_threads=T, separation=0.7)[0]
        for name, Z in L.items():
            r = score(Z); r["n_comm"] = m.n_comm_; res[f"{ds} | {name} | {s}"] = r
            print(f"{ds} | {name:20s} | {s}", {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}, flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
print()
for ds in DS:
    for name in [b + suf for suf in ("", " > 0.7", " + post") for b in ("default", "mid_in_comm", "translation", "rigid", "affine")]:
        v = [res[f"{ds} | {name} | {s}"] for s in range(SEEDS) if f"{ds} | {name} | {s}" in res]
        if v: print(f"{ds:10s} {name:20s} " + "  ".join(f"{a} {np.mean([x[a] for x in v]):.3f}" for a in ("knn15", "sp", "ari")) + f"  (n={len(v)}, {v[0]['n_comm']} communities)")
