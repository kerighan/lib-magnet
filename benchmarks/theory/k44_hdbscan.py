"""k44: the UMAP + HDBSCAN use case (BERTopic and friends). On data sets of Maixent's pybench collection chosen without
a PCA front end where possible (USPS 7.3k x 256, phonemes 4.5k x 256, letters 20k x 16, stellar 30k x 17 robust-scaled,
20 newsgroups 11.3k x 50 TF-IDF SVD) plus raw MNIST 70k, run UMAP, PaCMAP (the two cluster-friendly baselines) and
MAGNET (SGD; SGD, full-batch and hybrid + the post-process), then HDBSCAN on the 2-D layout with one rule for every
method (min_cluster_size = max(15, n / 200), min_samples default). Scored against the labels: ARI with the noise
points as one extra label (the honest number), ARI on the non-noise points, AMI, noise share, number of clusters,
and k-means ARI for reference; 3 seeds (UMAP unseeded, as elsewhere). Positive -> the paper, negative -> this log.
Usage: python benchmarks/theory/k44_hdbscan.py [datasets=usps,phonemes,letters,stellar,news20,mnist] [seeds=3]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
PYB = "/home/maixent/Dropbox (Reputation Squad)/DATAPROJECTS/lib/pybench"; sys.path.insert(0, PYB)
import numpy as np, hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from magnet import MAGNET
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); DS = kw.get("datasets", "usps,phonemes,letters,stellar,news20,mnist").split(","); SEEDS = int(kw.get("seeds", 3))
OUT = os.path.join(HERE, "k44_hdbscan.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def load(name):
    if name == "mnist":
        from datasets import mnist; X, y, _ = mnist(70000); return np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
    from pybench.clustering import datasets as D
    if name == "usps":
        import h5py
        with h5py.File(os.path.join(PYB, "pybench", "clustering", "datasets", "usps.h5"), "r") as hf: X = hf["train"]["data"][:]; y = hf["train"]["target"][:]
        return np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
    X, y = D.load({"letters": "letters", "phonemes": "phonemes", "stellar": "stellar", "news20": "20newsgroups"}[name])[:2]
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y).astype(int)
    if name == "stellar": idx = np.random.default_rng(0).choice(len(X), 30000, replace=False); X, y = X[idx], y[idx]
    return np.ascontiguousarray(X), y

def fit(method, X, s):
    if method == "UMAP": import umap; return umap.UMAP().fit_transform(X)
    if method == "PaCMAP": import pacmap; return pacmap.PaCMAP(random_state=s).fit_transform(X)
    if method == "MAGNET SGD": return MAGNET(random_state=s, n_threads=T).fit_transform(X)
    if method == "MAGNET SGD + post": return MAGNET(contraction=0.7, random_state=s, n_threads=T).fit_transform(X)
    if method == "MAGNET full-batch + post": return MAGNET(loss="tsne", contraction=0.7, random_state=s, n_threads=T).fit_transform(X)
    if method == "MAGNET hybrid + post": return MAGNET(loss="hybrid", contraction=0.7, random_state=s, n_threads=T).fit_transform(X)
    raise ValueError(method)
METHODS = ["UMAP", "PaCMAP", "MAGNET SGD", "MAGNET SGD + post", "MAGNET full-batch + post", "MAGNET hybrid + post"]

for ds in DS:
    X, y = load(ds); n = len(y); kcl = len(np.unique(y)); mcs = max(15, n // 200)
    print(f"\n=== {ds}: {X.shape}, {kcl} classes, HDBSCAN min_cluster_size {mcs}", flush=True)
    for m in METHODS:
        rows = []
        for s in range(SEEDS):
            k_ = f"{ds}/{m}/{s}"
            if k_ not in res:
                try:
                    t = time.time(); Y = np.asarray(fit(m, X, s), dtype=np.float32); dt = time.time() - t
                except Exception as e: print(f"  {m} seed {s} failed: {e}", flush=True); break
                cl = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(Y); noise = cl < 0
                lab_all = cl.copy(); lab_all[noise] = cl.max() + 1   # the noise as one extra cluster
                km = KMeans(kcl, n_init=5, random_state=0).fit_predict(Y)
                res[k_] = dict(ari=float(adjusted_rand_score(y, lab_all)), ari_nonnoise=float(adjusted_rand_score(y[~noise], cl[~noise])) if (~noise).sum() > 1 else 0.0,
                               ami=float(adjusted_mutual_info_score(y, lab_all)), noise=float(noise.mean()), clusters=int(len(set(cl[~noise]))), ari_kmeans=float(adjusted_rand_score(y, km)), t=dt)
                json.dump(res, open(OUT, "w"), indent=1)
            rows.append(res[k_])
        if rows:
            f = lambda a: f"{np.mean([r[a] for r in rows]):.3f}+-{np.std([r[a] for r in rows]):.3f}"
            print(f"  {m:<26} HDBSCAN ARI {f('ari')}  (non-noise {f('ari_nonnoise')}, AMI {f('ami')}, noise {np.mean([r['noise'] for r in rows]):.2f}, clusters {np.mean([r['clusters'] for r in rows]):.1f})  k-means ARI {f('ari_kmeans')}  {np.mean([r['t'] for r in rows]):.1f} s", flush=True)
