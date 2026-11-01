"""Linear references for the global column: PCA (dense) / truncated SVD (sparse) to 2-D, same metrics,
same cached subsamples as the benchmark table. Usage: python benchmarks/baselines_linear.py"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from datasets import mnist, fmnist, blobs, news20, news20_tfidf
from metrics import evaluate
from sklearn.decomposition import PCA, TruncatedSVD

out = {}
for loader in (mnist, fmnist, news20, news20_tfidf, blobs):
    X, y, key = loader()
    t = time.time()
    if sp.issparse(X):
        Y = TruncatedSVD(2, random_state=0).fit_transform(X)
        how = "SVD-2"
    else:
        Y = PCA(2, svd_solver="randomized", random_state=0).fit_transform(X)
        how = "PCA-2"
    dt = time.time() - t
    r = evaluate(X, Y.astype(np.float32), y, cache_key=key)
    r["time"] = dt; r["method"] = how
    out[key] = r
    print(key, how, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
json.dump(out, open(os.path.join(HERE, "cache", "linear_baselines.json"), "w"), indent=1)
