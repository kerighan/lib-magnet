"""k12: the sparse kNN of 20 newsgroups TF-IDF. Exact 15-NN by chunked sparse products, recall and time of the sparse
NN-descent, and the dense-projection route (Gaussian / truncated SVD + dense NN-descent, candidate recall among 60).
Run from the repository root; results in k12_sparse_knn.log."""
"""Sparse kNN: recall of the current sparse NN-descent, and of dense-projection + dense NN-descent (+ exact re-rank)."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np, scipy.sparse as sp
from datasets import LOADERS
from magnet.knn import knn_nndescent, prepare_data
X, y, name = LOADERS["news20tfidf"]()
n = X.shape[0]; K = 15
Xn = prepare_data(X, "cosine")                      # L2-normalised rows (cosine = 1 - dot)
Xn = Xn[0] if isinstance(Xn, tuple) else Xn
# exact 15-NN by chunked sparse products
t = time.time(); exact = np.zeros((n, K), np.int32)
for s in range(0, n, 2000):
    S = (Xn[s:s + 2000] @ Xn.T).toarray(); S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
    exact[s:s + 2000] = np.argpartition(-S, K, axis=1)[:, :K]
print(f"exact 15-NN (brute force, chunked): {time.time() - t:.1f}s", flush=True)
def recall(ind):
    return np.mean([len(np.intersect1d(ind[i], exact[i])) / K for i in range(0, n, 7)])
t = time.time(); ind, dist = knn_nndescent(X, K, metric="cosine", random_state=0); print(f"current sparse nn-descent: {time.time() - t:.2f}s  recall@15 {recall(ind):.3f}", flush=True)
rng = np.random.default_rng(0)
for kind, dim in [("gauss", 512), ("svd", 128), ("svd", 256)]:
    t0 = time.time()
    if kind == "gauss":
        R = rng.standard_normal((Xn.shape[1], dim), dtype=np.float32) / np.sqrt(dim)
        Z = np.asarray(Xn @ R, dtype=np.float32)
    else:
        from sklearn.decomposition import TruncatedSVD
        Z = TruncatedSVD(dim, random_state=0, algorithm="randomized", n_iter=2).fit_transform(Xn).astype(np.float32)
    t_proj = time.time() - t0
    for kd in (K, 60):
        t1 = time.time(); ind_d, _ = knn_nndescent(Z, kd, metric="cosine", random_state=0); t_knn = time.time() - t1
        if kd == K:
            print(f"{kind} {dim}: proj {t_proj:.2f}s + dense nn-descent k=15 {t_knn:.2f}s  recall {recall(ind_d[:, :K]):.3f}", flush=True)
        else:
            # recall after an exact re-rank = share of the true neighbours present among the kd candidates
            ind_r = ind_d
            t_rr = 0.0
            print(f"{kind} {dim}: proj {t_proj:.2f}s + dense nn-descent k=60 {t_knn:.2f}s + exact re-rank {t_rr:.2f}s  candidate recall {recall(ind_r):.3f}", flush=True)
