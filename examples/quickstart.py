"""Minimal end-to-end example: the three modes and the post-process on scikit-learn's digits (1797 x 64),
with the two evaluation axes of the paper computed on the result. Runs in a few seconds.
Usage: python examples/quickstart.py"""
import time
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
from magnet import MAGNET

X, y = load_digits(return_X_y=True)
X = X.astype(np.float32)

def knn_overlap(X, Y, k=15):
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return np.mean([len(set(u) & set(v)) / k for u, v in zip(a, b)])

for name, kw in [("sampled pass", {}), ("full-batch pass", {"loss": "tsne"}), ("sequence", {"loss": "hybrid"}),
                 ("sequence + post-process", {"loss": "hybrid", "contraction": 0.7})]:
    t = time.time(); Y = MAGNET(random_state=0, **kw).fit_transform(X); dt = time.time() - t
    print(f"{name:<24} {dt:5.1f} s   15-NN overlap {knn_overlap(X, Y):.3f}   distance Spearman {spearmanr(pdist(X), pdist(Y)).correlation:.3f}")
