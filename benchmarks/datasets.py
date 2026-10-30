"""Dataset loaders for the benchmarks. Every loader returns (X float32, labels or None, name)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(HERE, "cache")


def mnist(n=70000, seed=0):
    X = np.load(os.path.join(LIB, "magnet", "tests", "datasets", "mnist_data.npy"))
    y = np.load(os.path.join(LIB, "magnet", "tests", "datasets", "mnist_color.npy"))
    if n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
        X, y = X[idx], y[idx]
    return (X / 255.0).astype(np.float32), y, f"mnist{n}"


def fmnist(n=70000, seed=0):
    path = os.path.join(CACHE, "fmnist.npz")
    if not os.path.exists(path):
        from sklearn.datasets import fetch_openml
        d = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        np.savez(path, X=d.data.astype(np.float32), y=d.target.astype(int))
    d = np.load(path)
    X, y = d["X"], d["y"]
    if n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
        X, y = X[idx], y[idx]
    return (X / 255.0).astype(np.float32), y, f"fmnist{n}"


def blobs():
    X = np.load(os.path.join(LIB, "cmagnet", "data.npy")).astype(np.float32)
    return X, None, "blobs10k"


def news20(n_components=300):
    """20 newsgroups TF-IDF reduced by SVD (dense, cosine-friendly)."""
    path = os.path.join(CACHE, f"news20_svd{n_components}.npz")
    if not os.path.exists(path):
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        T = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english").fit_transform(d.data)
        Z = TruncatedSVD(n_components, random_state=0).fit_transform(T)
        np.savez(path, X=Z.astype(np.float32), y=np.asarray(d.target))
    d = np.load(path)
    return d["X"], d["y"], f"news20"


def news20_tfidf():
    """20 newsgroups raw TF-IDF (sparse CSR, ~18.8k x ~30k)."""
    import scipy.sparse as sp
    path = os.path.join(CACHE, "news20_tfidf.npz")
    ypath = os.path.join(CACHE, "news20_tfidf_y.npy")
    if not os.path.exists(path):
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        T = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english").fit_transform(d.data)
        sp.save_npz(path, sp.csr_matrix(T, dtype=np.float32))
        np.save(ypath, np.asarray(d.target))
    return sp.load_npz(path).tocsr(), np.load(ypath), "news20tfidf"


def sbm_graph():
    import pickle
    G = pickle.load(open(os.path.join(LIB, "cmagnet", "graph.pkl"), "rb"))
    y = np.repeat(np.arange(6), 5000)
    return G, y, "sbm30k"


LOADERS = {"mnist": mnist, "fmnist": fmnist, "blobs": blobs, "news20": news20, "news20tfidf": news20_tfidf,
           "sbm": sbm_graph}
