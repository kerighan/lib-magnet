"""Quick correctness tests: python -m pytest tests -q"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pytest
from magnet import MAGNET, find_ab_params
from magnet.knn import knn_hnsw, knn_nndescent, smooth_knn, fuzzy_graph
from magnet.spectral import spectral_init


def blobs(n=3000, d=20, centers=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, centers, n)
    C = rng.normal(0, 10, (centers, d))
    return (C[y] + rng.normal(0, 1, (n, d))).astype(np.float32), y


def test_ab_params_match_umap():
    a, b = find_ab_params(1.0, 0.1)
    assert abs(a - 1.577) < 0.01 and abs(b - 0.895) < 0.01


def test_knn_excludes_self_and_is_sorted():
    X, _ = blobs(500)
    ind, dist = knn_hnsw(X, 10)
    assert ind.shape == (500, 9) and dist.shape == (500, 9)
    assert not (ind == np.arange(500)[:, None]).any()
    assert (np.diff(dist, axis=1) >= -1e-6).all()
    # exact neighbours for a few points
    D = ((X[:5, None, :] - X[None, :, :]) ** 2).sum(-1)
    for i in range(5):
        exact = set(np.argsort(D[i])[1:10])
        assert len(exact & set(ind[i])) >= 8


def test_smooth_knn_rows_sum_to_log2k():
    X, _ = blobs(2000)
    ind, dist = knn_hnsw(X, 15)
    vals = smooth_knn(dist, 15)
    assert vals.shape == dist.shape
    assert np.allclose(vals.sum(1), np.log2(15), atol=1e-2)
    assert (vals[:, 0] == 1.0).all()  # nearest neighbour always has full membership


def test_fuzzy_graph_symmetric():
    X, _ = blobs(1000)
    ind, dist = knn_hnsw(X, 10)
    P = fuzzy_graph(ind, dist, 10)
    assert abs(P - P.T).max() < 1e-6
    assert P.diagonal().max() == 0
    assert P.data.max() <= 1.0 + 1e-6


def test_rows_follow_input_order():
    """Row i of the embedding must be point i (the v2 bug was a permutation)."""
    X, y = blobs(3000, centers=6)
    Y = MAGNET(n_epochs=150, random_state=0).fit_transform(X)
    assert Y.shape == (3000, 2) and np.isfinite(Y).all()
    # the embedded neighbours of a point must belong to its own blob
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(Y).kneighbors(Y[:300], return_distance=False)[:, 1:]
    same = (y[nn] == y[:300, None]).mean()
    assert same > 0.97


def test_graph_input_row_order():
    """networkx graph with shuffled node insertion: rows follow list(G.nodes)."""
    G = nx.random_partition_graph([200, 200, 200], 0.1, 0.002, seed=0)
    nodes = list(G.nodes)
    perm = np.random.default_rng(0).permutation(nodes)
    H = nx.Graph()
    H.add_nodes_from(perm)
    H.add_edges_from(G.edges)
    Y = MAGNET(n_epochs=200, random_state=0).fit_transform(H)
    assert Y.shape == (600, 2)
    order = list(H.nodes)
    assert order == list(perm)
    labels = np.array([n // 200 for n in order])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    assert (labels[nn] == labels[:, None]).mean() > 0.9


def test_sparse_input_and_components():
    A = sp.block_diag([sp.csr_matrix(np.ones((50, 50)) - np.eye(50)) for _ in range(3)]).tocsr()
    Y = MAGNET(n_epochs=100, random_state=0).fit_transform(A)
    assert Y.shape == (150, 2) and np.isfinite(Y).all()
    # the three disconnected cliques must land in three separate places
    centers = np.array([Y[i * 50:(i + 1) * 50].mean(0) for i in range(3)])
    spread = np.array([Y[i * 50:(i + 1) * 50].std(0).max() for i in range(3)])
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.linalg.norm(centers[i] - centers[j]) > 3 * max(spread[i], spread[j])


def test_spectral_init_is_smooth_on_a_cycle():
    G = nx.cycle_graph(400)
    P = MAGNET().build_graph(G)
    Y = spectral_init(P, 2, random_state=0)
    # consecutive nodes should be close compared to the size of the layout (smooth init)
    step = np.linalg.norm(np.diff(Y, axis=0), axis=1).mean()
    diameter = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).max()
    assert diameter > 30 * step


def test_single_thread_is_reproducible():
    X, _ = blobs(800)
    Y1 = MAGNET(n_epochs=50, random_state=3, n_threads=1).fit_transform(X)
    Y2 = MAGNET(n_epochs=50, random_state=3, n_threads=1).fit_transform(X)
    assert np.allclose(Y1, Y2)


def test_other_dimensions_and_metrics():
    X, _ = blobs(1000)
    Y = MAGNET(n_components=3, metric="cosine", n_epochs=50).fit_transform(X)
    assert Y.shape == (1000, 3) and np.isfinite(Y).all()
    Y = MAGNET(n_components=2, init="pca", n_epochs=50).fit_transform(X)
    assert Y.shape == (1000, 2)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_nndescent_matches_brute_force(metric):
    X, _ = blobs(4000, d=30, centers=8, seed=1)
    ind, dist = knn_nndescent(X, 15, metric=metric, random_state=0)
    assert ind.shape == (4000, 14) and dist.shape == (4000, 14)
    assert not (ind == np.arange(4000)[:, None]).any()
    assert (np.diff(dist, axis=1) >= -1e-5).all()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=15, metric=metric, algorithm="brute").fit(X)
    ed, ei = nn.kneighbors(X[:500])
    ed, ei = ed[:, 1:], ei[:, 1:]
    recall = np.mean([len(np.intersect1d(ind[i], ei[i])) / 14 for i in range(500)])
    assert recall > 0.97
    assert np.allclose(dist[:500, 0], ed[:, 0], atol=1e-3)


def test_nndescent_handles_duplicates_and_small_n():
    X = np.repeat(np.random.default_rng(0).normal(size=(10, 5)).astype(np.float32), 50, axis=0)  # 500 rows, 10 distinct
    ind, dist = knn_nndescent(X, 10)
    assert ind.shape == (500, 9) and np.isfinite(dist).all() and (dist[:, 0] == 0).all()
    X, _ = blobs(60, d=5)
    ind, dist = knn_nndescent(X, 15)
    assert ind.shape == (60, 14)


def test_nndescent_cosine_zero_rows_are_not_everyones_neighbours():
    X, _ = blobs(2000, d=20, centers=4, seed=2)
    X[:100] = 0.0
    ind, dist = knn_nndescent(X, 15, metric="cosine")
    zero_neighbours = (ind[100:] < 100).mean()
    assert zero_neighbours < 0.01
    assert (dist[100:] < 0.99).all()


def test_sparse_data_input_and_transform():
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 1500)
    C = sp.random(4, 500, density=0.1, random_state=0, format="csr")
    X = sp.vstack([C[y[i]] + sp.random(1, 500, density=0.02, random_state=i, format="csr") for i in range(1500)]).tocsr()
    m = MAGNET(n_epochs=150, metric="cosine", random_state=0).fit(X)
    assert m.embedding_.shape == (1500, 2)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(m.embedding_).kneighbors(m.embedding_, return_distance=False)[:, 1:]
    assert (y[nn] == y[:, None]).mean() > 0.9
    Yq = m.transform(X[:50])
    assert Yq.shape == (50, 2)
    assert np.linalg.norm(Yq - m.embedding_[:50], axis=1).mean() < 0.5 * np.sqrt((m.embedding_ ** 2).sum(1).mean())


def test_transform_dense_places_points_near_their_neighbours():
    X, y = blobs(4000, d=20, centers=6, seed=3)
    m = MAGNET(n_epochs=150, random_state=0).fit(X[:3500])
    Yq = m.transform(X[3500:])
    assert Yq.shape == (500, 2) and np.isfinite(Yq).all()
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(10).fit(m.embedding_, y[:3500])
    assert (clf.predict(Yq) == y[3500:]).mean() > 0.95
    # the same point transformed lands on top of its training position
    Ys = m.transform(X[:100])
    spacing = np.median(np.linalg.norm(np.diff(m.embedding_[np.argsort(m.embedding_[:, 0])], axis=0), axis=1))
    assert np.linalg.norm(Ys - m.embedding_[:100], axis=1).mean() < 20 * spacing


@pytest.mark.parametrize("kw", [dict(loss="infonce", learning_rate=2), dict(optimizer="adam", learning_rate=0.1),
                                dict(clip="norm"), dict(n_epochs=(60, 40)), dict(n_epochs="auto"),
                                dict(exaggeration=2.0, repulsion_strength=2.0)])
def test_optimizer_and_loss_options_run(kw):
    X, y = blobs(1500, d=10, centers=5, seed=4)
    params = dict(n_epochs=80, random_state=0)
    params.update(kw)
    m = MAGNET(**params).fit(X)
    assert np.isfinite(m.embedding_).all()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(m.embedding_).kneighbors(m.embedding_, return_distance=False)[:, 1:]
    assert (y[nn] == y[:, None]).mean() > 0.9


def test_fft_repulsion_matches_brute_force():
    from magnet.tsne import FFTRepulsion2D
    rng = np.random.default_rng(0)
    Y = np.ascontiguousarray(rng.normal(0, 8, (1500, 2)).astype(np.float32))
    D2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    Q = 1.0 / (1.0 + D2)
    np.fill_diagonal(Q, 0)
    F_exact = ((Q ** 2)[:, :, None] * (Y[:, None, :] - Y[None, :, :])).sum(1)
    sumQ, F = FFTRepulsion2D(n_interp=5)(Y)
    assert abs(sumQ - Q.sum()) / Q.sum() < 1e-3
    assert np.linalg.norm(F - F_exact) / np.linalg.norm(F_exact) < 2e-2


def test_tsne_loss_separates_blobs():
    X, y = blobs(2000, d=10, centers=5, seed=5)
    m = MAGNET(loss="tsne", n_epochs=300, random_state=0).fit(X)
    assert np.isfinite(m.embedding_).all()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(m.embedding_).kneighbors(m.embedding_, return_distance=False)[:, 1:]
    assert (y[nn] == y[:, None]).mean() > 0.95
