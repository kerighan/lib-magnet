# MAGNET

**Fast dimensionality reduction and graph layout, built on attraction and repulsion.**

MAGNET turns a data matrix, a sparse matrix or a graph into a 2-D (or 3-D) layout the way UMAP
and t-SNE do: a neighbour graph pulls close points together, everything else pushes apart. It
rebuilds that pipeline from scratch in C++ and makes the one choice that matters explicit: how the
repulsion is estimated.

* `loss="umap"` (default): negative sampling, lock-free multithreaded SGD. **MNIST 70k in 5.6 s**
  against 16 s for UMAP, with better neighbourhood preservation.
* `loss="tsne"`: exact repulsion over all pairs through FFT interpolation. **MNIST 70k in 11 s**
  against 102 s for openTSNE and 196 s for scikit-learn's t-SNE, with the best neighbourhood
  preservation of all the methods tested, three times that of UMAP in less time.
* 1M points in 55 s and 3 GB. Native NN-descent (1.2 s for 70k x 784), spectral initialisation by
  Chebyshev-filtered subspace iteration, `transform()` for new points, sparse inputs, networkx graphs.

![MNIST 70k: UMAP, MAGNET, MAGNET with exact repulsion](benchmarks/plots/compare_mnist70000_tsne.png)

![quality against time](benchmarks/plots/tradeoff.png)

## Install

```bash
pip install .            # C++17 compiler needed; numpy, scipy, tqdm; hnswlib optional
python -m pytest tests -q
```

## Quickstart

```python
from magnet import MAGNET

Y = MAGNET().fit_transform(X)                        # X: (n, d) array; row i of Y is X[i]
Y = MAGNET(loss="tsne").fit_transform(X)             # exact repulsion, t-SNE-grade local structure
Y = MAGNET(metric="cosine").fit_transform(T)         # sparse CSR matrix (TF-IDF), no densification
Y = MAGNET().fit_transform(G)                        # networkx graph, rows follow list(G.nodes)
Y = MAGNET().fit_transform(A)                        # scipy sparse adjacency (weights = similarities)

m = MAGNET().fit(X)
Y_new = m.transform(X_new)                           # out-of-sample points, training layout fixed
m.embedding_, m.graph_, m.knn_indices_, m.knn_dists_
```

## Two regimes, one knob

t-SNE, UMAP and LargeVis optimise the same kind of objective and differ mostly in the balance
between attraction and repulsion (Boehm, Berens & Kobak 2022); the negative-sampling estimator of
UMAP is biased towards weak repulsion, which is what makes its clusters compact (Damrich &
Hamprecht 2021). MAGNET exposes this directly:

| | repulsion | look | when |
|---|---|---|---|
| `loss="umap"` | sampled (5 random points per positive sample) | compact clusters, UMAP-like | speed, 3-D, graphs, millions of points |
| `loss="tsne"` | exact, all pairs, FFT-accelerated (2-D) | t-SNE-like, finest local structure | best neighbourhood preservation |
| `loss="tsne", exaggeration=4` | exact | UMAP-like again | the two regimes really are one spectrum |

`repulsion_strength` (default 2) and `exaggeration` move along the spectrum within each mode.

## How it works

```
data (n, d) ──NN-descent──▶ kNN ──smooth kNN──▶ fuzzy graph ─┐
   or CSR     (C++, SIMD)                                    ├─▶ init ──▶ optimiser ──▶ layout
networkx graph / sparse adjacency ───────────────────────────┘
```

1. **Neighbours.** NN-descent with a random-projection forest for the initial candidates, lock-free
   multithreaded local joins (each heap is only written by its owner thread), AVX-512 / AVX2 distance
   kernels picked at runtime, dense or CSR rows. Recall 0.995 on MNIST at k = 15.
2. **Graph.** UMAP's smooth-kNN kernel and fuzzy union, vectorised. Graph inputs are symmetrised and
   degree-normalised so that hubs do not monopolise the sampling budget.
3. **Initialisation.** Leading non-trivial eigenvectors of the normalised adjacency, by a
   Chebyshev-filtered subspace iteration in C++ (0.2 s for 20k points; ARPACK needs seconds and
   sometimes fails to converge). Disconnected components are laid out separately and placed according
   to their centroids. PCA is used instead for `loss="tsne"` on data (better global structure).
4. **Optimiser.** Either Hogwild SGD with an explicit sampling budget per node (`samples_per_node`,
   so the cost of an epoch does not depend on the graph), negatives, and an optional mid-range
   attraction through short random walks; or full-batch gradient descent with the t-SNE objective,
   early exaggeration, gains and momentum, where the repulsive field is the convolution of the point
   density with a radial kernel, evaluated by Lagrange interpolation on a grid and FFT.

## Benchmarks

Measured on a 20-core i9-9900X, several seeds, defaults everywhere. Local structure: 15-NN
preservation and trustworthiness. Global structure: Spearman correlation of pairwise distances and
random-triplet accuracy on a sample. Label structure: 10-NN accuracy in 2-D. The full table against
UMAP, openTSNE, PaCMAP, TriMap and scikit-learn's t-SNE is in [benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md).

**MNIST 70k**

| | time | 15-NN pres. | trustworthiness | dist. Spearman | 10-NN acc. |
|---|---|---|---|---|---|
| **MAGNET** | **5.6 s** | 0.128 | 0.953 | 0.321 | 0.967 |
| **MAGNET `loss="tsne"`** | 11.1 s | **0.352** | 0.967 | **0.335** | 0.967 |
| MAGNET `loss="tsne"`, fast preset (`n_epochs=450, fft_box=1.5`) | 7.3 s | 0.324 | 0.967 | 0.340 | 0.968 |
| UMAP 0.5.11 | 18.2 s | 0.116 | 0.951 | 0.300 | 0.967 |
| openTSNE 1.0.4 | 102.5 s | 0.347 | **0.968** | 0.330 | 0.965 |
| sklearn t-SNE (Barnes-Hut) | 195.7 s | 0.350 | 0.968 | 0.338 | 0.967 |
| TriMap | 57.0 s | 0.088 | 0.933 | 0.184 | 0.964 |

**Other data sets** (MAGNET default / UMAP)

| dataset | time | 15-NN pres. | trust. | dist. Spearman |
|---|---|---|---|---|
| Fashion-MNIST 70k | 5.2 s / 11.1 s | 0.136 / 0.133 | 0.974 / 0.975 | 0.56 / 0.58 |
| 20 newsgroups TF-IDF, sparse 18.8k x 24k | 5.7 s / 22.2 s | 0.145 / 0.140 | 0.730 / 0.716 | 0.144 / 0.145 |
| 15 Gaussian blobs 10k x 200 | 1.3 s / 9.2 s | 0.049 / 0.046 | 0.971 / 0.971 | 0.55 / 0.13 |
| SBM graph, 30k nodes, 782k edges | 4.9 s / 36 s | | | 10-NN acc. 1.00 / 0.26 |

**Scaling** (50 anisotropic clusters in 50 dimensions)

| n | MAGNET | UMAP | peak RSS |
|---|---|---|---|
| 100k | 7 s | 37 s | 0.4 / 2.1 GB |
| 250k | 16 s | 55 s | 0.8 / 3.0 GB |
| 500k | 30 s | 85 s | 1.6 / 5.1 GB |
| 1M | 55 s | | 3.1 GB |

`transform()` on MNIST (fit 60k, transform 10k): 0.6 s and 10-NN accuracy 0.962 against 14.8 s and
0.956 for UMAP.

Regenerate everything with `benchmarks/full_table.py`, `benchmarks/plot_tradeoff.py`,
`benchmarks/compare_plots.py`, `benchmarks/scaling.py` and `benchmarks/graph_demo.py`.

## Graphs

Any networkx graph or sparse adjacency matrix is laid out directly, in linear time: communities of
25 to 3000 nodes in 1.4 s, an 8k-node power-law graph in 1.1 s, a 60 x 60 grid comes out as a sheet.

![graphs](benchmarks/plots/graph_demo.png)

## Parameters

| parameter | default | meaning |
|---|---|---|
| `n_components` | 2 | output dimension (`loss="tsne"` is 2-D only) |
| `n_neighbors` | 15 | neighbourhood size (counts the point itself, as in UMAP) |
| `metric` | `"euclidean"` | or `"cosine"` |
| `min_dist`, `spread` | 0.1, 1.0 | shape of the low-dimensional kernel (UMAP semantics) |
| `loss` | `"umap"` | `"umap"` (sampled repulsion), `"tsne"` (exact), `"infonce"` (contrastive softmax) |
| `exaggeration`, `repulsion_strength` | 1, 2 | attraction / repulsion multipliers |
| `n_epochs` | auto | SGD epochs (500 / 300 / 200 by size) or t-SNE iterations (450); also `(constant, anneal)` or `"auto"` |
| `samples_per_node` | 5 | attractive samples per node and per epoch |
| `negative_sample_rate` | 5 | negatives per positive sample |
| `walk_len`, `mid_weight`, `mid_rate`, `mid_phase` | 4, 0.1, 0.25, 0.5 | mid-range attraction through random walks (`mid_weight=0` disables) |
| `init` | `"auto"` | PCA for `loss="tsne"` on data, spectral otherwise; or `"spectral"`, `"pca"`, `"random"`, an array |
| `knn` | `"nndescent"` | or `"hnsw"` (needs hnswlib) |
| `pca` | None | project wide data to this many components before the neighbour search |
| `kernel_b` | None | fix the kernel exponent (`1` = Cauchy: 30% faster SGD, ~10% less 15-NN preservation) |
| `fft_interp`, `fft_box` | 3, 1.0 | grid of the exact repulsion (`fft_box=1.5` is the fast preset) |
| `mid_scale` | `"fixed"` | or `"spread"` / `"rolewalk"`: per-node length of the mid-range walks from the graph's diffusion spread or from RoleWalk roles |
| `component_placement` | `"centroid"` | or `"rolewalk"`: disconnected components of a graph placed by their structural-role signature |
| `optimizer`, `clip` | `"sgd"`, `"coord"` | or `"adam"`; `"norm"` clips the step norm |
| `input` | `"auto"` | a sparse square matrix is an adjacency; `"data"` or `"graph"` to force |
| `graph_norm` | `"sym"` | degree normalisation of graph inputs |
| `transform_epochs` | 50 | epochs of `transform()` |
| `n_threads`, `random_state`, `verbose` | -1, None, False | |

Runs are reproducible only with `n_threads=1`: the SGD is lock-free.

## What was tried and did not make the cut

Measured with the harness, kept as options: constant-then-anneal and automatic epoch schedules
(lose local structure against a plain linear decay), per-node Adam, norm clipping, InfoNCE
(more global, less local), a bit-level `pow` approximation (no gain), the Cauchy kernel as default.
A per-node diffusion scale for the mid-range walks (`mid_scale="spread"` or `"rolewalk"`) changes
nothing measurable on MNIST: the mid-range term itself matters (distance correlation 0.35 against
0.29 without it) but not its per-node length. What did make the cut: `repulsion_strength=2` (+8%
neighbourhood preservation, nothing lost), the centroid placement of disconnected components, PCA
initialisation for the exact-repulsion mode, and for graphs the placement of disconnected components
by RoleWalk signature (`component_placement="rolewalk"`: structurally similar components end up next
to each other).

## History

MAGNET started in 2019 as a random-walk embedding with a full-batch optimiser; v3 (2026) is a
complete rewrite around per-edge SGD, a native neighbour search and the exact-repulsion mode.
RoleWalk, the author's structural-role embedding, is the natural companion for graph inputs.

## Cite

```
@software{magnet,
  author = {Chenebaux, Maixent},
  title  = {MAGNET: fast dimensionality reduction and graph layout},
  year   = {2026},
  url    = {https://github.com/kerighan/lib-magnet}
}
```
