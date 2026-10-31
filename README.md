# MAGNET

**Fast dimensionality reduction and graph layout, built on attraction and repulsion.**

MAGNET turns a data matrix, a sparse matrix or a graph into a 2-D (or 3-D) layout the way UMAP
and t-SNE do: a neighbour graph pulls close points together, everything else pushes apart. It
rebuilds that pipeline from scratch in C++ and makes the one choice that matters explicit: how the
repulsion is estimated.

* `loss="umap"` (default): negative sampling, lock-free multithreaded SGD. **MNIST 70k in 4.4 s**
  against 18 s for UMAP, 15-NN preservation 0.19 against 0.12 and a distance correlation of 0.41
  against 0.30: the best global structure of the seven methods benchmarked.
* `loss="tsne"`: exact repulsion over all pairs through FFT interpolation. **MNIST 70k in 9 s**
  against 102 s for openTSNE and 196 s for scikit-learn's t-SNE, with the same neighbourhood
  preservation as those two, three times that of UMAP in less time.
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
| `loss="umap"` | sampled (5 random points per positive sample) | UMAP-like, with the global structure of PaCMAP | speed, 3-D, graphs, millions of points |
| `loss="tsne"` | exact, all pairs, FFT-accelerated (2-D) | t-SNE-like, finest local structure | best neighbourhood preservation |
| `loss="tsne", exaggeration=4` | exact | UMAP-like again | the two regimes really are one spectrum |

`repulsion_strength` (default 2) and `exaggeration` move along the spectrum within each mode.

### Two exponents, not one

The low-dimensional kernel `1 / (1 + a d^2b)` does two jobs. On an edge of the neighbour graph its
gradient pulls with a force proportional to `d^(2b-1)`; on a random far point it pushes with a force
proportional to `d^-(2b+1)`. UMAP uses one exponent for both (fitted to `min_dist`, about 0.8 to
0.9). Sweeping the two separately (`kernel_b` for attraction, `repulsion_b` for repulsion; MNIST and
Fashion-MNIST 20k, 1-D to 8-D, 20 newsgroups, synthetic manifolds of known dimension; the scripts are
`benchmarks/tail_law.py`, `tail_split*.py`, `tail_force_octaves.py`) separates two clean effects:

* **The attraction exponent sets local fidelity, and its optimum is 1/2 in every target dimension.**
  15-NN preservation peaks at `kernel_b` = 0.5 on images and 0.6-0.7 on text in 1-D, 2-D, 3-D alike,
  whatever the repulsion exponent (MNIST 20k, 2-D, repulsion fixed at 1: 0.280 / **0.295** / 0.288 /
  0.243 for attraction 0.4 / 0.5 / 0.6 / 0.8; the same ranking in 3-D and on Fashion-MNIST). At
  b = 1/2 the attractive force no longer depends on the distance: the loss is linear in d near zero, an
  L1 spring. Below it the force diverges as d goes to 0 and neighbours jitter at the clipping scale;
  above it the force vanishes near zero and negatives displace the neighbours. This is a crowding
  effect: on a synthetic 2-D manifold embedded in 2-D the optimum moves to the L2 end (attraction 1.0,
  15-NN 0.724 against 0.648 at 0.5), on a 10-D manifold in 2-D it is at 0.4-0.5 (0.213 against 0.068
  at 1.0). Real data (TwoNN dimension 14 to 20) is always in the crowded regime, and more points at
  fixed `n_neighbors` means more crowding (coupled optimum 0.6 on MNIST 5k, 0.5 at 20k and 70k).
* **The repulsion exponent sets the tail felt by far points: compactness, speed and a little of
  everything else.** At fixed attraction, the Cauchy tail (`repulsion_b=1`, the kernel of t-SNE)
  matches or beats heavier tails on every fidelity metric: MNIST 20k 15-NN 0.290 to 0.296 in 2-D and
  0.315 to 0.330 in 3-D, Fashion-MNIST 0.283 to 0.299 in 3-D, distance correlation +0.01 to +0.02,
  label accuracy unchanged. It costs the compact look (silhouette 0.48 to 0.32 on MNIST 20k) and
  removes the `pow` from the inner loop: MNIST 70k 5.3 s to 4.3 s, 3 seeds. Lighter tails keep helping
  the local metric in higher dimensions (the optimum sits near m/2: 2-4 in 4-D, 3-4 in 6-D), by
  margins too small to be worth a dimension-dependent default.

The old "optimal b" was the compromise between the two: coupled, it climbs with the target dimension
(0.52 / 0.59 / 0.76 / 1.2 / 1.6 / 1.9 / 2.3 for 1 to 8 components, mean of three data sets, roughly
0.3 m) because the repulsion side tolerates lighter tails as the volume grows while the attraction
side does not move. Decoupled, the defaults are `kernel_b=0.6` (0.5 for images, 0.7 for text) and
`repulsion_b=1` in every dimension; `repulsion_b=None` gives back the coupled kernel and UMAP's
compact blobs. A tail scale-free in the volume sense (b = m/2, van der Maaten's Student-t with m - 1
degrees of freedom) is what the repulsion side prefers; the attraction side obeys a different
principle (the L1 point), which is why a single exponent could never satisfy both. The exponent
cannot be learned by gradient on the same loss (it sharpens the kernel into a step); it is a
per-dataset choice. `n_neighbors` stays at 15 for robustness, knowing that fewer neighbours give more
local detail on dense data: from 50 down to 5, 15-NN preservation rises monotonically on MNIST,
Fashion-MNIST and 20 newsgroups (0.228 / 0.248 / 0.285 / 0.297 / 0.304 at 50 / 30 / 15 / 10 / 8 on
MNIST 20k), trustworthiness and global structure stay flat, label accuracy peaks between 8 and 15, and
the run is 5% faster at 10; the exponents do not interact with it. Graph inputs are unaffected (SBM:
label accuracy 1.00 either way), `transform()` keeps 0.96 of 10-NN accuracy on MNIST test points.

The derivation behind the two numbers, and its tests, are in [docs/kernel_exponents.md](docs/kernel_exponents.md):
the attraction is a near-field robust location problem (bounded influence needs 2b <= 1, convexity
2b >= 1, hence the L1 point under crowding and L2 without), the repulsion is a far-field question of
which scale carries the loss (`d^(m - 2b)` per octave, scale-free at m/2). Two consequences were
tested. The attraction optimum is a step at q = m, not a slope in q (synthetic clusters of dimension
2 / 3 / 5 / 10 / 20 in 2-D want 1.0 / 0.5 / 0.5 / 0.4 / 0.4; 3 / 5 / 10 in 3-D want 1.0 / 0.5 / 0.4),
so homogeneous real data (MNIST, Fashion-MNIST, 20 newsgroups: under 1% of the points have a local
dimension below 6) gain nothing from a per-point exponent, but data mixing manifolds of different
dimensions do, provided the repulsion stays at Cauchy (the former per-point mode coupled the two and
never beat a constant): on five 2-D plus five 10-D clusters in 2-D, 15-NN preservation 0.46 with the
true dimension against 0.42 for the best constant (+9%; +12% on 3-D plus 10-D clusters in 3-D), the
uncrowded points getting the L2 result and the crowded ones the L1 result. `kernel_b="local"` now
applies that law (1 where the neighbourhood-smoothed local dimension fits in `n_components`, 1/2
elsewhere, with dimprint's calibrated estimator when it is installed: on the synthetic mixtures it
matches or beats the oracle, 0.545 against 0.540 and 0.459 against 0.441), `kernel_b="adaptive"` reads
the same split off the layout itself (the points whose edges all stay short at 3/4 of the run finish
at L2; it works on graph inputs, where the local dimension is not available), and an array of
exponents is the oracle path when the structure is known. On real data the heterogeneity exists where
the theory says: COIL-20 (20 objects photographed on a turntable, 1-D circles) has a third of its
points uncrowded and wants b = 1 outright (15-NN kept 0.77 against 0.69 at 0.5; the adaptive mode gets
0.72, the local dimension misses the circles), 58% of Cora's nodes and 27% of the US airports' are
uncrowded (adaptive +0.06 and +0.01 edge preservation over the same constant, 0.57 against 0.45 on
stochastic blocks with chains attached), while MNIST, Fashion-MNIST, 20 newsgroups and the Paul15
single-cell data are crowded everywhere and the adaptive mode stays neutral there (MNIST 0.293 against
0.292). A Huber attraction (L2 below a scale c, L1 above) has no universal c (MNIST wants the
smallest, 2-D manifolds want 0.2-0.5), as Huber's theorem says, and was not kept. Why text sits at
0.6-0.7 is still open: it is neither the hubs (the unsatisfiable edges end at low-degree points, and
damping hubs moves nothing) nor the degree (the optimum is 0.6-0.7 in every in-degree quintile of 20
newsgroups and 0.5 in every quintile of MNIST). `kernel_b="auto"` (subsample sweep) drifts with the sample
size. Kernels with
exponential or lighter tails confirm the crowding theory the hard way: on MNIST 20k, 15-NN
preservation 0.285 (power) against 0.16 (sech), 0.11 (Gaussian) and 0.10 (1 - tanh(d^2)). Reshaping
the neighbour-distance profile on the input side (`neighbor_power`, `neighbor_rank_exponent`) changes
nothing: the smooth-kNN calibration absorbs monotone transforms. The exact-repulsion mode behaves the
other way round: a heavier Student-t tail (`tsne_dof` < 1) raises trustworthiness and compactness but
lowers 15-NN preservation (0.405 at dof 1, 0.354 at dof 0.5 on MNIST 20k), as Kobak et al. found; its
default stays the Cauchy kernel.
Cluster compactness is a look, not a fidelity measure (it can always be bought with neighbourhood
quality); if you want UMAP's tight blobs, `repulsion_b=None` (the coupled kernel) or
`late_exaggeration=1.5` each give a silhouette of 0.51 on MNIST 70k against 0.52 for UMAP and 0.40
for the default, while staying ahead of UMAP on every other metric (15-NN 0.19 / 0.18 against 0.12).

### The written UMAP loss, optimised exactly (`loss="umap_exact"`)

A research mode: the cross-entropy UMAP writes down, with its repulsion summed over all pairs by the
same FFT machinery, scaled by `repulsion_strength`. On MNIST 20k the loss as written (weight 1)
destroys the layout (15-NN preservation 0.002, label accuracy 0.25); the sampled optimiser's quality
is only approached once the repulsion is divided by about 1000 (weight 0.001: 0.263 against 0.286),
which is the order of n / (negative_sample_rate x samples per node) = 20000 / 25 predicted by
Damrich & Hamprecht for the implicit down-weighting of negative sampling. Two conclusions: UMAP's
embeddings come from that bias, not from its stated objective, and removing the sampling noise
without changing the balance (the "debiased UMAP") brings nothing: in the exact setting the
normalised repulsion of t-SNE (`loss="tsne"`, 0.405) is the one that works.

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
4. **Mid-range pairs.** For every point, the second closest of six random points measured in the
   data (PaCMAP's mid-near pairs, 0.2 s for 70k x 784 in C++): a weak spring towards them, annealed
   to zero over the first half of the optimisation, tells the layout where the clusters go relative
   to each other. For graphs, short random walks play that role.
5. **Optimiser.** Either Hogwild SGD with an explicit sampling budget per node (`samples_per_node`,
   so the cost of an epoch does not depend on the graph) and negatives; or full-batch gradient
   descent with the t-SNE objective,
   early exaggeration, gains and momentum, where the repulsive field is the convolution of the point
   density with a radial kernel, evaluated by Lagrange interpolation on a grid and FFT.

## Benchmarks

Measured on a 20-core i9-9900X, several seeds, defaults everywhere. Local structure: 15-NN
preservation and trustworthiness. Global structure: Spearman correlation of pairwise distances and
random-triplet accuracy on a sample. Label structure: 10-NN accuracy in 2-D. The full table against
UMAP, densMAP, openTSNE, PaCMAP, TriMap, PHATE, h-NNE, scikit-learn's t-SNE and t-SNE-CUDA is in
[benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md) (NCVis is missing: its PyPI package does not build
here). t-SNE-CUDA runs on an RTX 2070 that was shared with another job during the measurement, and
its defaults (1000 iterations, learning rate 200) are under-converged at 70k points, which is why it
sits below the CPU t-SNEs on neighbourhood preservation.

**MNIST 70k**

| | time | 15-NN pres. | trustworthiness | dist. Spearman | triplet acc. | 10-NN acc. |
|---|---|---|---|---|---|---|
| **MAGNET** | **4.4 s** | 0.193 | 0.957 | **0.413** | **0.644** | 0.967 |
| MAGNET, `kernel_b=0.5` (images) | 4.6 s | 0.215 | 0.958 | 0.424 | 0.648 | 0.964 |
| MAGNET, `repulsion_b=None` (coupled kernel, UMAP's compact look: silhouette 0.51 against 0.52 for UMAP and 0.40 for the default) | 5.5 s | 0.190 | 0.958 | 0.405 | 0.641 | 0.966 |
| MAGNET, `late_exaggeration=1.5` (compact preset, silhouette 0.51) | 4.7 s | 0.181 | 0.956 | 0.386 | 0.637 | 0.967 |
| **MAGNET `loss="tsne"`** | 9.2 s | 0.349 | 0.966 | 0.348 | 0.627 | 0.966 |
| UMAP 0.5.11 | 18.2 s | 0.116 | 0.951 | 0.300 | 0.607 | 0.967 |
| densMAP (umap-learn 0.5.11) | 43.2 s | 0.070 | 0.923 | 0.357 | 0.621 | 0.956 |
| openTSNE 1.0.4 | 102.5 s | 0.347 | **0.968** | 0.330 | 0.619 | 0.965 |
| PaCMAP | 24.8 s | 0.093 | 0.946 | 0.307 | 0.614 | **0.976** |
| TriMap | 57.0 s | 0.088 | 0.933 | 0.184 | 0.597 | 0.964 |
| PHATE 2.0 | 43.2 s | 0.080 | 0.865 | 0.291 | 0.611 | 0.861 |
| h-NNE | 9.1 s (5.9 s warm) | 0.233 | 0.940 | 0.261 | 0.601 | 0.949 |
| sklearn t-SNE (Barnes-Hut) | 195.7 s | **0.350** | **0.968** | 0.338 | 0.622 | 0.967 |
| t-SNE-CUDA 3.0.1 (GPU) | 13.8 s | 0.169 | 0.967 | 0.169 | 0.567 | 0.957 |

h-NNE is the one fast method that beats the default MAGNET on neighbourhood preservation (0.233
against 0.193 on MNIST, 0.154 against 0.134 on 20 newsgroups, the other way round on Fashion-MNIST,
0.173 against 0.202); it pays for it on every other column (trustworthiness 0.940, distance
correlation 0.261, label accuracy 0.949). `kernel_b=0.5` closes a third of that gap and
`loss="tsne"` all of it.

**Other data sets** (MAGNET default / best competitor on that column)

| dataset | time | 15-NN pres. | trust. | dist. Spearman |
|---|---|---|---|---|
| Fashion-MNIST 70k | **4.4 s** / 15.0 s UMAP | 0.202 / 0.327 sklearn t-SNE | 0.980 / 0.983 openTSNE | **0.730** / 0.672 TriMap |
| 20 newsgroups SVD-300, cosine | **1.3 s** / 5.9 s UMAP | 0.134 / 0.192 sklearn t-SNE | 0.760 / 0.772 openTSNE | 0.059 / 0.232 sklearn t-SNE |
| 20 newsgroups TF-IDF, sparse 18.8k x 24k | **5.7 s** / 22.2 s UMAP | 0.145 / 0.140 UMAP | 0.730 / 0.716 UMAP | 0.144 / 0.145 UMAP |
| 15 Gaussian blobs 10k x 200 | **1.1 s** / 2.3 s UMAP | 0.067 / 0.114 sklearn t-SNE | 0.972 / 0.975 sklearn t-SNE | **0.690** / 0.666 PaCMAP |
| SBM graph, 30k nodes, 782k edges | **4.9 s** / 36 s UMAP | | | 10-NN acc. 1.00 / 0.26 |

**Scaling** (50 anisotropic clusters in 50 dimensions)

| n | MAGNET | UMAP | peak RSS |
|---|---|---|---|
| 100k | 7 s | 37 s | 0.4 / 2.1 GB |
| 250k | 16 s | 55 s | 0.8 / 3.0 GB |
| 500k | 30 s | 85 s | 1.6 / 5.1 GB |
| 1M | 55 s | | 3.1 GB |

`loss="tsne"` on the same data: 17 s at 250k and 56 s at 1M points (600 iterations; at 1M the
attraction pass over the 40M edges and the position update dominate, the FFT repulsion step costs
0.1 s). The scatter, spectral products and gather of that step run in C++; the transforms use
scipy's pocketfft, which was never the bottleneck.

`transform()` on MNIST (fit 60k, transform 10k): 0.6 s and 10-NN accuracy 0.962 against 14.8 s and
0.956 for UMAP.

Regenerate everything with `benchmarks/full_table.py`, `benchmarks/render_table.py`, `benchmarks/plot_tradeoff.py`,
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
| `min_dist`, `spread` | 0.05, 1.0 | shape of the low-dimensional kernel (UMAP semantics; 0.05 beats UMAP's 0.1 on every metric here) |
| `late_exaggeration`, `late_phase` | 1.0, 0.2 | attraction multiplier over the last epochs; `1.5` is the "compact clusters" preset |
| `loss` | `"umap"` | `"umap"` (sampled repulsion), `"tsne"` (exact), `"infonce"` (contrastive softmax) |
| `exaggeration`, `repulsion_strength` | 1, 2 | attraction / repulsion multipliers |
| `n_epochs` | auto | SGD epochs (500 / 300 / 200 by size) or t-SNE iterations (450); also `(constant, anneal)` or `"auto"` |
| `samples_per_node` | 5 | attractive samples per node and per epoch |
| `negative_sample_rate` | 5 | negatives per positive sample |
| `mid_source`, `n_mid_pairs`, `mid_jump` | `"auto"`, 5, 0.5 | mid-range partners from the data (PaCMAP-style pairs), from random walks on the graph, or `"hybrid"` walks on both layers |
| `walk_len`, `mid_weight`, `mid_rate`, `mid_phase` | 4, 0.1, 0.25, 0.5 | strength and schedule of the mid-range attraction (`mid_weight=0` disables) |
| `init` | `"auto"` | PCA for `loss="tsne"` on data, spectral otherwise; or `"spectral"`, `"pca"`, `"random"`, an array |
| `knn` | `"nndescent"` | or `"hnsw"` (needs hnswlib) |
| `pca` | None | project wide data to this many components before the neighbour search |
| `kernel_b` | 0.6 | attraction exponent (0.5 for images, 0.7 for text, any dimension); None = UMAP's fit; `"auto_dim"` = the former coupled defaults; `"local"` = 1 where the local dimension fits in `n_components`, 1/2 elsewhere (mixed-dimension data); `"adaptive"` = the same split read off the layout at 3/4 of the run (graphs too); an array = one exponent per point |
| `adaptive` | `(0.5, 1.0, 0.75, 0.0, 2.0, 2)` | for `kernel_b="adaptive"`: base and high exponents, switch epoch fraction, crowding threshold, far-edge reference (times the median edge length), minimum degree |
| `repulsion_b` | 1.0 | repulsion exponent (Cauchy tail); None couples it to `kernel_b` (UMAP's compact look, 20% slower) |
| `kernel` | `"power"` | or `"gaussian"`, `"sech"` (exponential tails, for comparison) |
| `tsne_dof` | 1.0 | Student-t degrees of freedom of `loss="tsne"` (< 1 = heavier tail, Kobak et al. 2019) |
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
Random walks on a two-layer graph (kNN edges plus mid-near edges, `mid_source="hybrid"`): one
mechanism for every scale on paper, but each walk step after a mid-range jump diffuses back into
local structure and dilutes the inter-cluster signal (distance correlation 0.39 against 0.42 on
MNIST, 0.68 against 0.71 on Fashion-MNIST, local unchanged). A per-node diffusion scale for the
mid-range walks (`mid_scale="spread"` or `"rolewalk"`) changes nothing measurable on MNIST: the mid-range term itself matters (distance correlation 0.35 against
0.29 without it) but not its per-node length. For graph inputs, mid-near pairs measured in an 8-d
spectral embedding (`mid_source="spectral"`) lose to the random walks (0.31 against 0.34 on the kNN
graph of MNIST evaluated against the hidden data): walks stay the default for graphs. What did make the cut: `repulsion_strength=2` (+8%
neighbourhood preservation, nothing lost), the centroid placement of disconnected components, PCA
initialisation for the exact-repulsion mode, PaCMAP-style mid-near pairs in place of random walks for
data inputs (distance correlation +30% on MNIST, +25% on Fashion-MNIST, for 3% of 15-NN preservation),
and for graphs the placement of disconnected components by RoleWalk signature
(`component_placement="rolewalk"`: structurally similar components end up next to each other).

## History

MAGNET started in 2019 as a random-walk embedding with a full-batch optimiser; v3 (2026) is a
complete rewrite around per-edge SGD, a native neighbour search and the exact-repulsion mode, and
the split of the kernel exponent into an attraction exponent (the L1 point, 1/2) and a repulsion
exponent (the Cauchy tail).
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
