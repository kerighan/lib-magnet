# MAGNET

**Fast dimensionality reduction and graph layout, built on attraction and repulsion.**

MAGNET turns a data matrix, a sparse matrix or a graph into a 2-D (or 3-D) layout the way UMAP
and t-SNE do: a neighbour graph pulls close points together, everything else pushes apart. It
rebuilds that pipeline from scratch in C++ and makes the one choice that matters explicit: how the
repulsion is estimated.

**In short.** Local fidelity and global arrangement pull apart, so MAGNET does them in two passes with two
objectives: a sampled SGD pass with a decoupled, heavy-tailed attraction places the clusters, a full-batch FFT
pass with t-SNE's objective refines the neighbourhoods, and an optional label-free post-process moves the
communities of the graph to raise the agreement of pairwise distances (`contraction=0.7`). MNIST 70k: the
full-batch pass reaches Barnes-Hut t-SNE's neighbourhood preservation (0.349 against 0.350) in 4.9 s against 175
(interleaved timings on a quiet 20-thread machine, `benchmarks/timing.py`: sampled pass 3.6 s, sequence 7.3, with
the post-process 9.0; UMAP 8.5, PaCMAP 31, openTSNE 93);
the sequence has the highest 15-NN and the highest pairwise Spearman of the neighbour embeddings tested
(0.353 / 0.446 over five seeds); with the post-process given to every method of a twelve-method panel on five
seeds, the sequence keeps the best composite balance of local, global and cluster structure on three data sets
of four (on MNIST the post-processed t-SNE layouts converge). Two papers: the exhaustive document
(`paper/main.tex`, every experiment and negative result) and the short one (`paper/human/main.tex`).

![The complete pipeline next to UMAP and Barnes-Hut t-SNE](https://raw.githubusercontent.com/kerighan/lib-magnet/master/benchmarks/plots/pipeline_mnist_fmnist.png)

* `loss="umap"` (default): negative sampling, lock-free multithreaded SGD. **MNIST 70k in 4.1 s**
  against 11 s for UMAP 0.5.12, 15-NN preservation 0.19 against 0.12 and a distance correlation of 0.41
  against 0.30: the highest pairwise distance Spearman among the unstaged neighbour embeddings benchmarked (PCA stays ahead).
* `loss="tsne"`: repulsion over all pairs through FFT interpolation ("exact" below is shorthand for all pairs, not for an exact gradient: the interpolation is accurate to its grid). **MNIST 70k in 7 s**
  against 102 s for openTSNE and 196 s for scikit-learn's t-SNE, with the neighbourhood
  preservation of the second (0.349 against 0.350; openTSNE 0.347), three times UMAP's in less time.
* 1M points in 44 s and 3.3 GB. Native NN-descent (1.2 s for 70k x 784), spectral initialisation by
  Chebyshev-filtered subspace iteration, `transform()` for new points, sparse inputs, networkx graphs.

![MNIST 70k: UMAP, MAGNET, MAGNET with exact repulsion](https://raw.githubusercontent.com/kerighan/lib-magnet/master/benchmarks/plots/compare_mnist70000_tsne.png)

![quality against time](https://raw.githubusercontent.com/kerighan/lib-magnet/master/benchmarks/plots/tradeoff.png)

## Install

```bash
pip install git+https://github.com/kerighan/lib-magnet    # or, from a clone: pip install .
python -m pytest tests -q
```

A C++17 compiler is needed (gcc, clang or MSVC); the dependencies are numpy, scipy, scikit-learn and tqdm.
Optional extras: `[hnsw]` (hnswlib, an alternative kNN backend), `[graphs]` (networkx inputs), `[leiden]`
(python-igraph, the Leiden backend of the post-process; the default Louvain is built in), `[clustering]`
(hdbscan, for `kernel_b="consistent"` on old scikit-learn), `[test]`. One research option uses the author's
unpublished dimprint library when it is installed: `kernel_b="local"` (a biased MLE fallback is used without it). `examples/quickstart.py` runs
the three modes and the post-process on scikit-learn's digits in a few seconds.

## Quickstart

```python
from magnet import MAGNET

Y = MAGNET().fit_transform(X)                        # X: (n, d) array; row i of Y is X[i]
Y = MAGNET(loss="tsne").fit_transform(X)             # exact repulsion, t-SNE-grade local structure
Y = MAGNET(loss="hybrid").fit_transform(X)           # SGD layout refined by the exact repulsion: both at once (2-D)
Y = MAGNET(loss="tsne", contraction=0.7).fit_transform(X)   # + community placement and separation: all three axes (needs igraph)
Y = MAGNET(metric="cosine").fit_transform(T)         # sparse CSR matrix (TF-IDF), no densification
Y = MAGNET().fit_transform(G)                        # networkx graph, rows follow list(G.nodes)
Y = MAGNET().fit_transform(A)                        # scipy sparse adjacency (weights = similarities)

m = MAGNET().fit(X)
Y_new = m.transform(X_new)                           # out-of-sample points, training layout fixed
m.embedding_, m.graph_, m.knn_indices_, m.knn_dists_

# scikit-learn estimator: clone / get_params / set_params, pipelines, grid searches
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Y = make_pipeline(StandardScaler(), MAGNET(loss="hybrid")).fit_transform(X)

# the community post-process on any layout (here a UMAP one), with MAGNET's graph
from magnet import hierarchical_contraction
Z, log = hierarchical_contraction(Y_umap, MAGNET().fit(X).graph_, X=X)
```

Invalid parameters raise a `ValueError` that names the accepted values. The fit is reproducible for a fixed
`random_state` with `n_threads=1`; with more threads the lock-free SGD gives slightly different layouts per run.

## Two regimes, one knob

t-SNE, UMAP and LargeVis optimise the same kind of objective and differ mostly in the balance
between attraction and repulsion (Boehm, Berens & Kobak 2022); the negative-sampling estimator of
UMAP is biased towards weak repulsion, which is what makes its clusters compact (Damrich &
Hamprecht 2021). MAGNET exposes this directly:

| | repulsion | look | when |
|---|---|---|---|
| `loss="umap"` | sampled (5 random points per positive sample) | UMAP-like, with the global structure of PaCMAP | speed, 3-D, graphs, millions of points |
| `loss="tsne"` | exact, all pairs, FFT-accelerated (2-D) | t-SNE-like, finest local structure | best neighbourhood preservation |
| `loss="hybrid"` | sampled, then exact for 300 iterations at the exact mode's scale | the global structure of the first with the local structure of the second | the best 2-D layout MAGNET makes (MNIST 70k: 15-NN kept 0.353, distance Spearman 0.444, 12.5 s) |
| `loss="tsne", exaggeration=4` | exact | UMAP-like again | the two regimes really are one spectrum |

`repulsion_strength` (default 2) and `exaggeration` move along the spectrum within each mode, and
`mid_weight` trades local for global structure within the SGD mode (MNIST 70k, 0 / 0.1 / 0.3 / 1.0:
15-NN kept 0.206 / 0.193 / 0.187 / 0.168, distance Spearman 0.285 / 0.413 / 0.427 / 0.478). Under
`loss="hybrid"` the SGD pass only has to place the clusters, so it runs at `mid_weight=0.5`: its
local cost is erased by the exact pass, its global gain is kept (+0.03 to +0.04 Spearman). The two
modes' strengths do not blend inside one loss (a constant far-field repulsion, a bounded mid-range
term or a longer schedule in the exact mode each cost local structure before buying any global, and
the exaggeration knob trades local for compactness, not for global; `docs/kernel_exponents.md`,
section 6), but they compose in sequence: the global arrangement is set by the SGD pass and carried
through the init, the exact pass only sharpens the neighbourhoods, provided it starts at its own
scale and without the mid-range spring.

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
side does not move. Decoupled, the defaults are `kernel_b=0.6` at n >= 20 000 points (0.5 for images, 0.7 for
text; 0.8 at n <= 10 000 and log-linear between, `kernel_b="size"`: the heavy tail tears small data sets into
micro-clusters, see "The SGD mode at small n" below) and `repulsion_b=1` in every dimension; `repulsion_b=None` gives back the coupled kernel and UMAP's
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

Measured on a 10-core (20-thread) i9-9900X, several seeds, defaults everywhere. Local structure: 15-NN
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
| **MAGNET** | **4.1 s** | 0.194 | 0.956 | **0.414** | **0.644** | 0.966 |
| MAGNET, `kernel_b=0.5` (images) | 4.6 s | 0.215 | 0.958 | 0.424 | 0.648 | 0.964 |
| MAGNET, `repulsion_b=None` (coupled kernel, UMAP's compact look: silhouette 0.51 against 0.52 for UMAP and 0.40 for the default) | 5.5 s | 0.190 | 0.958 | 0.405 | 0.641 | 0.966 |
| MAGNET, `late_exaggeration=1.5` (compact preset, silhouette 0.51) | 4.7 s | 0.181 | 0.956 | 0.386 | 0.637 | 0.967 |
| **MAGNET `loss="tsne"`** | 7.0 s | 0.349 | 0.966 | 0.349 | 0.627 | 0.967 |
| **MAGNET `loss="hybrid"`** | 8.6 s | **0.355** | 0.960 | **0.445** | **0.652** | 0.962 |
| UMAP 0.5.12 | 11.2 s | 0.117 | 0.951 | 0.297 | 0.606 | 0.967 |
| densMAP (umap-learn 0.5.12) | 37.2 s | 0.068 | 0.921 | 0.364 | 0.623 | 0.945 |
| openTSNE 1.0.4 | 102.5 s | 0.347 | **0.968** | 0.330 | 0.619 | 0.965 |
| PaCMAP | 24.8 s | 0.093 | 0.946 | 0.307 | 0.614 | **0.976** |
| TriMap | 57.0 s | 0.088 | 0.933 | 0.184 | 0.597 | 0.964 |
| PHATE 2.0 | 43.2 s | 0.080 | 0.865 | 0.291 | 0.611 | 0.861 |
| h-NNE | 9.1 s (5.9 s warm) | 0.233 | 0.940 | 0.261 | 0.601 | 0.949 |
| sklearn t-SNE (Barnes-Hut) | 195.7 s | **0.350** | **0.968** | 0.338 | 0.622 | 0.967 |
| t-SNE-CUDA 3.0.1 (GPU) | 13.8 s | 0.169 | 0.967 | 0.169 | 0.567 | 0.957 |

`loss="hybrid"` is the first row of this table to lead both the local and the global columns at once:
15-NN kept equal to the best t-SNE (0.355 against 0.350), distance correlation above the SGD mode
(0.444 against 0.413), 16 times faster than scikit-learn's t-SNE. Fashion-MNIST: 0.328 / 0.752
(t-SNE 0.327 / 0.634, SGD 0.202 / 0.730); 20 newsgroups 0.181 / 0.073 (t-SNE 0.192 / 0.232, which
keeps the global lead there; the global column of that set moves by 0.02 between seeds); blobs
0.124 / 0.684 (t-SNE 0.114 / 0.381, SGD 0.067 / 0.690). The exact pass refines the SGD layout, it
does not repair one: the SGD pass must run its full schedule (100 epochs: 15-NN 0.276), and the knobs
of the exact pass (scale, learning rate, iterations, dof, exaggeration) buy at most 0.005 of 15-NN and 0.015 of Spearman for 5-6 s more, or hurt (`docs/kernel_exponents.md`, section 6.1). The time of the hybrid is the exact pass's
repulsion (the FFT grid is set by the layout's extent, 162 boxes from the first iteration; the box width
is 0.7 since the re-timing of 2026-11-04, see below: FIt-SNE's unit box costs 0.004 of 15-NN for 15 % less
time). Two presets trade a little local structure for it: `fft_box=1.5` (measured before that change:
MNIST 8.5 s, 15-NN 0.350; Fashion-MNIST 8.1 s, 0.323) and `hybrid_iter=150, tsne_learning_rate=n/6`
(6.4 s, 0.344); together 5.5 s at 0.341 (section 6.2 of the note).
h-NNE is the one fast method that beats the default MAGNET on neighbourhood preservation (0.233
against 0.193 on MNIST, 0.154 against 0.134 on 20 newsgroups, the other way round on Fashion-MNIST,
0.173 against 0.202); it pays for it on every other column (trustworthiness 0.940, distance
correlation 0.261, label accuracy 0.949). `kernel_b=0.5` closes a third of that gap and
`loss="tsne"` all of it.

**Other data sets** (MAGNET default / best competitor on that column)

| dataset | time | 15-NN pres. | trust. | dist. Spearman |
|---|---|---|---|---|
| Fashion-MNIST 70k | **4.4 s** / 15.0 s UMAP | 0.202 / 0.327 sklearn t-SNE | 0.980 / 0.983 openTSNE | **0.730** / 0.672 TriMap |
| 20 newsgroups SVD-300, cosine | **1.3 s** / 2.7 s UMAP | 0.134 / 0.192 sklearn t-SNE | 0.760 / 0.772 openTSNE | 0.059 / 0.232 sklearn t-SNE |
| 20 newsgroups TF-IDF, sparse 18.8k x 24k | **1.6 s** / 4.5 s UMAP | **0.240** / 0.145 UMAP | **0.736** / 0.712 UMAP | 0.121 / 0.143 UMAP |
| 15 Gaussian blobs 10k x 200 | **1.1 s** / 2.4 s UMAP | 0.067 / 0.114 sklearn t-SNE | 0.972 / 0.975 sklearn t-SNE | **0.690** / 0.666 PaCMAP |
| SBM graph, 30k nodes, 782k edges | **5.0 s** / 11.7 s UMAP | | | 10-NN acc. 1.00 / 1.00 |

**Scaling** (50 anisotropic clusters in 50 dimensions)

| n | MAGNET | UMAP | peak RSS |
|---|---|---|---|
| 100k | 6.5 s | 9.5 s | 0.5 / 1.7 GB |
| 250k | 12 s | 23 s | 0.9 / 2.1 GB |
| 500k | 23 s | 62 s | 1.7 / 2.7 GB |
| 1M | 44 s | | 3.3 GB |

`loss="tsne"` on the same data: 17 s at 250k and 56 s at 1M points (600 iterations; at 1M the
attraction pass over the 40M edges and the position update dominate, the FFT repulsion step costs
0.1 s). The whole repulsion step (scatter, FFTs, kernel products, gather) runs in C++ with persistent
buffers (`src/fftconv.cpp`, pocketfft header-only, rows and columns split over MAGNET's thread pool).

`transform()` on MNIST (fit 60k, transform 10k): 0.6 s and 10-NN accuracy 0.962 against 14.8 s and
0.956 for UMAP.

**Sparse inputs.** The sparse NN-descent computes a cosine or euclidean distance between two CSR rows;
a merge join of their sorted indices costs one unpredictable branch per non-zero of either row. The
kernel now scatters the row it is about to compare against many partners (the pivot of a leaf clique
or of the local join, the query of the mid-near pairs) into a dense per-thread buffer once, and every
partner's dot product is a branch-free gather over its own non-zeros: the kNN graph of 20 newsgroups
TF-IDF (18.8k x 24k, 57 non-zeros per row) takes 0.67 s instead of 2.3 s at the same recall (0.86 of
the exact 15-NN), the fit 1.6 s instead of 3.5. Dense projections of the rows (Gaussian 128-512
dimensions, truncated SVD 128-256) followed by the dense NN-descent were tried first and keep 12-37 %
of the true neighbours (61 % at best among 60 candidates): text neighbourhoods hang on a few shared
rare terms that any projection flattens. Buying recall with more internal neighbours (45: 0.89,
60: 0.89) changes nothing on the layout (15-NN kept 0.240 in every case), so the default stays at 30
(`benchmarks/theory/k12_sparse_knn.py`).

**openTSNE's 102 s.** Its defaults on MNIST 70k (`benchmarks/opentsne_audit.py`; perplexity 30, so 90
neighbours, Annoy index, PCA init, FFT gradient, 250 + 500 iterations at learning rate n/12) split into
24 s of affinities, 2 s of init, 12 s of early exaggeration and 76 s of main phase, 0.15 s per
iteration. `n_jobs` 10 or 20 and the pynndescent index change nothing; 125 + 250 iterations give 0.329
of 15-NN kept in 56 s (0.347 at the defaults); uniform k = 15 affinities, MAGNET's graph size, 0.330 in
100 s. The comparison is fair as it stands: MAGNET's full-batch mode runs 600 iterations in 7 s because
its neighbour search is 20x cheaper (15 neighbours by NN-descent, 1.1 s) and its FFT iteration 10x.

**Graph inputs and UMAP.** The SBM line gives UMAP the graph as a sparse precomputed distance matrix
(`metric="precomputed"`, 1 on every edge); an earlier version of this table fed it the adjacency rows as a
feature matrix under the euclidean metric, which is not a graph layout and scored 0.26-0.38 of 10-NN
accuracy. Under the right protocol UMAP reaches 1.00 like MAGNET (`benchmarks/sbm_umap_fair.py`: rows as
features 9.1 s / 0.38; precomputed sparse 11.7 s / 1.00, silhouette 0.85; UMAP's own
`simplicial_set_embedding` on the adjacency 15.6 s / 1.00, silhouette 0.86; MAGNET 5.0 s / 1.00,
silhouette 0.81, edge recall@15 0.019 against 0.014-0.015). The difference on this graph is the time,
not the separation of the blocks.

**Linear reference.** PCA (or truncated SVD) to 2-D, scored with the same metrics and subsamples
(`benchmarks/baselines_linear.py`): distance Spearman 0.488 on MNIST, 0.873 on Fashion-MNIST, 0.319 on
20 newsgroups, 0.588 on the blobs, with 15-NN kept at 0.01-0.03. Every neighbour embedding in the tables
is below PCA on that global metric on the three real data sets and above it only on the blobs; "best
global structure" above means best among neighbour embeddings.

**Where the global score comes from.** On the same MNIST layouts (`benchmarks/global_strata.py`),
the distance Spearman split by class and by tercile of the input distance, and reduced to the 45
distances between class centroids: PCA is ahead on every stratum (same class 0.526, different class
0.452, short / intermediate / long 0.392 / 0.131 / 0.231, centroids 0.814). The hybrid ties PCA on the
intermediate tercile (0.132) and is second on the other pairwise strata (0.501 / 0.412 / 0.371 / 0.158),
but its arrangement of the ten class centroids is the worst among the methods with centroid measurements (0.421; UMAP 0.752, SGD 0.635,
full-batch 0.669). MAGNET's global advantage over the other neighbour embeddings is in pairwise distances
within and between classes, not in the placement of the classes relative to each other. One layout per
method, and the centroid column is a rank correlation over 45 pairs: over 5 seeds the hybrid lands at
0.51 ± 0.08 on it (SGD 0.635 ± 0.002, full-batch 0.66 ± 0.01), and the staged baselines below show that
the refinement pass itself moves the centroids by -0.07 to +0.01. It is the stiff mid spring of the
hybrid's SGD pass (0.5 against 0.1), not the pass, that trades the class-level arrangement for the
pairwise one.

**Neighbourhood scale.** On the six MNIST layouts of the figure, kNN preservation at k = 5 / 15 / 50 / 100
(`benchmarks/extra_metrics.py`): hybrid 0.351 / 0.351 / 0.319 / 0.313, full-batch 0.333 / 0.348 / 0.325 /
0.325, openTSNE 0.375 / 0.348 / 0.327 / 0.329, SGD 0.159 / 0.193 / 0.219 / 0.249, UMAP 0.071 / 0.117 /
0.176 / 0.226, PaCMAP 0.052 / 0.094 / 0.158 / 0.211. The hybrid leads at k = 5-15, openTSNE at k = 50-100.
Continuity (trustworthiness with the two spaces swapped) is 0.944-0.953 for every method at k = 15, the
SGD mode's 0.953 the highest.

**Staged baselines: the pass is generic, the start is MAGNET's.** Is the hybrid's gain the staging (any
global layout, then a full-batch pass) or the SGD layout it starts from? `benchmarks/staged_baselines.py`
applies the same pass (300 full-batch iterations, no early exaggeration, no mid spring, hand-over scale
31 sqrt(n / 70000), MAGNET's 15-NN graph) to PCA-2, a random layout, UMAP, PaCMAP, openTSNE and MAGNET's
SGD layouts of MNIST and Fashion-MNIST 70k. Every neighbour-embedding start lands at 0.353-0.356 of
15-NN on MNIST and 0.327-0.330 on Fashion-MNIST, the full-batch mode's own level, in 8-13 s, and keeps
its own global structure: the distance Spearman moves by -0.02 to +0.03 (UMAP 0.296 -> 0.299, PaCMAP
0.306 -> 0.311, openTSNE 0.329 -> 0.326, MAGNET SGD at mid 0.5 0.459 -> 0.442 on MNIST; on Fashion-MNIST
UMAP and PaCMAP even gain 0.02-0.03). PCA-2 reaches 0.315 only and loses 0.07-0.09 of Spearman, a random
start 0.065: the pass refines, it does not repair. The hybrid's lead on both axes is therefore a general
recipe (a short full-batch pass at the right scale on top of any layout that has neighbourhoods) plus a
specific start: MAGNET's SGD layout with a stiff mid spring has the highest pairwise Spearman among the unstaged
neighbour embeddings (0.459 / 0.774 on MNIST / Fashion-MNIST against UMAP 0.296 / 0.572, PaCMAP 0.306 /
0.582, openTSNE 0.329 / 0.617), and the pass erases the local cost of that stiffness. openTSNE's
optimiser can do the pass as well (MAGNET's layout handed to it at the same scale, 300 iterations without
exaggeration: 0.346 / 0.428 on MNIST), 4-5 times slower. Hand-over scale and iterations of the pass
(`benchmarks/theory/k14_hybrid_centroids.py`): scale 31 costs the SGD layout 0.015 of Spearman, scale 62
costs 0.004 and gains 0.004 of 15-NN for three times the time (the FFT grid follows the layout's extent:
10 / 30 / 121 s per pass at scales 31 / 62 / 124); scale 62 with 150 iterations keeps the whole Spearman
of the SGD layout at 0.349 of 15-NN in 17 s. The mid spring is confirmed fatal inside the pass
(15-NN 0.04-0.24), the learning rate and the kernel's degrees of freedom buy nothing.

**Seeds.** `benchmarks/seeds.py threads=all` (5 seeds, summary by `seeds_summary.py`; the baselines on
an idle machine, the MAGNET rows re-run on 2026-11-04 after the optimisation round and the box-width change
with a training job sharing the machine, so their times are upper bounds: the SGD row, whose code did not
change, went from 3.6 to 4.5 s and UMAP control runs took 9.8-11.8 s against 8.4):
MNIST 15-NN / Spearman MAGNET SGD 0.192 ± 0.002 / 0.416 ± 0.004, full-batch 0.349 ± 0.001 / 0.352 ± 0.003,
hybrid 0.353 ± 0.002 / 0.446 ± 0.014, UMAP 0.117 ± 0.001 / 0.299 ± 0.005, openTSNE 0.347 ± 0.001 /
0.331 ± 0.002, PaCMAP 0.094 ± 0.001 / 0.310 ± 0.006; Fashion-MNIST hybrid 0.329 ± 0.000 / 0.750 ± 0.003,
SGD 0.204 / 0.724, full-batch 0.325 / 0.698, openTSNE 0.323 / 0.620, UMAP 0.133 / 0.579, PaCMAP 0.099 /
0.582 (all ± 0.003 or less). Times on MNIST: SGD 4.5 s, full-batch 6.2 s, hybrid 8.4 s (3.6 / 8.2 / 10.9
idle before the round), UMAP 8.4 s, openTSNE 93 s, PaCMAP 30 s. The one unstable column is the hybrid's
centroid Spearman on MNIST (0.53 ± 0.11).

**Graphs table** (`benchmarks/graph_table.py`, MAGNET against UMAP's `simplicial_set_embedding` on the
adjacency, 3 seeds, edge preservation = share of a node's graph neighbours among its nearest layout
neighbours): Cora 0.509 / 0.469 (0.4 s both), US airports 0.349 / 0.232 (0.1 / 0.9 s), Europe airports
0.328 / 0.271, SBM + chains 0.483 / 0.535 (adaptive exponent 0.573), SBM 30k 0.017 / 0.013 (1.5 / 10.4 s
on the sparse adjacency). UMAP is ahead on the 10-NN label accuracy of the airport graphs (0.62 against
0.48 on the US graph), whose labels are activity levels rather than communities.

**Is the start MAGNET's, or just a stiff mid-near term?** `benchmarks/staged_seeds.py` (5 runs per cell,
MNIST / Fashion-MNIST, same pass for every start): PaCMAP with its mid-near term stiffened by a sweep on
MNIST (`pacmap_sweep.py`, MN_ratio 4) reaches 0.346 / 0.648 of Spearman after the pass; MAGNET's SGD at
spring 0.5 reaches 0.449 / 0.750 (1.0: 0.456 / 0.764), UMAP 0.310 / 0.605, openTSNE 0.327 / 0.617, and
MAGNET's SGD without the spring 0.301 / 0.588. The start matters beyond the attraction's presence, by
0.10 on both data sets; what it pays is the MNIST centroid column (0.47 ± 0.16 at spring 0.5 against 0.63
at 0.1 and 0.66-0.69 for UMAP / PaCMAP). The pass drifts slowly through 3000 iterations (convergence not reached)
(`benchmarks/theory/k15_persistence.py`: Spearman 0.470 at 300 iterations, 0.446 at 3000, 15-NN 0.367,
KL still decreasing); the FFT repulsion is accurate to 3-4 % RMS on the force and 0.1 % on Z at the
default grid (`k16_fft_check.py`). Details in `docs/kernel_exponents.md`, section 6.5.

**Scale, drift and a scaffold** (`docs/kernel_exponents.md`, section 6.6). A virial identity for the full-batch
objective gives the KL-optimal dilation of any start with two FFT evaluations per scale (SGD starts ~90 x
their std on MNIST 70k, UMAP ~120, PCA 10; s* ~ n^0.42), explains why the pass expands every start handed
over at 31 and why 62 keeps more of the start's Spearman; the `loss="hybrid"` global preset is scale 62
with `fft_box=1.5` (0.354 / 0.468 in 14 s against 0.356 / 0.457 in 10 s). Freezing K >= 32 Laplacian modes
during the pass holds the start's Spearman through 1000 iterations at a 15-NN cost of 0.02; a rest-length
stress on the barycentres of 64 label-free cells does the same at no local cost beyond 0.005 and no time,
and with input-derived targets (2-D MDS of the cells' means) lifts the centroid Spearman by 0.06-0.10 for
0.007-0.013 of 15-NN (300 iterations; the setup costs about 10 s). In the SGD stage the same scaffold destroys the spring's pairwise Spearman.

**Two controls** (`benchmarks/theory/k26_backend_control.py`). Given the same affinity matrix, start, scale
and budget, openTSNE's optimiser and MAGNET's pass return nearly identical fidelity scores (within 0.001 on every metric)
(5 starts, MNIST and Fashion-MNIST; nearly identical fidelity scores, not identical coordinates), MAGNET's
in 7 s against 36 s: the pass is the objective and the hand-over, not the implementation. Scale x exaggeration from one start: the hand-over scale is what keeps
the arrangement (tiny init: -0.04 Spearman, -0.02 15-NN), early exaggeration costs 0.006 of 15-NN at the
matched scale. Negative results on the same page of the docs (section 6.7): conditional KL by blocks,
metric budgets, mid-near heat flow in place of the spring.

**Frozen-rule transfer** (`benchmarks/theory/k27_transfer.py`, docs section 6.8). With every setting frozen
(spring 0.5, scale 31 sqrt(n/70000), 300 iterations, PaCMAP's stiff MN_ratio 4), on data never used in
development (covertype 20k / 70k / 250k, letter 20k, RCV1 70k text): the pass equalises 15-NN across starts
and MAGNET's start keeps the highest pairwise Spearman on all five cases, 0.07-0.13 above the stiffened
PaCMAP start, 0.16-0.38 above UMAP, 0.03-0.28 above standalone openTSNE. Caveats: PCA-2 refined by the
same pass wins on letter (nearly linear data), and spring 0.1 transfers at least as well as 0.5 (and much
better on covertype's class centroids): the hybrid's 0.5 is an MNIST choice. At 250k the sequence takes
40 s at 8 threads (UMAP 107 s, PaCMAP 146-265 s).

**Separability, the third axis** (`benchmarks/separability.py`, docs section 6.9). Cluster recovery on the
layout (labels used only to score: silhouette, ARI of k-means with k = number of classes, ARI and noise of
HDBSCAN) reverses the podium of the two fidelity metrics: on MNIST PaCMAP 0.914 / UMAP 0.778 of k-means
ARI, MAGNET SGD 0.748, full-batch 0.802, openTSNE 0.726, and the hybrid last at 0.509 (HDBSCAN finds 2
clusters and discards 77 % of the points). On a composite (each of 15-NN, Spearman, ARI divided by the
panel's best, geometric mean) the all-round row is `loss="tsne"`: 0.848 on MNIST (openTSNE 0.810, hybrid
0.807), 0.816 on Fashion-MNIST (openTSNE 0.824). The hybrid's pass has a knob for this axis,
`hybrid_tail=(100, 2.0)` (100 final iterations at exaggeration 2; it was the "clusters" preset until
`contraction=0.7` replaced it, see below): on MNIST 15-NN
0.305, Spearman 0.432, k-means ARI 0.707, HDBSCAN ARI 0.764 against 0.355 / 0.452 / 0.615 / 0.097 without
it, same time (`benchmarks/theory/k28_exaggeration_tail.py`). Not the default.

**Full-batch mode: parameter search and graph variants** (`benchmarks/fullbatch_search.py`,
`benchmarks/theory/k29_graph_variants.py`, docs 6.10-6.11). On four data sets scored by the three-axis
composite, the default is within 0.002 of the best single-parameter change (29 tried) and on a plateau; the
levers trade axes (iterations: local and global up, ARI down; exaggeration tail: the reverse). The one
consistent combination, `n_epochs=900, hybrid_tail=(100, 1.5)`, gains 0.009 of composite at twice the time
and is the quality preset. Twelve kNN-graph alterations (fuzzy intersection / mix, mutual kNN, shared
nearest neighbours, local scaling, Sinkhorn, hub capping, degree normalisation, uniform weights, t-SNE's
perplexity-30 affinities) in the same optimiser: none beats the fuzzy union; t-SNE's own affinities score
lower than the fuzzy graph on MNIST, Fashion-MNIST and 20 newsgroups.

**Community placement and separation** (`contraction=0.7`; `magnet/contraction.py`, docs 6.12,
`benchmarks/theory/k30`-`k39`). An optional post-process of a finished layout that targets one of the reported
metrics directly. The communities of the fuzzy graph (Louvain in C++, resolution 0.3; 10-11 on MNIST, ARI 0.89-0.92 against the
labels, labels never used in execution) are translated as rigid bodies, largest first and only those above 1 % of
the points, by a coordinate search of the Spearman correlation between the input distances and the layout
distances of a 500-point sample under a mixing guard (the fraction of a 5000-point sample whose nearest sample
point is in another community may not rise above its initial value + 0.002); then every community is
contracted about its centroid by 0.7. A contraction keeps the distance order within a community and can only
change neighbours across its border; k-means recovers the communities more easily by construction, so the ARI
gain measures how well the graph's communities match the labels, not new clusters. MNIST 70k, `loss="tsne"`,
three seeds, Spearman evaluated on a sample the search never saw (`k39`, final post-process: 0.7 box, strength-ordered
Louvain, density-contrast merge): 15-NN 0.349 -> 0.347, Spearman 0.350 -> 0.469 (+- 0.001; hybrid 0.48, PCA-2 0.49),
k-means ARI 0.76 -> 0.914 (+- 0.004; PaCMAP 0.91); triplet accuracy (0.627 -> 0.649) and class-centroid Spearman
(0.66 -> 0.70), which the step does not fit, rise too; composite against the unprocessed panel's bests 0.839 -> 0.982
(+- 0.001). Placement alone gets the Spearman (0.482) but lowers the ARI to 0.69, separation alone gets the ARI (0.920)
at Spearman 0.33: the two parts are complementary. Fashion-MNIST (three seeds, 4 communities after the merge)
0.326 / 0.697 / 0.340 -> 0.325 / 0.679 / 0.441 (+- 0.004), composite 0.820 -> 0.875 (hybrid 0.830 -> 0.886); letter
(classes are not clusters) within 0.01 of 15-NN and Spearman; covertype 20k Spearman 0.386 -> 0.545 at -0.002 of
15-NN, 250k 0.343 -> 0.402 (138 communities, step 6 s after a 25 s layout). The step is method-agnostic: a Barnes-Hut
t-SNE layout receives the same gains (MNIST 0.350 / 0.338 / 0.829 -> 0.347 / 0.463 / 0.917, three seeds), so the rows "+ contraction" in
`composite_table.py` (first three places on the mean of both composites) describe the configurations
tabulated, not a method comparison. 3 s on 70k (2 of Leiden; the layouts themselves take 8 s for MAGNET and
200 s for sklearn's t-SNE), 9 s on covertype 250k with 141 communities; off by default. Its settings were chosen
with the benchmark scores in view and it can open visual gaps along community borders of connected data
(Fashion-MNIST).

**Timing protocol.** Every method runs with its defaults and all cores; UMAP and densMAP run without
`random_state` (setting it makes their layout single-threaded: 49 s instead of 10 s on MNIST), MAGNET
with a seed (its threads do not depend on it). The numba methods are warmed up before the clock on
data of the same dtype and on enough points (6000) to compile the NN-descent path as well: a float64
warm-up on 500 points left the float32 and NN-descent compilations inside the timed call, which had
put UMAP at 18 s on MNIST and 36 s on the SBM graph in earlier versions of this table (10-11 s both).

**Optimisation round of 2026-11-04 (full-batch and contraction).** Profiling the full-batch pass showed the
kernel spectrum recomputed at every iteration (the box width followed the layout's extent, so the cache key
never hit: now the box keeps its nominal width, FIt-SNE's convention), ~0.4 ms of thread creation per
kernel call (now a persistent pool, `src/pool.hpp`, pthread primitives because `std::condition_variable`
pulls a libstdc++ symbol conda lacks), a generic-dimension attraction loop (now a 2-D path), a modulo per
element in the update, three numpy temporaries per iteration (now one fused `tsne_step`), and ~30 MB of
scipy FFT temporaries per iteration (now `FFTConv2D`: in-house pocketfft with persistent buffers, products
in explicit real arithmetic since `std::complex` multiplication calls `__mulsc3`, grid channels
interleaved for the scatter / gather). Repulsion on MNIST's final 140-box grid: 15-18 ms -> 6 ms per
iteration at 8 threads (4.8 at 20), early grid 4.4 -> 2.1 ms. The contraction step's Leiden (python-igraph,
1.9 s with the conversion) is replaced by a C++ Louvain (`src/graph.cpp`, 0.35 s, equal or better label
ARI of the communities on MNIST, seed-dependent like Leiden: a class is split in two in some runs) and the
fuzzy graph's bandwidth search by a C++ per-row binary search (112 -> 16 ms). MNIST 70k on a loaded
machine (load 3-4), 8 threads: `loss="tsne"` 8.9 -> 5.3 s, hybrid 11.3 -> 7.0 s, `loss="tsne",
contraction=0.7` 10.6 -> 6.4 s (step 1.2 s), SGD unchanged at 4.3 s; 20 threads: 4.6 / 6.5 / 5.6 s
(all with the unit box, see next paragraph).

**Re-timing, and the FFT box width (`k40`, 2026-11-04).** Re-running the tables after the round showed the
fixed unit box (FIt-SNE's convention, adopted in the round so that the kernel spectrum could be cached)
had cost 15-NN: 0.126 -> 0.118 on the blobs, 0.001 on MNIST. The earlier grid, whose width followed the
layout's extent, had been an accidental middle ground, and the cause is the interpolation itself: three
Lagrange points on a unit box leave a force error of 3.5 % RMS on MNIST's final layout (1.2 % on the
blobs'), and 15-NN rises monotonically as the box narrows (blobs 0.118 / 0.129 / 0.133 / 0.136 / 0.136 at
widths 1 / 0.8 / 0.7 / 0.6 / 0.5; MNIST 0.346 / 0.348 / 0.350 / 0.349 / 0.350) while the FFT cost grows
as 1 / width^2. Four or five points per box reach the same accuracy at a higher cost (the scatter and
gather touch 16-25 cells per point), dithering the grid's origin does nothing, and the finer grid matters
over the whole post-exaggeration phase (0.7 for its last 200 iterations only: 0.129). Box width 0.7 is
the default since (`fft_box`): MNIST full-batch 0.346 -> 0.349 = Barnes-Hut's 0.350, Fashion 0.3215 ->
0.325, hybrid 0.3525 -> 0.355, for ~15 % more time (the column FFT passes, 4 ms of the 12 per iteration
on the final grid). The box count is rounded up to a multiple of four so that the spectrum survives a
few iterations of growth (120 -> 31 recomputations on MNIST), and the kernel spectra are stored real
(the kernels are even). The benchmark tables above are re-measured with it, on a machine shared with a
training job (the SGD row, unchanged in code, serves as the control: 3.6 -> 4.5 s on the seeds table).

**The SGD mode at small n (`k44`-`k45`, 2026-11-04).** On Maixent's pybench collection the HDBSCAN use case (UMAP's
strongest practical argument) showed the SGD mode tearing small data sets into micro-clusters: USPS 7k 60 HDBSCAN
clusters with 23 % noise, phonemes 4.5k 34, MNIST subsamples 56-78 at n <= 10k against 8 at n >= 20k, while
UMAP gives 6-8 everywhere. The cause is the attraction exponent: the force on a neighbour is ~ d^(2b-1), steep at
short range for b = 0.6, so neighbours glue into tight clumps that only the graph's redundancy knits together,
which small data sets lack; b = 0.8 holds them together (USPS 9 clusters, HDBSCAN ARI 0.79 against UMAP's 0.74 with
15-NN 0.337 against 0.316; MNIST 10k 7 clusters, 0.59-0.63 against UMAP's 0.52-0.61; 3 seeds) at a 15-NN cost of
0.02-0.03, and b = 1.0 collapses MNIST 70k into 3 clusters. Nothing else fixes it (n_epochs, learning rate, n_neighbors, one
thread, the coupled kernel, `"adaptive"`, `"local"`: tested). Default since: `kernel_b="size"`, 0.6 at n >= 20k,
0.8 at n <= 10k, log-linear between; the 70k and 20k tables are unchanged, the blobs (10k) and 20 newsgroups
(18.8k) SGD rows were re-run. The principled version is `kernel_b="consistent"` (section 6.19 of the note): the
heaviest tail whose layout does not tear the graph's own communities into density clusters they do not have
(HDBSCAN on a 10k sample against the post-process's communities, ARI >= 0.7, raised while that improves); it picks
0.7 on USPS, 0.8 on MNIST 5k, 0.6 on MNIST 20k-70k, letter, covertype, blobs and text, and 0.7 on Fashion 70k, where
the labels prefer 0.6 (classes that are not density clusters): the option for clustering pipelines, at one extra
fit per exponent tried. With it, HDBSCAN on the SGD layout beats UMAP at 7k (0.81 against 0.74), 10k and 70k
(MNIST 70k: SGD + post-process 0.92 against 0.66), loses on phonemes 4.5k to PaCMAP (0.63 against 0.80); the
post-process does not help below 10k (it fuses touching classes: USPS 0.51).

**A deterministic Louvain (`k42`, 2026-11-04).** The random visiting order of the Louvain was the one seed-dependent
part of the post-process, with a spectacular failure mode (a class split in two: MNIST ARI after the step 0.83 instead
of 0.92 on one seed in three). Visiting the nodes by decreasing strength (best-connected first), with no RNG, finds the
ten MNIST classes on every graph tried (label ARI 0.924-0.929 against 0.89-0.92 +- 0.01-0.05 for the random order,
min 0.81), is neutral on covertype, slightly better on letter and slightly lower on Fashion (0.464 against 0.484),
at the same cost; downstream, the step's MNIST ARI goes from 0.831-0.919 over the three layouts to 0.910-0.919. A
medoid consensus of 5 random runs does the same at 6x the cost, Leiden is as unstable as the random Louvain, CPM is
unusable on these graphs (27-340 communities), and resolution 0.3 is the knee (0.2 merges classes, 0.5 splits them).
`order="strength"` is the default; `order="random"`, `consensus=k`, `backend="leiden"` stay available.

**Why modularity fails on kNN graphs, and a density-contrast merge (`k43`).** On a kNN graph every node has about
the same strength, so the modularity's null model is nearly a constant and the objective is internal weight minus a
penalty in (community size)^2 (it coincides with CPM at the matched scale, checked). A large geometric community is
therefore split (its internal cuts are thin: the field-of-view limit) and a small one swallowed (the resolution
limit), and no resolution fixes both. What tells an artificial cut from a real boundary is the density across it:
the boundary nodes on each side send a share of their strength across; a cut through a uniform cloud is symmetric
and large on both sides (0.14-0.16 on MNIST's same-class splits), a real boundary is a density dip and at least one
side reaches little (<= 0.09). `interface_merge`: Louvain at resolution 0.5, then merge interfaces while
min(share_AB, share_BA) >= 0.10, which is half the no-contrast reference (the share across the thinnest cut of a
uniform cloud is ~0.2 for this graph, d = 2-20, k = 10-30, simulated; MNIST's own communities 0.17-0.19) while real
class boundaries sit at half of it: the midpoint between no contrast and a full gap (0.08 merges true boundaries,
0.14 misses splits; 0.10-0.12 are identical on MNIST). MNIST: 11-14 communities -> 10 on every graph and resolution tried, label ARI
0.92-0.93; Fashion merges touching classes (13 -> 6); letter and covertype are left alone (size-free). Downstream
the composite is equal or better on the four sets (MNIST 0.981 -> 0.982, Fashion 0.883 -> 0.889, letter 0.851 ->
0.847, covertype 0.832 -> 0.847). It is the default of `contraction` (`merge_tau=0.10`; `None` restores plain
Louvain).

**Where the SGD time goes, and `parallel="owner"`.** On MNIST 70k the default fit is 1.1 s of kNN graph,
0.9 s of graph construction and spectral init, and 2-3 s of SGD. One thread runs the SGD at 47 cycles per
force evaluation, but 10 cores gave only 3.7x and 20 threads 4.3x: with lock-free (hogwild) updates every
sampled edge writes two or three cache lines owned by other cores, every negative sample reads a line
another core has just written, and the slowest thread sets the pace of each epoch (half the wall-clock
time is spent in transfers and waiting). `parallel="owner"` splits the nodes into chunks of about 4096
edges that the threads claim dynamically, writes only the chunk's own nodes and reads every other node
from a snapshot of the layout frozen for the epoch (double-buffered, one barrier per epoch, persistent
threads). The attraction then moves the head of the edge only, scaled by 1.5 (the mirrored edge moves the
other end), and the pull of the mid-range spring on the partner is delivered when the partner is the head
(reverse pairs): without that term the global structure collapses (Spearman 0.31), it is the inter-cluster
spring. Result on MNIST, 20 threads with other jobs running: SGD 1.45 s against 2.98 s, fit 3.4 s against
5.1 s; 15-NN kept 0.189 against 0.193, trustworthiness 0.948 against 0.957, distance Spearman 0.451
against 0.415, triplets 0.656 against 0.645. Same trade on Fashion-MNIST (0.202 / 0.974 / 0.782 against
0.205 / 0.979 / 0.731), 20 newsgroups (0.133 / 0.742 / 0.090 against 0.134 / 0.755 / 0.046) and blobs; SBM
graph 2.9 s against 4.0 s at the same accuracy. The layout is identical for any `n_threads` given the graph
and the seed. The hogwild mode stays the default until the hybrid's `mid_weight` is re-tuned for it (the
owner mode shifts the attraction / mid-range balance; `benchmarks/theory/k11_owner.py`).

Regenerate everything with `benchmarks/full_table.py`, `benchmarks/render_table.py`, `benchmarks/plot_tradeoff.py`,
`benchmarks/compare_plots.py`, `benchmarks/scaling.py` and `benchmarks/graph_demo.py`.

## Graphs

Any networkx graph or sparse adjacency matrix is laid out directly, in linear time: communities of
25 to 3000 nodes in 1.4 s, an 8k-node power-law graph in 1.1 s, a 60 x 60 grid comes out as a sheet.

![graphs](https://raw.githubusercontent.com/kerighan/lib-magnet/master/benchmarks/plots/graph_demo.png)

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
| `parallel` | `"hogwild"` | `"owner"`: threads own node chunks and read the others from a per-epoch snapshot; the SGD runs 2x faster, the layout is deterministic for any `n_threads`, global structure up (MNIST Spearman 0.451 against 0.415), local slightly down (15-NN 0.189 against 0.193, trustworthiness 0.948 against 0.957) |
| `walk_len`, `mid_weight`, `mid_rate`, `mid_phase` | 4, None (0.1; 0.5 under `loss="hybrid"`), 0.25, 0.5 | strength and schedule of the mid-range attraction (`mid_weight=0` disables) |
| `init` | `"auto"` | PCA for `loss="tsne"` on data, spectral otherwise; or `"spectral"`, `"pca"`, `"random"`, an array |
| `knn` | `"nndescent"` | or `"hnsw"` (needs hnswlib) |
| `pca` | None | project wide data to this many components before the neighbour search |
| `kernel_b` | `"size"` | attraction exponent: 0.6 at n >= 20k (0.5 for images, 0.7 for text, any dimension), 0.8 at n <= 10k, log-linear between (`kernel_b_` holds the value used; graph inputs 0.6); None = UMAP's fit; `"auto_dim"` = the former coupled defaults; `"local"` = 1 where the local dimension fits in `n_components`, 1/2 elsewhere (mixed-dimension data); `"adaptive"` = the same split read off the layout at 3/4 of the run (graphs too); an array = one exponent per point |
| `adaptive` | `(0.5, 1.0, 0.75, 0.0, 2.0, 2)` | for `kernel_b="adaptive"`: base and high exponents, switch epoch fraction, crowding threshold, far-edge reference (times the median edge length), minimum degree |
| `repulsion_b` | 1.0 | repulsion exponent (Cauchy tail); None couples it to `kernel_b` (UMAP's compact look, 20% slower) |
| `kernel` | `"power"` | or `"gaussian"`, `"sech"` (exponential tails, for comparison) |
| `hybrid_tail`, `tsne_early_exaggeration` | (0, 1.0), 12 | exaggerated tail (iterations, factor) of the full-batch pass (`n_epochs=900, hybrid_tail=(100, 1.5)` = quality preset; `(100, 2.0)` compacts clusters at a local cost, superseded by `contraction`); early-exaggeration factor of `loss="tsne"` |
| `contraction` | None | community placement and separation of the final layout: a float g = the graph's Louvain communities (strength-ordered, resolution 0.5, interfaces without density contrast merged) are translated as rigid bodies by a label-free ascent of the input-vs-layout distance Spearman (overlap guard), then each contracted towards its centroid by g (0.7 = the "clusters" preset); a dict passes `resolution`, `separation`, `placement`, `n_sample`, `tol`, `sweeps` |
| `tsne_dof` | 1.0 | Student-t degrees of freedom of `loss="tsne"` (< 1 = heavier tail, Kobak et al. 2019) |
| `fft_interp`, `fft_box` | 3, 0.7 | grid of the exact repulsion (FIt-SNE's unit box: -0.004 of 15-NN on MNIST for 15% less time; `fft_box=1.5` is the fast preset: -3% 15-NN in `loss="tsne"`, -1% in `loss="hybrid"` for 30% less time) |
| `mid_scale` | `"fixed"` | or `"spread"`: per-node length of the mid-range walks from the graph's diffusion spread |
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
mid-range walks (`mid_scale="spread"`, or a per-role length from structural-role signatures) changes nothing measurable on MNIST: the mid-range term itself matters (distance correlation 0.35 against
0.29 without it) but not its per-node length. For graph inputs, mid-near pairs measured in an 8-d
spectral embedding (`mid_source="spectral"`) lose to the random walks (0.31 against 0.34 on the kNN
graph of MNIST evaluated against the hidden data): walks stay the default for graphs. What did make the cut: `repulsion_strength=2` (+8%
neighbourhood preservation, nothing lost), the centroid placement of disconnected components, PCA
initialisation for the exact-repulsion mode, PaCMAP-style mid-near pairs in place of random walks for
data inputs (distance correlation +30% on MNIST, +25% on Fashion-MNIST, for 3% of 15-NN preservation).

## History

MAGNET started in 2019 as a random-walk embedding with a full-batch optimiser; v3 (2026) is a
complete rewrite around per-edge SGD, a native neighbour search and the exact-repulsion mode, and
the split of the kernel exponent into an attraction exponent (the L1 point, 1/2) and a repulsion
exponent (the Cauchy tail).

## Cite

```
@software{magnet,
  author = {Chenebaux, Maixent},
  title  = {MAGNET: fast dimensionality reduction and graph layout},
  year   = {2026},
  url    = {https://github.com/kerighan/lib-magnet}
}
```
