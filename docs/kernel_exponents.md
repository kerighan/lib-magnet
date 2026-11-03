# The two kernel exponents: derivation and tests

MAGNET's SGD mode minimises the UMAP loss with the low-dimensional kernel `w(d) = 1 / (1 + a d^(2b))`.
Since commit `1471e5c` the attraction and the repulsion use two different exponents (`kernel_b`, default
0.6, and `repulsion_b`, default 1.0). This note argues why the two roles want different exponents (asymptotic arguments, stated with their assumptions and checked against measurement; the crowding argument gives a direction, not an optimum), says
which parts of the derivation are checked and which are only consistent with the data, and records what
the derivation gave back to the algorithm. Scripts: `benchmarks/theory/`.

## 1. One exponent, two regimes

For an edge (i, j) at embedded distance d the attraction is the gradient of `-log w`:

    F_att(d) = 2 a b d^(2b-1) / (1 + a d^(2b))

and the repulsion from a negative sample is the gradient of `-log(1 - w)`:

    F_rep(d) = 2 b / (d (1 + a d^(2b)))        (times repulsion_strength and the negative rate)

The kernel scale `d_k = a^(-1/2b)` is 0.62 at the default `min_dist` / `spread` (a ~ 1.8). The two forces
depend on b in opposite regimes:

| | near field, d << d_k | far field, d >> d_k |
|---|---|---|
| attraction | `2ab d^(2b-1)`: the exponent sets the shape | `2b / d`: redescending, the same for every b up to the factor 2b |
| repulsion | `2b / d`: the same for every b up to the factor 2b | `2b / (a d^(2b+1))`: the exponent sets the tail |

So a single exponent fixes the attraction's *near* field and the repulsion's *far* field at the same time,
two regimes with nothing in common. This is the structural reason for the split; the rest of the note says
what each regime wants.

## 2. Attraction: the near field is a robust location problem

Where a neighbour settles. A neighbour j of i is pulled with `F_att` and, like any point, pushed by the
negatives with `F_rep` (near field, `2 b_r g / d`, g = repulsion_strength x negative rate). The net force
on the pair is positive (attractive) only beyond `d_eq = (b_r g / (a b))^(1/2b)`, so every neighbour that
can be satisfied sits on a shell at `d_eq` (this is what `min_dist` controls). The stiffness of that
equilibrium is `4 b b_r g / d_eq^2 > 0` for every b: no instability anywhere, the exponent does not act
through stability.

Crowding. When the intrinsic dimension q of the data exceeds the target dimension m, the k nearest
neighbours of i cannot all sit on the shell: a fraction of them ends in the far field, at distances of
several `d_k`, where every kernel pulls with `2b/d`. The embedded distance distribution of the kNN edges at
the defaults (`benchmarks/theory/e2_contamination.py`, 20k points, 2-D) makes the contamination explicit:

| data | local dim (MLE) | 15-NN kept | share of edges beyond d_k | beyond 2 | measured attraction optimum (Cauchy repulsion) |
|---|---|---|---|---|---|
| Gaussian clusters q = 2 | 2.1 | 0.670 | 0.06 | 0.01 | 1.0 |
| q = 3 | 3.1 | 0.511 | 0.14 | 0.01 | 0.5 |
| q = 5 | 5.4 | 0.333 | 0.23 | 0.02 | 0.5 |
| MNIST 20k | 16 | 0.288 | 0.24 | 0.08 | 0.5 |
| Fashion-MNIST 20k | 17 | 0.278 | 0.24 | 0.05 | 0.5-0.6 |
| q = 10 | 10.6 | 0.163 | 0.36 | 0.00 | 0.4 |
| q = 20 | 18.9 | 0.084 | 0.46 | 0.00 | 0.4-0.5 |
| 20 newsgroups | 40 | 0.133 | 0.34 | 0.20 | 0.6-0.7 |
| MNIST 5k | 16 | 0.382 | 0.35 | 0.12 | 0.6 (flat: 0.368 / 0.376 / 0.368 at 0.5 / 0.6 / 0.7) |

Why the L1 point. In the near field the attraction loss is `rho(d) = a d^(2b)`, a location problem: i
goes where the sum of `rho` over its neighbours is smallest. Two classical facts about the family `|d|^p`:
the influence `rho'(d) = p d^(p-1)` is bounded iff p <= 1, and `rho` is convex iff p >= 1 (for p < 1 the
Fermat-Weber minimiser sits on one of the data points: points collapse onto a single neighbour). p = 2b = 1
is the only exponent with both properties, the geometric median. In force terms, the grip on a satisfied
neighbour at `d_n` relative to the drag of an unsatisfiable one at `d_f` is
`a d_n^(2b-1) d_f`: 3.6 at b = 1/2 and 0.36 at b = 1 for `d_n = 0.1, d_f = 2`. With L2 (b = 1) the point
is dragged towards the mean of all its neighbours, satisfiable or not; with L1 every neighbour casts a
vote of the same size and the point stays at the median of the satisfiable ones. Huber's minimax result
for contaminated location estimation says the same thing in statistical language: the optimal loss is
quadratic below a threshold and linear above, and the threshold shrinks to zero as the contamination
grows. Without contamination (q <= m) the quadratic loss is the efficient one, hence b = 1 there.

What this predicts, and what was measured:

1. The optimum is at 1/2 as soon as a sizeable share of neighbours is unsatisfiable, and at 1 when the
   share is small. Measured: the synthetic series goes 1.0 / 0.5 / 0.5 / 0.4 / 0.4 for q = 2 / 3 / 5 / 10
   / 20 in 2-D, and 1.0 / 0.5 / 0.4 for q = 3 / 5 / 10 in 3-D (`e5_law.py`): the switch is a step at
   q = m, not a slope in q. The image sets (contamination 0.24) sit at 0.5. The law in m from the first
   study (0.6 in 2-D, 0.8 in 3-D) was the compromise with the repulsion, not a property of the attraction.
2. Text and small samples are the two points the share of far edges does not order: 20 newsgroups has
   0.34 beyond `d_k` and an optimum of 0.6-0.7. Its contamination is bimodal (0.20 of the edges beyond 2,
   hubs that no kernel can satisfy and that every kernel treats alike), so the contamination that matters,
   the edges between `d_k` and 2 where the kernels differ, is 0.14, the lowest of the real sets. MNIST 5k
   stays unexplained (0.23 in that band, optimum flat around 0.6).
3. Below 1/2 the loss is concave and the force diverges at 0: 0.4 loses to 0.5 on MNIST (0.280 against
   0.296) but wins slightly on the most crowded synthetic sets, where collapse onto a neighbour costs less
   than the drag.

## 3. Repulsion: the far field is a question of which scale carries the loss

With negative sampling the expected repulsive loss on i is `g sum_j -log(1 - w_ij)`, and in the far field
`-log(1 - w) ~ w ~ 1 / (a d^(2b_r))`. For a cloud of density `rho` the shell at distance d holds
`rho d^(m-1) dd` points, so the loss carried by the octave [d, 2d] is

    L(octave) ~ d^(m - 2 b_r)

scale-free at `b_r = m/2`: Cauchy in 2-D, 1.5 in 3-D, 2 in 4-D, 3 in 6-D. Van der Maaten's heuristic for
the Student-t degrees of freedom (`nu = m - 1`, tail `d^-(nu+1)`) is the same number. Below m/2 the
far-field loss is dominated by the largest scale and the optimiser spends the repulsion budget inflating
the cloud (in the coupled sweep the radius in units of the 1-NN distance is 2^11.7 at b = 0.4 and 2^9.1 at
b = 1), the clusters become compact and the local structure is squeezed; above m/2 the repulsion is local
and the global arrangement is free, the t-SNE regime. Measured optima (fidelity): 1.0 in 2-D, 1.0-1.5 in
3-D, 2-4 in 4-D, 3-4 in 6-D, the law m/2 within the flatness of the optimum.

Where the law applies, measured on fixed layouts (`k13_tail_crossover.py`, 256 anchors against all
points, every exponent evaluated on the same coordinates, octaves of d / d_kr with d_kr = a^(-1/2b_r)
the crossover of the kernel, fit window a d^2b_r > 100 and d < R/2): at MAGNET's working scale no octave
of a MNIST layout (2-D or 3-D) is in the window, so the asymptotic regime is simply not reached on the
layout itself. Binned by the 1-NN distance, as an earlier version of this analysis did, the fitted
octaves sit below the crossover, where the force is 2 b_r / d whatever b_r, and the slope comes out at
+0.1 to +0.2 for every exponent, which looked like a failure of the law and was a failure of the window.
Rescaled by 16-64 (equivalent to shrinking a) the window opens to 4-5 octaves and the slope follows
alpha - 1 - 2 b_r within 0.05, alpha being the occupancy exponent of the layout in the window (d log N /
d log d): 1.6-1.7 for MNIST in 2-D and in 3-D alike (a clustered layout does not fill its space at those
scales), 1.8 for a 2-D Gaussian cloud and 2.7 for a 3-D one (finite cloud). So the power counting is
right, m must be read as alpha < m, and at the scale the optimiser works at there is no asymptotic
window for the law to act in: it cannot fix the optimum by itself, consistent with the measured optima
sitting at or above alpha/2 and m/2. Log: `k13_tail_crossover.log`; figure `paper/figures/tail_crossover.png`.

What is checked and what is not. The per-octave *force* was measured on fixed clouds with the pure power
law (`e1b_powerlaw.py`, 40k points, slopes of log2(force per octave) against the octave): it scales as
`d^(m - 1 - 2 b_r)` (2-D, b_r = 0.5 / 1 / 1.5 / 2: measured +0.07 / -0.97 / -2.01 / -3.06 on a uniform
square, predicted 0 / -1 / -2 / -3), i.e. the loss law times the lever arm d, as it should. In MAGNET's
own regime the far field spans only three octaves (radius ~10, `d_k` 0.6) and the embedding is clustered,
so the exponent of a smooth cloud is a trend, not a prediction of the exact optimum; this is why the gain
of heavier-than-Cauchy tails in 4-D and 6-D is small and the default stays 1.0 in every dimension.

The coupled optimum (0.52 / 0.59 / 0.76 / 1.2 / 1.6 / 1.9 / 2.3 for m = 1..8, roughly 0.3 m) is the
compromise between an attraction that wants 1/2 everywhere and a repulsion that tolerates m/2.

## 4. What the derivation gives back to the algorithm

The split itself (section 1) is the first consequence: 4.3 s instead of 5.3 s on MNIST 70k, 15-NN
preservation 0.192 instead of 0.189, and a `pow`-free repulsion loop.

Per-node attraction exponent. Section 2 says the exponent is a property of each point's crowding, so
data that mix manifolds of different dimensions should gain from `b_i = 1` on the uncrowded points and
`1/2` on the others, with the repulsion left at Cauchy (the former `kernel_b="local"` mode coupled the two,
which is why it never beat a constant). 15-NN preservation on 20k points, 2 seeds, Cauchy repulsion
(`e4_oracle_local.py`, `e6_estimator.py`):

| data | best global | oracle per-node (true q) | local MLE dimension, threshold m + 0.5 | threshold 4 (m = 2) |
|---|---|---|---|---|
| 5 clusters q = 2 + 5 clusters q = 10, 2-D | 0.422 (b = 0.5) | 0.449-0.464 | 0.430-0.433 | 0.447-0.452 |
| 5 x q = 3 + 5 x q = 10, 3-D | 0.402 (0.6) | 0.452-0.454 | 0.407-0.408 | |
| q = 2, 3, 5, 10 mixed, 2-D | 0.462 (0.5) | 0.472-0.489 | 0.463-0.464 | |

The oracle gains 7-12 %: the q = 2 points get what b = 1 gives them (0.69-0.73 against 0.64 at b = 0.5)
while the q = 10 points keep what b = 0.5 gives them (0.20 against 0.07 at b = 1). The estimator is the
weak link: the Levina-Bickel MLE on 15 neighbours reads 2.6 for q = 2, 3.8 for q = 3 and 10.2 for q = 10,
and the step of the law sits exactly at q = m, so a threshold at m + 0.5 keeps only 39 % of the q = 2
points and a looser one would misfile the q = 3 points (which lose 0.16 under b = 1). Homogeneous real
data have nothing to gain: on MNIST, Fashion-MNIST and 20 newsgroups fewer than 1 % of the points have a
local dimension below 6.

Huber attraction. Section 2 also says a loss that is quadratic below a scale c and linear above should
match the power kernels when c is set right. Implemented as a research kernel (`e3_huber.py`,
`w = 1 / (1 + a h_c(d))`, `h_c(d) = c^2 (sqrt(1 + d^2/c^2) - 1)`, L2 below c, L1 above, the usual
redescending far field; research build only): on MNIST 20k the best c is the smallest tested (0.02:
0.299 / 0.299 against 0.297 / 0.294 for b = 1/2) and quality falls monotonically with c (0.288 at 0.1,
0.236 at 0.5); on the q = 2 clusters the best c is 0.2-0.5 (0.707-0.719, the level of b = 1: 0.706-0.712)
and c = 0.02 is at 0.656; on q = 10, c = 0.02 equals b = 1/2 (0.197-0.200). There is no universal
transition scale, which is what Huber's theorem says (the threshold depends on the contamination), so the
kernel was not kept: the per-point exponent is the right form of the same idea.

Reading the crowding off the layout (`kernel_b="adaptive"`). The contamination of section 2 can be
measured on the layout itself instead of estimated from the input distances. Measured on a converged
layout (share of a point's graph edges longer than `d_k`, `g1_gate.py`), it separates the regimes on
real data where the theory predicts them:

| data | n | uncrowded points (share <= 0.05) | median crowding | note |
|---|---|---|---|---|
| COIL-20 (64 x 64, 20 objects x 72 views) | 1440 | 0.31 | 0.13 | objects 11 and 12 at 0.00; local dimension (dimprint) reads 2.3-8 for these 1-D circles |
| Paul15 single-cell, 50 / 20 / 10 PCs | 2730 | 0.01 / 0.01 / 0.02 | 0.61 / 0.52 / 0.47 | crowded everywhere |
| Cora citation graph | 2708 | 0.58 | 0.00 | low-degree nodes |
| US airports graph | 1190 | 0.27 | 0.50 | small airports at 0.00, hubs at 0.84 |
| Europe airports graph | 399 | 0.10 | 0.83 | |
| SBM + chains (synthetic) | 3000 | 0.46 | 0.30 | chains 0.00, blocks 0.48 |
| MNIST 6k | 6000 | 0.06 | 0.33 | digits 0, 1, 6 at 0.22, digits 3, 8, 9 at 0.39 |
| 1-D / 2-D / 10-D clusters | 6000 | 0.25 | 0.32 | 0.00 / 0.12 / 0.50 by group |

Mid-run the layout is still contracting (median edge length 1.8 at 40 % of the epochs against 0.1-0.5 at
the end), so the absolute reference `d_k` flags everything; the share of edges longer than twice the
median edge length separates the groups as well as the final measure does (1-D / 2-D / 10-D clusters at
60 % of the epochs: 71 % / 66 % / 2 % of the points with no far edge). The implementation runs every
point at the crowded exponent, measures that share at 3/4 of the epochs, and switches the points with
no far edge and at least two neighbours to L2 for the rest of the run, with no restart of the schedule
(`g2_adaptive.py`, `g3_mindeg.py`; 2-3 seeds, 15-NN kept for data, edge preservation for graphs):

| data | constant 0.5 | constant 0.6 | constant 1.0 | oracle | local (dimprint) | adaptive, base 0.5 | adaptive, base 0.6 |
|---|---|---|---|---|---|---|---|
| 1-D + 2-D + 10-D clusters, 6k | 0.493 | 0.498 | 0.474 | 0.540 | **0.545** | 0.526 | 0.524 |
| 5 x 2-D + 5 x 10-D clusters, 20k | 0.421 | 0.417 | 0.392 | 0.441 | **0.459** | 0.442 | 0.433 |
| COIL-20 | 0.694 | 0.717 | **0.772** | | 0.701 | 0.720 | 0.729 |
| MNIST 20k | 0.292 | 0.282 | 0.200 | | 0.294 | 0.291 | 0.280 |
| Paul15, 20 PCs | 0.324 | 0.324 | 0.275 | | 0.321 | 0.325 | 0.326 |
| Cora graph | 0.442 | 0.507 | 0.470 | | | 0.501 | **0.522** |
| US airports graph | 0.333 | 0.344 | 0.297 | | | 0.344 | 0.339 |
| SBM + chains | 0.449 | 0.484 | 0.532 | 0.571 | | **0.573** | 0.568 |

The adaptive mode stays within 0.005 of its base exponent on homogeneous data (airports, base 0.6: 0.344 to 0.339) and better where the data is
heterogeneous; the local-dimension mode is the better of the two when the low-dimensional regions are
flat enough for the estimator (the synthetic mixtures, where it beats the 1.0 / 0.5 oracle) and useless
when they are not (COIL-20) or when there is no data matrix (graphs). The minimum degree matters on
graphs: chains (degree 2) gain 0.25 of edge preservation under L2, leaves (degree 1) lose it. What
remains between the adaptive mode and the oracle on the mixtures (0.526 against 0.540, 0.442 against
0.441) is the share of uncrowded points the mid-run measure misses (60-70 %); COIL-20 shows the other
limit, a data set where every point would rather be at L2 and the mode only moves a third of them.

## 5. Open

* Why text settles at 0.6-0.7 is still open, and two explanations are now ruled out (`h1_hubs.py`,
  `h2_degree.py`, 20 newsgroups with MNIST 20k as the control, 2 seeds). Hubs: the edges that end far
  from their point do not go to hubs, their endpoints have 0.77-0.81 of the mean in-degree on both data
  sets, and damping the hubs on the graph side (fuzzy intersection, degree cap at 30, symmetric degree
  normalisation) leaves the text optimum at 0.6-0.7 and MNIST's at 0.5. Degree: the in-degree of the
  kNN graph is extreme on text (69 % of the points are listed by fewer than 5 others, 20 % by none,
  against 12 % and 0 % on MNIST), but the optimum is 0.6-0.7 in every in-degree quintile on text and 0.5
  in every quintile on MNIST, so it is a property of the whole data set, not of the poorly connected
  points. Two side results worth keeping: the fuzzy intersection raises 15-NN preservation on text from
  0.135 to 0.163 at the cost of trustworthiness (0.76 to 0.71) and label accuracy (0.51 to 0.44), and a
  per-point exponent decreasing with the in-degree (0.8 below 5, 0.6 below 15, 0.5 above) gains 2 % on
  text (0.132 against 0.129), all of it on the well-connected points (0.355 against 0.335 in the top
  quintile), and is neutral on MNIST: a point's exponent acts on its neighbours' pulls as much as on
  its own (the edge uses the mean of the two exponents). MNIST 5k (0.6, flat) is not explained either.
* The repulsion law is a loss-per-octave argument for a smooth cloud; what fixes the optimum in a
  clustered embedding with three far-field octaves is not derived, and the measured optimum is at or a
  little above m/2 in every dimension tested, never below.
* The adaptive switch is a step at 3/4 of the run; a second measurement later, or a graded exponent
  between 1/2 and 1 by the crowding share, could take more of the oracle gain. COIL-20 says the
  crowding share should also be allowed to push a whole data set to L2.
* A per-point *repulsion* exponent does not follow: the layout is m-dimensional everywhere.

## 6. Two regimes in one layout (`loss="hybrid"`)

The sampled optimiser gives the best global structure among the neighbour embeddings of the benchmark
(MNIST 70k distance Spearman 0.41; PCA to 2-D reaches 0.488 on that metric, 0.873 on Fashion-MNIST,
`benchmarks/baselines_linear.py`; split by class, by tercile of the input distance or reduced to the 45
inter-centroid distances, PCA stays ahead on every stratum and the hybrid ties it on intermediate distances
only, `benchmarks/global_strata.py`; the hybrid layout of that table also placed the ten class centroids
worst of the panel, 0.421 against UMAP 0.752, which section 6.4 traces to the run rather than to the
pass) and the exact one the best local structure (15-NN kept 0.35); each is mediocre on the other
axis (0.19 and 0.34). Can one loss fade from one regime to the other? What was tried (`k1`-`k7`, MNIST
70k unless said otherwise):

* Where the global structure comes from. In the SGD mode it is entirely the mid-range term: without it
  the distance Spearman is 0.285, below t-SNE's 0.342; 0.1 / 0.3 / 1.0 give 0.413 / 0.427 / 0.478 at
  15-NN 0.193 / 0.187 / 0.168. That weight is already a continuous local-global fade inside the SGD mode.
* The same term does nothing for the exact mode: stronger (0.3, 1.0, 3.0) or kept on through the late
  phase, it lowers 15-NN preservation (to 0.13, 0.08, 0.01) before the Spearman moves; as a
  bounded-influence spring per partner (force w d / (1 + d^2 / c^2), the PaCMAP form) the picture is the
  same (`k5`). The exact mode's late phase does not tolerate an extra force: its local structure is the
  balance of the normalised repulsion and the attraction, and the layout scale (31 against 4) turns
  any linear spring into a strong contraction.
* A constant-strength far-field repulsion added to the exact mode (`tsne_far`, the Cauchy tail beyond
  0.1 x RMS radius, gated out of the early phase, vanishing as r^8 below) is a dead end: in 2-D the
  t-SNE and UMAP repulsions share the Cauchy tail and differ only by the 1/Z normalisation, so the sum
  is a t-SNE with a different attraction-repulsion ratio; the measured window is 3e-8 (nothing) to
  1e-7 (+0.02 Spearman, -0.02 trustworthiness) to 1e-6 (layout destroyed).
* The exaggeration knob of the exact mode (1.5 / 2 / 4) trades local structure for compactness
  (15-NN 0.278 / 0.195 / 0.121, silhouette 0.44 / 0.52 / 0.57) and leaves the Spearman at 0.335.
* Sequence, the wrong way: the exact layout finished by 60-150 SGD epochs reaches Spearman 0.436 but
  the local structure falls back to the SGD level (0.13-0.17). The SGD forces rewrite the layout at
  their own noise floor.
* Sequence, the right way: the SGD layout refined by the exact mode. Three details decide it. Started
  from t-SNE's tiny init the layout re-expands and loses the arrangement (Spearman 0.29-0.39);
  started at its own scale (std 4) the normalised repulsion collapses it; started at the exact mode's
  scale (std 31 at 70k, 31 sqrt(n / 70000) in general) with no early exaggeration it keeps the
  arrangement. The mid-range spring must be off in that pass (with it, `k2`: collapse to std 3.5,
  15-NN 0.06). The learning rate n/12 and 300 iterations give 15-NN kept 0.355-0.356 and Spearman
  0.411-0.416 (two seeds); lr 1000 gives 0.340 / 0.419, 150 iterations 0.281 / 0.414, 600 iterations
  0.352 / 0.412.

Benchmark rows (`run_hybrid_rows.py`, the seeds of BENCHMARK.md): MNIST 0.355 / 0.414 in 15 s
(SGD 0.193 / 0.413 in 4.4 s, exact 0.349 / 0.348 in 9 s, scikit-learn t-SNE 0.350 / 0.338 in 196 s);
Fashion-MNIST 0.326 / 0.712; 20 newsgroups 0.183 / 0.092 (the exact mode's 0.108 and Barnes-Hut's
0.232 stay ahead on the global axis there); blobs 0.124 / 0.682 (the exact mode alone: 0.125 / 0.340).
The fade the question asked for exists, then, but across scales of the optimiser rather than inside
the loss: the sampled pass owns the far field, the exact pass owns the near field, and the hand-over
is the init at the right scale.

### 6.1 Parametrisation of the hybrid (`k8_mid_sweep.py`, `k9_mid_validate.py`)

Since the exact pass rewrites the local structure, the SGD pass no longer has to protect it: it can
be tuned for the global arrangement alone. MNIST 70k, seed 0, exact pass at scale 31, lr n/12,
300 iterations (seed noise, measured on two seeds: 0.003 on both columns):

| SGD pass | SGD alone, 15-NN / Spearman | after the exact pass, 15-NN / Spearman | time |
|---|---|---|---|
| `mid_weight=0.1` (SGD default) | 0.193 / 0.413 | 0.356 / 0.410 | 11.0 s |
| `mid_weight=0.3` | 0.187 / 0.430 | 0.354 / 0.423 | 11.6 s |
| `mid_weight=0.5` | 0.176 / 0.455 | 0.355 / 0.449 | 12.3 s |
| `mid_weight=1.0` | 0.171 / 0.474 | 0.351 / 0.452 | 10.8 s |
| `mid_weight=2.0` | 0.154 / 0.487 | 0.350 / 0.453 | 10.7 s |
| `mid_weight=1.0`, 150 epochs | 0.127 / 0.464 | 0.336 / 0.445 | 11.9 s |
| `mid_weight=1.0`, 100 epochs | 0.066 / 0.399 | 0.276 / 0.401 | 11.0 s |

The local cost of the stiffer spring (15-NN 0.193 down to 0.154) is erased by the exact pass, its
global gain (+0.04) is kept. Saturation at 0.5-1.0. A shorter SGD pass does not work: the exact pass
refines a layout, it does not repair one (100 epochs: 15-NN 0.276, trustworthiness 0.71), so the time
of the hybrid stays SGD plus exact. On the other data sets (`k9`, hybrid at `mid_weight` 0.1 / 0.5 /
1.0): Fashion-MNIST 15-NN 0.328 / 0.328 / 0.331, Spearman 0.716 / 0.750 / 0.770; 20 newsgroups
0.183 / 0.182 / 0.180, 0.073 / 0.094 / 0.095; blobs 0.123 / 0.124 / 0.124, 0.696 / 0.713 / 0.673.
0.5 is never below 0.1 on any column of any data set, 1.0 loses 0.04 on the blobs: the hybrid's
default is `mid_weight=0.5` (the SGD mode keeps 0.1; the constructor default is None, resolved per
mode, the value used is in `mid_weight_`).

The knobs of the exact pass buy at most 0.005 of 15-NN and 0.015 of Spearman for 5-6 s more, or hurt (`mid_weight=1.0`
SGD pass; 15-NN / Spearman / time): scale 22: 0.344 / 0.443 / 8.9 s; scale 40: 0.356 / 0.459 /
16.2 s; lr n/24: 0.346 / 0.458; lr n/6: 0.356 / 0.446; 450 iterations: 0.357 / 0.443 / 17.0 s;
dof 0.8: 0.343 / 0.447; exaggeration 1.2: 0.343 / 0.448. Scale 40 and 450 iterations buy 0.005 of
15-NN for 5-6 s; dof below 1 and late exaggeration cost local structure, as in the exact mode alone.
The hybrid has one knob worth turning, and it is in the SGD pass.

### 6.2 Where the time goes (`k10_speed.py`)

MNIST 70k, 20 threads, CPU shared with other jobs (timings within 15 %):

| mode | kNN graph | init + graph | layout | of which repulsion | total |
|---|---|---|---|---|---|
| SGD (300 epochs) | 1.1 s | 0.9 s | 2.9 s | (sampled) | 4.0-5.7 s |
| exact (600 iterations) | 1.1 s | | 7.7 s | 5.5 s | 9-11 s |
| hybrid (SGD + 300 exact) | 1.0 s | 0.9 s | 2.9 + 9.1 s | 8.0 s | 12.5-14 s |

The exact mode's repulsion costs 0.9 s per 100 iterations while the layout is small and 2.8 s for the
last 100: the FFT grid follows the layout's extent (140 boxes of width 1 at the end). The hybrid pays
the large grid from its first iteration (162 boxes), which is why its 300 iterations cost more than the
exact mode's 600. What buys time:

* Hybrid, wider boxes: `fft_box=1.5` cuts the pass to 8.5 s at MNIST 15-NN 0.349-0.352 (base
  0.351-0.356), Fashion-MNIST 8.1 s at 0.322-0.324 (base 0.327-0.329), 20 newsgroups 3.2 s at 0.179
  (0.182), blobs 2.0 s at 0.114-0.117 (0.122-0.124); Spearman unchanged everywhere. About -1 % of local
  structure for -30 % of time; 2.0 costs the same on MNIST but is the limit (3.0: 0.335). Not the
  default, which stays on the fidelity side; documented as the preset.
* Hybrid, fewer iterations at a larger step: 150 iterations at lr n/6 give 0.343-0.346 in 6.4 s, 200 at
  n/8 0.346 in 7 s; with `fft_box=1.5` as well, 5.5 s at 0.341. The exact pass converges in 150
  iterations when the step is doubled, where 150 at the default step gave 0.281 (section 6).
* Exact mode: no equivalent. Box 1.5 for the whole run or for the late phase only (research edit,
  reverted) both cost 0.017 of 15-NN (0.331 against 0.348) for 1.5-2 s; 2.0 costs 0.043. The late
  phase is where the local structure forms, and it needs the fine grid; the hybrid tolerates the coarse
  grid because it refines an existing structure.
* SGD mode: nothing dominates. The attraction's `pow` costs 0.25 s (b = 1 runs in 3.72 s against 3.97),
  three negatives instead of five 0.27 s, the mid-range term 0.4 s, 200 epochs instead of 300 0.7 s;
  10 threads are slower than 20 (4.37 s, hyper-threading pays). The loop is at about 39 core-cycles per
  force evaluation at the 3.5 GHz base clock (2.6 G evaluations in 2.9 s on 10 physical cores; the
  earlier figure of 70 counted the 20 hyper-threads as cores), against 47 on one thread (`k11`); a fast `pow` and a cheaper RNG would
  give 10-20 %, not a factor.

### 6.3 Staged baselines: the pass is generic, the start is MAGNET's (`benchmarks/staged_baselines.py`)

Is the hybrid's gain the staging (any global layout, then a full-batch pass) or the SGD layout it starts
from? The same pass (300 full-batch iterations, no early exaggeration, no mid spring, hand-over scale
31 sqrt(n / 70000), MAGNET's 15-NN graph) applied to seven starts on MNIST and Fashion-MNIST 70k
(4 threads; the UMAP, PaCMAP, openTSNE and SGD starts are the seed-0 layouts of the paper's figure):

| start (MNIST 70k) | 15-NN before -> after | Spearman before -> after | centroids before -> after |
|---|---|---|---|
| PCA-2 | 0.010 -> 0.315 | 0.488 -> 0.399 | 0.814 -> 0.811 |
| random | 0.000 -> 0.065 | 0.000 -> 0.008 | 0.31 -> 0.33 |
| UMAP | 0.117 -> 0.353 | 0.296 -> 0.299 | 0.752 -> 0.681 |
| PaCMAP | 0.094 -> 0.354 | 0.306 -> 0.311 | 0.684 -> 0.633 |
| openTSNE | 0.348 -> 0.354 | 0.329 -> 0.326 | 0.607 -> 0.614 |
| MAGNET SGD, mid 0.5 (= hybrid) | 0.178 -> 0.356 | 0.459 -> 0.442 | 0.524 -> 0.492 |
| MAGNET SGD, default (mid 0.1) | 0.193 -> 0.356 | 0.418 -> 0.414 | 0.635 -> 0.617 |

| start (Fashion-MNIST 70k) | 15-NN | Spearman | centroids |
|---|---|---|---|
| PCA-2 | 0.018 -> 0.315 | 0.873 -> 0.799 | 0.971 -> 0.962 |
| UMAP | 0.133 -> 0.330 | 0.572 -> 0.607 | 0.879 -> 0.879 |
| PaCMAP | 0.098 -> 0.327 | 0.582 -> 0.599 | 0.885 -> 0.882 |
| openTSNE | 0.323 -> 0.329 | 0.617 -> 0.614 | 0.882 -> 0.876 |
| MAGNET SGD, mid 0.5 (= hybrid) | 0.193 -> 0.329 | 0.774 -> 0.753 | 0.958 -> 0.931 |
| MAGNET SGD, default | 0.203 -> 0.328 | 0.729 -> 0.716 | 0.939 -> 0.908 |

Three facts. (i) The pass is generic: every neighbour-embedding start lands at 0.353-0.356 of 15-NN on
MNIST and 0.327-0.330 on Fashion-MNIST, the full-batch mode's own level, in 8-13 s. (ii) It keeps the
start's global structure: the Spearman moves by -0.02 to +0.03 (it even rises for UMAP and PaCMAP on
Fashion-MNIST), and the centroid arrangement by -0.07 to +0.01. The global column after the pass is the
global column of the start. (iii) It refines, it does not repair: PCA-2 reaches 0.315 only and loses
0.07-0.09 of Spearman, a random start reaches 0.065. So the hybrid's lead on both axes decomposes into a
general recipe (a short full-batch pass at the right scale, without exaggeration, on top of any layout
that already has neighbourhoods) and a specific start: MAGNET's SGD layout with a stiff mid spring has
the best global structure among the neighbour embeddings (0.459 / 0.774 against UMAP 0.296 / 0.572,
PaCMAP 0.306 / 0.582, openTSNE 0.329 / 0.617), and the pass erases the local cost of that stiffness.
The converse also holds: handed MAGNET's SGD layout at the same scale, openTSNE's optimiser (its own
perplexity-30 affinities, 300 iterations at momentum 0.8, no exaggeration) reaches 0.346 / 0.428 on
MNIST and 0.322 / 0.743 on Fashion-MNIST, 4-5 times slower (55 / 44 s at 4 threads against 12 / 8 s);
and PCA handed to openTSNE's optimiser does better than PCA handed to ours (0.333 / 0.376 against
0.315 / 0.399 on MNIST). Log: `benchmarks/cache/staged_baselines.log`, JSON alongside.

### 6.4 The pass's own knobs, and the centroid question (`k14_hybrid_centroids.py`)

One SGD layout (MNIST 70k, mid 0.5, seed 0, 4 threads), the pass re-run from a copy of it under each
variant. Columns: 15-NN kept, Spearman (all / short / intermediate / long tercile), centroid Spearman,
time of the pass.

| pass | 15-NN | Spearman | short / mid / long | centroids | time |
|---|---|---|---|---|---|
| SGD layout before the pass | 0.178 | 0.428 | 0.366 / 0.111 / 0.157 | 0.594 | |
| default (scale 31, 300 it) | 0.353 | 0.413 | 0.353 / 0.105 / 0.152 | 0.596 | 10.1 s |
| 25 / 75 / 150 iterations | 0.277 / 0.328 / 0.343 | 0.428 / 0.427 / 0.423 | | 0.594 / 0.595 / 0.587 | 1.6 / 2.9 / 5.2 s |
| scale 15 | 0.338 | 0.395 | 0.339 / 0.099 / 0.145 | 0.586 | 5.9 s |
| scale 62 | 0.357 | 0.424 | 0.363 / 0.110 / 0.155 | 0.595 | 29.6 s |
| scale 62, 150 it | 0.349 | 0.428 | 0.366 / 0.111 / 0.157 | 0.595 | 17.1 s |
| scale 124 | 0.338 | 0.427 | | 0.594 | 121 s |
| mid spring kept on (0.1, not annealed) | 0.118 | 0.310 | | 0.646 | |
| mid spring 0.5 (annealed / not) | 0.238 / 0.044 | 0.108 / 0.195 | | 0.20 / 0.06 | |
| learning rate 1000 / 20000 | 0.340 / 0.335 | 0.419 / 0.414 | | | |
| dof 0.5 | 0.279 | 0.395 | | 0.600 | |

The pass costs 0.015 of Spearman at scale 31 and 0.004 at scale 62, where it also gains 0.004 of 15-NN,
for three times the time: the FFT grid follows the layout's extent, so the cost of an iteration grows
with the square of the hand-over scale (10 / 30 / 121 s for 31 / 62 / 124). Scale 62 with 150
iterations is the one setting that keeps the whole Spearman of the SGD layout (0.428) at 0.349 of
15-NN, for 17 s; the default stays at 31 and 300 (the paper's "+0.005 15-NN, +0.015 Spearman for 5-6 s"
knob). The mid spring is confirmed fatal in the pass (0.04-0.24 of 15-NN), the learning rate and the dof
buy nothing. On the centroids: here the pass leaves them where the SGD put them (0.594 -> 0.596, and
0.586-0.600 under every sane variant), and the three starts of the staged table move by -0.07 to +0.01.
The 0.421 of the strata table is the hybrid layout of the figure, one run; the SGD start itself lands at
0.52-0.59 of centroid Spearman from one Hogwild run to the next with the same seed (0.524 in
`staged_baselines`, 0.594 here), and that metric is a rank correlation over 45 pairs, so a single layout
says little about it. The seed table (`benchmarks/seeds.py`) carries the dispersion.

## Scripts (`benchmarks/theory/`)

`e1_force_spectrum.py` (per-octave repulsive force on MAGNET's own embeddings), `k13_tail_crossover.py` (the same binned by the kernel crossover, with the shell-count-adjusted prediction), `e1b_powerlaw.py`
(pure power law on smooth clouds; `e1b.log`), `e2_contamination.py` (edge-distance distribution per
data set), `e3_huber.py` (needs the research build with the Huber kernel; the C++ and Python patch is
described in the script's docstring), `e4_oracle_local.py`, `e5_law.py`, `e6_estimator.py`,
`e7_embedding_estimator.py` (two-phase version, superseded), `g1_gate.py` (crowding distribution on
real data), `g2_adaptive.py`, `g3_mindeg.py`, `h1_hubs.py`, `h2_degree.py` (the text question), `k1_hybrid.py` to `k7_hybrid_general.py`, `k8_mid_sweep.py` (parametrisation of the hybrid on MNIST: mid_weight, SGD epochs, seed; exact-pass scale, lr, iterations, dof, exaggeration), `k9_mid_validate.py` (mid_weight of the hybrid on Fashion-MNIST, 20 newsgroups, blobs) `run_hybrid_rows.py` (section 6) and `k14_hybrid_centroids.py` (section 6.4; the staged baselines are `benchmarks/staged_baselines.py`); the `.log` files are the runs quoted here (4 threads, 2-3
seeds). The e3 / e4 / e6 / e7 scripts take the package directory and the repository as arguments; g1 and
g2 take the repository and a data directory holding `coil20/coil-20-proc/*.png` (Columbia CAVE) and
`paul15/paul15.h5` (falexwolf.de/data); Cora and the airport graphs are read from local copies.
