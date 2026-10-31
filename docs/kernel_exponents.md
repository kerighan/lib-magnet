# The two kernel exponents: derivation and tests

MAGNET's SGD mode minimises the UMAP loss with the low-dimensional kernel `w(d) = 1 / (1 + a d^(2b))`.
Since commit `1471e5c` the attraction and the repulsion use two different exponents (`kernel_b`, default
0.6, and `repulsion_b`, default 1.0). This note derives why the two roles want different exponents, says
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

Reading the crowding off the embedding. The contamination of section 2 can be measured on a partial
layout instead of estimated from the input distances: after 40 % of the epochs at the global exponent,
the share of a node's edges beyond `d_k` is 0.00 for the q = 2 points, 0.07-0.12 for q = 3, 0.20 for q = 5
and 0.32 for q = 10 (`e7_embedding_estimator.py`), a cleaner separation than the local dimension gives.
Finishing from that layout with the per-point exponent did not gain (+0.00 to +0.01 on the mixtures at
threshold 0.05, -0.02 on MNIST), but the second phase restarts the learning-rate schedule, a cost of the
same size as the gain being measured (the two-phase run with a constant exponent, the control, was not
done). The clean version updates the per-point exponent between epoch chunks inside one run; not done.

## 5. Open

* Why text settles at 0.6-0.7: the bimodal contamination (hubs) is consistent with section 2 but not
  derived from it, and MNIST 5k (0.6, flat) is not explained at all.
* The repulsion law is a loss-per-octave argument for a smooth cloud; what fixes the optimum in a
  clustered embedding with three far-field octaves is not derived, and the measured optimum is at or a
  little above m/2 in every dimension tested, never below.
* A per-point *repulsion* exponent (the local dimension of the embedding cloud rather than of the data)
  follows from section 3 and was not tested.
* A calibrated local-dimension estimator (the MLE reads 30 % high at k = 15) would release more of the
  7-12 % oracle gain; so would the in-run embedding-side estimator above.

## Scripts (`benchmarks/theory/`)

`e1_force_spectrum.py` (per-octave repulsive force on MAGNET's own embeddings), `e1b_powerlaw.py`
(pure power law on smooth clouds; `e1b.log`), `e2_contamination.py` (edge-distance distribution per
data set), `e3_huber.py` (needs the research build with the Huber kernel; the C++ and Python patch is
described in the script's docstring), `e4_oracle_local.py`, `e5_law.py`, `e6_estimator.py`,
`e7_embedding_estimator.py`; the `.log` files are the runs quoted here (20k points, 4 threads, seeds 0
and 1). The e3 / e4 / e6 / e7 scripts take the package directory and the repository as arguments.
