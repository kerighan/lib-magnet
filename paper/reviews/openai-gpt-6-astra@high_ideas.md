## Diagnosis

The opportunity is not another attraction–repulsion law. It is to **separate three things that the present forces confound: neighbourhood identity, coarse geometry, and embedding scale**. Your projected experiment is the strongest evidence: coarse motion can be suppressed without destroying local repair, but freezing coordinates suppresses too much.

Three mathematical cautions first:

- Write \(k_{ij}=(1+s^{2}d_{ij}^{2})^{-1}\), \(Z=\sum k_{ij}\), \(q_{ij}=k_{ij}/Z\). Then
  \[
  \frac{\partial F(sY)}{\partial\log s}
  =2(\mathbb E_pu-\mathbb E_qu)
  =2\left(\frac{\sum k_{ij}^{2}}Z-\sum p_{ij}k_{ij}\right).
  \]
  The equilibrium dilation depends on **\(P\) and the shape \(Y\)**, not on \(P\) alone. It need not be unique or finite; the full-batch rows ending at 124 with negative derivative have not bracketed it.
- Freezing Laplacian coordinates proves preservation of those coordinates, **not preservation of Spearman**. “A refinement that cannot drift” overstates the result.
- Positive zero-rest-length springs are not intrinsically frustrated: alone they seek consensus. The relevant issues are nonreciprocal updates, stochasticity, and competition with the other forces.

Below are eight proposals, ranked by expected research value given your measurements. Numerical thresholds are **experiment acceptance criteria, not predicted results**. Let \(m\) denote graph edges and \(M\) FFT grid sites.

---

## 1. Moment-separated geometry: put rest lengths on barycentres, not on individual points

**Principle.** For two populations,
\[
\mathbb E\|X_a-X_b\|^2
=\|\mu_a-\mu_b\|^2+\operatorname{tr}\Sigma_a+\operatorname{tr}\Sigma_b.
\]
Thus pairwise-distance geometry confounds separation of means with within-population dispersion; a spring between individual points acts on both. Partition the graph into label-free cells and give their barycentres a nonzero-rest-length stress, whose gradient translates each cell uniformly and therefore leaves every within-cell displacement unchanged under that term alone.

### Construction

Obtain \(B=64\) or \(128\) balanced cells from spectral coordinates or graph partitioning. Define
\[
c_a(Y)=\frac1{n_a}\sum_{i\in a}y_i,\qquad
\mu_a=\frac1{n_a}\sum_{i\in a}x_i,\qquad
D_{ab}=\|\mu_a-\mu_b\|.
\]
Use
\[
S_{\rm coarse}(Y)
=\frac1{|\mathcal E_c|}
\sum_{(a,b)\in\mathcal E_c}
w_{ab}
\left(
\frac{\|c_a-c_b\|-\alpha D_{ab}}
{\alpha D_{ab}+\delta}
\right)^2,
\qquad
F_{\rm new}=F_{\rm KL}+\lambda S_{\rm coarse}.
\]
Take all cell pairs initially. Fix \(\alpha\) by matching coarse RMS distances to the starting layout; **do not jointly minimise unnormalised stress over a free \(\alpha\)**, which admits collapse.

This also gives a useful rigidity statement. At a realizable zero-stress configuration, the stress Hessian is \(2R^\top WR\), where \(R\) is the rigidity matrix; for an infinitesimally rigid framework its only null directions are rigid motions. This is local stability, not a claim of uniqueness for inconsistent high-dimensional distances.

### Why here

This directly attacks the MNIST **\(0.47\pm0.16\) centroid failure**, without asking the same individual-point force to repair both neighbourhoods and means. Unlike your unsuccessful springs, it has no systematic contraction once the target coarse distances are attained.

It also supplies geometry from the input rather than merely preserving the stiff start’s mistakes. A fuzzy locally rescaled graph cannot, in general, identify original-space centroid distances; that information must enter somewhere.

### One-day experiment

1. Fix one graph and one partition across seeds; never use labels to construct cells.
2. Add the term only during the pass, from both \(s=0.1\) and \(s=0.5\) starts.
3. Set \(\lambda\) by initial lifted-gradient ratios
   \[
   \frac{\|\lambda\nabla_Y S_{\rm coarse}\|}
        {\|P_{\rm cell}\nabla_YF_{\rm KL}\|}
   \in\{0.1,0.3,1\},
   \]
   where \(P_{\rm cell}\) averages a vector field within cells.
4. Run three seeds; expand the winner to five.

**Gate:** MNIST centroid Spearman \(>0.65\), SD \(<0.04\), with 15-NN \(\ge0.350\) and pairwise Spearman \(\ge0.45\). Then test whether the same scaffold allows a weaker or shorter first stage.

### Cost and risk

Extra cost \(O(n+B^2)\) per iteration; input means cost \(O(nd)\) once. The main risks are mixed cells, cross-cell neighbour damage, and a coarse stress incompatible with the desired 2-D topology. For cosine data, use normalized input vectors if invoking the squared-Euclidean moment identity.

**This is my first experiment.** It targets the conspicuous failure with a force acting in the right subspace.

---

## 2. Make the first stage a reversible, implicit graph flow

**Principle.** A directed mean-partner pull is not generally the Euclidean gradient of a scalar energy; symmetrizing its expected rates produces a Laplacian energy
\[
E_{\rm mid}(Y)=\tfrac12\operatorname{tr}(Y^\top L_{\rm mid}Y).
\]
Its proximal step is
\[
Y^+=(I+\tau sL_{\rm mid})^{-1}\widetilde Y,
\]
which attenuates Laplacian mode \(\lambda\) by \((1+\tau s\lambda)^{-1}\), rather than the potentially unstable explicit multiplier \(1-\tau s\lambda\). This is backward-Euler heat flow: unconditionally stable and nonexpansive for the isolated spring step.

### Why here

Your owner-parallel implementation already halves SGD time and increases pairwise Spearman from approximately \(0.415\) to \(0.451\). That is much more consequential than the exponent gain.

The experiment should distinguish two hypotheses:

- the stiff spring supplies useful graph diffusion;
- asynchronous, degree-dependent spring impulses happen to select useful—but unstable—basins.

An explicit reversible flow gives the first hypothesis a mathematical identity and removes a source of scheduling dependence. It does **not** prove that the nonlinear embedding will have a unique solution.

### One-day experiment

Form the directed mid-near sampling rates \(r_iM_{ij}\), including the source’s update frequency. Symmetrize:
\[
c_{ij}=\tfrac12(r_iM_{ij}+r_jM_{ji}).
\]
Run the existing owner-parallel local SGD, then apply one spring proximal step per epoch. Use 3–5 diagonally preconditioned CG iterations, or initially solve to relative residual \(10^{-2}\).

Compare:

1. current lock-free spring;
2. owner mode with the current spring;
3. owner mode with deterministic reciprocal explicit springs;
4. owner mode with reciprocal implicit springs.

Match **integrated smoothing strength**, not the numerical value \(s=0.5\): calibrate attenuation on several nontrivial eigenmodes of \(L_{\rm mid}\). Otherwise this is another uncontrolled attraction-strength comparison.

**Gate:** retain stiff-start pairwise Spearman within \(0.005\), cut first-stage optimisation time by at least 30%, and materially reduce centroid dispersion. Also inspect whether all variance remains after the spring has been made deterministic.

### Cost and risk

An approximate proximal step costs \(O(Jm_{\rm mid})\), with roughly five mid-near edges per vertex and \(J\) sparse iterations. No shared-coordinate writes are needed.

The principal risk is that the useful effect is not diffusion but nonlinear basin selection caused by asynchronous impulses. That negative result would still identify what the current recipe actually exploits. Symmetrization changes the effective objective, so calibrating the rates is essential.

---

## 3. Spend FFT accuracy according to the **net descent direction**, not repulsive-force accuracy

**Principle.** For an approximate gradient \(\widetilde g\), the condition
\[
\|\widetilde g-g\|\le\eta\|\widetilde g\|,\qquad \eta<1,
\]
implies
\[
\langle g,\widetilde g\rangle\ge(1-\eta)\|\widetilde g\|^2;
\]
hence \(-\widetilde g\) is a descent direction. This is standard inexact-gradient theory, but the relevant denominator is the **net KL gradient**, not the repulsive force, because attraction and repulsion increasingly cancel.

### Why here

Your existing result is already unusually favourable:

- box 1.0, 300 iterations: \(0.356/0.457\), 9.7 s;
- box 1.5, 300 iterations: \(0.353/0.461\), 5.0 s.

You are paying for accuracy before you know it matters. Conversely, the quoted 3–4% repulsive-force error may be large relative to the late net gradient.

The opportunity is a solver that is coarse while neighbourhood repair is large, then accurate only when cancellation demands it.

### One-day experiment

Every 20–30 iterations:

1. Evaluate a reference field using five interpolation points and box width 1.
2. Compare **net gradients** for box widths \(1,1.5,2\), using RMS error and directional agreement.
3. Choose the cheapest setting with empirical relative error below \(0.25\).
4. Check the actual momentum/gains proposal \(v\): if
   \[
   \langle g_{\rm ref},v\rangle\ge0,
   \]
   reset momentum or reject the proposal.

Audit the reference occasionally using exact repulsion on 64–128 query points against all \(n\), rather than treating the finer FFT as exact.

First compare this against the simple schedule “200 coarse iterations + 100 fine iterations”; the adaptive machinery must beat that baseline.

**Gate:** a 300-iteration pass in 5–6 s with at most \(0.002\) loss of 15-NN relative to the fine-grid pass.

### Cost and risk

Normal cost remains \(O(m+n+M\log M)\); a fine audit every 25 iterations adds roughly 7% of a fine iteration’s cost averaged over the run. The error check is empirical, not a uniform mathematical certificate.

A possible second step is a particle-mesh split
\[
k=k_{\rm near}+k_{\rm far},\qquad
k_{\rm near}(r)=\chi(r/R)k(r),
\]
with exact spatial-hash near interactions and a smooth far field on a coarser grid. Compute both \(Z\) and its derivative from the same split. Its risk is dense clusters: the near-pair count can erase the FFT saving.

**Do not assume a fixed-size grid becomes accurate merely by rescaling coordinates and kernel together.** The resolution requirement is dimensionless.

---

## 4. Replace frozen coordinates by a **coarse-metric budget**

**Principle.** Instead of imposing \(U_K^\top Y=U_K^\top Y_0\), minimise KL subject to a small number of geometric budgets
\[
G_r(Y)\le\varepsilon_r^2,
\]
where \(G_r\) measures changes of coarse distances at resolution \(r\). Projecting \(-g\) onto the feasible tangent cone gives
\[
v=\operatorname{proj}_{T_Y}(-g),
\qquad
\frac{dF}{dt}=\langle g,v\rangle=-\|v\|^2,
\]
by Moreau’s cone decomposition: local repair remains a descent flow, while only motions violating the budget are removed.

### Why here

Your \(K=32\) result protects pairwise Spearman but costs approximately \(0.022\) of 15-NN at 1000 iterations. Freezing all coarse coordinate coefficients forbids useful motions as well as harmful ones.

A metric budget allows coarse translation, rotation, some shear, and redistribution of error, provided the actual protected geometry remains within tolerance. It is an explicit answer to “how much global structure may local optimisation spend?”

### One-day experiment

Use the cells from idea 1 at two resolutions, say \(B=32,128\). Let
\[
G_B(Y)=
\frac1{|\mathcal E_B|}
\sum_{a<b}
\left[
\frac{\|c_a(Y)-c_b(Y)\|-\ell_{ab}}
{\ell_{ab}+\delta}
\right]^2,
\]
with \(\ell_{ab}\) taken from the starting scaffold. Try RMS budgets \(2\%,5\%,10\%\).

For the first implementation:

- project the proposed update using the one- or two-constraint quadratic program;
- perform a normal correction to restore nonlinear feasibility;
- backtrack if KL increases.

A tangent step alone is not sufficient: it can leave a curved constraint surface at second order. Apply the correction **after** momentum and gains.

**Gate:** at 1000 iterations, recover at least \(0.010\) of the hard-projection 15-NN loss while keeping pairwise Spearman within \(0.005\) of the start.

### Cost and risk

Cell-based gradients cost \(O(n+B^2)\), much less than repeatedly multiplying by 32–128 dense eigenvectors. The small constraint QP is negligible.

This protects the chosen geometry, not arbitrary Spearman. Starting from a bad centroid arrangement protects a bad arrangement; use idea 1 to learn a better scaffold first. Aggregate quadratic budgets can also hide a few severely distorted distances, so inspect their maximum and upper quantiles.

---

## 5. Quotient out dilation by fixing **graph-edge kernel occupancy**

**Principle.** Define a homogeneous bandwidth \(h(Y)\) implicitly by
\[
\sum_{ij}p_{ij}
\frac{\|y_i-y_j\|^2}{h(Y)^2+\|y_i-y_j\|^2}=c,
\qquad 0<c<1,
\]
and use
\[
q_{ij}(Y)\propto
\left(1+\frac{\|y_i-y_j\|^2}{h(Y)^2}\right)^{-1}.
\]
Since \(h(aY)=a\,h(Y)\), the resulting objective is scale invariant; Euler’s identity gives
\[
\langle Y,\nabla F_c(Y)\rangle=0,
\]
so its continuous Euclidean gradient flow has no radial expansion.

### Why here

This replaces the dataset-size rule \(31\sqrt{n/70000}\) by a dimensionless statement: **what fraction of attraction is operating beyond the kernel crossover?**

It does not confuse the KL-optimal dilation with the best quality–time hand-over. Nor does it pretend that the local/global scale dial can disappear entirely: the meaningful parameter is now \(c\), not an absolute plotting radius.

Unlike periodically rescaling a standard t-SNE trajectory, this defines an actual scale-invariant objective. The derivative of \(h(Y)\) must be included.

### One-day experiment

For each existing start, measure
\[
c_0=\sum p_{ij}\frac{d_{ij}^2}{1+d_{ij}^2}
\]
at scale 31. Implement the implicit derivative of \(h\), then compare \(c=c_0\) and \(c_0\pm0.1\), within \((0,1)\).

The scalar root is monotone; warm-started Newton needs only graph-edge reductions. Run at \(n=5\mathrm{k},20\mathrm{k},70\mathrm{k}\), and keep the plotting RMS fixed by discrete retraction.

**Gate:** eliminate dependence on initial absolute scale and obtain a transferable occupancy setting with less than \(0.005\) 15-NN loss relative to separately tuned hand-overs.

### Cost and risk

Extra cost \(O(m)\) per bandwidth/derivative update; no additional FFT is intrinsically required.

This removes **pure dilation**, not shape drift. Also, a fixed plotting radius does not guarantee a fixed computational grid: if \(h(Y)\) shrinks as shape changes, the required grid resolution increases. The quality advantage is speculative; the scale-invariance identity is exact.

---

## 6. A curvature-aware local solver, with coarse modes protected

**Principle.** For the attractive term
\[
A(Y)=\sum_{ij}p_{ij}\log(1+d_{ij}^{2}),
\]
its instantaneous Hessian satisfies
\[
\nabla^2 A(Y)\preceq 4L_{p/(1+d^2)}\otimes I.
\]
Therefore
\[
v=-B^{-1}g,\qquad
B=\tau I+4L_{p/(1+d^2)}\otimes I,\quad \tau>0,
\]
is always a descent direction, since \(g^\top v=-g^\top B^{-1}g<0\); it uses the actual graph-force stiffness rather than sign-based gains.

### Why here

You spend 300 iterations largely repairing local configurations, and 3000 iterations still improve 15-NN. Some of this may be conditioning rather than an intrinsically slow statistical process.

However, a naive inverse-Laplacian preconditioner accelerates low-frequency motion—exactly the motion you do not want. Apply it within the protected geometry of idea 4, or initially on the complement of the existing frozen modes. Otherwise it could efficiently destroy the start.

### One-day experiment

Replace gains and momentum by a 3–5-iteration PCG solve for the direction. Update edge coefficients every 5–10 iterations; test
\[
\tau\in\{0.1,1,10\}\times
\operatorname{median}\operatorname{diag}
\bigl(4L_{p/(1+d^2)}\bigr).
\]
Use the next computed KL to detect failed steps and increase \(\tau\); do not spend several extra FFT evaluations on every line search.

Compare against both the normal pass and a diagonal-only version of \(B\). If diagonal scaling captures all the gain, keep the simpler method.

**Gate:** attain the usual \(0.353\)-level 15-NN in at most 150 iterations and reduce total pass time by at least 25%.

### Cost and risk

Adds \(O(Jm)\) per step for \(J\) sparse iterations. It pays only if iteration count falls sufficiently.

The Laplacian is an attraction-curvature surrogate, **not an upper bound on the full KL Hessian**. Repulsion, negative curvature, and changing neighbour configurations can limit its usefulness. This is an efficiency proposal, not a claim of a new embedding objective.

---

## 7. Treat vertex mass as a nuisance parameter: Sinkhorn-balanced repulsion

**Principle.** Let \(\pi_i=\sum_jp_{ij}\), and replace ordinary \(q\) by
\[
q^\theta_{ij}
=\frac{e^{\theta_i+\theta_j}k_{ij}}
{\sum_{kl}e^{\theta_k+\theta_l}k_{kl}}.
\]
For fixed \(Y\), minimising \(\mathrm{KL}(P\|Q^\theta)\) over \(\theta\) is convex and enforces \(\sum_jq^\theta_{ij}=\pi_i\); equivalently, \(Q^\theta\) is the entropic scaling of \(K\) to those marginals. The envelope theorem then gives the usual-looking embedding gradient, but with the balanced \(q\).

### Why here

UMAP membership strengths are not perplexity-calibrated probabilities. After global normalization, variation in fuzzy degree becomes variation in attractive vertex mass, while ordinary repulsion assigns mass according to embedding density.

That mismatch can make geometry compensate for degree. Profiling out vertex propensities asks a cleaner question: **after accounting for how much probability each vertex owns, which neighbours should receive it?**

This is especially worth trying on text and heterogeneous-degree graphs. It may also improve neighbourhoods away from the graph’s native \(k=15\), although that is a hypothesis, not a consequence of balancing.

### One-day experiment

Keep \(P\) unchanged. Warm-start damped symmetric Sinkhorn updates every five iterations until marginal error is below approximately 1%.

Weighted FFT fields suffice: sources carry \(a_j=e^{\theta_j}\), and the force at \(i\) is multiplied by \(a_i\). Subtract self-interactions consistently.

Measure:

- 15-NN and 50-NN;
- preservation stratified by fuzzy degree;
- correlation between degree and layout density;
- MNIST, Fashion, and 20 newsgroups.

**Gate:** remove degree-stratified fidelity differences and improve 50-NN by approximately \(0.008\)—enough to close the reported MNIST gap—without sacrificing more than \(0.003\) at 15-NN.

### Cost and risk

Same asymptotic FFT cost, with extra scalar weighted convolutions for balancing. Two updates every five iterations could add tens of percent; warm starts and less frequent balancing are important.

This equalizes **probability marginals**, not force magnitudes. It may remove useful density information or create extreme scaling factors. Sinkhorn machinery is known; the distinctive methodological claim would be that fuzzy-graph embedding should condition on vertex mass rather than turn it into geometry.

---

## 8. Replace the coarse part of KL, not merely add a competing global loss

**Principle.** Partition ordered point pairs into blocks \(b=(a(i),a(j))\). The KL chain rule gives exactly
\[
\mathrm{KL}(P\|Q)
=
\mathrm{KL}(P_b\|Q_b)
+
\sum_b P_b\,
\mathrm{KL}\!\left(P(\cdot\mid b)\|Q(\cdot\mid b)\right).
\]
Keep the conditional term for neighbourhood organisation, but replace the block-mass term by input-derived coarse geometry:
\[
F_{\rm conditional}
=
\sum_bP_b\,\mathrm{KL}(P(\cdot\mid b)\|Q(\cdot\mid b))
+\lambda S_{\rm coarse}.
\]

### Why here

Adding stress to full KL asks two objectives to fight over coarse geometry. This construction first removes KL’s coarse preference, then installs the geometry you actually want.

For Cauchy kernels the conditional repulsion uses
\[
Z_{ab}=\sum_{i\in a,j\in b}k_{ij},
\]
and the coefficient \(P_{ab}/Z_{ab}\), instead of the single coefficient \(1/Z\). When two compact cells are well separated, their conditional kernel distribution is nearly uniform, so the leading force on their separation cancels: coarse placement is left primarily to the metric term.

This is the most conceptually distinctive proposal here: **local fidelity is a conditional probability problem; global fidelity is a metric problem**.

### One-day experiment

Do not attempt the fast production version first.

On a fixed 5k subset, use exact pairwise kernels and \(B=16\) cells. Compare:

1. ordinary KL;
2. ordinary KL plus coarse stress;
3. conditional KL plus the identical stress.

Use the same starts, optimiser, and wall-time accounting. If variant 3 does not beat variant 2 on the Pareto frontier, abandon the additional complexity.

Then try a two-level hierarchy: retain intermediate conditional KL terms, and replace only the top-level block-mass term. This gives a principled place to weight mesoscopic neighbourhood fidelity—potentially \(k=50\)—without merely averaging several affinity matrices.

### Cost and risk

Exact pilot: \(O(n^2)\). A direct FFT implementation with \(B\) source-cell masks costs approximately \(B\) field evaluations, which is unacceptable at \(B=64\).

A scalable version needs local FFTs plus far-cell multipoles. For separated cells,
\[
Z_{ab}
=
n_an_b\,k(c_a-c_b)
+\frac{n_an_b}{2}
\nabla^2k(c_a-c_b):
(\Sigma_a^Y+\Sigma_b^Y)
+\cdots .
\]
That expansion provides a route, not a free speed guarantee.

The major statistical risk is overlap: blocks with \(P_{ab}=0\) have no conditional repulsion, and correct barycentre distances alone do not prevent their point clouds from interpenetrating. Boundary interactions and overlap control must be checked explicitly.

**Highest conceptual upside; lowest immediate implementation certainty.**

---

## What I would actually run first

Do not build all eight into one system.

1. **Immediate quality test:** coarse barycentre stress, with the existing pass. This decides whether the centroid failure is cheaply repairable without paying local fidelity.
2. **Immediate speed tests:** retuned owner-parallel first stage; coarse-to-fine FFT pass. Your measurements already make both credible.
3. **Mathematical core:** replace hard spectral freezing by a coarse-metric budget. This turns “stop before drift” into an explicit constrained optimisation problem.
4. **Only after that:** test conditional KL on 5k points. A clear win over “KL + stress” would justify a genuinely new objective and a fast hierarchical implementation.

The strongest resulting identity would be:

> **Learn a label-free coarse metric scaffold, then optimise conditional neighbourhood structure inside a quantified geometric budget.**

That is more than a staged combination of UMAP-like and t-SNE-like dynamics. It specifies what information belongs at each scale, what the optimiser is allowed to change, and what its computational effort should resolve.