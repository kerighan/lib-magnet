## 1. VENUE TIER

These are rough editorial judgments, not calibrated acceptance rates. They assume a completed manuscript with the **present evidence**, not submission of the outline. The probabilities are not mutually exclusive.

| Tier | Realistic assessment now | What changes the assessment |
|---|---|---|
| **(a) arXiv** | Straightforward dissemination; not an acceptance test. I would **not** classify this as necessarily “arXiv only.” | Remove unsupported theorem/optimality claims before posting; otherwise the preprint establishes claims you will later need to retract. |
| **(b) JOSS / JMLR MLOSS** | **JOSS: roughly 60–85%**, conditional on software eligibility and repository quality. **MLOSS: 30–55%**. This is currently the most natural home. | Installation, wheels, documentation, tests against reference implementations, reproducibility contracts, maintenance, licensing, and evidence of utility beyond the benchmark. I cannot assess these from the outline. |
| **(c) TMLR / ECML-PKDD / SDM / AISTATS / TVCG / EuroVis** | **TMLR: 15–30%; ECML/SDM: 10–25%; AISTATS: 5–10%; TVCG/EuroVis: 5–15%.** These venues are not interchangeable: the present draft lacks both AISTATS-level theory and a visualization-task evaluation for TVCG/EuroVis. | One defensible scientific claim; controlled comparisons against strong native implementations and staged baselines; substantially better uncertainty reporting; validation beyond the development datasets. |
| **(d) NeurIPS / ICML / ICLR / JMLR regular** | **Low single digits** with the current framing. JMLR regular is particularly unlikely without a much more complete scientific account. | A general, predictive result that survives controls—not merely additional speed tables. For example, a transferable account of when scale-matched refinement preserves global organization, demonstrated across initializers, backends, datasets, and sample sizes. |

**Moving up one tier:**  
- **Software → research:** establish causality and generality, not just benchmark leadership.  
- **Mid-tier research → top-tier:** identify something genuinely new about neighbor embedding that other implementations can exploit. More datasets alone will not do that.  
- A narrower, correct empirical paper has better prospects than the current combination of strong “derived optimum” claims and largely heuristic arguments.

## 2. THE THREE WEAKNESSES A STRONG REVIEWER WILL ATTACK FIRST

### 1. The claimed derivation of optimal exponents is not a derivation of optimal embedding quality

The abstract says the optima are **derived**, with a **step at \(q=m\)**, and the introduction says they **match measurement**. Neither follows.

- The near-field approximation \(\rho(d)\approx ad^{2b}\) is being used to invoke robustness properties concerning **large residuals**, where that approximation does not apply. For the actual loss
  \[
  \rho(d)=\log(1+ad^{2b}),
  \]
  the attraction is redescending at infinity for every \(b>0\), and its score is globally bounded for every \(b\ge 1/2\). The proposed uniqueness argument for \(b=1/2\) therefore does not characterize the actual loss.
- Convexity and bounded influence of a location-estimation surrogate do not establish an optimum for **15-NN preservation in a jointly moving, repelling configuration**.
- \(q>m\) does not imply a particular contamination fraction, let alone a discontinuous optimum. Conversely, \(q\le m\) does not guarantee an undistorted global embedding.
- Your own measurements give **\(b=0.4\)** on crowded synthetic datasets and **\(0.6\)–\(0.7\)** on text. Those are compatible with a heuristic preference for subquadratic attraction, not a universal \(1/2\) law.
- The smooth-cloud tail experiment verifies a power-counting calculation. It does **not** establish that \(b_r=m/2\) maximizes fidelity on optimized clustered layouts.

**Likely review sentence:** “The theoretical claims substantially exceed what is established, and the experiments are presented as confirmation despite material exceptions.”

### 2. The main empirical gains are not attributed to the proposed conceptual contribution

The exponent split changes MNIST preservation from **0.189 to 0.192**: an absolute gain of **0.003**, comparable to your stated seed noise. Its clearer contribution is the **5.3 → 4.3 s** computational improvement.

Meanwhile:

- Removing the borrowed mid-range mechanism reduces Spearman from **0.413 to 0.285**. The manuscript explicitly says this term supplies **all** the SGD mode’s global advantage.
- The largest local improvement comes from a **t-SNE-style refinement pass**, not the exponent argument.
- The hybrid is a globally informed initialization followed by local refinement. That is close to established initialization, multiscale-affinity, and staged-optimization ideas.

The missing decisive controls are **PaCMAP → FFT t-SNE**, **UMAP → FFT t-SNE**, and **PCA/spectral initialization → FFT t-SNE**, with matched handover scale, affinity construction, and refinement budget. Also run **MAGNET initialization → openTSNE refinement**.

**Likely review sentence:** “The results demonstrate a useful implementation and combination of known components, but do not isolate a new algorithmic principle.”

### 3. The headline superiority is smaller and less secure than the presentation suggests

Specific problems:

- MNIST hybrid **0.353** versus sklearn t-SNE **0.350** is a **0.003** difference, equal to the reported seed noise. Fashion-MNIST **0.328 versus 0.327** is from **one seed**.
- “Leads local and global” means two selected columns. On MNIST, hybrid trustworthiness is **0.961**, below openTSNE/sklearn **0.968**; label accuracy is **0.960**, below PaCMAP **0.976**.
- Even the global lead is panel-dependent: your own SGD sweep reaches Spearman **0.478–0.487**, above the headline hybrid **0.444**, at lower neighborhood preservation.
- **9.2 s versus openTSNE’s 102.5 s** demands investigation of threads, affinities, initialization, convergence, and interpolation accuracy. openTSNE already contains native accelerated code; this is not simply C++ beating Python.
- The native CPU baseline **uwot** is missing.
- “Defaults for every method” conceals that MAGNET’s defaults and hybrid scale were selected using these same datasets.

**Likely review sentence:** “The proposed method is extensively developed on the evaluation suite, while competitors receive defaults-only treatment and insufficient implementation/protocol scrutiny.”

## 3. THE TAIL-FORCE MISMATCH

### Most likely cause: the fitted octaves are not the asymptotic far field

The prediction for **scalar force magnitude per octave** is correctly
\[
S(d)\propto d^{m-1-2b_r}.
\]
This is **not** explained by confusing loss with force; you already accounted for that exponent difference.

But the prediction assumes:

1. \(ad^{2b_r}\gg1\);
2. approximately homogeneous \(m\)-dimensional shell occupancy;
3. scalar force accumulation, not vector cancellation;
4. the ideal force, without clipping or distance regularization.

Your log bins are defined by **\(d/\operatorname{median}(1\text{-NN distance})\)**, not by the kernel crossover. Those are very different scales.

For illustration, if \(R/d_k\approx16\) and \(R/d_1=2^{9.1}\), then \(d_k/d_1\approx2^{5.1}\). Octaves 1–5 are mostly **below the kernel crossover**. They should not exhibit the far-tail exponent. This is consistent with the note saying there are only about three far-field octaves.

A useful diagnostic identity is
\[
\frac{d\log S}{d\log d}
\approx
\underbrace{\frac{d\log N_{\rm octave}(d)}{d\log d}}_{\alpha(d)}
-1-
\frac{2b_r\,ad^{2b_r}}{1+ad^{2b_r}}.
\]

- Homogeneous cloud, far field: \(\alpha=m\), yielding the proposed slope.
- Near field: the last term nearly vanishes, making the slope approximately **independent of \(b_r\)**.
- Clustered layouts: \(\alpha(d)\) varies with distance and changes when the embedding is reoptimized at another exponent.

That explains the qualitative signature—especially the nearly constant positive 2-D slope—better than “theoretical slope is approximately right.” The smaller 3-D scale separation can place the same fitted octaves across the crossover.

### Cheap discriminating experiment: no embedding reruns required

On one saved layout, choose perhaps 128–256 anchor points and evaluate distances to all other points.

1. **Plot shell counts and force sums against \(d/d_{k,r}\)**, where \(d_{k,r}=a^{-1/(2b_r)}\), not \(d/d_1\).
2. Separate:
   - pair counts;
   - mean scalar force per pair;
   - total scalar force;
   - norm of the vector sum.
3. On the **same fixed coordinates**, evaluate every \(b_r\). Do not let geometry change with the tested kernel.
4. Compare the full force with the artificial pure-tail force \(C d^{-(2b_r+1)}\).
5. Rescale the coordinates so the fitted bins satisfy, for example, \(ad^{2b_r}>100\). If there is no sufficiently populated boundary-free interval, report that the layout has **no measurable asymptotic window**.
6. Overlay the shell-count-adjusted prediction above.

Also audit:

- **Log units:** fitting \(\ln S\) against \(\log_2d\) gives slope \(s\ln2\), not \(s\); \(\log_{10}S\) gives \(s\log_{10}2\). This can explain magnitude discrepancies, not the 2-D sign failure.
- **Force definition:** scalar gradient coefficient versus coefficient times displacement; clipping; epsilon in the distance denominator.
- **Fit indices:** the printed octave shares do not make the stated fit interval transparent. Save actual bin centers, fit masks, and regression coordinates.

Vector cancellation cannot explain the main discrepancy because the **magnitude** slopes also disagree. Per-layout normalization by a single total cannot change a log slope.

### Fatal or reframeable?

- **Not fatal** to the library or hybrid algorithm.
- **Fatal to the current statement that the asymptotic theory predicts and matches real-layout force spectra**, unless corrected.
- Reframe it as **conditional asymptotic analysis plus a diagnostic of where its assumptions fail**. The smooth-cloud result is a sanity check, not empirical validation of an optimal-exponent theorem.
- “Scale-free loss budget” is defensible under explicit assumptions. “Therefore fidelity is optimal at \(m/2\)” is not established.

## 4. MISSING EXPERIMENTS, RANKED BY VALUE PER HOUR

Approximate effort assumes the current benchmark harness is usable.

### 1. Protocol audit and cheap reference checks — hours

Before generating more results:

- Verify physical/logical cores, thread settings, metric propagation, preprocessing, affinity normalization, self-neighbor exclusion, and timing boundaries.
- Compare the FFT gradient with an \(O(n^2)\) reference on a small dataset, including normalization and finite-difference checks.
- Audit whether every news20 implementation actually uses the intended cosine geometry.
- Add **PCA / truncated SVD** as global-structure baselines. Their omission is serious when claiming superior distance Spearman.
- Run the tail diagnostic above.

These can invalidate or rescue major claims faster than another dataset sweep.

### 2. Repeat the decisive rows — several machine-hours

Use **at least five independent runs**, preferably ten for the MNIST superiority claim:

- MAGNET SGD, full-batch, hybrid;
- UMAP;
- openTSNE;
- PaCMAP;
- uwot once available.

Report individual runs or mean ± SD, plus intervals on relevant differences. Use a common evaluation sample; distinguish embedding randomness from metric-subsampling randomness.

Do not treat millions of evaluated pairs as independent replicates. With Hogwild and unseeded UMAP, “same seed” does not imply paired optimizer randomness or deterministic runs.

Prioritize Fashion-MNIST and news20: one seed is inadequate, and news20 already shows **0.02 global-score variation**.

### 3. Add uwot and audit openTSNE — roughly half a day to a day

**uwot is required for “fastest CPU” claims.** Include:

- documented default behavior;
- an explicitly configured fast parallel setting;
- a reproducible setting if reproducibility is part of MAGNET’s contribution.

For **openTSNE**, report:

- `n_jobs`;
- FFT versus Barnes–Hut selection;
- affinity construction and neighbor count;
- initialization;
- iteration count and learning-rate convention;
- interpolation/grid settings.

Include a **time–quality curve**, not just default completion times. MAGNET’s own grid-width results show that numerical accuracy materially affects speed and preservation.

**FIt-SNE:** worthwhile as a second native t-SNE implementation if openTSNE remains anomalously slow. It need not be another main-table row if it adds no information beyond a correctly configured openTSNE.

### 4. Staged baselines — about one day; highest conceptual value

Run:

- PCA/spectral → t-SNE-style refinement;
- UMAP/uwot → refinement;
- PaCMAP → refinement;
- MAGNET → openTSNE refinement.

Use the same refinement settings and a small common handover-scale sweep. Where possible, use the same normalized affinity matrix.

These answer:

- Does MAGNET supply an unusually good initialization?
- Is the effect simply a generic warm-start benefit?
- Does the result transfer beyond MAGNET’s implementation?
- Is scale matching essential, or is 300-step refinement merely retaining initialization through incomplete convergence?

Also test longer refinement. Persistence over 300 iterations is not evidence of persistence at convergence.

### 5. Component ablations and shared-input comparisons — about one day

Separate the contributions of:

1. NN-descent;
2. fuzzy versus perplexity affinities;
3. initialization;
4. exponent split;
5. repulsion strength;
6. mid-range term;
7. optimizer/backend.

Provide two benchmark panels:

- **End-to-end native pipelines:** legitimate software comparison.
- **Matched graph/affinities and initialization:** scientific comparison of layout kernels.

For exponent sweeps, hold the crossover scale fixed, preferably parameterizing
\[
w(d)=\frac1{1+(d/s)^{2b}},
\]
and distinguish shape changes from changes in force amplitude and attraction–repulsion balance.

The current split ablation does not demonstrate a meaningful quality improvement; it does demonstrate a possible implementation advantage.

### 6. Evaluate the deterministic owner kernel — hours to a day

It is mentioned in the library description but absent from the evidence. Report:

- speed, quality, and memory against Hogwild;
- repeatability at fixed thread count;
- behavior across thread counts;
- any changed update semantics.

A native reproducible parallel implementation could be a stronger software contribution than the **0.003** exponent-split gain. Compare reproducible with reproducible, not deterministic MAGNET against deliberately single-threaded UMAP alone.

### 7. Expand metrics using existing layouts — hours

- Neighborhood preservation at **\(k=5,15,50,100\)**, not just the training graph’s neighborhood scale.
- Trustworthiness **and continuity**.
- Global distances/triplets stratified into within-cluster, between-cluster, and intermediate-distance relations.
- Per-group results on heterogeneous datasets.
- Neighborhood stability under small input perturbations where exact nearest-neighbor ranks are fragile.

Keep silhouette labeled compactness. Do not treat preservation, trustworthiness, and label accuracy as interchangeable measures of “local quality.”

### 8. Add genuinely new data — one to several days

Freeze defaults first. Prioritize:

1. **A real continuous/branching dataset**, such as a larger single-cell dataset with documented preprocessing.
2. **A non-image embedding dataset**, e.g. sentence/document embeddings, providing a stronger text test than one SVD pipeline.
3. **A large real dense or sparse dataset**, not only 50-D Gaussian clusters, for scaling.
4. Synthetic manifolds varying curvature, density, noise, and neighborhood size around the claimed crowding transition.

MNIST and Fashion-MNIST are not two independent validations of a universal law. The Gaussian blobs are useful controls, not convincing evidence of real-world scalability.

### Timing fairness: the presentation reviewers will accept

C++ versus Numba is a **valid end-to-end software comparison**, provided it is not presented as an isolated algorithmic speedup. Numba also produces native machine code; openTSNE already uses native kernels.

Report:

- **10 physical cores / 20 hardware threads**, if the CPU really is an i9-9900X;
- equal thread budgets and BLAS/OpenMP settings;
- dedicated-machine timing, medians and spread;
- warm steady-state timing separately from cold-start/JIT timing;
- dtype, preprocessing, peak RSS, ANN recall, and per-phase costs;
- CPU-only and GPU results in separate panels.

**cuML:** include it if claiming broad practical speed leadership or maintaining a GPU comparison. It is more informative than relying exclusively on an older t-SNE-CUDA implementation and RTX 2070. It is optional if the claim is explicitly **CPU performance**. Include transfer costs and device synchronization in end-to-end GPU timing.

The timing note explicitly mentions other concurrent jobs and ±15% variation. Those runs are suitable for exploration, not final speed claims.

## 5. THE STRONGEST THESIS

### Defensible conceptual claim

> **Global organization and local neighborhood refinement can be optimized sequentially: a sufficiently developed global layout, handed over at a compatible scale, can retain much of its organization while a second objective recovers local neighborhoods. This permits the first stage to sacrifice local fidelity temporarily rather than permanently.**

This is stronger than “faster and about as good,” but narrower than “the local/global tradeoff exists across scales, not inside a loss.” Your own SGD mid-weight sweep already demonstrates a tradeoff **inside one stage**.

### Existing evidence that supports it

The most informative result is the **mid-weight sweep**, not the leaderboard:

- Increasing mid-weight from **0.1 to 2.0** changes SGD preservation **0.193 → 0.154**, while Spearman improves **0.413 → 0.487**.
- After refinement, preservation stays around **0.356 → 0.350**, while Spearman improves **0.410 → 0.453**.
- Thus, much of the local cost is repaired while part of the global gain survives.
- Reverse ordering gives approximately **0.436 global**, but only **0.13–0.17 local**.
- A prematurely terminated first stage is not repaired: **100 epochs → 0.276 preservation**.
- Fashion-MNIST supports the pattern; news20 limits its generality.

These observations support a **staged optimization strategy**, not yet a claim that sampled optimization inherently “owns the far field” or full-batch optimization inherently “owns the near field.”

### Abstract direction

> Neighbor embeddings often trade neighborhood fidelity against large-scale organization. We investigate a staged alternative: first construct a globally organized layout using transient mid-range attraction, then refine neighborhoods with a normalized t-SNE-style objective. On the evaluated image datasets, scale-matched refinement largely removes the first stage’s neighborhood penalty while retaining much of its global organization. Ablations identify initialization scale, stage order, and first-stage maturity as important conditions; the benefit is not universal on text data. We implement this strategy in MAGNET, a C++/Python library with native neighbor search, sampled and FFT-accelerated layouts, and configurable attraction and repulsion kernels.

Add headline timings only after the protocol audit.

**Novelty warning:** globally informative initialization and staged forces are established ideas. The research contribution must be the **specific transferable conditions and controlled evidence**, not the fact that a warm start can help t-SNE.

## 6. OTHER ISSUES THE AUTHORS APPEAR TO HAVE MISSED

### Mathematical and objective-definition problems

- **FFT interpolation is not exact.** Use “full-batch, FFT-interpolated repulsion” or “unsampled repulsion.” FIt-SNE, Barnes–Hut, and t-SNE-CUDA are not “exact methods” in the numerical sense.
- **UMAP and t-SNE do not optimize the same attraction-plus-binary-repulsion loss.** t-SNE contains a global normalization term. At \(a=b=1\), their scalar repulsions have forms proportional to
  \[
  \frac{1}{d(1+d^2)}
  \quad\text{versus}\quad
  \frac{d}{Z(1+d^2)^2}.
  \]
  They share a far-tail power, not an identical force differing only by \(1/Z\). Adding a gated far-field force is therefore not exactly “t-SNE with another normalization.”
- **Negative sampling does not generally minimize the nominal full UMAP cross-entropy.** State the sampling distribution, endpoint-update rules, degree weighting, and effective objective. Your own Damrich citations make this omission conspicuous.
- **The equilibrium-shell argument is unjustified.** A graph neighbor does not receive repulsion corresponding to all negative samples. With your definition \(g=2\times5=10\), \(a\approx1.8\), \(b=0.6\), \(b_r=1\), the proposed near-field equilibrium is approximately **6.4**, versus \(d_k\approx0.61\). That violates the approximation used to derive it. Positive isolated-pair stiffness also does not establish stability of a many-point layout.
- “Influence bounded iff \(p\le1\)” needs qualification: for \(p<1\), \(pd^{p-1}\) diverges at zero. “For \(p<1\), the multivariate minimizer sits on a data point” is also false in general.
- **The Huber check confounds shape with strength.** Your pseudo-Huber function has \(h_c(d)\sim c\,d\) in its linear region. Changing \(c\) at fixed \(a\) changes attraction amplitude and kernel crossover, not just the quadratic threshold.
- The note says force scaling is loss scaling **times** the lever arm. It is the reverse: force is loss scaling **divided by \(d\)**.
- “The layout is \(m\)-dimensional everywhere” is not a justification against local repulsion adaptation. Ambient dimension is not local occupancy dimension—the tail mismatch itself illustrates this.

### Prior work and claim scope

- Independently configurable attraction and repulsion are longstanding in **force-directed graph drawing**, including Noack’s energy/LinLog framework. Do not claim the generic split as new.
- Generalized Student/heavy-tailed kernels already vary tail behavior while retaining quadratic near-field behavior. Your claimed coupling is a property of the particular family \(1/(1+ad^{2b})\), not an unavoidable property of neighbor embedding.
- Discuss **globally informative t-SNE initialization, multiscale affinities, PaCMAP’s staged schedule**, and CNE substantively, not only as citations.
- If graph layout remains a major claim, compare with actual graph-layout methods such as **sfdp or ForceAtlas2**, or narrow that section to supported input functionality.

### Concrete inconsistencies and protocol red flags

- “t-SNE-CUDA is the only faster exact method” contradicts the table: **13.8 s versus MAGNET 9.2 s**. “Loses on every column” is also false: its MNIST trustworthiness is **0.967 versus 0.966**.
- “Exact-pass knobs move results within seed noise” contradicts **scale 22 versus 40: 0.344 versus 0.356 preservation; 0.443 versus 0.459 Spearman**. Both differences exceed **0.003**.
- “Adaptive never worse within 0.002” contradicts airports, base \(0.6\): **0.344 → 0.339**. Also, COIL-20’s best fixed exponent gives **0.772**, above adaptive **0.720–0.729**. Compare against the best fixed policy under an equal tuning budget.
- The adaptive table mixes base-exponent choices across datasets. State the base explicitly. The “oracle” is an oracle **dimension-based rule**, not an upper bound; estimated policies exceeding it are unsurprising.
- **i9-9900X is 10-core/20-thread, not 20-core.** Recheck the “70 cycles per force evaluation” calculation if it counts SMT threads as independent physical cycle budgets.
- \(a\approx1.8,b\approx0.6\) are not the usual fitted UMAP `min_dist=0.1, spread=1` parameters. Label MAGNET’s kernel scale correctly, and distinguish attraction and repulsion crossover scales after splitting.
- **SBM accuracy 1.00 versus UMAP 0.26** is a protocol alarm. Did UMAP optimize the same adjacency graph, or was adjacency passed as features or as precomputed distances? Similarities are not distances. Without this clarification, the graph comparison is uninterpretable.
- Specify whether 15-NN evaluation searches the full dataset or only the 2,000-point evaluation subset. These are different statistics. Define whether label accuracy is leave-one-out or held-out, and exclude self-neighbors.
- High trustworthiness with low neighbor overlap is not contradictory: it can mean substituted neighbors are nearby in rank. Conversely, higher overlap can come with worse large-rank intrusions, as your text fuzzy-intersection result already demonstrates.
- Audit Hogwild’s C++ memory semantics. Unsynchronized conflicting non-atomic accesses are language-level data races; “lock-free” does not itself establish correctness or portability.

**Bottom line:** the strongest current research evidence concerns **staged refinement**, the strongest software evidence concerns **native end-to-end performance**, and the weakest part is the assertion of **derived universal exponent optima**. Organize the paper accordingly.