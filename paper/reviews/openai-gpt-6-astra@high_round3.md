## 1. THE THESIS

**The revised empirical thesis survives. The old estimator-based explanation does not reappear as a defensible mechanism. The new PaCMAP experiment materially weakens the “defaults-only tuning artefact” objection; it does not establish universal start superiority.**

### Claim-by-claim audit of the abstract and contributions

| Claim | Verdict |
|---|---|
| A spring-conditioned sampled layout followed by short, scale-matched refinement combines high 15-NN preservation with substantial retention of pairwise Spearman | **Supported on the two image datasets.** This is now the central defensible result. |
| The three hand-over settings constitute a successful configuration, not necessary conditions | **Supported as an empirical recipe.** The explicit disclaimer in contribution 1 is appropriate. Their individual causal roles remain incompletely separated. |
| Refinement brings the tested neighbour-embedding starts to approximately the same 15-NN level | **Supported**, including by the new repeated comparisons. “Full-batch level” should mean competitive with full-batch results, not numerical equality with the standalone full-batch mode. |
| Refinement leaves the starts’ Spearman ordering intact | **False as written.** There are rank reversals in both the single-run and repeated tables. |
| The global column is predominantly inherited from the start | **Supported for the unmodified pass and pairwise Spearman**, not for arbitrary notions of global geometry or the scaffolded variants. |
| The spring, rather than sampling alone, supplies MAGNET’s observed Spearman advantage | **Supported conditionally on this implementation.** Removing the spring removes the advantage. That does not isolate sampling’s contribution under a fixed objective. |
| MAGNET supplies the strongest pairwise-Spearman start among the tested configurations | **Supported.** The PaCMAP sweep is meaningful evidence. “Best start among neighbour embeddings” without the configuration, metric and dataset restrictions remains unsupported. |
| Fast implementation with audited default-pipeline timings | **Supported on the reported panel**, especially the idle-machine image results. The fastest configuration is sampled MAGNET, not the hybrid. |
| Pairwise and centroid Spearman tell different stories; PCA leads; MNIST centroid geometry deteriorates under the stiff spring | **Supported for the methods and datasets actually evaluated on those metrics.** “Last among neighbour embeddings” must not silently include methods with no centroid results. |

### (a) “The sequence leads both axes under three hand-over conditions”

**The useful operating point survives; unrestricted leadership does not.**

The repeated staged experiment gives the cleanest evidence:

- **MNIST, spring 0.5:** 15-NN **0.179 → 0.354**, Spearman **0.463 → 0.449**.
- **Fashion, spring 0.5:** 15-NN **0.192 → 0.328**, Spearman **0.773 → 0.750**.

That is substantial neighbourhood reconstruction with a comparatively small reduction in one global statistic.

The native-pipeline five-run table also supports the reported endpoint: **0.353 / 0.455** and **0.328 / 0.750**. Adding sklearn closes a real hole in the previous comparison. Nevertheless:

1. **The strict local lead is small and panel-dependent.** Against sklearn, the mean differences are **0.003** on MNIST and **0.001** on Fashion. Against *staged* alternatives, the hybrid does not lead: Fashion UMAP → refinement reaches **0.331 ± 0.001**, versus **0.328** for spring-0.5 MAGNET in `staged_seeds`.

2. **The local advantage is specifically at the training graph’s neighbourhood scale.** In `kscale`, openTSNE beats the hybrid at:
   - \(k=5\): **0.375 versus 0.351**;
   - \(k=50\): **0.327 versus 0.319**;
   - \(k=100\): **0.329 versus 0.313**.

   MNIST trustworthiness is also worse: **0.960 versus 0.968** in `seeds`.

3. **The global lead is not over all staged outputs.** Fashion PCA → refinement gives **0.315 / 0.799**, versus **0.329 / 0.753** for the stiff sampled start in `staged`. This is a genuine Pareto alternative. Its final output is a nonlinear neighbour embedding; excluding linear references does not exclude it.

4. **It is not a dataset-independent result.** News20: hybrid **0.181 / 0.073**, sklearn **0.192 / 0.232**. Blobs: hybrid **0.124 / 0.684**, versus separate MAGNET maxima **0.126** and **0.690**.

5. **“Global” cannot mean class-level geometry.** MNIST centroid Spearman is **0.509 ± 0.075** for the native hybrid versus **0.728 ± 0.014** for UMAP. The repeated staged experiment is harsher: **0.466 ± 0.158** for the stiff MAGNET start after refinement versus **0.687 ± 0.020** for stiff PaCMAP.

The conditions are a working configuration, not three established mechanisms:

- Tiny scale and exaggeration remain confounded.
- Scale 31 is a **finite-budget reference scale**, not an equilibrium scale. Your virial experiment establishes that distinction.
- Failure of zero-rest-length springs at 0.1 and 0.5 does not establish that attraction must be absent. The scaffold is your own counterexample to any broader prohibition.

**Surviving wording:** “On MNIST and Fashion-MNIST, the sequence attains competitive 15-NN preservation while retaining most of a spring-conditioned start’s pairwise-distance correlation.”

### (b) “The pass is generic across starts and optimisers”

**Cross-start transfer is now substantially supported. Optimiser-independence is not established.**

The repeated table improves this claim materially: neighbour-embedding starts finish at **0.352–0.355** on MNIST and **0.327–0.331** on Fashion. This is no longer resting solely on single runs.

But “ordering intact” is factually wrong:

- Fashion, repeated table: UMAP starts below default PaCMAP, **0.572 < 0.582**, and finishes above it, **0.605 > 0.600**.
- MNIST: stiff PaCMAP starts below openTSNE, **0.319 < 0.330**, and finishes above it, **0.346 > 0.327**.

The defensible statement is that **large differences between starts persist**, not that the ordering is invariant.

Nor is the result generic over arbitrary starts. At 300 iterations, random layouts reach only **0.065 / 0.062** in 15-NN. PCA reaches **0.315** but loses substantial Spearman. These are limitations of this particular schedule, not evidence that refinement intrinsically cannot reconstruct neighbourhoods.

Cross-backend transfer remains a different, weaker result:

- MNIST MAGNET start → openTSNE: **0.346 / 0.428**, versus **0.356 / 0.442** with your refinement.
- Fashion: **0.322 / 0.743**, versus **0.329 / 0.753**.

Those results show that another pipeline can exploit the start. They do not establish interchangeable optimisers: affinities, implementation and possibly effective update conventions differ. The paper now acknowledges this correctly; the abstract’s “does the same” wording should reflect it.

### (c) “The global gain is entirely the start’s, and MAGNET has the best start”

**“Predominantly” survives. “Entirely” does not. The bounded comparative superiority claim now has evidence.**

The pass actively changes Spearman:

- MNIST stiff PaCMAP: **0.319 → 0.346**, a gain of **0.027**.
- Fashion UMAP: **0.572 → 0.605**, a gain of **0.033**.
- Fashion spring-0.5 MAGNET: **0.773 → 0.750**, a loss of **0.023**.

The strongest attribution is therefore: **the stiff start supplies most of MAGNET’s advantage, and refinement trades some of that advantage for neighbourhood preservation.**

The PaCMAP comparison should be credited for what it establishes. After refinement:

- MNIST: **0.449 ± 0.015 versus 0.346 ± 0.017**, difference **0.103**.
- Fashion: **0.750 ± 0.001 versus 0.648 ± 0.001**, difference **0.102**.

On MNIST, an approximate independent-run Welch interval for the difference is roughly **[0.080, 0.126]**. This is not a marginal seed effect. It is no longer reasonable to dismiss the observed advantage simply because PaCMAP was left at defaults.

The remaining restrictions are substantive:

- `MN_ratio` changes the number of mid-near pairs; it is not the same intervention as changing a linear spring coefficient. PaCMAP’s bounded mid-near force and MAGNET’s linear force remain different.
- The experiment compares **two tuned pipelines**, not a clean factorial intervention on a common “stiffness” variable.
- The tuning budget needs an explicit ledger: configurations, seeds, selection statistic and selection stage.
- MAGNET at spring **1.0** exceeds spring **0.5** on post-pass Spearman: **0.456 versus 0.449**, and **0.764 versus 0.750**. Thus spring 0.5 is a chosen compromise, not the maximiser of that column.
- No analogous superiority has been established on unseen datasets or on centroid geometry.

**The claim I now doubt most is optimiser-independence in (b), not the measured MAGNET–PaCMAP gap.** The smallest decisive control is:

> Take the five cached stiff starts on each image dataset. Give both backends the **same affinity matrix, coordinates, scale, exaggeration and iteration budget**, reconcile gradient/learning-rate conventions, and report paired changes in 15-NN, Spearman, centroids and KL at 300 iterations. Use a fixed thread count and report numerical accuracy.

That is twenty refinement runs, no new start search. It resolves the present affinity/backend confound. It would not prove universality over optimisers, but it would justify the claimed transfer much more cleanly.

Separately, four runs from one cached start—**tiny/reference scale × exaggeration 1/12**—would settle the currently conflated hand-over explanation.

---

## 2. WHAT IS STILL WRONG

### A. Direct numerical and reporting problems

| Claim/location | Problem | Correction |
|---|---|---|
| Abstract: Spearman changes “within \(-0.02/+0.03\)” | Single-run Fashion UMAP gains **0.035**; repeated stiff MAGNET loses **0.023**. | Use the actual range and identify the table, or say “changes by a few hundredths.” |
| “Full-batch level” is **0.353–0.356 / 0.327–0.330** | Standalone full-batch in `seeds` is **0.347 / 0.323**. Repeated staged Fashion UMAP reaches **0.331**. | Say “competitive with, or slightly above, standalone full-batch 15-NN,” and stop mixing single-run ranges with repeated means. |
| “The pass … takes 0.01–0.02 from ours … in every run” | MNIST spring 0 gains **0.025**; spring 0.1 loses only **0.003**. Fashion spring 0 gains **0.022**; springs 0.5/1 lose **0.023/0.025**. Means cannot establish “every run.” | Specify the stiffnesses and report paired per-run changes. |
| “The two starts with the best centroids (UMAP, PaCMAP)” | True for the listed neighbour starts on MNIST, not Fashion: stiff MAGNET starts at **0.958**, versus **0.879/0.885** for UMAP/PaCMAP. | Restrict the statement to MNIST. |
| Hybrid has the lowest continuity at \(k=15\) | `kscale`: PaCMAP **0.944**, hybrid **0.945**. | “Among the lowest,” or name the actual minimum. |
| Sampled 15-NN is above UMAP by “60–70%” | Five-run image values imply **64% on MNIST**, **54% on Fashion**. Main text/blobs ratios are about **37%/43%**. | Give dataset-specific percentages or remove the range. |
| sklearn’s five layouts are “identical” | `seeds` reports centroid SD **0.001** on both datasets. Rounded zeros elsewhere do not establish identical coordinates. | Say “nearly identical on the reported fidelity metrics”; verify whether PCA initialisation itself varied. |
| Virial text: PCA at scale 31 is “below its equilibrium … by a factor three” | Reported fixed-shape optimum is **10**. Scale 31 is **above**, not below, by about 3.1. | Reverse the statement. |
| Virial text: sampled optima are “85–93” | `virial` reports spring 0.1 at **102**. | Reconcile the prose with the actual runs. |
| Projected refinement costs **0.012** at 300 iterations for \(K=32\) or 128 | `projected`: **0.354−0.342=0.012**, but **0.354−0.340=0.014**. | Report **0.012–0.014**. |
| Adaptive exponents stay within **0.005** on homogeneous data | `adaptive`, MNIST 20k: **0.282 → 0.294/0.291**, changes **0.012/0.009**. | Restrict the statement to the specific tested configurations for which it holds. |
| Persistence table caption defines KL | There is no KL column. | Add the objective trace/column or remove the caption claim. The decreasing-objective assertion should have a visible numerical reference. |

The retained documentation also needs updating. Sections 6.3–6.4 still contain the old categorical claims—“the global column … is the global column of the start,” “it refines, it does not repair,” and the dismissal of centroid variability through the number of centroid pairs. The paper and repository should not tell different scientific stories.

### B. The new theory and constraints need narrower claims

**1. The virial identity is correct; several interpretations are not justified by it.**

For the stated normalised objective, the derivative calculation is valid. However:

- A root is a **stationary dilation**, not automatically the KL-optimal dilation. Establish the minimum by a sign change/objective scan, and distinguish a local minimum from a global minimum along the ray.
- “Equilibrium” here is **conditional on a frozen shape**. It is not the equilibrium scale of the jointly optimised layout.
- The notation mixes a dimensionless multiplier \(s\) with target standard deviations 31, 62 and 93. State explicitly that \(Y_0\) has unit standard deviation, or use a separate symbol for target scale. Do not also use \(s\) for spring stiffness in the same discussion.
- The identity diagnoses radial expansion pressure. It does **not** by itself explain why scale 62 retains more Spearman. That requires information about shape-changing dynamics and the finite optimisation budget.
- Three MNIST sizes yielding an exponent near **0.42** do not validate the hand-over law. You correctly say the identity does not select the hand-over; do not then suggest that the measured dilation exponent validates the operational rule.

Also, “normalised repulsion collapses” the small-scale start is mechanically misleading. A repulsive force does not itself contract the layout. If the total dynamics lose cluster arrangement, say that and show what “collapse” means.

**2. Neither continuation experiment reaches convergence.**

The stiff trajectory changes from **0.480** initially to **0.470** at 300 and **0.446** at 3000, while the objective is reportedly still decreasing. This establishes **finite-time drift**, not the limiting Spearman.

Replace:

> “The full-batch objective does not preserve … at convergence”

with:

> “Retention is not invariant along refinement; measurable drift continues through 3000 iterations, and convergence has not been established.”

The additional loss from 300 to 3000 is **0.024**; the loss from the initial start is **0.034**. Keep those comparisons distinct.

**3. Projection preserves coordinates, not global quality.**

The invariant \(U^\top Y=U^\top Y_0\) follows from re-projection. The text correctly states this locally, but contradicts that qualification with “a refinement that cannot drift.”

- Only \(K=32\) and 128 were tested successfully—not every \(K\ge32\).
- Failure at \(K=8\) does not establish that “the Spearman lives in more than eight modes.” Spearman is a nonlinear function of all pairwise distances, not a spectral coefficient statistic.
- Define “leading Laplacian eigenvectors” as the **smallest-eigenvalue modes** if that is what is intended.
- Include basis-construction time in any cost comparison.

**4. The scaffold is not demonstrated to be drift-free or cost-free.**

The reported example is **0.469 → 0.465**, with 15-NN **0.354 → 0.348** relative to unconstrained refinement. This is favourable retention with a small local cost—not unchanged geometry at no cost.

A finite stress penalty does not enforce exact coarse distances. Even exact coarse distances would not fix pairwise Spearman over individual points. No long scaffolded trajectory is reported.

There is also a reproducibility problem in the weight definition:

> With targets equal to the start’s own distances, \(\nabla S(Y_0)=0\).

A weight calibrated at the hand-over by a ratio involving that gradient is therefore zero or undefined, depending on the convention. State **when and how calibration is performed**, including regularisation, \(\delta\), and whether the coefficient subsequently remains fixed.

Finally, the scaffold gradient is constant within a cell, but **per-point gains can turn it into nonuniform updates**. Specify whether scaffold updates are applied as separate rigid translations or passed through the full-batch optimiser’s gains. The same issue motivated re-projection in the preceding section.

### C. Statistical and protocol issues

- **Report before-SDs and paired change distributions.** `staged_seeds` gives before-means and after-SDs. That is insufficient to support “in every run” or to attribute centroid variance quantitatively to the start rather than refinement. The saved paired layouts should make this cheap.
- **Stop using “three standard deviations” as a significance argument.** The sklearn comparison needs uncertainty on the difference, with exact unrounded values. Its near-determinism does not remove uncertainty in MAGNET or establish cross-dataset generalisation.
- **Five runs are adequate for the large Spearman contrast, not for elaborate distributional claims.** The central PaCMAP comparison does not require indiscriminate additional seeds. OpenTSNE has only **two** repeated staged runs, however; do not describe the entire staged panel as five-run evidence.
- **Run variability and evaluation-sample variability are different.** Fixed evaluation samples are appropriate for paired comparisons. They do not quantify uncertainty over query/sample selection. The millions of pair distances are not independent observations.
- **Give the tuning ledger.** Fashion is held out from the stated PaCMAP selection, but it was already part of MAGNET’s development. It is not an untouched validation dataset for the overall procedure.
- **Define text geometry per method.** “Cosine for every method that accepts one” leaves methods that do not accept it on potentially different input geometry. State row normalisation, distances used by each method, and evaluation geometry. The News20 failure is important enough to require an unambiguous protocol.
- **Call label accuracy transductive if the full dataset was embedded before classifier cross-validation.** It is not an out-of-sample embedding test.

### D. Timing and implementation claims

The default-pipeline timing claim is now defensible. Do not undo that by demanding or implying equal optimisation work:

- The idle-machine comparison **10.9 versus 8.4 versus 93.5 seconds** is a legitimate comparison of those configured pipelines.
- It is not a controlled backend speed comparison, nor a comparison at matched objective value or a common quality frontier.
- The four-dataset “fastest” statement still relies partly on shared-machine, one/two-run measurements. State that; the five-run image table does not validate all four datasets’ timing rankings.
- Scaffold timings need setup costs and repeated measurements before “no extra time” is a claim.
- The exact-sum check validates repulsion at selected layouts. It does not validate the complete gradient, every scale, or optimisation trajectories. A small full-gradient/finite-difference test remains useful.
- The scale-62 preset also changes FFT box width. With **10–14%** force RMS error reported for width 1.5 on the checked layout, do not attribute the preset’s quality change solely to scale. Test the preset at its own operating coordinates against the finer grid.

Two appendix statements remain mathematically or computationally wrong:

1. **Removing `pow`:** \(b=0.6\) is not a half-integer or integer. Setting \(b_r=1\) removes the general power from the **repulsive evaluations**, not necessarily the entire force loop.

2. **Fixed-force equilibrium:** the appendix calls \(g\) a fixed external force but gives
   \[
   d_{\rm eq}=(b_rg/ab)^{1/(2b)}.
   \]
   That formula balances attraction against a force proportional to \(2b_rg/d\), not a constant force. For constant radial force \(g\), the stated near-field attraction instead gives
   \[
   2ab\,d^{2b-1}=g.
   \]
   Redefine \(g\) consistently or delete this paragraph. Its claimed positive stiffness for every \(b\) depends on the former, different force model.

Also call the first stage **UMAP-style negative sampling with split kernels**, not literally UMAP’s cross-entropy: the two exponents differ, and the negative-sampling effective weighting is not the full cross-entropy weighting. Specify which endpoints negative updates move.

---

## 3. VENUE TIER NOW

| Tier | Assessment |
|---|---|
| **arXiv only** | No longer the appropriate ceiling. The narrow empirical claim is publishable in principle. Correct the factual and mathematical issues before posting. |
| **JOSS / JMLR MLOSS** | Natural software targets, conditional on repository quality, tests, documentation and reproducibility. The paper alone cannot establish those. |
| **TMLR** | **A credible target after revision.** The repeated competing-start experiment and persistence analysis materially strengthen the scientific case. I would still request major revision for the present overinterpretations and protocol gaps. |
| **ECML-PKDD / SDM** | Plausible for a focused empirical/algorithmic paper. More credible than before; external validation remains the main weakness. |
| **AISTATS** | Still weak. The virial identity is useful but elementary; it does not provide a predictive hand-over rule. The projection invariant is standard constrained optimisation, not a substantial statistical theory. |
| **TVCG / EuroVis** | Limited movement. The metric disagreement is relevant, but there is still little evidence connecting the proposed operating point to visualization tasks or interpretability. |
| **NeurIPS / ICML / ICLR** | Still a stretch. The contribution is a measured operating regime and implementation, not yet a broadly validated new principle. Adding small theorems does not change that. |
| **JMLR regular** | Not yet. It needs a more complete, transferable scientific account rather than additional MNIST mechanisms. |

**What moved:** confidence in the middle research tier, especially TMLR. The PaCMAP experiment removes a major objection; the continuation experiments identify early stopping honestly. Neither establishes broad generality.

**The single addition with most upward leverage:** a **frozen-rule, held-out transfer study** on a genuinely new non-image dataset at several sizes, comparing MAGNET, tuned PaCMAP and PCA starts under the same refinement, with time–quality trajectories. Freeze the spring, scale rule and stopping budget beforehand. Test whether the recipe transfers—not whether another sweep can find a favourable operating point.

That would contribute more than another exponent experiment or another MNIST scaffold variant.

---

## 4. THE PAPER AS TEXT

### Structure

**Staging before exponent theory is correct.** The current manuscript still contains too much experimental history and too many partially developed secondary methods.

Recommended main-text sequence:

1. Narrow problem and positioning relative to informative initialisation and staged embedding.
2. Objectives, algorithm, and a short evaluation-protocol description.
3. Repeated staged comparison, including tuned PaCMAP and spring removal.
4. Hand-over controls, finite-time persistence and a short scale interpretation.
5. Metric dependence, failure datasets and runtime.
6. Limitations and conclusion.

Specific edits:

- **Define the metrics before the staged tables.** At present the reader encounters many pages of metric comparisons before the evaluation protocol.
- Make `staged_seeds` the primary comparative table; put the older single-run table in the appendix, retaining only its unique PCA/random/backend rows in the main text.
- Move the detailed twelve-row panels to the appendix; retain a compact main table containing repeated image results and the text/blobs failures.
- Compress the hand-over shopping list into an ablation table.
- Keep neighbourhood-scale and centroid failures in the main text. They constrain the thesis directly.
- Keep the virial identity to roughly half a page in the main text; move derivation details and scale sweeps out.
- **Move projected refinement and the scaffold to an exploratory appendix for this version.** Neither is necessary to establish the staged recipe. The scaffold’s calibration and persistence are insufficiently specified for a new headline contribution.
- Move graph-layout experiments to the appendix unless graph embedding is made a separate research contribution with appropriate competitors.
- Delete the standalone main-text exponent section; retain a short implementation paragraph pointing to the appendix.

### Abstract

**Mostly honest in its central framing, but overloaded and still imprecise.** It no longer rests on “sampling gives global structure.” That is the important correction.

Problems remaining:

- “UMAP’s objective” is too literal for the split-kernel sampled implementation.
- The retention range is numerically inaccurate.
- “Does the same” obscures cross-backend differences.
- The repeated metric ranges and endpoint values make it read like compressed table commentary.
- The implementation parenthesis distracts from the thesis.
- “The implementation … is the fastest CPU method” should identify the sampled mode.
- The tuned PaCMAP result is now more informative than the defaults-only start claim.

Aim for roughly 180–220 words: procedure, two measured endpoints, repeated competing-start result, metric-specific limitation, runtime and dataset failure. The FFT implementation details and exponent split need not appear.

### Sentences reviewers will quote back

These are in the current version, not merely inherited from the previous reviews:

- **“Leaves their pairwise Spearman ordering intact.”** Directly contradicted by `staged_seeds`.
- **“The balance of the two forces is the one axis that separates…”** Too absolute, particularly in a paper that changes kernel shape, affinities, sampling and schedules.
- **“Ours … drops the exaggeration, which is what lets a non-linear start’s pairwise Spearman survive.”** The experiment combines scale and exaggeration changes.
- **“The full-batch objective does not preserve … at convergence.”** Convergence was not reached.
- **“A refinement that cannot drift.”** Only selected coordinates are invariant.
- **“A drift-free pass cheaper and better on 15-NN than the projection.”** Finite-horizon retention, incomplete timing comparison, no invariance guarantee.
- **“The hybrid takes the top of both columns.”** Only under a restricted choice of configurations and metrics.
- **“The global structure it keeps is the start’s … that structure is the attraction’s.”** Replace ownership language with the measured conditional effect: adding the spring supplies the observed pairwise-Spearman advantage in this pipeline.

The conclusion should say **“predominantly inherited,” “tested neighbour-embedding starts,” and “qualitative transfer to openTSNE.”**

---

## 5. BEFORE ARXIV

Estimates are hands-on work with the existing scripts and cached layouts; machine time is additional.

| Rank | Action | Estimated work |
|---|---|---|
| **1** | Correct the numerical contradictions and remove unsupported invariance, convergence and universal-ranking language. Fix the virial scale direction/units and appendix force balance. | **½–1 day** |
| **2** | Produce a canonical run manifest: configuration, cached-layout ID, graph/affinity ID, evaluation sample, threads, preprocessing and timing boundaries. Publish the PaCMAP tuning ledger and distinguish exploratory from final runs. | **½–1 day** |
| **3** | Regenerate paired pre/post summaries, including before-SDs and change intervals. Reconcile “identical” sklearn runs and report small SDs without rounding them misleadingly to zero. | **½ day** |
| **4** | Run the common-affinity backend control. Add the four-cell scale/exaggeration diagnostic if retaining causal language about the hand-over. | **½–1½ days**, plus several machine-hours |
| **5** | Fix scaffold calibration and update semantics. Either move it to an explicitly preliminary appendix, or add paired long trajectories with setup-inclusive timings. | **2–3 hours** to scope/move; **½–1 day** to validate further |
| **6** | Add a small complete-gradient reference test; check the coarse-grid preset at its actual scale. Clarify text geometry and transductive label evaluation. | **½–1 day** |
| **7** | Shorten and reorder the manuscript; synchronise the abstract, captions, conclusion and repository documentation. Remove stale second-draft interpretations. | **½–1 day** |
| **8** | Verify clean installation, pinned versions, benchmark commands, cached-result provenance and reproducibility claims. | **½–1 day** |

Items 4–6 are not all mandatory if the corresponding claims are narrowed or the exploratory material is removed. The preprint need not await a broad new benchmark.

**I would explicitly leave out before arXiv:**

- Further MNIST searches for an optimal spring or scaffold.
- A new adaptive-exponent feature or an attempted universal exponent theorem.
- Jointly learned scaffolds and residual-budget variants.
- More million-point synthetic timings.
- A full rerun of every baseline merely to increase seed counts.
- A visualization user study.
- uwot benchmarking, **provided the speed claim remains explicitly panel-restricted**.
- Any claim of convergence, drift-free global preservation, necessary hand-over conditions, or universal best-start status.

The paper to post is a **reproducible, metric-specific staged-refinement study with a strong bounded competing-start comparison**. The new evidence supports that paper. It does not yet support a general solution to the local/global embedding trade-off.