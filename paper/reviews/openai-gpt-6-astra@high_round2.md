## 1. THE THESIS

### (a) “A sampled pass followed by a short full-batch pass leads both axes under three hand-over conditions.”

**The useful empirical result survives; the leadership claim and the necessity of the conditions do not.**

What survives is specific: a mature, spring-conditioned start can recover approximately full-batch **15-NN overlap** while retaining much of its **pairwise-distance Spearman**.

- MNIST, staged table: stiff-spring SGD goes from **0.178 / 0.459 to 0.356 / 0.442**. That is a substantial local improvement for a comparatively small global decrease.
- Fashion-MNIST: **0.193 / 0.774 → 0.329 / 0.753**.
- The five-run results support the final operating point: **0.353 ± 0.001 / 0.455 ± 0.014** on MNIST and **0.328 ± 0.001 / 0.750 ± 0.001** on Fashion.

But “leads both axes” is narrower than the prose admits:

1. **The strict local lead over sklearn is not established.** MNIST is **0.353 versus 0.350**, Fashion **0.328 versus 0.327**; sklearn still has only two and one runs, respectively, and is absent from the five-run table.
2. **It does not lead local fidelity generally.** In the neighbourhood-scale table, openTSNE wins at:
   - \(k=5\): **0.375 versus 0.351**;
   - \(k=50\): **0.327 versus 0.319**;
   - \(k=100\): **0.329 versus 0.313**.
   
   Hybrid also loses on trustworthiness: MNIST five-run **0.960 versus 0.968** for openTSNE.
3. **It does not dominate the staged baselines.** On Fashion, **PCA → refinement scores 0.315 / 0.799**, versus stiff-SGD → refinement at **0.329 / 0.753**. This is a meaningful Pareto alternative. Its final output is a nonlinear neighbour embedding; excluding *linear* references does not exclude this staged method.
4. **It does not lead both columns across datasets.** News20: hybrid **0.181 / 0.073**, sklearn **0.192 / 0.232**, MAGNET full-batch **0.184 / 0.110**. Blobs: hybrid **0.124 / 0.684**, versus separate MAGNET maxima **0.126** and **0.690**.
5. **It does not preserve the maximum global score available from its own starts.** The stiff starts score **0.459 / 0.774** before refinement; the MNIST spring sweep reaches **0.478** at stiffness 1.0.

The three hand-over conditions are **a successful configuration, not established necessary conditions**:

- Scale 31 is not uniquely “the optimiser’s own scale.” In the knob experiment, scale 62 improves both reported scores: **0.353 / 0.413 → 0.357 / 0.424**, at higher cost.
- The rule \(31\sqrt{n/70000}\) is not validated by two image datasets both containing 70k points.
- Failure with springs of 0.1 or 0.5 does not establish that *any* residual spring fails.
- The no-exaggeration condition needs a quantitative controlled comparison, not an explanatory assertion.

**Defensible wording:** “On the two image datasets, scale-matched refinement reaches competitive 15-NN overlap while retaining most of the start’s pairwise-distance correlation.”

### (b) “The pass is generic across starts and optimisers.”

**This is the strongest surviving claim, provided “generic” means the tested, sufficiently developed neighbour-embedding starts.**

The common-refinement experiment is informative:

- Five neighbour-embedding starts land within **0.353–0.356** on MNIST and **0.327–0.330** on Fashion.
- The ordering of their pairwise Spearman scores largely survives.
- PCA and random starts identify a limitation of this particular short refinement schedule.

However:

- These are **single-run staged comparisons**. The five-run native-pipeline table does not establish repeatability of the cross-start result.
- Cross-optimiser transfer is qualitative, not numerical equivalence. MAGNET SGD → openTSNE gives **0.346 / 0.428**, versus MAGNET refinement **0.356 / 0.442** on MNIST; Fashion is **0.322 / 0.743 versus 0.329 / 0.753**.
- That comparison changes **affinities as well as optimiser**: perplexity-30 affinities versus MAGNET’s fuzzy 15-NN graph. It demonstrates that another pipeline can exploit the start, not that the backends are interchangeable under an identical objective.
- “It refines, it does not repair” is too categorical. PCA improves from **0.010 to 0.315** on MNIST and **0.018 to 0.315** on Fashion. That is substantial neighbourhood reconstruction. Random failure at a large initial scale, without exaggeration, establishes failure of that schedule—not an impossibility of repair.

**Verdict:** conditional cross-start transfer survives; backend independence remains incompletely controlled.

### (c) “The global gain is entirely the start’s, and MAGNET’s stiff-spring SGD is the best neighbour-embedding start.”

**The first clause should become “predominantly”; the second should become “best among the tested default starts on the two image datasets.”**

The pass is not globally inert:

- Fashion UMAP: **0.572 → 0.607**, a gain of **0.035**.
- Fashion PaCMAP: **0.582 → 0.599**, a gain of **0.017**.
- MAGNET stiff start: losses of **0.017** on MNIST and **0.021** on Fashion.

The incremental spring advantage is also attenuated:

- MNIST, stiff versus default spring: the pre-pass advantage is **0.459 − 0.418 = 0.041**; post-pass it is **0.442 − 0.414 = 0.028**.
- Fashion: **0.045** before, **0.037** after.

Thus the start supplies the ordering and most of the advantage, but the second stage changes its magnitude.

“Best global structure” remains metric-dependent. On MNIST:

- Stiff SGD has pairwise Spearman **0.459**, above UMAP’s **0.296**.
- Its centroid Spearman is **0.524**, far below UMAP’s **0.752**.
- After refinement those centroid scores are **0.492 versus 0.681**.

The stiff spring does not simply “place the clusters better.” It buys one distance statistic partly at the expense of class-centroid geometry.

Finally, “best start” is a defaults-only comparison against competitors after substantial development of MAGNET’s spring and schedule on these datasets. PaCMAP supplies the borrowed mid-near mechanism but receives no comparable tuning. News20 already defeats any dataset-independent version of the claim.

**The smallest direct test of the claim I doubt most—MAGNET-specific start superiority:** give PaCMAP a small, explicit mid-near-strength sweep, matched to the tuning budget used for MAGNET’s spring, then apply the identical MAGNET refinement. Use five independent starts per setting, recording paired pre/post 15-NN, pairwise Spearman and centroid Spearman. Freeze the setting selected on MNIST before evaluating Fashion. If a tuned PaCMAP start reaches the same frontier, the contribution is the staged recipe and implementation, not a uniquely superior MAGNET start.

Two particularly cheap diagnostics should accompany that test:

- Continue the saved Fashion PCA start to 600 and 1,200 refinement iterations: **0.315 / 0.799** is too competitive to dismiss after one budget.
- Continue a saved stiff-SGD start beyond 300 iterations and report Spearman, centroids and objective—not just 15-NN. Global retention through early stopping is a legitimate recipe, but is not evidence that the full-batch objective preserves global geometry at convergence.

## 2. WHAT IS STILL WRONG

### A. Direct numerical or textual contradictions

| Claim | Evidence contradicting or qualifying it | Required correction |
|---|---|---|
| “The hybrid leads at \(k=5\) and 15.” | Neighbourhood-scale table: openTSNE **0.375**, hybrid **0.351** at \(k=5\). | Hybrid leads only at \(k=15\) in that table. |
| Spring 0.5 is “never below 0.1 on any column of any data set.” | Staged MNIST centroid Spearman after refinement: **0.492 versus 0.617**. | Name the metrics and datasets for which the statement holds. |
| MAGNET is “3–9 times faster on every graph.” | Cora: **0.4 s versus 0.4 s**. | Report a tie at displayed precision; give finer timings if needed. |
| Five-run timings are “10–25% below” the earlier timings. | PaCMAP MNIST **30.5 versus 24.8 s**, Fashion **30.2 versus 24.2 s**: approximately **23–25% higher**. | Investigate configuration/runtime differences; do not attribute every change to machine idleness. |
| openTSNE’s audited **102 s** comprises the listed phases. | **24 + 2 + 12 + 76 = 114 s**. | Identify this as a different audit run, or provide an internally summing breakdown from one run. |
| The converse optimiser is “four to five times slower.” | Staged table: **55.2/11.8 = 4.68**, **44.1/7.7 = 5.73**. | Approximately 4.7× and 5.7×, under the stated differing-affinity protocol. |
| UMAP \(a\approx1.58,b\approx0.9\) gives \(d_k\approx0.62\). | \(1.58^{-1/1.8}\approx\mathbf{0.776}\). | Correct the crossover and any calculations using it. |
| Repulsion optima “sit at or above … \(m/2\).” | The stated 3-D range includes **1.0**, below **1.5**. | Remove the universal inequality. |
| Dimension oracle gains “7–12%” on mixed-dimension data. | Adaptive table, second synthetic dataset: **0.421 → 0.441**, approximately **4.8%**. | State the baseline and dataset-specific gains. |

Also fix the UMAP timing sentence. **Setting** `random_state` forces the seeded serial optimisation path; **omitting** it allows parallel execution. The current relative clause says the opposite.

The adaptive-exponent prose still lacks a declared baseline policy. On COIL-20, “local” is **0.701**, versus **0.694** at \(b=0.5\), **0.717** at \(b=0.6\), and **0.772** at \(b=1\). “Gains 0.01–0.06” is not interpretable without saying which fixed policy is being replaced.

### B. The main mechanistic statement still confounds objective and estimator

The opening claim that “the estimator of the repulsion fixes which side wins” is not established.

You change:

- the sampled effective objective;
- repulsion normalisation;
- attraction and repulsion kernels;
- the mid-range term;
- optimisation dynamics and schedule;
- layout scale.

UMAP cross-entropy repulsion and normalised t-SNE repulsion are not merely sampled and full evaluations of one force. For the Cauchy case, their scalar magnitudes are proportional to

\[
F_{\mathrm{UMAP,rep}}(d)\propto \frac{1}{d(1+d^2)},\qquad
F_{\mathrm{tSNE,rep}}(d)\propto \frac{d}{Z(1+d^2)^2}.
\]

They share an asymptotic tail, not the complete force law. Consequently, adding another repulsive term is not generally “a t-SNE with another attraction–repulsion ratio.”

Your own ablation is especially damaging to the estimator-only narrative: removing the spring gives SGD Spearman **0.285**, below UMAP’s **0.297** and MAGNET full-batch’s **0.348**. Sampling alone does not supply the claimed global advantage.

Define both objectives explicitly, including:

- normalisation of the fuzzy graph for the t-SNE objective;
- negative-sampling distribution and endpoint updates;
- degree-dependent weighting;
- clipping/regularisation;
- the exact spring and its schedule.

The current background presents one force family and then switches objective without a sufficiently explicit mathematical boundary.

### C. Statistical evidence still misses the decisive comparisons

1. **The closest local competitor is missing from the five-run table:** sklearn t-SNE. Add it or stop claiming a strict local win.
2. **The stiff SGD start itself is missing from that table.** The five-run SGD row uses the default spring. You therefore lack the paired evidence needed to attribute the hybrid’s centroid variance specifically to the stiff first stage rather than to first-stage/second-stage interaction.
3. **The central staged table is single-run.** Repeat the important staged rows, not all twelve native methods indiscriminately.
4. **“Seeds are numba’s” is not a reproducibility specification.** Explain how worker RNG streams are initialised. Otherwise call these five independent unseeded runs. A shared integer seed does not pair Hogwild trajectories.
5. **Report uncertainty on differences, not “many standard deviations.”** The large Spearman advantage over UMAP is not in doubt; the small local-ranking claims require the correct comparison. For perspective, an approximate five-run \(t\)-interval is:
   - hybrid MNIST Spearman: **0.455 ± 0.017**;
   - hybrid MNIST centroids: **0.509 ± 0.093**.
   
   These are uncertainty estimates for the mean under the run protocol, not cross-dataset generalisation intervals.
6. **Do not print SD \(0.000\)** where the intended message is nonzero but small dispersion.
7. Graph results have three runs but no spread; adaptive results have “2–3 seeds” without row-specific counts or spread. Neither supports fine rankings confidently.

The differing same-seed Hogwild starts in the supplied documents—Spearman **0.459 versus 0.428**, centroid **0.524 versus 0.594**—are not necessarily contradictions. They demonstrate why cached-layout identifiers and genuinely paired pre/post measurements are necessary.

### D. Timing fairness is improved, not finished

The end-to-end default comparison is legitimate. The claimed decomposition into algorithmic/backend speedups is not yet controlled.

- **90 Annoy neighbours versus 15 NN-descent neighbours** does not isolate a 20× neighbour-search improvement. Match neighbour count, metric and recall for that claim.
- A uniform \(k=15\) affinity experiment is not the same as passing MAGNET’s weighted affinity matrix to openTSNE.
- **“10× per FFT iteration”** needs matched coordinates, objective, gradient convention, interpolation accuracy and grid policy. Layout extent itself changes cost strongly in your scale experiment.
- The standard pipelines use different iteration schedules: MAGNET full-batch **600**, hybrid refinement **300**, openTSNE default **750**. Report time–quality trajectories rather than treating default completion time as equal optimisation work.
- All headline timings should come from an idle machine with consistent boundaries. The PaCMAP reversal shows that the old and new timing panels cannot simply be interchanged.
- Add a small exact-gradient/finite-difference reference check for the FFT backend. Matching one quality statistic does not establish equal numerical accuracy.

The statement **“SGD is fastest among the listed CPU implementations on the four tabulated datasets”** is supported. It is not equivalent to fastest CPU neighbour embedding generally, nor to the hybrid being fastest. uwot remains an important missing native competitor; its absence is a limitation of coverage, not a logical refutation of the explicitly panel-restricted statement.

### E. Metrics and graph protocols still require definitions

- Specify whether 15-NN ground truth searches the full dataset or only the 2,000-point evaluation sample; whether it is exact; and whether evaluation reuses the training ANN graph.
- Define trustworthiness/continuity reference populations, self-exclusion, tie handling, and label-accuracy train/test or leave-one-out protocol.
- Separate embedding randomness from evaluation-subsample randomness.
- For news20, document how every implementation receives equivalent cosine geometry. The sparse TF-IDF speed paragraph and the SVD-300 main benchmark concern different representations and must not be read as one fidelity comparison.
- Define graph edge preservation mathematically: layout neighbourhood size, degree normalisation, macro/micro averaging, weights and isolates.

The primary direct-adjacency UMAP comparison is now interpretable. The retained **`metric="precomputed"`** comparison is not automatically equivalent because minimum degree exceeds 15. Treating entries as distances and constructing a kNN fuzzy graph can change both the edge set and weights. Either demonstrate equivalence or remove those secondary **5.0 versus 11.7 s** numbers.

Finally, centroid Spearman is deterministic conditional on a layout. “Only 45 pairs” does not make the observed loss disposable measurement noise. The pairs share ten centroids and are dependent; the large run-to-run variation is a property of the layouts under this metric. The default SGD row’s **0.002** SD shows that the statistic need not fluctuate by **0.075**.

### F. Remaining mathematical problems

- **“Bounded influence iff \(p\le1\)” remains wrong without qualification.** For \(0<p<1\), the score \(p d^{p-1}\) diverges at zero. Say “bounded in the large-residual tail,” or restrict the claim appropriately.
- The equilibrium-shell paragraph still says **“every satisfiable neighbour”**, introduces an undefined \(g\), and states a near-field balance as though it described the full configuration. Define \(g\), state the approximation, verify that the proposed equilibrium lies inside both relevant crossover scales, and remove the many-point stability implication.
- Sparse measurements at a few intrinsic dimensions do not establish a discontinuous **“step at \(q=m\)”**. The figure caption is stronger than the qualified main text.
- Exponent sweeps still need to distinguish changes in shape, crossover and force amplitude.
- Splitting exponents does not itself remove `pow` for arbitrary exponents. The computational simplification applies to particular choices, such as the default half/integer exponents.

The revised tail diagnostic addresses the original mismatch. What it establishes is the conditional power law on fixed, appropriately rescaled layouts—not an explanation of fidelity optima at the working scale.

## 3. VENUE TIER NOW

These judgments assume correction of the factual and protocol statements above; they are not assessments of an unseen software repository.

| Tier | Assessment now |
|---|---|
| **arXiv only** | No longer the appropriate ceiling. There is a coherent empirical research claim, but this version should not be posted unchanged. |
| **JOSS / JMLR MLOSS** | Still natural targets. JOSS depends mainly on software eligibility, testing, documentation and maintenance evidence. MLOSS is plausible if the performance and reproducibility claims survive implementation-level checks. |
| **TMLR** | **Now a credible research target**, rather than principally a software paper looking for a scientific thesis. The staged controls materially improve the case. Current obstacles are claim scope, development-set tuning and incomplete causal controls. |
| **ECML-PKDD / SDM** | Plausible, but borderline on novelty and external validation. A focused empirical paper is more competitive than the current multi-contribution package. |
| **AISTATS** | Still weak. The exponent analysis is heuristic, and the staging result lacks a sufficiently general explanatory or predictive account. |
| **TVCG / EuroVis** | Little movement. Better embedding metrics are not a visualization-task evaluation. The disagreement between pairwise and class-level geometry needs task relevance, not merely another leaderboard. |
| **NeurIPS / ICML / ICLR** | Still a stretch. The evidence supports a useful combination and operating regime; it does not yet establish a general new principle. |
| **JMLR regular** | Not currently a realistic target without a substantially more complete scientific account and broader validation. |

**What moved:** the middle research tier, especially TMLR. Software-tier suitability is largely unchanged; top-tier novelty is largely unchanged.

**The one addition with the highest upward leverage:** a **held-out transfer experiment for a frozen hand-over rule**, rather than another leaderboard. Use an unseen non-image, non-cluster-only dataset at several sample sizes; freeze spring, scale rule and refinement budget beforehand; compare MAGNET and a tuned competing start using shared affinities and both backends. The question is whether the rule predicts a useful region without retuning.

That would test the proposed contribution. More exponent sweeps or another synthetic million-point timing would not.

## 4. THE PAPER AS TEXT

### Structure

**Staged optimisation before exponent theory is correct.** But the main empirical evidence should also precede the extended exponent discussion.

Recommended order:

1. Problem, related-work positioning and narrow claim.
2. Explicit objectives and minimal algorithm.
3. Staged controls and hand-over experiments.
4. Main results, uncertainty, metric-scale analysis and failure cases.
5. Brief implementation/performance section.
6. Limitations and conclusion.
7. Appendix: exponent analysis, adaptive policies, detailed engineering and supplementary benchmarks.

The current algorithm section interrupts the thesis with cache-line ownership, sparse dot-product engineering and `transform()`. Those are relevant to a software paper, but they do not belong before the central experiment.

**Move or shorten:**

- Move most of the exponent theory and all per-point-exponent experiments to an appendix; retain a short summary of what they do and do not explain.
- Move sparse-search internals, core-cycle counts, owner-mode implementation details and transform timings to the implementation appendix.
- Compress “Nothing blends inside one loss” to an ablation table. The failed configurations are useful; the universal heading is not.
- Keep the global-strata and neighbourhood-scale results in the main text. They directly limit the thesis.
- Treat graph layout as supported functionality unless you add graph-layout competitors.
- Replace the related-work citation inventory with actual comparative prose. Explain the difference from informative initialisation, PaCMAP’s staging, multiscale affinities and CNE.

### Abstract

**Not yet honest enough, and overloaded.** The measured headline operating points are defensible. The opening mechanism, cluster-placement language, genericity and leadership claims exceed the evidence. The closing PCA qualification does not repair those earlier assertions.

A suitable core would be:

> We study a staged neighbour-embedding procedure: construct a layout using sampled optimisation with transient mid-range attraction, then refine it with a scale-matched t-SNE-style objective. On MNIST and Fashion-MNIST, a 300-iteration refinement pass substantially increases 15-NN overlap while retaining most of the start’s pairwise-distance Spearman correlation. Similar refinement benefits occur for UMAP and PaCMAP starts and transfer qualitatively to openTSNE. MAGNET’s stiff-spring start provides the highest pairwise correlation among the tested default neighbour-embedding starts, but not the best class-centroid geometry. The hybrid reaches 0.353/0.455 and 0.328/0.750 in 15-NN overlap/Spearman over five runs, at approximately 1.3× and 1.2× UMAP’s runtime. The result is not universal on text, and PCA remains stronger on pairwise-distance correlation on the real datasets.

Put the exponent analysis in one final subordinate sentence, if anywhere in the abstract.

### Sentences a reviewer will quote back

- **“The estimator of the repulsion fixes which side wins.”** Objective, normalisation, spring and optimiser are confounded.
- **“The all-pairs gradient of t-SNE gives tight neighbourhoods and no arrangement.”** Universal, undefined and inconsistent with informative-initialisation literature.
- **“Without touching the arrangement.”** UMAP centroids **0.752 → 0.681**; MAGNET stiff **0.524 → 0.492**.
- **“The sequence works under three conditions … and fails outside them.”** Only selected violations were tested.
- **“Nothing blends inside one loss.”** A finite negative ablation is not an impossibility result.
- **“The global column after the pass is the global column of the start.”** Fashion UMAP changes by **+0.035**; PCA by **−0.074**.
- **“This term is all of the SGD mode’s global structure.”** Spearman without it is **0.285**, not zero; at most it supplies the observed advantage over particular baselines.
- **“The implementation … runs the whole sequence in the time UMAP takes.”** The five-run MNIST comparison is **10.9 versus 8.4 s**.

The title’s division into “global” and “local” passes also needs softening. “Staged neighbour embedding with scale-matched refinement” describes the evidence without pretending each stage exclusively owns one type of structure.

## 5. BEFORE ARXIV

Estimates are hands-on work assuming the existing scripts and cached layouts are usable; machine time is additional.

| Rank | Action | Estimated work |
|---|---|---|
| **1** | Correct the direct contradictions, crossover arithmetic, objective definitions and universal language. Restrict leadership to named metrics, datasets and configurations. | **½–1 day** |
| **2** | Establish one canonical result manifest: configuration, run ID, saved-layout ID, thread count, preprocessing, timing boundary and evaluation sample. Regenerate prose numbers and tables from it. Clearly separate old exploratory timings from final timings. | **½ day** |
| **3** | Complete the decisive uncertainty evidence: five-run sklearn rows; stiff-SGD pre/post metrics; repeated UMAP/PaCMAP → common refinement. Report paired changes and retain individual runs. | **½–1 day**, plus several machine-hours |
| **4** | Resolve the performance audit: summing phase timings, PaCMAP’s reversal, common-affinity backend comparison and a small exact-gradient check. If matching numerical work is not feasible, remove the 20× search and 10× FFT attribution claims. | **1–2 days** |
| **5** | Run the small competing-start tuning test from §1. Include the cheap PCA continuation and longer-refinement trajectories. If omitted, explicitly withdraw any claim stronger than “best among the tested default starts.” | **½–1 day**, plus several machine-hours |
| **6** | Define evaluation metrics completely; fix graph edge-preservation reporting; remove or segregate the nonequivalent precomputed-graph timing. State which text representation each result evaluates. | **½ day** |
| **7** | Reorganise and shorten: staged evidence first, theory/engineering mostly supplementary, substantive related work, rewritten abstract and conclusion. | **½–1 day** |
| **8** | Check artifact claims: installation, benchmark commands, version pinning, owner-mode reproducibility tests and the memory-safety assumptions of concurrent SGD updates. | **½–1 day**, potentially longer if defects emerge |

**Conditional priority:** add uwot before making an unrestricted CPU-speed claim. Otherwise retain the panel-qualified claim and identify the omission prominently. That is approximately another **½–1 day**, not a reason to delay a narrowly framed preprint indefinitely.

**What I would leave out before arXiv:**

- A full new twelve-method benchmark on many datasets.
- Another million-point synthetic scaling result.
- Further attempts to rescue a universal optimal-exponent theory.
- New adaptive-exponent features.
- A visualization user study.
- Additional GPU comparisons, unless GPU competitiveness remains a headline claim.
- Claims that all three hand-over conditions are necessary, that the pass preserves arrangement unchanged, or that MAGNET has solved the local/global tradeoff.

The preprint should establish **a useful, reproducible staged operating regime**, with its metric-dependent benefits and failures stated alongside the headline results. The broader predictive theory and held-out transfer study are the next paper-quality increment, not prerequisites for disseminating that narrower result.