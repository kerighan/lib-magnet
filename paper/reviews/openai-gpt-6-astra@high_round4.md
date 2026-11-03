## 1. THE THESIS

**The narrow empirical thesis survives. “Leads both axes,” “generic optimisers,” and “best start” still require restrictions. The new controls resolve the previous backend/affinity confound and the scale/exaggeration confound for the tested configurations; they do not establish universality.**

### Abstract and contributions: claim audit

| Claim | Verdict |
|---|---|
| The sequence reconstructs neighbourhoods while retaining most of a spring-conditioned start’s pairwise Spearman | **Supported on MNIST and Fashion-MNIST.** This is the defensible thesis. |
| Three hand-over conditions explain the result | **A successful configuration is supported; three necessary conditions are not.** In particular, omitting exaggeration is not necessary at scale 31. |
| The pass works across the tested neighbour-embedding starts | **Supported.** Not across arbitrary starts, and only under this finite-budget schedule. |
| The result transfers between MAGNET’s and openTSNE’s implementations under common affinities | **Supported at the level of the reported metrics.** “The same layout” is not established by those metrics. |
| The global column is predominantly inherited from the start | **Supported for pairwise Spearman.** “Entirely” would be false. |
| Adding the mid-range attraction supplies MAGNET’s observed Spearman advantage | **Supported within this pipeline.** This is not an isolation of sampling, kernel choice, pair construction or force law. |
| MAGNET exceeds the selected stiff PaCMAP pipeline by approximately 0.10 after refinement | **Supported for the measured configurations.** “Same tuning budget” and a clean “\(2\times2\) design” overstate the control. |
| The reported shipped endpoints and image-dataset runtime ratios | **Supported by `seeds`.** They are configured-pipeline comparisons, not equal-work comparisons. |
| PCA leads; pairwise and centroid geometry disagree; MNIST centroids deteriorate | **Supported for the evaluated methods.** Restrict centroid rankings to methods with centroid measurements. |
| The sampled mode is fastest across the four tabulated datasets | **Supported as an observed panel result.** The text/blobs rankings have much weaker timing replication than the image results. |

Thus contribution 1 is mostly supported, with narrower backend and hand-over wording; contribution 2 supports the measured decomposition but not matched tuning or a common-factor factorial interpretation; contribution 3 is supported within its stated timing scope; contribution 4 is supported within the centroid-evaluated panel. The secondary claim about removing `pow` from **repulsive evaluations** is now correctly stated.

### (a) Does the sequence lead both axes?

**Not across the paper’s own tested configurations. It supplies a useful operating point.**

The repeated staged table establishes the central effect:

- MNIST, \(\kappa=0.5\): **15-NN 0.179 → 0.354; Spearman 0.463 → 0.449**.
- Fashion: **0.192 → 0.328; 0.773 → 0.750**.

The attacks on “leads both” are decisive:

1. **It does not lead local preservation among staged alternatives.** Fashion UMAP → refinement reaches **0.331 ± 0.001**, versus **0.328** for the shipped MAGNET start.

2. **It does not lead every local metric or neighbourhood scale.** In `kscale`, openTSNE beats the hybrid at \(k=5\), **0.375 versus 0.351**, and at \(k=100\), **0.329 versus 0.313**. In `seeds`, MNIST trustworthiness is **0.968 versus 0.960**.

3. **It does not maximise pairwise Spearman among the tested outputs.** Fashion PCA → refinement gives **0.315 / 0.799**, versus **0.329 / 0.753** for the stiff sampled start in `staged`. That output is itself a nonlinear neighbour embedding; excluding linear references does not exclude it. Within MAGNET, \(\kappa=1\) reaches **0.764**, above the shipped **0.750**.

4. **Refinement sacrifices some of the stiff start’s global score.** Fashion **0.773 → 0.750**. Consequently, the hybrid cannot dominate its own first stage on both axes.

5. **The result does not transfer uniformly across datasets.** News20: hybrid **0.181 / 0.073**, Barnes–Hut **0.192 / 0.232**. Blobs: hybrid **0.124 / 0.684**, versus separate MAGNET maxima **0.126** and **0.690**.

The hand-over control now supports a substantially cleaner explanation:

- Without exaggeration, changing tiny scale to 31 gains **0.019 of 15-NN and 0.041 of Spearman**.
- With exaggeration, the same scale change gains **0.051 and 0.103**.
- At scale 31, adding exaggeration changes the result by **−0.006 / +0.004**.
- At tiny scale, it changes it by **−0.038 / −0.058**.

That is **a scale effect with a substantial scale–exaggeration interaction**, not evidence that exaggeration is irrelevant. I would qualify:

> “a \(2\times2\) diagnostic shows that the scale, not the exaggeration, is what keeps the start’s arrangement.”

Better:

> “In one cached-start diagnostic, scale 31 retains more Spearman with or without exaggeration; omitting exaggeration is not necessary for retention at that scale.”

Likewise, failure of sustained springs at 0.1 and 0.5 supports switching off that spring in the shipped pass, not a necessary prohibition on every nonzero mid-range force.

**Surviving claim:** competitive 15-NN preservation with substantial retention of a strong start’s pairwise Spearman, on the two image datasets.

### (b) Is the pass generic across starts and optimisers?

**Across the tested neighbour-embedding starts: yes. Across two implementations of essentially the same optimisation procedure: yes, to similar measured quality. Across arbitrary starts or optimiser families: no.**

The repeated endpoints occupy narrow local ranges:

- MNIST: **0.352–0.355**.
- Fashion: **0.327–0.331**.

But the schedule does not erase initialization dependence: random starts reach only **0.065 / 0.062** in the single-run table; PCA reaches **0.315**, with substantial Spearman loss.

The common-affinity control resolves the objection previously raised. It establishes that the large earlier backend gap is not reproduced when \(P\), initialization and schedule are controlled. The approximately fivefold timing difference—**6.4–7.5 seconds versus 33–39 seconds**—is useful implementation evidence.

It does **not** establish:

> “openTSNE’s optimiser returns the same refined layout”

or:

> “the two optimisers return the same layout to within the metrics’ resolution”.

Two layouts can have almost identical input–embedding Spearman and 15-NN preservation while differing materially from each other. Moreover, mean differences around **0.0009** are not a bound of **0.001 on every pair of runs**.

Use **“nearly identical reported fidelity scores”**. If retaining “same layout,” supply aligned coordinate residuals and direct cross-layout neighbourhood/distance comparisons. No such extra measurement is needed if the wording is narrowed.

Also distinguish **backend transfer** from optimiser independence: these are two implementations using matched t-SNE momentum/gains conventions, not evidence about Adam, different preconditioning or arbitrary optimisation schedules.

### (c) Is the global gain entirely the start’s, and is MAGNET the best start?

**“Predominantly” survives. “Entirely” does not. The measured MAGNET–PaCMAP advantage survives; universal or tuning-equivalent superiority does not.**

The pass changes the global metric actively:

- MNIST stiff PaCMAP: **0.319 → 0.346**, gain **0.027**.
- Fashion UMAP: **0.572 → 0.605**, gain **0.033**.
- Fashion MAGNET \(\kappa=1\): **0.789 → 0.764**, loss **0.025**.

The MAGNET–stiff-PaCMAP gap actually **shrinks** through refinement:

- MNIST: **0.144 before → 0.103 after**.
- Fashion: **0.149 before → 0.102 after**.

The attraction ablation supports the conditional attribution: removing it reduces post-pass Spearman from **0.449 to 0.301** on MNIST and from **0.750 to 0.588** on Fashion.

The post-pass PaCMAP contrast is not a marginal seed fluctuation: **0.449 ± 0.015 versus 0.346 ± 0.017**, with the stated approximate Welch interval around **[0.080, 0.126]**. That interval concerns the selected configurations, not uncertainty from model selection or transfer to new datasets.

The remaining tuning problem is explicit in your ledger:

> “PaCMAP’s mid-near term, stiffened with the same tuning budget…”

PaCMAP was selected on **pre-pass Spearman**; MAGNET was selected using **post-pass metrics**, with broader development on both datasets. Eight versus six listed configurations is not the important discrepancy; **the selection target and accumulated development differ**. Furthermore, `MN_ratio` changes pair counts, not the coefficient of MAGNET’s linear force.

Calling this a “\(2\times2\) design” while subsequently admitting that it is not a common-stiffness factorial design is unnecessary. Call it a **method-specific strength comparison with repeated runs**.

**The residual claim I doubt most is tuning-equivalent “best start,” not backend transfer.**

**Smallest useful experiment:** send all eight cached PaCMAP sweep starts through the same pass, select using the same predeclared **post-pass** criterion as MAGNET, and repeat the selected setting on fresh starts. Apply that criterion to MAGNET’s cached candidate set too; report \(\kappa=0.5\) separately as the shipped compromise. This directly tests the remaining selection-stage confound without another broad search. It cannot establish universal best-start status.

---

## 2. WHAT IS STILL WRONG

### Factual and interpretive corrections

| Location / claim | Problem and concrete correction |
|---|---|
| Abstract: “\(-0.023\) to \(+0.035\) across eight starts” | This mixes cohorts and misses the Fashion \(\kappa=1\) change **0.789 → 0.764 = −0.025**. Across the eight configurations in `staged_seeds`, changes in dataset-specific means run from **−0.025 to +0.033**. The **+0.035** is from the single-run UMAP result. Identify the table and say **changes in run means**, not a bound on individual runs. |
| Opening of staged refinement: “The sampled mode has the highest pairwise Spearman … and the full-batch mode the highest 15-NN preservation” | False for the displayed benchmark. In `seeds`, hybrid Spearman is **0.455**, sampled **0.413**; Barnes–Hut 15-NN is **0.350**, full-batch **0.347**. Say that these are the respective strengths **of the two unstaged MAGNET modes on the image datasets**. |
| “Sampled mode” results paragraph: “highest pairwise Spearman … on MNIST, Fashion-MNIST and the blobs” | Again false with the hybrid included: **0.413 < 0.455**, **0.724 < 0.750** in `seeds`. Restrict the comparator set explicitly. |
| Limitations: “the full-batch objective does not preserve the start’s Spearman at convergence” | Still present despite the corrected persistence section. `persist` establishes **0.480 → 0.446 through 3000 iterations**, with decreasing objective—not convergence. Replace with “retention drifts through 3000 iterations; the limiting value is unknown.” |
| Hybrid paragraph: “it is the margin of refining on the 15-NN graph that the metric reads, not of the sequence” | The openTSNE-start comparison changes affinities, scale and optimisation budget. It shows that MAGNET’s first stage is **not necessary for the local margin**; it does not isolate the graph as its cause. The continued increase to **0.367 at 3000 iterations** makes budget dependence particularly relevant. |
| Backend paragraph: “the 0.010 / 0.014 … is the effect of its perplexity-30 affinities” | The new control strongly supports affinity choice as the explanation, but the old comparison used different runs/thread conditions. Do not assign those exact historical differences exclusively to affinities without the corresponding paired affinity swap. Also the preceding dataset order is MNIST/Fashion, for which the old differences are **0.014 / 0.010**. |
| “h-NNE … is behind on every other column” | True for the cited MNIST comparison, not across datasets. News20 h-NNE beats sampled MAGNET on Spearman, **0.102 versus 0.059**, and triplets, **0.535 versus 0.529**. Add “on MNIST.” |
| Centroid rankings: “worst … of the panel” / “last of the neighbour embeddings” | Centroids are not reported for the full twelve-row panel. Use **“among methods with reported centroid measurements.”** |
| News20 caption: “cosine” | The setup now correctly documents mixed fitting geometries and Euclidean evaluation. The caption still suggests a common cosine experiment. Label it accordingly. The failure is of these configured pipelines against that evaluation geometry, not a geometry-independent result about text. |
| Appendix: oracle gains “over the best fixed exponent” | In the second synthetic row, **0.421 at \(b=0.5\)** exceeds the quoted baseline **0.417 at \(b=0.6\)**. Say “over the default \(b=0.6\),” or use the actual best-fixed baseline. |

Two smaller technical reporting points:

- The virial derivation now defines \(s\) as the **target standard deviation of a unit-standard-deviation shape**. Make the table caption use that wording rather than “in units of the start’s standard deviation,” which can be read as a multiplier of the original cached coordinates.
- Clarify whether the `persist` values labelled KL include \(\sum p_{ij}\log p_{ij}\). “Edge terms plus \(\log Z\)” alone is the objective **up to an additive constant**, not the complete KL expression given earlier.

### Statistics and timing

- **The five-run backend control and the two-run openTSNE-start row are different experiments.** The former does not upgrade the latter to five independent openTSNE-generated starts. The table is explicit; surrounding summaries must remain so.
- `staged_seeds` still lacks **before-SDs and SDs of paired changes**. Differences of before/after means give the mean change, but not its uncertainty. The assertion that centroid variability is attributable quantitatively to the first stage needs those paired summaries, not just after-SDs.
- Fixed evaluation samples make comparisons consistent, but the reported SDs quantify **embedding-run variability only**. Neither the approximately two million pair distances nor the 45 centroid distances are independent replicates.
- Values displayed as **±0.000** should be labelled as rounded small SDs, not zero variability.
- The shared-machine tables should say **runs**, not uniformly “seeds,” because UMAP is intentionally unseeded.
- The new fivefold backend timing is defensible as **time to similar reported fidelity under the tested settings**. It is not yet a matched-force-accuracy speed comparison: MAGNET’s FFT accuracy is measured, but corresponding openTSNE interpolation error is not. Give both grid settings and timing boundaries; do not describe the result as identical numerical computation.
- The strong image timing conclusions need no further blanket rerun. The four-dataset “fastest” conclusion remains a lower-confidence observed ranking outside the repeated image experiment.

### The repository still contains claims that the paper has withdrawn

The supplied documentation is not synchronized:

- §6.3: **“has the best global structure among the neighbour embeddings”**—not metric- or configuration-restricted.
- §6.5: **“takes 0.01–0.02 from ours”**—false at \(\kappa=0\), where the pass adds **0.025 / 0.022**, and at \(\kappa=1\), where it removes **0.017 / 0.025**.
- §6.5: **“does not preserve the start at convergence”**—convergence unestablished.
- §6.6: **“the sqrt(n) form of the rule is right”**—the stationary-dilation measurements do not validate the operational hand-over rule.
- §6.6: **“\(K\ge32\)”**—only 32 and 128 were tested successfully.
- §6.6: **“a drift-free pass at no extra cost”**—contradicted by the finite-horizon nature of the scaffold, its local cost and approximately ten-second setup.
- §6.7: **“The spring’s effect is not graph diffusion.”** Failure of the tested between-segment heat-flow substitutes does not establish this. Linear spring updates are themselves closely related to Laplacian diffusion; interleaving, stochastic sampling, annealing and normalisation matter. Say the tested substitutes did not reproduce the spring pipeline’s result.

The three new negative experiments are appropriately exploratory. They establish failures of implementations/settings, not impossibility results or categorical mechanism exclusions.

---

## 3. VENUE TIER NOW

| Tier | Assessment |
|---|---|
| **arXiv only** | Not the appropriate ceiling. Post after the short claim-correction pass below. |
| **JOSS / JMLR MLOSS** | Plausible software targets, conditional on installation, tests, documentation and maintained reproducibility. The pasted paper cannot establish those properties. |
| **TMLR** | **Credible submission after correction.** The specific controls previously missing are now present. The principal remaining scientific weakness is transfer beyond the development datasets, not an unresolved backend confound. |
| **ECML-PKDD / SDM** | Plausible for a focused empirical/algorithmic presentation. |
| **TVCG / EuroVis** | Still weaker without visualization-task evidence. Metric disagreements are relevant, but high pairwise Spearman is not itself demonstrated visualization utility. |
| **AISTATS** | Weak fit at present: neither a substantial statistical result nor a predictive optimisation theory. |
| **NeurIPS / ICML / ICLR** | Still a stretch. Controlled assembly of known ingredients plus implementation speed, demonstrated mainly on two related image benchmarks, does not yet establish a broadly transferable principle. |
| **JMLR regular** | Not yet: the scientific account remains too narrow and development-set-dependent. |

**Movement:** increased confidence within the middle research tier, especially TMLR; not a jump to the top tier. The backend control also narrows the novelty correctly: the refinement is not a new optimisation mechanism.

**Highest-leverage addition:** a frozen-rule transfer study on genuinely untouched non-image data at several sizes. Freeze \(\kappa\), the scale rule and the stopping budget beforehand; compare MAGNET, post-pass-selected PaCMAP and PCA starts under the common pass, with time–quality trajectories. This tests the actual unresolved proposition: whether the recipe transfers without another dataset-specific search.

---

## 4. THE PAPER AS TEXT

**Staged optimisation before exponent theory is the right structure.** The remaining problem is experimental chronology overwhelming the argument.

Recommended changes:

1. Define metrics and the essential protocol **before** the staged results.
2. Make `staged_seeds` the primary start-comparison table.
3. Put the common-affinity control and the actual scale × exaggeration experiment into compact tables. These are central evidence and currently buried in prose.
4. Move most of the older single-run table to the appendix, retaining its unique PCA/random comparisons.
5. Replace the hand-over shopping list with a compact ablation table. Keep detailed failed searches in the repository.
6. Keep centroid failures, neighbourhood-scale dependence and News20 in the main text.
7. Reduce the virial discussion to the identity and its explicit non-predictive limitation; move the scan details out.
8. Move graph benchmarking and most exponent discussion to the appendix. The exploratory constraints are already in the appropriate place.
9. Remove correction history from the main narrative—old warm-up mistakes and withdrawn graph-input comparisons belong in an audit/provenance appendix.

**The abstract is mostly honest in direction, but too dense and not fully accurate.** It combines procedure, retention range, backend control, tuned-baseline comparison, endpoint metrics, three limitations and two speed claims. Aim for approximately 200 words. Keep the measured image result, the competing-start result, implementation transfer, runtime and one compact metric/dataset limitation sentence. Remove the misleading retention range and “same layout”; qualify the tuning claim.

Sentences reviewers can still quote back:

> “openTSNE’s optimiser returns the same refined layout five times slower”

Metrics establish similar measured quality, not layout identity.

> “PaCMAP’s own mid-near term, stiffened with the same tuning budget”

The disclosed selection procedures are not matched.

> “A \(2\times2\) design … tests whether the start matters beyond that attraction’s presence”

It compares method-specific tuned pipelines; it does not factor out a common attraction intervention.

> “The sampled mode has the highest pairwise Spearman … and the full-batch mode the highest 15-NN preservation”

Contradicted by the paper’s broader tables unless the comparator set is restricted.

> “the full-batch objective does not preserve the start’s Spearman at convergence”

Contradicted by the manuscript’s own statement that convergence was not reached.

The conclusion should retain **“predominantly inherited,” “tested neighbour-embedding starts,” and “similar reported fidelity under common affinities.”** It should not return to ownership language about “the global structure” or implementation-independent optimisation in general.

---

## 5. GO / NO-GO

**NO-GO literally as written; GO after a small factual and claim-scope patch. No additional experiment is an arXiv blocker.**

Shortest blocking list:

1. **Correct the numerical summary and unrestricted rankings.**  
   Replace abstract **“−0.023 to +0.035 across eight starts”** with **“mean changes of −0.025 to +0.033 across the eight tested neighbour-embedding start configurations”**, or simply “a few hundredths.” Restrict the two “sampled highest / full-batch highest” statements to the intended comparator set.

2. **Replace layout identity with measured-quality agreement.**  
   Change every **“same layout”** claim to **“nearly identical reported fidelity scores under common affinities and matched schedules.”** Keep the approximately fivefold timing result. Soften the exact historical attribution to perplexity-30 affinities.

3. **Remove false tuning equivalence and factorial language.**  
   Replace **“same tuning budget”** with an explicit description of the limited method-specific sweeps and different selection criteria. Rename the start experiment from **“\(2\times2\) design”** to a repeated tuned-start comparison. The measured **0.103 / 0.102** advantage can remain.

4. **Remove the surviving convergence claim and synchronize the linked documentation.**  
   Replace the Limitations sentence with **“Spearman continues to drift through 3000 iterations; convergence and the limiting value are unknown.”** Remove the repository’s corresponding convergence, validated-\(\sqrt n\), universal-\(K\), drift-free/no-cost and categorical “not graph diffusion” claims.

Everything else above is a revision opportunity, not a reason to delay this preprint. The postable paper is a **bounded staged-refinement result with a measured competing-start advantage and a fast implementation**, not a general resolution of local versus global embedding quality.