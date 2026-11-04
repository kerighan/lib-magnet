## Decision: **NO-GO for the exact v3 files**

The remaining release blockers are **the false timing-variability statement and unresolved calibration/companion provenance inconsistencies**—items 5 and 6 below. The central sampled-stage specification gap is closed. No additional benchmark campaign is required.

This is an audit of the supplied source, tables, summaries and captions. Repository contents, installation, tag resolution, rendered page count and PNG contents are not verified.

## 1. CLAIMS

### Abstract

| Claim | Assessment |
|---|---|
| Neighbourhood overlap and pairwise-distance rank correlation favour different operating points “in the configurations we test.” | **Supported at this empirical scope.** This is not an unavoidable local–global trade-off or a theorem about neighbour embeddings. |
| A sampled pass with decoupled kernels and transient mid-range attraction “places the clusters,” followed by 300 full-batch iterations on the same fuzzy graph. | **The algorithmic description is supported; “places the clusters” remains unnecessarily teleological.** The evidence concerns an initial arrangement and inherited distance ranks, not correct cluster locations. **Replacement:** “MAGNET constructs an initial arrangement using a sampled pass with decoupled attraction and repulsion kernels and transient mid-range attraction, then applies 300 full-batch iterations of an FFT-interpolated \(t\)-SNE objective on the same fuzzy graph to refine measured neighbourhood overlap.” |
| Translation increases sampled distance-rank correlation under a sampled cross-community nearest-neighbour guard; fixed contraction follows; execution is label-free, settings development-selected. | **Supported as a description of the procedure and selection process.** It does not promise that the final layout improves evaluation Spearman or preserves full-data 15-NN overlap. Fashion sequence Spearman actually falls from **0.751 to 0.705** after the complete post-process. The revised wording accommodates this. |
| Standalone full-batch MNIST overlap **0.349 versus 0.350**, runtime **4.9 versus 175 s**, graph construction included. | **Supported by the separate quality and timing families.** This compares shipped configurations on one machine, not equal objectives, equal optimisation accuracy or general implementation efficiency. |
| On both image datasets, the unprocessed sequence has higher mean Spearman than UMAP, PaCMAP, openTSNE and Barnes–Hut, with overlap comparable to the \(t\)-SNE baselines and above UMAP/PaCMAP. | **Supported.** The comparator restriction matters: this is not the best Spearman in the panel. PCA is higher on both image sets; PCA followed by the pass is also higher on Fashion. |
| Under identical post-processing, highest observed four-development-set mean composite among twelve configurations, winning three dataset means; ARI enters only on image datasets. | **Supported as an observed ranking, subject to correcting the contradictory reference provenance.** The calibration panel can contain sixteen configurations while the evaluated panel contains twelve, but the paper must say so consistently. |
| Paired difference from post-processed Barnes–Hut **\(0.023\pm0.047\)** across 20 runs; MNIST **\(-0.046\pm0.022\)**; standalone \(t\)-SNE-type composites **0.971–0.976**, sequence **0.930**. | **Supported.** These are SDs of paired differences, not confidence intervals. The pooled SD includes between-dataset heterogeneity. Twenty runs are not twenty independent dataset-level replications. |
| Report, code and artifacts are public at a pinned revision. | **Documented, not independently verified.** The tag and manifest now identify a release in the manuscript. A full commit hash would strengthen archival identification but is not a renewed blocker merely because a tag is used. |

**Matched-control interpretation:** the defensible headline is **highest observed mean of this specified composite on these four development datasets**. It is not established general superiority over post-processed Barnes–Hut. On MNIST, the sequence’s higher Spearman, **0.504 versus 0.464**, accompanies substantially lower \(k\)-means ARI, **0.728 versus 0.918**, and a lower composite. Neither statistical superiority nor statistical equivalence follows by comparing the pooled mean with its SD.

### Conclusion

| Claim | Assessment |
|---|---|
| Empirical exponent rule, transient mid-range term, 300-iteration refinement, box width 0.7, empirical hand-over and optional post-process. | **Supported as the released configuration, not as generally optimal settings.** |
| Unprocessed sequence has Barnes–Hut-like overlap on the two image datasets, taking **7.3/6.7 s versus 175/174 s**; **9.0/7.5 s** with post-processing. | **Supported.** Configuration and timing boundary are now explicit. |
| Higher mean pairwise Spearman than the four named comparators. | **Supported for the unprocessed sequence on these two datasets.** |
| Highest observed four-development-set mean, three dataset wins, pooled paired difference and MNIST reversal. | **Supported at the stated scope.** The round-8 omission of the pooled effect is repaired. “Both run standard deviations” is acceptable in context; “SDs of paired differences” would be more precise. |
| Density-contrast merging reduces selected over-partitioning failures on tested graphs; extension beyond these graphs remains open. | **Supported at this limited scope.** It does not establish a derived threshold, class recovery in general, or correction of modularity’s limitations generally. |

No other abstract/conclusion replacement is required on claim-support grounds. The principal remaining defects are elsewhere in the manuscript and companion abstract.

## 2. STORY AND STRUCTURE

### There are three contributions now, not five

The appropriate short-paper argument is:

1. **A sampled start and finite-budget full-batch refinement:** what is inherited, what is repaired, and why the hand-over matters.
2. **An optional community transformation:** its operational objective, safeguards and distortions.
3. **A matched evaluation:** raw axes, dataset-specific effects and limits of the composite ranking.

The exponent rule, interpolation audit and merge reference should remain **supporting design decisions**, not additional headline contributions.

### Cut, merge or relocate

- **Exponent paragraph:** retain the default rule, its empirical status and one failure-mode sentence. Move the detailed HDBSCAN counts, optional adaptive criterion and USPS/Fashion exceptions to the companion.
- **Community paragraph:** retain Louvain settings, the crossing-share definition, threshold and failure case. Cut most of the modularity/constant-Potts motivation; its present compression creates a conceptual problem discussed below.
- **Hand-over:** retain the operational scaling rule and finite-budget interpretation. Detailed scale comparisons and the radial derivation need not be duplicated in this companion.
- **Interpolation:** retain width 0.7 and the fact that it was audited. The numerical error sweep and subsystem details can live entirely in the long report.
- **Control:** keep the figure in the main text. Moving the full composite table to the appendix resolves the earlier duplication issue.
- **Configuration catalogue:** the quality-preset rows are peripheral to the shipped three-mode story. Leave them in the long report unless required to document the calibration references.
- **Conclusion:** the final merge sentence is defensible but elevates a supporting heuristic again. Move it to Limitations and end with the metric- and dataset-dependent interpretation of the recipe.
- **Abstract:** shorten the operational detail and the standalone-layout convergence numbers before cutting any qualification of the matched result.

### Essential content

Already present:

- method schematic;
- definitions of the three primary evaluation axes and their sampling;
- explicit hand-over;
- sampled-stage settings and standalone/sequence distinction;
- control pairing and shared post-processing instance;
- raw image-set ARI;
- integrated related work;
- reproducibility statement, revision tag and dependency manifest.

Still worth clarifying, without reopening the central specification blocker:

- Write the epoch schedule as a complete piecewise rule—e.g. **“300 otherwise”**, if that is implemented. “300 epochs at 70k” leaves the 20k configurations implicit.
- Identify the precise definitions or companion sections for **triplet accuracy** and **class-centroid Spearman**.
- Give the fixed \(k\)-means algorithm seed, not only the sampling seed.
- Avoid “each at its library defaults” for the whole panel: PCA-2 plus the custom pass and the MAGNET modes are prescribed configurations. Say **“external baselines at the stated library defaults; custom configurations as specified above.”**

## 3. CORRECTNESS AND CONSISTENCY

### A. Numerical and provenance audit

| Item | Finding and required action |
|---|---|
| **Frozen references: twelve versus sixteen configurations** | **Unresolved contradiction.** Metrics says “the best unprocessed value of the twelve-configuration panel”; provenance says “per-axis maxima of the sixteen unprocessed configurations.” These are different calibration populations. Assuming the provenance paragraph is authoritative, replace the former with: **“Frozen references are the per-axis maxima over the sixteen unprocessed calibration configurations generated by `benchmarks/composite_table.py` at seed 0, distinct from the twelve-configuration matched-control panel.”** |
| **Identity of calibration maxima** | The generating script and seed are now named, and the three MNIST sources are identified. The other seven axis maxima lack named configuration sources in the supplied text. A compact reference-provenance table or exact artifact/configuration keys would close the traceability chain. The essential blocker is the contradictory panel definition, not a demand to reproduce calibration experiments. |
| **Timing variability** | **Still false.** “And within 3% elsewhere” conflicts with MNIST sampled **3.4–3.7 around 3.6**, sequence + post **8.6–9.3 around 9.0**, and openTSNE **90.3–95.0 around 93.3**. Delete the blanket clause. The timing table already supplies the relevant ranges. |
| **Barnes–Hut post-processed mean** | The earlier **0.884/0.883** disagreement is repaired: text and table now say **0.883**. |
| **Sequence post mean 0.907** | The displayed dataset means average to **0.90625**. This still does **not** prove an error: an average of the underlying unrounded values can round to 0.907. Generate the aggregate and paired effect from the same artifact; do not mechanically replace it with 0.906. |
| **0.907 − 0.883 versus paired difference 0.023** | Not necessarily inconsistent: separately rounded means need not subtract to the rounded paired difference. The four displayed paired differences average to **0.023**. |
| **Raw-axis SD caption** | The substantive round-8 understatement is corrected: Spearman variability up to approximately **0.04** and ARI variability around **0.07** are acknowledged. Treat these as rounded summaries—“up to approximately” is safer than exact upper bounds. The supplied composite summary does not independently verify every raw-axis SD. |
| **Three-run demonstration versus five-run control** | Now distinguished. Demonstration full-batch ARI **0.761 → 0.914** and k41 **0.791 → 0.915** are not contradictory when their experiment families remain explicit. |
| **Quiet versus shared-machine timing** | Now distinguished. Sequence **7.3 s** in the timing suite and **11.2 s** in k41 are not interchangeable but are not an internal contradiction. |
| **Post-processing time range** | The text now distinguishes configuration means **1.0–1.9 s** from individual replicates **0.8–4.5 s**. The supplied summary supports the former; the latter requires the cited run artifacts. No contradiction is established from the supplied material. |
| **Pipeline ARI citation** | Corrected: post-processed **0.728**, rounded **0.73**, now points to the configurations/control result rather than the raw sequence row **0.596**. |
| **Other experiment families** | “Three principal evaluation families” plus the additional script identifiers repairs the earlier claim that all quoted numbers came from only three families. More local citations would improve readability. |

### B. Short/long consistency

1. **PCA exception: resolved.** The long abstract now acknowledges covertype, where sequence **0.536**, sampled **0.592** and TriMap **0.748** exceed PCA **0.397**.

2. **MNIST matched reversal: resolved.** The long abstract now says the margin reverses and supplies **\(-0.046\pm0.022\)**.

3. **Transfer terminology: resolved in substance.** Letter/covertype are no longer presented as an independent held-out population. The frozen-rule transfer experiments remain conditional on development-family provenance.

4. **Sequence means: source distinction remains missing.**  
   Long abstract: **0.353/0.446**, **0.329/0.750**.  
   Short k41 table: **0.354/0.443**, **0.329/0.751**.  
   These can be different five-run families, but the long abstract does not identify that distinction. Either name its experiment family or use k41 consistently. Do not silently replace legitimate results from another experiment.

5. **Fastest-method claim: still incorrect in the companion.**

   > “The sampled mode is the fastest neighbour-embedding row … (PCA is faster, and h-NNE matches it on Fashion-MNIST in the five-run control).”

   In that control, h-NNE is **5.1 s**, sampled MAGNET **5.8 s**. “Matches” does not describe the reported means, and the unconditional fastest claim is false.

   **Replacement:** “In the five-run control, the sampled mode has the lowest mean fit time among the neighbour-embedding configurations on three of the four datasets; on Fashion-MNIST, h-NNE is faster, at 5.1 s versus 5.8 s. PCA is faster on all four.”

6. **“An eleventh of openTSNE’s” runtime is not traceable to either supplied suite.** The quiet-suite unprocessed ratios are approximately **1/12.8** and **1/12.6**; the k41 ratios approximately **1/8.8** and **1/8.2**. Identify the source suite or replace this with the explicit quiet-suite times.

7. **Matched-start Spearman bounds:** short **−0.02 to +0.04**, long **−0.025 to +0.033**. If these refer to the same experiment, quote the same bounds or explicitly describe the short version as approximate.

### C. Interface counts

The arithmetic objection is resolved:

- resolution 0.5: \(51+1=52\) interfaces among 11 communities, below \(\binom{11}{2}=55\);
- resolution 1.0: \(68+3=71\) among 13, below \(\binom{13}{2}=78\);
- pooled: **119 different-class and 4 same-class observations**.

Two clarifications remain:

- “Unordered pair … at least 20 boundary nodes on the \(A\) side” is not fully defined unless the rule assigning \(A\) is specified. State the actual orientation/filter rule; do not change it silently to a two-sided filter.
- Observations from two partitions of the same graph are not independent replications. Four same-class interface observations support a limited diagnostic, not a calibrated general classifier of valid merges.

### D. Scope, terminology and mathematical interpretation

**“Any neighbour-embedding start” remains too broad.**

> “The pass repairs any neighbour-embedding start…”

**Replace:** “Across the neighbour-embedding starts tested in the matched-start experiment, the pass reaches 15-NN overlap of 0.352–0.356 on MNIST and 0.327–0.331 on Fashion-MNIST.”

Likewise, replace “the global column after the pass is the start’s” with **“much of the measured pairwise-Spearman ordering across starts is retained.”** Changes of a few hundredths are not literal preservation.

**The introduction still imposes an unsupported method ordering.**  
“PaCMAP and TriMap … land between” is not a consistent description of the supplied raw table.

**Replace:** “These methods weight local, mid-range or triplet relations differently; their positions on the measured overlap–Spearman axes depend on the dataset, initialisation and settings.”

**Exponent rule:** the round-8 necessity claim is removed. The evidence supports an empirical default under the reported graph construction and HDBSCAN settings, not a universal sample-size transition. HDBSCAN cluster counts alone should not be presented as a dataset-independent measure of fragmentation.

**Merge threshold:** the **0.17–0.28** reference range and heuristic status of **0.10** are now stated. The threshold remains development-informed; “half a simulated reference” is a motivation, not a derivation.

**Constant-Potts argument:** the sentence connecting an approximately constant null model to retaining “the resolution and field-of-view limits” is misleading. CPM is specifically associated with resolution-limit-free community detection in the relevant technical sense. Equal strengths make the modularity null term constant **on a fixed graph**; they do not establish the claimed general equivalence of limitations. Moreover, a strength coefficient of variation of **0.33** does not establish a quantitatively negligible null-model variation.

Prefer:

> “No tested modularity resolution avoided both splitting large communities and merging small ones. We therefore use the following empirical merge heuristic.”

If retaining the CPM comparison, distinguish the fixed-graph algebraic analogy from claims about resolution limits.

**Radial identity:** the displayed derivative is correct for normalized \(p\), excluded diagonals and \(q=k/Z\). The revised interpretation—diagnostic radial pressure, not selection of the finite-budget scale—is appropriately limited. Two minor corrections:

- define “unit standard deviation” consistently with the hand-over’s **first-coordinate** convention;
- a 13-point scan establishes the lowest value among scanned scales, not a certified global minimum along the ray. Use **“the lowest evaluated value along the scanned ray.”**

**Other terminology:** “whose classes are not clusters” is stronger than needed. Say the class labels are **not treated as density-component ground truth**. Also label **0.54–0.68** explicitly as the competitors’ **four-set mean composites**, not a range covering every dataset result.

## 4. FIGURES AND TABLES

### Space allocation

- **Method schematic:** retain. It supplies the essential overview.
- **Matched-control figure:** retain. It displays the actual central comparison.
- **Pipeline montage:** weakest remaining main-text figure. Its caption describes UMAP, Barnes–Hut, Barnes–Hut + post and the complete pipeline, but **not the unprocessed sequence**. Consequently it does not show what the sequence’s own post-process changes. If retained, replace the UMAP panel with the unprocessed sequence and make it a paired before/after comparison for Barnes–Hut and the sequence. Otherwise move it to the companion.
- **Small-\(n\) and interface figures:** supporting material only. Their evidence is already appropriately subordinate in the appendix.
- **Box-width figure:** absent from this revision. Do not repeat the obsolete demand to remove it.

### Caption corrections

**Method**

The caption still omits the central operational details:

- transient mid-range attraction in the first pass;
- empirical hand-over \(31\sqrt{n/70000}\), defined through first-coordinate SD;
- **3.6 s includes graph construction**;
- the full-batch stage is 300 iterations without exaggeration.

A compact version of these facts is sufficient.

**Pipeline**

The earlier defects—wrong ARI reference, ambiguous display/control provenance and overbroad “labels unused”—are repaired. Remaining concerns:

- show the unprocessed sequence if this figure is intended to explain the pipeline;
- do not attach a three-run class-centroid change to these separate display runs without identifying the source family;
- retain the warning that independent panel scaling prevents visual comparison of absolute inter-cluster distances.

**Control**

The reference-vector cross-reference is now present. The caption explicitly assigns mean ± SD to the plain circles, but not clearly to the post diamonds. Either:

- show and label **mean ± run SD for both**, or
- state that post dispersion is represented only by the five individual dots.

Individual post dots are informative, but are not automatically an SD display.

**Interfaces**

Define the \(A\)-side filtering convention and make clear that pooling is over **partition–interface observations**, not independent graph realizations.

**Small-\(n\)**

The HDBSCAN implementation/version, effective default semantics and sampling are now supplied. Explain that the reported “5–10” count variability comes from additional replicate checks, since the plotted subsample sweep itself has one run per condition.

### Is the sorted-dot control the right form?

**Yes.** Use a per-dataset dumbbell display: connected plain/post means, visible run dispersion, method labels and common composite limits where practical. Sorting separately by post mean is acceptable.

Keep the paired effects visible:

- MNIST: **−0.046 ± 0.022**
- Fashion: **+0.020 ± 0.015**
- letter: **+0.071 ± 0.003**
- covertype: **+0.047 ± 0.013**

Do not replace these with only the pooled ranking or relabel SD as confidence intervals.

### Should the seeds table include \(k\)-means ARI?

**Yes.** There is no matched-start/seeds table in this short revision. The raw-method table already includes ARI on the image sets, but that does **not** expose ARI changes across initialisations under the same refinement.

The long report’s matched-start table should show before/after ARI, preferably mean ± SD, if it supports the interpretation that the pass “repairs” starts. Two-axis repair is not three-axis recovery. No additional short-paper table is necessary.

## 5. RELATED WORK

A separately titled section is unnecessary. All works specifically missing in round 8 are now cited. Required positioning, one line each:

- **van der Maaten & Hinton (2008):** full-batch KL objective and the scope of the \(t\)-SNE comparison.
- **van der Maaten (2014):** Barnes–Hut baseline and tree-based acceleration.
- **McInnes, Healy & Melville (2018):** fuzzy graph, bandwidth calibration and sampled optimisation.
- **Dong, Charikar & Li (2011):** NN-descent; the previously incorrect author names are corrected.
- **Damrich & Hamprecht (2021):** effective negative-sampling objective, not merely nominal kernel formulas.
- **Linderman et al. (2019):** FFT/interpolation machinery and numerical approximation.
- **Poličar et al., openTSNE:** optimiser components and the matched-optimiser comparison.
- **Böhm, Berens & Kobak (2022):** attraction–repulsion operating points and limits of objective-novelty claims.
- **Kobak et al. (2019):** heavy-tailed kernels; distinguish attraction-only modification from coupled changes.
- **Kobak & Linderman (2021):** inherited geometry and initialisation dependence.
- **Wang et al. (2021), PaCMAP:** mid-near sampling and transient attraction.
- **Amid & Warmuth (2019), TriMap:** triplet-based alternative, not another mid-range pair loss.
- **Kruskal (1964):** rank-fitting placement; distinguish community-constrained translations from pointwise nonmetric MDS.
- **Newman & Girvan (2004):** modularity and its null model.
- **Blondel et al. (2008):** Louvain algorithm family.
- **Traag, Van Dooren & Nesterov (2011):** CPM comparison, with the resolution-limit distinction corrected.
- **Fortunato & Barthélemy (2007):** modularity’s resolution limit.
- **Schaub et al. (2012):** field-of-view/geometric limitations, distinguished from the resolution limit.
- **Traag, Waltman & van Eck (2019):** Leiden; connectivity guarantees do not settle the reported scale/partition instability.
- **Campello, Moulavi & Sander (2013):** HDBSCAN clustering methodology.
- **McInnes, Healy & Astels (2017):** actual HDBSCAN implementation and default semantics.
- **Narayan, Berger & Cho (2021):** densMAP baseline.
- **Sarfraz et al. (2022):** h-NNE baseline.
- **Hubert & Arabie (1985):** ARI and external-label evaluation.
- **Chenebaux, companion report:** introduction, experiment provenance and reproducibility, tied to the same release.

Huber remains unnecessary: the robust-estimation analogy is absent.

## 6. ROUND-8 BLOCKER CHECK AND RELEASE VERDICT

| Carried blocker | v3 status |
|---|---|
| **1. Incorrect implemented hand-over** | **Resolved.** The first-coordinate SD convention, \(31\sqrt{n/70000}\), empirical status and separation from the radial stationary point are explicit. |
| **2. Misleading matched-control headline** | **Resolved.** Abstract and conclusion both include **\(0.023\pm0.047\)** and the MNIST reversal; development-set scope and two-/three-axis distinction are stated. No general-superiority interpretation is licensed. |
| **3. False benchmark statements in the short paper** | **Resolved for the specific carried items.** Letter/covertype rankings, post-processing exceptions, image ARI and PCA qualification agree with the supplied tables. The remaining companion fastest-method statement is counted under item 6. |
| **4. Heuristics presented as necessities or derivations** | **Resolved for the round-8 demands.** Small-\(n\) necessity wording is removed; the uniform-cloud range is disclosed; the threshold is heuristic; the radial identity no longer selects the operating point. The CPM sentence still needs the localized correction above. |
| **5. Unqualified fidelity and timing claims** | **Not fully resolved.** The abstract’s fidelity comparison is repaired, but Results still says **“and within 3% elsewhere.”** The supplied timing table contradicts this. Delete that clause or regenerate a correct statement. |
| **6. Experiment provenance and transfer status** | **Not fully resolved.** Metrics says **“the best unprocessed value of the twelve-configuration panel”** while provenance says **“the per-axis maxima of the sixteen unprocessed configurations.”** The long abstract also retains **“h-NNE matches it on Fashion-MNIST”**, contrary to **5.1 versus 5.8 s**, and does not identify the family behind its different sequence means. Transfer terminology, interface arithmetic, principal experiment-family distinctions and the substantive SD-range problem are otherwise repaired. |
| **7. Incomplete/untraceable release** | **Resolved at the manuscript-documentation level.** The named `short-paper-v3` tag, artifact/script locations and frozen manifest replace “commit … and later” and installation-date provenance. Actual release availability and reproducibility remain unverified here. |
| **8. Additional sampled-stage specification gap** | **Resolved.** The text now identifies the mid-near pair sampler, spring update, probability, stiffnesses, decay interval, negative-sampling rate, learning-rate schedule, hand-over convention, placement candidates and treatment of small communities. The epoch-schedule wording needs a minor clarification, not another specification round. |

### New blocker?

**No independent new experimental or methodological blocker is established by the supplied material.** The twelve-versus-sixteen contradiction is an unresolved calibration-provenance blocker, not a reason to demand new experiments. The “any start” and CPM wording are additional localized scope/correctness corrections.

### Release decision

**NO-GO for these exact files; release after the bounded text/provenance corrections under items 5 and 6.** Do not hold release for a new significance analysis or a larger benchmark campaign. Do not convert the descriptive matched-control result into a superiority claim when making those corrections.

### Three highest-value non-blocking improvements

1. **Reduce the short paper to its three-part argument:** remove duplicated design audits and either replace or relocate the montage.
2. **Expose post-processed raw trade-offs:** especially MNIST Spearman versus ARI, with run variability; add ARI to the long report’s matched-start table.
3. **Show scalarisation sensitivity from existing results:** compare two-axis versus three-axis image composites and geometric versus minimum composites, while keeping dataset-specific paired differences primary.