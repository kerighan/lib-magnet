## 1. CLAIMS

**The central claim must be an empirical recipe with a metric-specific advantage—not a generally superior embedding or a principled solution to global–local structure.** The matched control supports the highest observed four-set mean, but its paired advantage over post-processed Barnes–Hut is **0.023 ± 0.047 across 20 runs**, with a **negative MNIST difference**. That qualification belongs in the abstract, not only in Results.

The audit below uses the supplied short paper, long-version abstract and k41 summary. The contents of `control_geo.tex` were not supplied; its actual cells cannot be checked here.

### Abstract

| Claim | Supported as stated? | Required wording |
|---|---|---|
| “Neighbour embeddings pay for local fidelity with global arrangement, and the other way round.” | **Too categorical.** The experiments show operating-point trade-offs, not an unavoidable law. “Global arrangement” also exceeds what pairwise Spearman measures. | “In the tested configurations, neighbourhood overlap and pairwise-distance rank correlation favour different operating points.” |
| The sampled heavy-tailed attraction places clusters; the full-batch pass refines neighbourhoods. | **Incomplete attribution.** The long abstract says the first-stage Spearman advantage comes from the **mid-range attraction**, and disappears without it. The short abstract omits that ingredient. | “MAGNET combines a sampled pass with decoupled attraction and repulsion kernels and a transient mid-range attraction, followed by 300 iterations of a full-batch, FFT-interpolated \(t\)-SNE objective.” |
| The optional label-free post-process raises distance agreement. | **Only as an optimisation target, not a general outcome.** Held-out Spearman falls for several configurations, including the sequence on Fashion and letter. Label-free execution does not imply label-free development. | “An optional post-process, label-free in execution, translates and contracts graph communities to optimise sampled distance-rank correlation under an overlap guard; its settings were selected on the development benchmarks.” |
| Full-batch reaches Barnes–Hut neighbourhood preservation, 0.349 versus 0.350, in 6 versus 186 seconds. | **Numerically supported for the reported MNIST protocol**, but “reaches” should mean similar measured 15-NN overlap, not established equivalence or general local fidelity. Timings are provisional and load-mismatched. | “On MNIST 70k, the standalone full-batch mode obtains 15-NN overlap of 0.349 versus 0.350 for Barnes–Hut \(t\)-SNE; provisional timings are 6.2 versus 186 seconds under the reported, unmatched-load protocol.” |
| The sequence has the highest pairwise-distance Spearman of the tested neighbour embeddings. | **Needs an explicit comparison scope.** Supported for the named conventional comparators on the image benchmarks; false as a statement over all tested data and configurations. Covertype and 20 newsgroups are counterexamples; PCA-2 plus refinement exceeds it on Fashion. | “On MNIST and Fashion-MNIST, the unprocessed sequence has higher mean pairwise-distance Spearman than UMAP, PaCMAP, openTSNE and Barnes–Hut \(t\)-SNE in the reported default-configuration comparison.” |
| Post-processing changes full-batch Spearman 0.35→0.47, ARI 0.76→0.91, at a 0.002 15-NN cost. | **Supported for the specified three-seed experiment**, not the five-run matched-control experiment. State that provenance. | “Across three MNIST runs, full-batch 15-NN overlap / Spearman / \(k\)-means ARI changes from 0.349 / 0.350 / 0.761 to 0.347 / 0.469 / 0.914.” |
| “Does the same to a Barnes–Hut layout.” | **Qualitatively supported; not identical changes.** Barnes–Hut loses 0.003 overlap and gains less ARI. | “Applying the same procedure to Barnes–Hut gives corresponding scores of 0.350 / 0.338 / 0.829 → 0.347 / 0.463 / 0.917.” Alternatively: “Barnes–Hut receives comparable final scores.” |
| The exponent maximising fidelity at 70k tears datasets below 15k and “must follow the sample size.” | **Unsupported as a necessity or universal threshold.** These are finite sweeps, largely single-run subsampling evidence. Dataset geometry, graph construction and clustering diagnostics are confounded with \(n\). | “In MNIST subsamples and several smaller benchmarks, the low-exponent sampled pass fragments the layout; we therefore use an empirical sample-size-dependent default, with an optional graph–layout agreement criterion.” |
| FIt-SNE’s unit box leaves 3.5% force error that costs 0.004 overlap. | **Supported only for the audited layout and settings.** Not an invariant property of FIt-SNE. Width 0.7 recovers about 0.003; the finest tested setting recovers about 0.004. | “On the audited MNIST layout, unit-width interpolation boxes produce 3.5% relative RMS force error; narrowing the boxes improves 15-NN overlap from 0.346 to 0.349–0.350.” |
| On kNN graphs the null model is nearly constant, causing splitting and swallowing; density-contrast merging repairs this with a threshold read from a uniform cloud. | **Overgeneralised and over-derived.** Approximately constant strengths are a condition, not a property of every symmetrised fuzzy kNN graph. The uniform-cloud experiment supplies a reference, not a uniquely justified threshold or a general repair guarantee. | “When graph strengths are nearly uniform, modularity approximates a constant-Potts objective and retains geometric scale limitations. We test a community-merging heuristic whose threshold is set to half an empirical uniform-cloud reference.” |
| Same post-process: best three-axis balance on three datasets of four; MNIST \(t\)-SNE layouts converge. | **Ranking supported; description misleading.** Two datasets use **two** axes. The converging MNIST layouts are the three **standalone** full-batch/\(t\)-SNE configurations, **not the sequence**. | “With identical post-processing, the sequence has the highest mean composite on three of four development datasets, using three axes on the image datasets and two elsewhere. Its paired advantage over post-processed Barnes–Hut is 0.023 ± 0.047 across 20 runs; on MNIST it instead trails by 0.046, while the three standalone \(t\)-SNE/full-batch configurations converge.” |

The abstract is carrying method, benchmark, exponent study, numerical audit, community-detection theory and matched-control interpretation. The last three technical digressions should not all survive its shortening.

### Conclusion

| Claim | Supported as stated? | Required wording |
|---|---|---|
| “Robust, size-aware attraction” and a full-batch pass “accurate to a 0.7 box.” | **Needs operational rather than theoretical language.** “Robust” is an interpretation; 0.7 is a grid parameter, not an accuracy guarantee. | “The method combines a sampled pass with an empirical size-dependent exponent and mid-range attraction, a 300-iteration full-batch refinement with interpolation-box width 0.7, and an optional community post-process.” |
| Community detection “needed a density-contrast correction that modularity cannot provide.” | **Too universal.** The experiments motivate this heuristic for the tested graphs; they do not establish that this correction is necessary or sufficient generally. | “For the tested graphs, a density-contrast merging heuristic reduces selected over-partitioning failures of modularity-based community detection.” |
| “The library runs MNIST 70k in 6–9 s.” | **Ambiguous configuration and timing boundary.** Those are roughly standalone/full-sequence times before post-processing in one suite, not complete-pipeline times. | “In the reported timing suite, standalone full-batch and sequence times are 6.2 and 8.4 seconds before optional post-processing, which adds approximately 1–2 seconds; these timings remain provisional.” State whether graph construction is included. |
| “Matches Barnes–Hut’s local fidelity.” | **Only at the reported neighbourhood size.** The long abstract explicitly excludes \(k=5\) and \(k\geq50\). | “It obtains similar 15-NN overlap to Barnes–Hut on the image benchmarks; this local comparison does not extend uniformly across neighbourhood sizes.” |
| “Best global arrangement of the neighbour embeddings tested.” | **Unsupported.** Pairwise Spearman is not global arrangement generally; the dataset-wide ranking is false. | “It improves pairwise-distance Spearman over the named default neighbour-embedding comparators on MNIST and Fashion-MNIST, without establishing superiority in other global-structure metrics or on every dataset.” |
| “Best measured composite balance … on four data sets.” | **Only if this means the four-set mean.** It does not win all four. | “With identical post-processing, it has the highest observed four-dataset mean composite, winning three dataset means and losing to post-processed Barnes–Hut on MNIST; the pooled paired difference is 0.023 ± 0.047.” |
| The matched control shows where gains come from the embedding stages and where they are “the post-process’s alone.” | **Causal overinterpretation.** The control measures post-processing increments and final rankings. It does not uniquely decompose causes. On MNIST the sequence retains higher Spearman but worse ARI and composite. | “The matched control shows that post-processing substantially changes the rankings: on MNIST, post-processed standalone full-batch, openTSNE and Barnes–Hut reach composites of 0.971–0.976, exceeding the sequence’s 0.930.” |
| “The exhaustive document, code and every result file are public.” | **Unverifiable from this manuscript.** There is no actual bibliographic entry or URL for the long version, repository or results. | Supply: “The companion report [citation], implementation [versioned URL], and experiment manifests and result files [archived URL] are publicly available.” Otherwise delete the assertion. |

## 2. STORY AND STRUCTURE

### Five claims are too many for this paper

The current structure is a compressed catalogue of the long version, not an eight-page argument. Use three contributions:

1. **Staged embedding:** what is optimised, why the first-stage mid-range term matters, what the second stage repairs, and what it inherits.
2. **Optional method-agnostic post-processing:** what changes, what is preserved, what can be distorted.
3. **Matched evaluation:** raw axes, common post-processing, dataset-dependent differences and limitations.

Treat the exponent rule, FFT audit and modularity analysis as supporting design decisions—not three additional headline discoveries.

### Cut, merge, move

- **Merge Sections 2–4 into one method section.** Retain the exponent schedule and grid width as operational defaults. Move the exponent sweeps, oracle/per-point variants, interpolation-order experiments and grid dithering to the companion report.
- **Move most of the modularity scaling argument and interface analysis out of the main text.** Retain the conditional near-regular-graph observation, the merge rule and its heuristic status.
- **Remove the catalogue of failed post-processes.** One sentence pointing to the long version suffices.
- **Remove most of the staged-refinement appendix here.** The long version already exists. Keep only details needed to disambiguate the shipped configuration.
- **Delete or complete the `[TO FILL]` appendix.** Do not release a companion paper with an unfinished results section.
- **Shorten the title.** It currently promises and describes three papers’ worth of machinery.

### Essential material missing or buried

- **A method schematic, not another layout montage:** shared graph → sampled stage with transient mid-range term → empirical rescaling → 300-step refinement → optional community placement/contraction. Show that the post-process also applies to competitors.
- **The actual hand-over rule in the main text.** The present main-text rule is contradicted by the appendix.
- **Matched-start evidence.** State that other neighbour-embedding starts also recover comparable 15-NN scores, and that openTSNE gives nearly identical fidelity with matched affinities and schedule. This bounds the algorithmic contribution.
- **The source of global improvement.** The long abstract’s “without the mid-range attraction, the sampled layout scores below openTSNE” is essential, not optional detail.
- **Early stopping.** The 300-iteration result is a finite-budget operating point; continued optimisation changes the trade-off. Put that fact in the method or limitations.
- **Metric definitions and preprocessing.** “15-NN preservation” is not self-defining, and input-distance Spearman cannot be interpreted without the input representation and metric.
- **A short related-work paragraph.** A separate section is unnecessary; substantive positioning is necessary.
- **A reproducibility box:** repository/commit, report version, scripts, data/subsample identifiers, preprocessing, hardware/threads, timing boundaries, stochastic-run protocol and result artifacts.

## 3. CORRECTNESS AND CONSISTENCY

### A. Methodological contradictions and overclaims

**1. The hand-over rule is wrong in the main text.**

Main text:

> “The stationary dilation … is where the full-batch pass starts.”

Appendix:

> “The identity … does not pick the hand-over.”  
> “The shipped rule, \(31\sqrt{n/70000}\), is a finite-budget reference…”

These cannot coexist. Replace the contribution and method description with:

> “The hand-over uses the empirical scale \(31\sqrt{n/70000}\); a radial derivative identity diagnoses scale pressure but does not select the finite-budget operating point.”

The derivative identity itself is consistent with the stated KL objective **provided** \(p\) is normalised, \(q_{ij}=k_{ij}/Z\), and diagonal terms are excluded. Specify these conditions. Also:

- The attractive expectation is an edge sum; the repulsive quantities use interpolation/FFT. “Both sides cost one FFT evaluation” is an imprecise cost statement.
- A stationary root along a fixed ray is neither a jointly optimal scale nor automatically a unique minimum.
- A 13-point scan supports the tested case, not a general minimum theorem.
- Remove the main-text claim that all smaller/larger scales collapse/lose the arrangement; the appendix itself reports useful alternatives.

**2. The attractive-force explanation is missing its domain of validity.**

For the usual positive-edge loss \(-\log\phi_b(d)\), the radial derivative is

\[
\frac{2ab\,d^{2b-1}}{1+ad^{2b}}.
\]

Thus \(d^{2b-1}\) describes the short-distance numerator/asymptote, not the whole force law. At large distance the force decays approximately as \(1/d\). Supply the implemented force and define \(a\).

“Huber’s,” “median-like,” and “robustness dial” are interpretations, not an established robust-estimation result. Huber loss is not simply this kernel. Either qualify the analogy or remove it.

**3. The size rule and adaptive criterion do not make “the same choices.”**

- USPS: size rule gives **0.8**, adaptive rule **0.7**.
- Fashion 70k: size rule gives **0.6**, adaptive rule **0.7**.

Replace:

> “The optional criterion often selects a similar range of exponents but differs from the size rule on USPS and Fashion-MNIST.”

The ARI threshold **0.7** is another heuristic operating point requiring provenance. “Epochs, learning rate, \(k\), threads … do not change this” must become “did not remove the effect over the tested ranges.”

“HDBSCAN finds three clusters” is not equivalent to “the embedding collapses into three clusters.” Cluster counts also depend on the changing `min_cluster_size` and on unspecified HDBSCAN settings.

**4. The density-contrast threshold is not derived.**

Uniform-cloud simulations suggest a no-contrast reference near **0.2**. Choosing **half** that reference is an additional modelling choice. Neither the simulation nor the gap endpoint determines **0.10** uniquely.

Replace:

> “We set the merge threshold heuristically to 0.10, half the approximately 0.20 crossing-share reference measured on simulated uniform clouds.”

Further qualifications:

- Fuzzy symmetrisation can produce heterogeneous strengths and hubs. Establish near-regularity empirically before invoking the constant-Potts approximation.
- The geometric cut scaling requires assumptions about dimension, geometry, neighbourhood construction and balanced cuts.
- The size penalty includes the graph-wide normalisation \(m\). Its scaling is not simply \(n_c^2\) when the total graph grows with \(n_c\).
- “Thinnest cut” needs an operational definition: Fiedler sign split, median split, balanced minimum, or something else.
- Define the relative-interface statistic explicitly when the two communities have different internal-cut references.
- “No resolution fixes both” should be scoped to the tested graphs/resolution sweep.
- “What the density says” should become “what this graph-based criterion returns.” It is not an independently established truth about density or classes.

**5. Community transformations do not preserve 15-NN overlap exactly.**

Translation and contraction preserve within-community distance order, but alter comparisons with points in other communities. Therefore:

> “their weak axis is 15-NN, which no rigid motion of communities changes”

is false. Replace:

> “The post-process does not reconstruct within-community neighbourhood order and generally does not remove these methods’ 15-NN deficit, although cross-community neighbour relations can change.”

The guard is a sampled nearest-neighbour cross-community statistic, not a bound on full-data 15-NN loss. Specify whether its 0.002 allowance is relative to the original layout or the preceding accepted move.

### B. Numerical and ranking audit

| Location | Problem and correction |
|---|---|
| Matched-control paragraph: “the three post-processed \(t\)-SNE layouts converge” | Name **full-batch MAGNET, openTSNE and Barnes–Hut**. Their MNIST composites are **0.976, 0.971, 0.976**. The sequence is **0.930**, not in that convergence group. |
| “On the two sets … the same ranking holds on the two axes” | False. On letter, sequence overlap **0.625** is below Barnes–Hut **0.642** and openTSNE **0.639**. On covertype, sequence Spearman **0.536** is below sampled MAGNET **0.592** and TriMap **0.748**. Delete or replace with dataset-specific results. |
| “The step lifts every configuration except … TriMap on letter” | False. Composite decreases also include **letter full-batch −0.009, letter sequence −0.016, letter PCA-2+pass −0.020, Fashion sampled −0.003**. Say “Most configuration–dataset combinations improve; exceptions include…” |
| Four-set UMAP/PaCMAP means | From printed k41 values, plain means are approximately **0.581 / 0.590**, not both 0.58; post means approximately **0.677 / 0.670**, not both 0.68. |
| Sequence post mean 0.907 | Printed dataset means average to **0.90625**. Recompute from unrounded artifacts. Rounding could explain 0.907; do not silently choose one. Likewise, 0.907−0.883 versus the paired mean 0.023 is not by itself evidence of an error. |
| Control-table caption: SDs “0.00–0.03” | False for the supplied panel: letter TriMap post SD is **0.059**, and covertype UMAP plain SD is **0.033**. Report actual row uncertainties or a correct range. The seeds table does **not** contain composite SDs. |
| Post-process cost “1–4 s” | k41 includes **5.0 s** for covertype PCA-2+pass and several **4.2 s** rows. Use **1.0–5.0 s in this panel**. |
| MNIST PaCMAP ARI “0.91” | k41 plain mean is **0.845**; post mean is **0.908**. If 0.91 comes from another unprocessed experiment, identify it. Do not mix it into the k41 interpretation. |
| Sequence MNIST ARI SD | Results says **±0.10**; Limitations says **±0.08**, “with or without the step.” Supply actual raw ARI SDs for each experiment. The k41 raw-axis summary does not directly provide them; do not copy a composite SD. |
| Full-batch ARI 0.761 versus 0.791 | Not necessarily contradictory: one is the three-seed demonstration, the other the five-run control. Label the experiment families consistently, including their different post scores. |
| Seeds table versus k41 | MNIST sequence **0.353/0.446** versus **0.354/0.443**; time **8.4 versus 11.2 s**. UMAP time **8.4 versus 16.2 s**. These may be distinct runs/protocols, but that must be explicit. Do not present all numbers as one “five seeds everywhere” experiment. |
| Interface counts | Text says **25 real boundaries**; caption says **119**. Explain whether these are class pairs versus community interfaces pooled across resolutions, and identify deduplication. Otherwise correct the count. |
| “Finds the ten classes” | Ten communities is not exact class recovery. State the number of communities and measured label agreement separately. |
| Stage-one global score | “Keeps … most” of **0.41** while reaching **0.45** obscures that the sequence uses a stiffer first-stage configuration. Compare against its actual start, not the standalone sampled default. |

**The PCA statement is materially wrong for the supplied control.**

> “Every neighbour embedding stays below PCA … with or without the step … [sequence] is the one exception.”

Counterexamples:

- Covertype plain PCA: **0.397**; sequence: **0.536**; TriMap: **0.748**.
- MNIST post sequence: **0.504**, but post PCA: **0.565**. The sequence exceeds the **frozen unprocessed** PCA reference of 0.488, not matched post-processed PCA.
- MNIST post sampled MAGNET also reaches **0.500**, so the sequence is not the only exception to that frozen reference.
- Several post-processed covertype configurations exceed contemporaneous PCA.

This also exposes a conflict with the long abstract’s broad PCA claim. Reconcile preprocessing/protocols if they differ; otherwise correct both documents.

### C. Evaluation definitions and missing qualifications

- Define overlap, for example
  \[
  L_{15}=\frac1{|Q|}\sum_{i\in Q}
  \frac{|N^{X}_{15}(i)\cap N^{Y}_{15}(i)|}{15},
  \]
  including whether neighbours are searched among all points and whether ground truth is exact.
- Define input preprocessing and distance metric per dataset, including PCA/standardisation/TF-IDF stages.
- Define whether Spearman uses all unordered pairs of the 2,000 sampled points, tie handling, and which evaluation samples are excluded from optimisation.
- Specify \(k\)-means fitting sample, `n_init`, randomisation and evaluation subset. Choosing \(k\) from class counts is an evaluation use of labels, even if the embedding is label-free.
- Write the composite formula and state **three axes on images, two on letter/covertype**, per-run calculation, then equal dataset weighting.
- Identify the exact experiment supplying each frozen reference. These are not uniformly the maxima of the displayed k41 means.
- Replace “five seeds everywhere” with **five runs**, distinguishing seeded methods from unseeded UMAP replicates.
- Report “0.000” SD as “<0.0005” where appropriate.
- Fashion is simultaneously described as a dataset “where the classes are clusters” and as touching, non-density-aligned classes. Treat ARI as external class recovery, not evidence that classes are density components.

**Transfer is not demonstrably held out as written.** Letter and covertype 20k are already development datasets. Give sample IDs, split definitions and tuning chronology. New seeds or sizes of a development dataset are not held-out dataset families.

Restore the long abstract’s material scope restrictions:

- the local advantage is specific to \(k=15\);
- the sequence trails six methods on 20 newsgroups;
- MNIST class-centroid geometry is poor and variable before post-processing;
- weaker stiffness transfers better than the shipped value;
- a refined PCA start wins on a nearly linear case.

These can fit in four sentences; omitting them changes the claim.

## 4. FIGURES AND TABLES

### Space allocation, from captions only

- **`fig_box_width` does not earn a full-width main-paper slot.** It documents an implementation audit, not the central staged-versus-post-process result. Move it to the companion report or reduce it to one small panel.
- **`layouts_mnist` is redundant with `fig_pipeline`.** Remove it or replace it with the method schematic.
- **`fig_pipeline` is a layout comparison, not a pipeline diagram.** If retained, include **post-processed Barnes–Hut**, not only plain Barnes–Hut. Otherwise the main visual comparison reproduces the unmatched comparison that Results correctly rejects.
- **`fig_interfaces` belongs in the appendix unless community detection remains a headline contribution.** Its caption currently cannot be reconciled with the boundary counts in the text.
- **`fig_small_n` is evidence for an empirical default, not a sample-size law.** Keep only if that default remains central.

### Caption deficiencies

- **Layouts:** exact configuration, stiffness/exponent, preprocessing, post-process status, axis-scaling convention and point-subsampling procedure.
- **Small-\(n\):** HDBSCAN `min_samples`, noise treatment, cluster-selection rule, nesting/randomisation of subsamples; “one run each” precludes uncertainty claims.
- **Box width:** which datasets occupy left/middle panels; exact seed count per point; normalisation of RMS force error; layout used for the exact-force audit; timing inclusions.
- **Interfaces:** what each dot represents, how labels identify “same-class,” why 25 versus 119 boundaries, internal-cut definition and the 0.10 threshold.
- **Pipeline layouts:** post-processing status of every comparator, ARI alongside the two favourable axes or a nearby reference to it. The claim that zeros move “because that is their input distance” is not established by centroid Spearman 0.66→0.70.
- **Control:** dataset per panel, reference vectors, two-axis versus three-axis composites, whether markers are means, uncertainty convention, and whether paired seed-level observations are shown.

### Is the sorted-dot control the right form?

**Yes, as a per-dataset comparison—not as evidence from a pooled ranking.** Use a dumbbell/slope display connecting plain and post means, common axis limits, visible run dispersion, and explicit method labels. Sorting by post mean is acceptable.

Add the sequence-minus-Barnes–Hut paired difference to each panel:

- MNIST: **−0.046 ± 0.022**
- Fashion: **+0.020 ± 0.015**
- letter: **+0.071 ± 0.003**
- covertype: **+0.047 ± 0.013**

Label these as **SD across paired runs**, not standard errors or confidence intervals. The pooled **0.023 ± 0.047** mixes between-dataset heterogeneity with run variability; twenty runs are not twenty independent datasets.

### Should the seeds table include \(k\)-means ARI?

**Yes.** Its omission hides the sequence’s principal failure on MNIST while the prose claims a three-axis balance. Include all three raw axes for the image datasets, with the same run provenance. Reduce method rows or move timing detail rather than omit ARI.

The control figure and full control table largely duplicate each other. Keep the figure plus a compact raw-axis table in the main paper; put the full configuration table in the companion material.

## 5. RELATED WORK

A compact paragraph can cover the essential positioning. Required citations and locations:

- **The actual long-version report, versioned:** introduction, experimental provenance and reproducibility statement; naming “the long version” is not a citation.
- **van der Maaten & Hinton (2008), \(t\)-SNE:** KL objective and local-fidelity framing; already cited, but connect it to the method.
- **van der Maaten (2014), Barnes–Hut \(t\)-SNE:** benchmark implementation family and runtime comparison.
- **McInnes, Healy & Melville (2018), UMAP:** fuzzy graph, bandwidth calibration and sampled optimisation; cite at those borrowed components.
- **Dong, Charikar & Li (2011), NN-descent:** graph construction.
- **Linderman et al. (2019), FIt-SNE; Poliĉar et al., openTSNE:** interpolation and optimiser machinery; distinguish the recipe from a new optimiser.
- **Böhm, Berens & Kobak (2022), attraction–repulsion spectrum:** decoupled force balance and positioning relative to neighbour-embedding families.
- **Damrich & Hamprecht (2021), UMAP’s true loss:** negative sampling and the distinction between nominal kernels and the effective sampled objective.
- **Kobak et al. (2019), heavy-tailed \(t\)-SNE kernels:** exponent-dependent cluster granularity; explain that the present attraction-only change is different.
- **Kobak & Linderman (2021), initialisation and global structure:** staged refinement and inherited global arrangement.
- **Wang et al. (2021), PaCMAP; Amid & Warmuth (2019), TriMap:** competing global/local mechanisms; TriMap uses triplets, not simply a mid-range pair term.
- **Kruskal (1964), nonmetric multidimensional scaling:** placement by distance ranks; distinguish community-constrained translation from pointwise rank fitting.
- **Newman & Girvan (2004), modularity; Blondel et al. (2008), Louvain:** null model and community-detection algorithm.
- **Traag, Van Dooren & Nesterov (2011), constant Potts model:** the claimed near-regular-graph equivalence.
- **Fortunato & Barthélemy (2007); Schaub et al. (2012):** resolution and field-of-view limitations; retain and avoid presenting these diagnoses as new.
- **Traag, Waltman & van Eck (2019), Leiden:** qualify the choice of Louvain and distinguish connectivity defects from the scale limitations discussed here.
- **Campello, Moulavi & Sander (2013), HDBSCAN, plus the implementation reference used:** fragmentation and clustering diagnostics.
- **Narayan, Berger & Cho (2021), densMAP; Sarfraz et al. (2022), h-NNE:** panel descriptions and captions.
- **Huber (1964):** only if the robust-estimation analogy remains; otherwise remove the attribution.
- **Hubert & Arabie (1985):** ARI definition in the evaluation paragraph.

## 6. GO / NO-GO FOR arXiv AS A COMPANION

### **NO-GO in the present form. Blockers only**

1. **Incorrect implemented method.**  
   Replace “the stationary dilation … is where the full-batch pass starts” with:
   > “The shipped hand-over scale is \(31\sqrt{n/70000}\); the radial identity is diagnostic, not the rule selecting this finite-budget scale.”

2. **Misleading matched-control headline.**  
   Abstract and conclusion must contain:
   > “The paired composite difference from post-processed Barnes–Hut is 0.023 ± 0.047 across 20 runs and reverses on MNIST.”  
   Change the MNIST convergence statement to name the standalone configurations and state **sequence 0.930 versus 0.971–0.976**.

3. **False benchmark statements.**  
   Delete or correct “same ranking” on letter/covertype; “every configuration” improves; and the universal PCA comparison. Correct PaCMAP’s k41 plain MNIST ARI to **0.845**, unless an explicitly different experiment is being cited.

4. **Empirical heuristics presented as necessities or derivations.**  
   Change “must follow sample size” to **“we use an empirical size-dependent default”**; change “threshold read off a uniform cloud” to **“heuristically set to half an empirical uniform-cloud reference”**; remove the universal “repairs” claim.

5. **Unqualified general-fidelity and timing claims.**  
   Replace “local fidelity” with **“15-NN overlap under the reported evaluation”** where appropriate. Restrict global superiority to named metrics/datasets. Mark **6.2/186 s** provisional and identify whether graph construction and post-processing are included; do not describe **6–9 s** as complete-pipeline time.

6. **Unresolved experiment provenance and transfer status.**  
   Distinguish the three-seed post-process demonstration, five-run seeds table and k41 control. Resolve the **0.10 versus 0.08 ARI SD**, **25 versus 119 interfaces**, and control-SD caption. Define the allegedly held-out letter/covertype cases or relabel them as additional configurations/subsamples.

7. **Incomplete, untraceable release.**  
   Remove `[TO FILL]`; provide the actual control table; add versioned links to the report, code and result artifacts. The sentence “every result file is public” cannot remain unsupported.

### Three most valuable non-blocking improvements

1. **Replace the five-claim catalogue with the three-part argument and one actual method schematic.**
2. **Put ARI in the seeds table and paired Barnes–Hut differences in the control figure; remove a redundant layout montage and the full-width grid audit.**
3. **Add a compact attribution/limitations paragraph:** mid-range attraction supplies the global gain; matched-start refinement is not implementation-specific; the result is early-stopped and \(k\)-specific; community post-processing can create visually misleading gaps.