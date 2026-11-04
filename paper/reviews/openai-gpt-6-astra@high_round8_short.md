## Decision: **NO-GO for the present files**

The remaining release blockers are specific claim, consistency and provenance corrections; they do not require another benchmark campaign. The central result is **the highest observed mean of a particular composite on four development datasets**, not established superiority over matched, post-processed Barnes–Hut.

This review checks the supplied source, tables, summary and captions. It does not verify the repository contents or inspect the PNGs.

## 1. CLAIMS

### Abstract

| Claim | Assessment and required wording |
|---|---|
| “In the configurations we test, neighbourhood overlap and pairwise-distance rank correlation favour different operating points…” | **Supported as an empirical operating-point observation.** Do not strengthen this into an unavoidable trade-off. The introduction currently does so; see §3. |
| Sampled pass with decoupled kernels and transient mid-range attraction, followed by 300 full-batch iterations on the same graph | **Supported as a method description**, although the first-stage mid-range term remains insufficiently specified. “Places the clusters” should mean produces the starting arrangement, not recovers correct inter-cluster geometry. |
| The post-process “translates and contracts … to optimise a sampled distance-rank correlation under an overlap guard” | **Not precise as stated.** Translation is searched against Spearman under the guard; contraction is a fixed, development-selected operation. The guard is not a 15-NN constraint, and the body does not say it is enforced on contraction. **Replace:** “An optional post-process, label-free in execution, translates graph communities to increase sampled distance-rank correlation under a sampled cross-community nearest-neighbour guard, then contracts them by a fixed, development-selected factor.” |
| Standalone full-batch MNIST overlap 0.349 versus 0.350, in 4.9 s versus 175 s | **Supported by the supplied raw and timing tables.** These are quality and timing measurements from separate experiment families. They establish similar measured 15-NN overlap and a library/default-configuration runtime comparison on this machine—not equivalent objectives, accuracy or implementation efficiency. |
| The unprocessed sequence has higher mean Spearman than UMAP, PaCMAP, openTSNE and Barnes–Hut on both image datasets “at a similar 15-NN” | **Spearman ranking supported; “similar” is false across that comparator list.** MNIST overlap is 0.354 versus 0.117 for UMAP and 0.094 for PaCMAP; Fashion is 0.329 versus 0.133 and 0.099. **Replace:** “On MNIST and Fashion-MNIST, the unprocessed sequence has higher mean pairwise Spearman than UMAP, PaCMAP, openTSNE and Barnes–Hut \(t\)-SNE; its 15-NN overlap is comparable to the \(t\)-SNE baselines and higher than UMAP’s and PaCMAP’s.” |
| Highest four-dataset mean “composite of 15-NN, Spearman and \(k\)-means ARI,” winning three dataset means | **Ranking supported; axis and dataset scope need correction.** Only the image datasets use ARI; all four are development datasets. **Replace:** “With identical post-processing of all twelve configurations, the sequence has the highest observed four-development-dataset mean composite, using overlap and Spearman throughout and additionally \(k\)-means ARI on the image datasets, and wins three dataset means.” |
| Paired difference \(0.023\pm0.047\) across 20 runs, reversing on MNIST; standalone layouts 0.971–0.976 versus sequence 0.930 | **Supported.** Identify the uncertainty as **SD across paired runs**. This pooled SD mixes dataset heterogeneity and run variation. Twenty runs are not twenty independent dataset-level tests. Neither “significantly better” nor “not significantly different” follows merely by comparing the mean with this SD. |
| Operational findings: size-dependent exponent, box width 0.7, community merge heuristic | **Supported as empirical design decisions**, not laws or generally optimal settings. “Empirical size-dependent default” is preferable to “sample-size-dependent attraction exponent.” This sentence can be deleted to shorten the abstract. |
| Report, code and results are public | **A repository URL and report citation are now supplied. Actual availability remains unverified here.** The release identification is not immutable. **Replace after pinning:** “The companion report, implementation and experiment artifacts are available at [release URL], corresponding to commit [exact commit].” |

The abstract should also say that post-processing settings were selected on the development benchmarks. “Label-free in execution” is an appropriate distinction, but not a substitute for that disclosure.

### Conclusion

| Claim | Assessment and required wording |
|---|---|
| Empirical exponent, transient mid-range term, 300-step refinement, width 0.7, empirical hand-over, optional post-process | **Supported as an operational summary.** |
| Similar Barnes–Hut 15-NN overlap on image benchmarks in a fraction of the time | **Supported**, but specify which configuration and timing boundary. **Use:** “On the two image benchmarks, the unprocessed sequence obtains similar 15-NN overlap to Barnes–Hut \(t\)-SNE and takes 7.3 and 6.7 seconds, versus 175.3 and 173.8 seconds, in the interleaved timing suite with graph construction included.” The post-processed sequence instead takes 9.0 and 7.5 seconds. |
| Improved pairwise Spearman over the named default comparators on MNIST and Fashion | **Supported for the unprocessed sequence.** Say “unprocessed sequence” and repeat the comparator names or point explicitly to the raw comparison; do not leave “recipe” to mean an unspecified optional configuration. |
| Highest observed four-set mean, three wins, MNIST loss \(0.046\pm0.022\) | **Supported, but the exact round-7 requirement remains incomplete:** the conclusion omits the pooled matched effect. **Replace:** “With identical post-processing, the sequence has the highest observed four-development-set mean composite and wins three dataset means; its paired difference from post-processed Barnes–Hut is \(0.023\pm0.047\) across 20 runs and is \(-0.046\pm0.022\) on MNIST, with both uncertainties denoting run SDs.” |
| Density-contrast merging reduces selected over-partitioning failures on tested graphs; wider applicability is open | **Supported at that scope.** Do not convert this into recovering true classes, correcting modularity generally, or deriving the threshold. |

**Matched-control interpretation:** the sequence’s MNIST post-processed Spearman advantage, 0.504 versus Barnes–Hut’s 0.464, coexists with substantially worse ARI, approximately 0.728 versus 0.918. The resulting composite loss is not a negligible convergence effect. The converging group contains **standalone full-batch, openTSNE and Barnes–Hut**, not the sequence.

## 2. STORY AND STRUCTURE

### The revision has three contributions, not five

The question’s “five-claim structure” no longer describes this manuscript. The three-part argument is the appropriate scope:

1. A sampled start and finite-budget full-batch refinement.
2. An optional, method-agnostic community transformation.
3. A matched evaluation showing metric- and dataset-dependent outcomes.

The exponent study, interpolation audit and modularity discussion should remain supporting design decisions. They still occupy disproportionate main-text space.

### Cut, merge or relocate

- **Reduce the exponent paragraph to the operational rule, its empirical status and one observed failure mode.** Move HDBSCAN counts, unsuccessful variants and adaptive-rule comparisons to the companion report.
- **Reduce the community-detection motivation.** Main text needs the graph construction, community algorithm, merge statistic, threshold and limitation—not a compressed community-detection paper.
- **Keep the empirical hand-over and early-stop statements.** Move the radial calculation and detailed scale comparisons entirely to the companion report if space is tight.
- **Keep the control figure; move the full control-composite table out of the main paper.** They duplicate the same ranking information.
- **Do not reproduce every supporting appendix merely because it exists.** The box audit, interface distributions and quality-preset catalogue can live in the long report. Retain only material needed to specify the released configuration.
- **Shorten the abstract.** Remove its final catalogue of exponent/grid/merge findings before removing any matched-control qualification.

### What is present, and what remains missing

Present in the revision:

- a method schematic;
- explicit metric definitions and evaluation samples;
- raw image-dataset ARI;
- matched post-processing and per-dataset paired differences;
- integrated related-work discussion;
- early stopping, matched-optimiser qualification and \(k\)-specific limitations;
- a reproducibility paragraph.

The essential remaining method gap is **the actual sampled-stage configuration**. The text names the ingredient responsible for the global gain but does not define its loss, pair sampler, weight or time schedule. It also does not distinguish operationally the standalone sampled mode from the stiffer start used by the sequence. “Intermediate input distance” and “transient mid-range attraction” are not reproducible instructions.

Supply either a short algorithm/configuration box or an exact, versioned algorithm-and-configuration reference covering:

- the mid-range loss/force, sampling rule, strength and decay schedule;
- negative-sampling rate and main learning-rate schedule;
- the standalone-sampled versus sequence-start differences;
- the precise definition of layout standard deviation used in \(31\sqrt{n/70000}\);
- placement candidate values and treatment of communities below 1%.

A short companion need not duplicate all implementation details, but it must identify its central recipe unambiguously.

## 3. CORRECTNESS AND CONSISTENCY

### A. Numerical and provenance audit

| Item | Finding |
|---|---|
| **Barnes–Hut post-processed four-set mean** | Results says **0.884**; the control table says **0.883**. These cannot both be the same rounded aggregate. Generate both from one unrounded artifact. |
| **Sequence post mean 0.907** | Displayed dataset means average to **0.90625**. This is **not by itself proof of an error**: averaging unrounded values could round to 0.907. Verify from artifacts rather than replacing it mechanically with 0.906. The paired mean 0.023 is consistent with the supplied per-dataset differences. |
| **Raw-table SD caption** | “0.000–0.01 except the sequence’s MNIST ARI” is not compatible with the supplied k41 variability. For example, covertype UMAP’s weak normalized axis is Spearman, with minimum composite \(0.482\pm0.053\); that corresponds approximately to raw Spearman SD **0.040**, not at most 0.01. Regenerate raw-axis uncertainties. Do not infer or copy them indiscriminately from composite SDs. |
| **Timing variability** | “The ranges … are within 5% except Barnes–Hut’s” is false. Fashion sampled is **3.2–4.5 s around 3.7**, and Fashion sequence is **6.4–7.2 around 6.7**. Delete the blanket statement or report the actual exceptions. The **156–184 s** Barnes–Hut range is Fashion’s; MNIST is **170.8–181.2**. |
| **Pipeline caption’s ARI citation** | It cites Table `raw` for complete-pipeline MNIST ARI \(0.73\pm0.07\). That table contains the **unprocessed** sequence’s **0.596**, not the post-processed **0.728**. Cite the configurations table/control paragraph and the artifact containing the SD. |
| **Three-run versus five-run full-batch scores** | **Now distinguished.** \(0.349/0.350/0.761\rightarrow0.347/0.469/0.914\) is the three-run demonstration; k41 gives \(0.349/0.350/0.791\rightarrow0.346/0.466/0.915\). Do not collapse them into one result. |
| **Quiet versus shared-machine timings** | **Now distinguished.** Sequence 7.3 s versus 11.2 s is not an internal contradiction when the suite is labelled. The method’s “4 s more” versus the schematic’s 3.7 s is ordinary rounding. Label the 3.6 s first-stage timing as including graph construction, so readers do not add the graph’s 1.2 s twice. |
| **Post-process runtime range** | “0.8–4.5 s on 70k points in the control runs” cannot be verified from the supplied summary, whose 70k row means are 1.0–1.9 s. Specify whether this is a range of individual replicates, another suite or another timing boundary. Do not use the 20k covertype 5.0 s row as a 70k counterexample. |
| **Frozen references** | The vectors are supplied, but “calibration runs of the companion report” is not an exact provenance identifier. Name the generating script/result file and configuration for each maximum. MNIST ARI reference **0.914**, for example, is not the maximum of the displayed unprocessed k41 means. That can be legitimate, but must be traceable. |
| **Other quoted experiments** | The early-stop trajectory, matched-start ranges, small-\(n\) studies and force audit are additional experiment families beyond the three introduced in “Runs and provenance.” Say “the three principal evaluation families” and identify the others where quoted. |

### B. The interface-count explanation is still insufficient

The appendix says:

> “The 25 real boundaries quoted in Sec. … are the adjacent pairs at resolution 0.5; the figure pools both resolutions.”

First, Section 3 no longer quotes 25: that cross-reference is stale.

More importantly, **pooling alone does not reconcile the numbers as written**. The figure has 13 communities at resolution 1.0, hence at most \(\binom{13}{2}=78\) distinct undirected community interfaces at that resolution. If there are 25 different-class adjacent pairs at resolution 0.5, the two partitions can contribute at most **103** different-class interfaces—not **119**.

Therefore at least one of the following is missing or wrong:

- 25 counts class pairs rather than community pairs;
- different filters are used;
- multiple graphs/runs are pooled;
- directed observations or duplicate measurements are counted;
- one of the counts is stale.

Regenerate the counts and define the observational unit and deduplication. The current explanation does not close the round-7 issue.

### C. Remaining overclaims and undefined operations

**1. The introduction still states an excessive trade-off claim.**

> “two things that, in every configuration we measured, trade against each other”

Replace with the abstract’s narrower operating-point statement. Pairwise Spearman also measures **all sampled pairs**, not specifically distances between different clusters. Replace the inter-cluster framing with “the ordering of sampled pairwise input distances.”

**2. The size rule is empirical in the method but necessary again in Limitations.**

> “Below 15k points the sampled pass needs the lighter attraction…”

Replace:

> “On the tested small-data cases, increasing the attraction exponent reduces observed fragmentation, although it does not remove every clustering failure.”

The evidence does not establish a universal 15k transition. Dataset geometry, graph construction and size-dependent HDBSCAN settings remain relevant.

Also identify the configuration and dataset behind:

> “the HDBSCAN use case is a clear win at 70k (ARI 0.92 against 0.66…)”

As written, this can be read as a result for the sequence, while the long abstract describes poor hybrid clustering. State the mode, dataset, run count and source, or remove this aside.

**3. The uniform-cloud reference is not uniformly near 0.20.**

The appendix’s Fiedler shares range from **0.17 to 0.28** over dimensions 2–50. Calling this “about 0.20, weakly dependent on dimension” understates the variation.

Use:

> “We choose 0.10 heuristically; the simulated Fiedler-cut reference ranges from 0.17 to 0.28 across the tested dimensions and is approximately 0.20 in the lower-dimensional cases.”

The selection of half a reference is now identified as heuristic. Keep that qualification.

**4. The radial identity is algebraically correct; two interpretations remain excessive.**

With normalized \(p\), excluded diagonals and \(q=k/Z\),

\[
\frac{\partial F}{\partial\log s}
=2\left(\mathbb E_p[u]-\mathbb E_q[u]\right)
\]

is correct. The appendix now distinguishes a frozen-shape radial stationary point from a jointly optimal scale.

However:

- The identity does not by itself “explain why the pass expands every start.” Its sign diagnoses radial pressure **at a specified layout**; subsequent shape changes, momentum and gains affect the trajectory.
- “The stationary dilation gives a worse trade-off than 31” is not a quality dominance result. The supplied values are **0.349/0.460 versus 0.350/0.450**: one axis improves and one decreases.

Replace the latter with:

> “Starting at the stationary dilation costs about five times as much and gives 0.349/0.460 overlap/Spearman versus 0.350/0.450 at scale 31; we therefore use 31 as an empirical finite-budget setting.”

**5. The whole post-process is not rigid-body motion.**

> “The post-process moves clusters as rigid bodies.”

Contraction is not rigid. Use:

> “The post-process translates communities and contracts each uniformly about its centroid.”

The later explanation of preserved within-community distance order and changed cross-community neighbours is appropriately scoped.

**6. “Deterministic” needs a condition.**

Strength-ordered visitation does not make approximate graph construction deterministic, nor necessarily resolve every Louvain tie. Say “deterministic for a fixed graph and specified tie-breaking,” if that is implemented.

**7. Clarify what is matched.**

“Graph and communities of that run” does not establish whether all methods in a paired replicate share one post-processing graph/community partition or independently reconstructed instances. Specify this, the shared search samples, and the pairing key. The control matches the post-processing procedure—not embedding objectives, computation budgets or necessarily every source of randomness.

### D. Short/long consistency

These are not all errors in the short paper; they are unresolved contradictions in the proposed companion release.

- **PCA:** the short paper’s covertype statement is correct. The long abstract’s claim that PCA remains ahead of every unprocessed neighbour embedding on real datasets is false for k41: sequence **0.536**, sampled **0.592**, TriMap **0.748**, versus PCA **0.397**.
- **MNIST matched control:** the long abstract says the margin “vanishes.” It **reverses**, by \(-0.046\pm0.022\). Replace “vanishes” with the measured reversal.
- **Transfer:** the short paper labels all four datasets as development sets; the long abstract calls letter and covertype cases “held-out.” Supply a tuning chronology that justifies a specific frozen-rule transfer experiment, or remove the held-out terminology from the long abstract.
- **Sequence means:** long **0.353/0.446, 0.329/0.750** versus k41 **0.354/0.443, 0.329/0.751** can be different runs, not contradictions. Identify the source explicitly in both documents.
- **Fastest CPU claim:** the long abstract’s literal twelve-configuration claim conflicts with the supplied k41 timings: PCA is faster, and Fashion h-NNE is **5.1 s** versus sampled MAGNET **5.8 s**. Restrict the panel/suite precisely or remove the claim.

## 4. FIGURES AND TABLES

### Which figures earn space?

- **Method schematic and matched-control figure:** retain.
- **Pipeline montage:** optional, but now includes post-processed Barnes–Hut, so the old unmatched-visual-comparison objection no longer applies. It must not substitute visual separation for evidence of input clusters.
- **Box-width figure:** still the lowest-priority figure for a short companion. It is now in the appendix, so the former objection to a full-width main-paper slot is no longer applicable. It can live solely in the long report.
- **Interface figure:** likewise belongs only in supporting material unless community detection becomes a principal contribution. Its counts currently need correction.
- **Small-\(n\) figure:** supports a default-selection observation, not a size law.
- The obsolete redundant layout figure is absent; do not repeat the round-7 demand to remove it.

### Remaining caption defects

- **Method:** identify the empirical hand-over scale and state that the sampled timing includes graph construction. From the caption alone, the central mid-range ingredient and rescaling are not auditable.
- **Pipeline:** correct the ARI table reference. Clarify whether these are k41 runs or separate display runs: “seed 0” for every panel conflicts with unseeded UMAP if they are k41 outputs. Replace “labels unused by every method” with “class labels are not supplied during embedding or post-processing execution”; development and evaluation did use label-derived scores.
- **Box width:** specify whether the error concerns the repulsive force or the complete KL gradient. Clarify whether “first 20,000 points” means 20,000 query forces interacting with all 70k points or a 20k subsystem. State timing inclusions.
- **Interfaces:** define the counting unit, pooling and deduplication; identify the 0.10 decision threshold in the caption. The Fiedler-cut definition is now supplied.
- **Small-\(n\):** the main HDBSCAN and sampling choices are now given. Write the effective `min_samples` value rather than relying only on a library default, or pin the implementation version.
- **Control:** the essential semantics are supplied. Add an explicit cross-reference to the reference vectors and ensure SD is shown for both plain and post values, not just individual post dots.

### Is the sorted-dot control the right form?

**Yes, for per-dataset comparison.** Prefer connected plain/post means—a dumbbell display—with common limits, method labels and visible run dispersion. Sorting by post mean is acceptable.

Retain the four paired differences:

- MNIST: **−0.046 ± 0.022**
- Fashion: **+0.020 ± 0.015**
- letter: **+0.071 ± 0.003**
- covertype: **+0.047 ± 0.013**

They must remain **means ± paired-run SD**, not confidence intervals. Do not let a pooled ranking obscure the sign reversal.

### Should the seeds table include \(k\)-means ARI?

**Yes—and the revised raw-axis table now does so for both image datasets.** There is no separate seeds table in this short revision. The outstanding issue is reliable uncertainty reporting, not adding an already-present column.

If the long version’s matched-start/seeds table supports claims about “repairing” arbitrary starts, it should also expose ARI. A two-axis repair is not three-axis quality recovery.

## 5. RELATED WORK

There is **no separately titled Related Work section**, but there is related-work content in the introduction and methods. A new section is unnecessary. Required citations/positioning, one line each:

- **Chenebaux, companion report — present:** introduction, experiment provenance and reproducibility; pin its version.
- **van der Maaten & Hinton (2008) — present:** KL objective and scope of the \(t\)-SNE comparison.
- **van der Maaten (2014) — present:** Barnes–Hut baseline and acceleration family.
- **McInnes, Healy & Melville (2018) — present:** fuzzy affinities, bandwidth calibration and sampled optimisation.
- **Dong, Charikar & Li (2011) — present but bibliographically incorrect:** NN-descent; fix “C. Moses” to **M. Charikar** and “W. Li” to **K. Li**.
- **Damrich & Hamprecht (2021) — present:** effective negative-sampling objective, not just nominal kernels.
- **Linderman et al. (2019) — present:** FFT/interpolation machinery and numerical audit.
- **Poličar et al., openTSNE — present:** optimiser machinery and matched-optimiser comparison.
- **Böhm, Berens & Kobak (2022) — present:** attraction–repulsion operating points and limits of a new-objective claim.
- **Kobak et al. (2019) — present:** heavy-tailed kernels; distinguish attraction-only changes from coupled kernel changes.
- **Kobak & Linderman (2021) — present:** inherited geometry and initialisation dependence.
- **Wang et al. (2021), PaCMAP — present:** transient/mid-near attraction positioning.
- **Amid & Warmuth (2019), TriMap — present:** triplet-based alternative, not merely another mid-range pair loss.
- **Kruskal (1964), nonmetric MDS — missing:** community placement by distance ranks; distinguish constrained translation from pointwise rank fitting.
- **Newman & Girvan (2004), modularity — missing:** null-model argument in the community paragraph.
- **Blondel et al. (2008) — present:** Louvain implementation family.
- **Traag, Van Dooren & Nesterov (2011) — present:** constant-Potts comparison.
- **Fortunato & Barthélemy (2007); Schaub et al. (2012) — present:** distinguish resolution-limit and geometric/field-of-view diagnoses.
- **Traag, Waltman & van Eck (2019) — present:** Leiden; distinguish connectivity guarantees from the reported instability/scale issue.
- **Campello, Moulavi & Sander (2013), plus the actual HDBSCAN implementation reference — theory present, implementation missing:** clustering diagnostics and default semantics.
- **Narayan, Berger & Cho (2021), densMAP — missing:** baseline panel.
- **Sarfraz et al. (2022), h-NNE — missing:** baseline panel.
- **Hubert & Arabie (1985) — present:** ARI definition and external-label evaluation.

Huber is no longer required: the robust-estimation analogy has been removed.

## 6. ROUND-7 BLOCKER CHECK AND RELEASE VERDICT

| Round-7 blocker | Status |
|---|---|
| **1. Incorrect implemented hand-over** | **Resolved.** The main text now gives \(31\sqrt{n/70000}\) and explicitly says the radial identity does not select it. The residual “every start” and scale-trade-off wording should still be corrected, but the implemented rule is no longer contradicted. |
| **2. Misleading matched-control headline** | **Not fully resolved against the stated round-7 requirement.** The abstract includes the pooled difference and reversal. The conclusion says “has the highest observed four-set mean composite … losing … on MNIST by \(0.046\pm0.022\)” but omits **\(0.023\pm0.047\)**. Add it; also make the abstract’s two-axis/three-axis distinction explicit. |
| **3. False benchmark statements** | **Resolved in the revised short paper.** The letter/covertype rankings, post-processing exceptions, PaCMAP ARI and short-paper PCA statement have been corrected. The contradictory long abstract remains a companion-consistency issue under blocker 6. |
| **4. Heuristics presented as necessities or derivations** | **Not fully resolved.** The main method now calls the rules empirical/heuristic, but Limitations still says: “Below 15k points the sampled pass needs the lighter attraction.” Remove the necessity claim. Correct the understated dimensional variation in the uniform-cloud reference. |
| **5. Unqualified fidelity and timing claims** | **Not fully resolved.** The new timing protocol replaces the old unmatched-load comparison, but the abstract still says “at a similar 15-NN” across dissimilar comparators, and Results falsely says timing ranges “are within 5% except Barnes–Hut’s.” |
| **6. Experiment provenance and transfer status** | **Not fully resolved.** Principal experiment families and the MNIST ARI variability are now distinguished, but the raw-SD caption is inconsistent with k41; the “25 … figure pools both resolutions” explanation does not reconcile the interface counts; calibration artifacts remain unidentified; and the long abstract still calls development-family cases “held-out.” |
| **7. Incomplete/untraceable release** | **Not fully resolved.** The placeholder is gone and the control table, report citation and repository URL are present. But “this paper corresponds to commit `2242d27` **and later**” does not identify an immutable release, and “as installed in November 2026” is not a sufficient environment identifier without a pinned manifest. Supply one exact revision for manuscript, report, scripts, artifacts and dependency manifest. |

### Additional blocker

**The central sampled-stage configuration is not operationally identified.** The sentence

> “a transient mid-range attraction on a sample of pairs at intermediate input distance”

names the source of the claimed advantage without specifying it. Supply the defining equations/schedule and standalone-versus-sequence settings, or an exact versioned algorithm/configuration pointer. This is specification work, not a request for new experiments.

### Three highest-value non-blocking improvements

1. **Remove duplication:** keep the method and control figures, move the full composite table and implementation-audit appendices to the long report, and use the recovered space for a concise argument.
2. **Expose post-processed raw trade-offs:** add a compact raw-axis display, especially MNIST Spearman versus ARI, rather than asking readers to infer the failure mode from composites.
3. **Show composite sensitivity using existing results:** indicate how rankings change without ARI or under the minimum composite, while keeping dataset-specific paired effects primary. This would separate a useful operating point from a ranking dependent on the chosen scalarisation.