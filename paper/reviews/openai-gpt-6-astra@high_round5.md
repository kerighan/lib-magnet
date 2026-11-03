## 1. LEGITIMACY

### Spearman: metric fitting, not automatically test-set leakage

The placement **directly fits one reported evaluation criterion**. That must be explicit. It is not inherently illegitimate: MDS can be evaluated by stress, and a rank-preserving embedding can be evaluated by rank agreement. “Optimising the test metric” and “using the test observations to optimise” are different accusations.

Here:

- Optimising Spearman on a search sample is legitimate.
- Evaluating on a separately drawn sample measures agreement beyond the sampled distances used for optimisation.
- Repeatedly choosing resolution, contraction, guard and ordering after inspecting benchmark scores is nevertheless **benchmark-guided method development**. An independent search sample does not erase that.
- These remain transductive layout-fidelity measurements, not predictive generalisation to unseen observations.

**Reporting protocol:** retain Spearman, mark the affected rows or column with “sample-Spearman fitted,” report search and evaluation sample sizes, and distinguish development data from frozen-rule evaluations. Prefer explicitly disjoint search/evaluation point sets, with the evaluation set excluded from parameter selection. A different global metric should supplement, not replace, Spearman.

The implementation does **not enforce disjointness**. Independent 500- and 2,000-point draws from 70,000 points share about 14 points on average. Independence and disjointness are not interchangeable; reconcile the research log’s “disjoint” with the actual benchmark sampler.

Triplet accuracy and centroid Spearman are useful corroboration, but:

- triplet accuracy is another distance-order criterion, not an entirely independent notion of fidelity;
- the evidence that **both** increase is the cached MNIST example;
- on Fashion, triplet accuracy decreases, **0.775 → 0.765**, while centroid Spearman increases.

### ARI: accept the measurement, reject the interpretation as independent cluster discovery

Contraction makes the supplied communities more compact. It does **not** raise label-based \(k\)-means ARI by mathematical construction: that additionally requires suitable community–label alignment, centroid arrangement, community counts and \(k\)-means behaviour. Your own controls demonstrate this:

- MNIST PaCMAP: **ARI 0.914 → 0.881**;
- Fashion PaCMAP: **0.483 → 0.460**.

I would accept the reported ARI gain as **improved label recovery by \(k\)-means on a graph-partition-conditioned layout**. I would not accept it as evidence that the embedding discovered previously absent clusters, or that the visible gaps faithfully represent input-space density.

The graph already contains substantial label-aligned structure: the quoted MNIST community ARI is 0.85. The post-process makes that structure easier for a particular downstream clustering procedure to recover.

Also distinguish **label-free execution** from **label-free selection of the recipe**. If \(g=0.7\) or resolution 0.3 was chosen after inspecting label-based ARI, say so.

### Appropriate framing

A sceptical but accurate description is:

> **A method-agnostic, graph-conditioned cluster-placement and separation post-process, directly targeting sampled distance-rank agreement and indirectly favouring compactness-based clustering.**

“Cosmetic” understates the measured changes in distance agreement. “Metric hacking” becomes justified if the paper hides the fitted criterion, presents contraction-induced gaps as discovered structure, or compares enhanced MAGNET rows against unenhanced competitors as evidence of MAGNET superiority.

Credit the **cheap restricted optimisation, guarded implementation, empirical gains and cross-method applicability**. Do not credit this as a MAGNET-specific advantage: the Barnes–Hut result directly argues otherwise.

---

## 2. WHAT THE RESULT MEANS

### “The three axes are not in conflict; the objectives are.”

**Not supported as a general statement.** The observations support a narrower conclusion:

> On these MNIST layouts, substantial gains in distance-rank agreement and \(k\)-means ARI are available at a small measured 15-NN cost.

There is still a local cost, so this is not literal Pareto dominance. Fashion also loses Spearman; covertype 250k loses **0.013** in 15-NN. Nothing establishes that these metrics are generally compatible, or that the previous search plateau was caused by the objective rather than optimisation, parameterisation or the search space.

The explanatory sentence is mathematically wrong:

- A community-wise similarity preserves **within-community distance order**, not the full layout’s neighbourhoods.
- Pairwise Spearman depends on **within- and between-community distances and their joint ordering**, not only cluster arrangement and relative size.
- \(k\)-means depends on centroid geometry, cluster sizes and internal dispersion—not compactness alone.
- A contraction is a similarity, **not a rigid transformation**.

### “The arrangement … is far from rank-optimal.”

There is substantial rank-agreement headroom on the tested MNIST layouts. The improvement is meaningful: roughly **+0.11** in the matched three-seed comparison.

However:

1. No optimum or optimality gap is estimated.
2. The quoted **0.342 → 0.460** endpoint includes **placement and separation**, not placement alone.
3. The result does not establish the same behaviour for arbitrary t-SNE-like layouts or datasets.

Use “substantial headroom under community-level adjustments,” not an unqualified rank-optimality claim.

A particularly informative existing comparison is:

- separation alone: **0.346 / 0.327 / 0.901**;
- placement followed by separation: **0.346 / 0.460 / 0.902**.

Thus adding placement before contraction substantially improves the final Spearman while leaving these other two measurements essentially unchanged. That is a defensible mechanism-level observation.

### Prior work that needs explicit discussion

The section currently presents familiar ingredients without locating them. At minimum:

| Line of work | References to discuss | Relevance |
|---|---|---|
| Ordinal/rank-based embedding | **Shepard (1962); Kruskal (1964), nonmetric MDS** | Optimising distance-order agreement is longstanding. Nonmetric stress is not identical to Spearman, but the conceptual objective is not new. |
| Coarse/multilevel layouts | Representative multilevel graph-layout work, e.g. **Harel and Koren (2002)** or **Walshaw (2003)** | Positioning aggregates and subsequently handling within-group geometry is established. |
| Hierarchical dimensionality reduction | **Sarfraz et al., h-NNE (2022)** | A particularly relevant comparison for using a hierarchy/partition to organise and separate groups. |
| Global neighbour embeddings | **Amid and Warmuth, TriMap; Wang et al., PaCMAP (2021); Zhou and Sharpee, global t-SNE (2022)** | Global constraints and mid-range information have already been used to improve cluster arrangement. These are not the same post-process. |
| Informative initialisation | **Kobak and Berens (2019); Kobak and Linderman (2021)** | Global arrangement can be supplied separately from local refinement. |
| Community detection | **Traag, Waltman and van Eck (2019), Leiden** | The partitioning algorithm and its objective need citation. |

It is **Shepard**, not “shepherd.” A Shepard diagram is a diagnostic; it is not itself a layout-refinement algorithm.

The research log’s failed class-centroid MDS experiment—Spearman **0.46 → 0.30**—is worth briefly reporting. It distinguishes centroid-distance fitting from fitting the rank order of all sampled point-pair distances. It does not establish novelty over the broader coarse-layout literature.

**Potential novelty:** the particular inexpensive combination of sampled Spearman coordinate search, a community-mixing guard, and subsequent uniform community contraction, with evidence that it improves several finished embeddings. No new optimisation primitive or general compatibility theorem is demonstrated. A priority claim about post-hoc community placement would require a closer literature comparison than is presently supplied.

---

## 3. PROTOCOL

### Classification of the issues

| Issue | Assessment |
|---|---|
| **Resolution 0.3** | A limitation, not inherently a blocker. One numerical resolution does not represent comparable granularity across datasets. Claims of robustness to community choice would require sensitivity measurements. Report actual community counts by run. |
| **Hand-set \(g=0.7\)** | Acceptable for an exploratory option if the selection history is disclosed. It is not “the one dial”: resolution, guard tolerance, sampling, candidate steps and sweeps are also choices. It cannot inherit the staged procedure’s frozen-rule transfer claim without evidence. |
| **Overlap guard** | A legitimate constraint, but not a neighbourhood-preservation guarantee. It measures sampled nearest-neighbour community mixing, not 15-NN preservation or geometric overlap. The code constrains mixing relative to its **initial** value and does not recheck after contraction. Describing it otherwise is a factual blocker. |
| **One search sample** | A limitation. About 250 adaptive evaluations on the same sample can overfit it. Independent evaluation is useful; repeated search samples would establish stability. The \(\binom{500}{2}\) distances are not independent observations. |
| **Moving composite references** | Mathematically permissible, but the present cross-table and headline comparisons are ambiguous. This is a reporting blocker, not evidence that the raw improvements are fictitious. |
| **Three seeds / single runs** | Enough for a scoped exploratory arXiv result, not broad robustness claims. Specify whether layout, Leiden and search-sampling seeds vary together. Single-run results must remain descriptive. |
| **“Cuts a continuum”** | A substantive limitation of the representation. The operation does not delete graph edges; it can manufacture visual gaps at partition boundaries in connected data. State that explicitly. |
| **Only MAGNET post-processed rows in the aggregate leaderboard** | The ranking is descriptive of an asymmetric set of configurations. It cannot establish method superiority. Either add matched post-processed baselines or remove that interpretation. |

### Implementation details missing from the prose

The placement does **not** move every community. It excludes communities smaller than **1% of the dataset**; contraction still applies to all communities. Consequently, at most 100 communities are eligible for placement, even when Leiden returns 141.

Also disclose:

- the fixed initial global centre and RMS radius;
- five radial candidates and the eight square-grid offsets—the diagonal offsets have length \(0.15\sqrt{2}R\), not \(0.15R\);
- the required sampled Spearman improvement of \(10^{-4}\);
- the guard’s initial-value reference and its application only during placement.

These details matter especially for the claimed scaling with community count.

### Numerical/provenance corrections required

**1. The headline mixes a cached baseline with a three-seed endpoint.**

The matched three-seed MNIST comparison in `k38` is:

| | 15-NN | Spearman | ARI |
|---|---:|---:|---:|
| Before | **0.347** | **0.350 ± 0.003** | **0.787 ± 0.010** |
| After | **0.346** | **0.461 ± 0.002** | **0.916 ± 0.009** |

The claimed starting triple **0.348 / 0.342 / 0.80** is the cached example, not those three runs. The composite means **0.848 → 0.976** are supported using the frozen original references.

**2. The Fashion cached row is not supported by the supplied logs.**

The table gives **0.321 / 0.684 / 0.452** after placement and separation. The cached `k37` run gives **0.320 / 0.675 / 0.486**. The table’s triplet and centroid measurements also need provenance.

Do not patch individual cells from different runs. Identify the actual run supporting the entire row, or regenerate it consistently.

**3. Community counts are misstated.**

`k37` reports **11 Fashion communities, 10 above 1%**, not 10 total. Across the three-seed study, counts are **9–11 on MNIST** and **10–11 on Fashion**.

**4. Composite references must be named.**

With the original reference Spearman 0.488:

\[
0.461/0.488=0.945,
\]

so “within 6%” is true **relative to the original unprocessed panel**.

With the expanded-panel reference 0.514:

\[
0.461/0.514=0.897,
\]

so the deficit is about **10.3%**, not 6%. Against the cached hybrid-plus-step value 0.523, it is about 11.9%. Those are different runs/panels and must not be silently interchanged.

Changing reference bests:

- does not alter within-dataset geometric-composite ordering when the same denominator is applied to every method;
- can alter cross-dataset mean rankings by changing dataset weights;
- can alter minimum-composite rankings.

Either freeze the original references throughout, allowing ratios above one, or clearly distinguish original-reference and expanded-reference scores. “0.976” is not an absolute quality scale.

**5. Timings need scoped denominators.**

The post-process itself is not 25 times slower on Barnes–Hut: `k38` reports **4.8 s**, versus approximately **6 s** on the repeated MAGNET layouts. If 25× refers to base-layout generation, say so and supply that comparison. Do not carry that ratio over to total pipelines.

Likewise, the repeated runs have MAGNET layout times around **11 s**, not 8 s. “8 s + 6 s” needs an explicitly identified timing protocol rather than joining measurements from different runs. For other methods, include any additional graph-construction cost in an end-to-end comparison.

---

## 4. TEXT

The new contribution is **item 4 in the pasted enumeration**. The literal item 5 also needs a scope correction because the new results invalidate its unqualified PCA claim.

### Section 4.6

**Original heading**

> “Community placement and separation: the three axes at once”

**Replace with**

> “Community placement and separation: a metric-targeted post-process”

---

**Original**

> “The three axes are not in conflict; the objectives are.”

**Replace with**

> “On the tested MNIST layouts, community-level adjustments substantially improve pairwise-distance Spearman and \(k\)-means ARI at a small measured cost in 15-NN preservation; this does not establish general compatibility of the three criteria.”

---

**Original**

> “A similarity applied to one cluster of a layout leaves its neighbourhoods unchanged, a rank statistic of the pairwise distances only sees how the clusters are arranged and how large they are relative to their spacing, and \(k\)-means sees how compact they are.”

**Replace with**

> “A similarity within a community preserves its internal distance ordering, but can change neighbours across community boundaries. Community translations and contractions also change the joint ordering of within- and between-community distances and the partition recovered by \(k\)-means.”

---

**Original**

> “contracting every class of the hybrid’s MNIST layout rigidly towards its centroid by 0.4, labels used”

**Replace with**

> “uniformly scaling each labelled class about its centroid by 0.4, using labels in this oracle diagnostic”

**Original**

> “The plateau of the parameter and graph searches … is a plateau of the objective.”

**Replace with**

> “This oracle finds a different metric trade-off from those reached by the preceding searches; it does not identify the cause of their plateau.”

---

**Original**

> “The label-free step … treats the communities of the fuzzy graph as rigid bodies.”

**Replace with**

> “The step translates graph communities as rigid bodies during placement, then uniformly contracts each community about its centroid. Its execution uses no class labels.”

Add the implementation qualification:

> “Placement considers only communities containing at least 1% of the points, whereas contraction applies to all communities; the mixing guard is referenced to the initial layout and is not rechecked after contraction.”

---

**Original**

> “The communities are Leiden’s at resolution 0.3 … (9 on MNIST …; 10 on Fashion-MNIST …)”

**Replace with**

> “Leiden is run on weighted modularity at resolution 0.3. The cached runs in `k37` contain 9 communities on MNIST and 11 on Fashion-MNIST, of which 9 and 10 respectively meet the placement size threshold; the three-seed study yields 9–11 and 10–11 communities.”

Attach the community–label ARIs only to the particular partitions for which they were measured.

---

**Original**

> “On MNIST the arrangement of a \(t\)-SNE-like layout’s clusters is far from rank-optimal: nine translations lift the full-batch layout’s Spearman from 0.342 to 0.460 …; the separation then takes \(k\)-means ARI from 0.80 to 0.90.”

**Replace with**

> “On the cached MNIST full-batch layout, placement followed by contraction changes 15-NN / Spearman / ARI from 0.348 / 0.342 / 0.802 to 0.346 / 0.460 / 0.902. Contraction alone gives 0.346 / 0.327 / 0.901, showing that including placement substantially improves the final distance-rank agreement. These measurements do not isolate placement alone.”

Follow with:

> “For this combined transformation, triplet accuracy increases from 0.626 to 0.646 and class-centroid Spearman from 0.669 to 0.739.”

---

**Original**

> “Over three seeds … every axis within 6% of the panel’s best, in 8 s of layout plus 6 of step.”

**Replace with**

> “Across three matched seeds, the full-batch means change from 0.347 / 0.350 / 0.787 to 0.346 / 0.461 / 0.916 in 15-NN / Spearman / ARI. Using the frozen original-panel references 0.352 / 0.488 / 0.914, the composite changes from \(0.848\pm0.002\) to \(0.976\pm0.004\), and each final axis is at least 94% of its reference. The post-process takes approximately 6 s given the graph and layout.”

Define the reported ± values as run standard deviations, not uncertainty intervals.

---

**Original**

> “In the method × data set tables the three rows with the step take the first three places…”

**Replace with**

> “The three post-processed MAGNET configurations have the highest aggregate composites among the configurations tabulated here. Because corresponding post-processed competitors are absent from this aggregate table, these ranks do not establish a MAGNET-specific advantage.”

Better still, remove the podium sentence and foreground paired changes.

---

**Original**

> “The step is a post-processing of a layout, not an objective…”

**Replace with**

> “This is an optional post-processor with an explicit sampled-Spearman placement objective, not a replacement for the base embedding objective.”

**Original continuation**

> “…applied to the Barnes–Hut \(t\)-SNE layout … it scores like ours … at 25 times the time…”

**Replace with**

> “Applied to a Barnes–Hut \(t\)-SNE layout using the same graph communities, it yields 0.349 / 0.461 / 0.910 on MNIST, demonstrating that the benefit is not specific to MAGNET; the post-processing itself takes 4.8 s in this run.”

---

**Original**

> “Its cost grows with the number of communities…”

**Replace with**

> “Placement cost depends on the number of communities meeting the 1% size threshold, the candidate count and the sweeps; graph partitioning and contraction additionally depend on the full graph and dataset.”

---

**Figure caption**

> “On MNIST the zeros move away from the other digits, as the input distances say.”

**Replace with**

> “The translations are selected by sampled distance-rank agreement; the subsequent contraction deliberately increases separation at graph-community boundaries.”

### Abstract

**Original**

> “The three axes are nevertheless compatible: a label-free post-processing … takes the full-batch MNIST layout from … 0.348 / 0.342 / 0.80 to 0.346 / 0.461 / 0.92 over three seeds…”

**Replace with**

> “An optional label-free post-process translates graph communities to improve sampled distance Spearman, then contracts them about their centroids. Across three matched MNIST seeds, the full-batch mean 15-NN / Spearman / ARI changes from 0.347 / 0.350 / 0.787 to 0.346 / 0.461 / 0.916, with Spearman evaluated on a separately drawn sample. Comparable gains on a Barnes–Hut \(t\)-SNE layout show that this is a method-agnostic adjustment.”

If retaining “within 6%,” append **“of the frozen original, unprocessed panel’s bests.”**

**Original**

> “PCA stays ahead of every neighbour embedding on pairwise Spearman on the real data sets”

**Replace with**

> “Before the optional community post-processing, PCA stays ahead of every neighbour embedding in the panel on pairwise Spearman on the real data sets.”

### New contribution

**Original**

> “The three axes are compatible…”

and

> “It lifts a Barnes–Hut \(t\)-SNE layout the same way, at 25 times the time.”

**Replace the contribution with**

> “**A metric-targeted community post-process.** We evaluate sampled-Spearman placement under a community-mixing guard, followed by uniform community contraction. On MNIST this substantially improves distance-rank agreement and \(k\)-means ARI at a small 15-NN cost, with similar gains on a Barnes–Hut \(t\)-SNE layout. The procedure is method-agnostic, uses no labels during execution, and can impose visual gaps at graph-community boundaries.”

For literal contribution 5, replace:

> “PCA leads every neighbour embedding on both”

with:

> “Among the configurations without community post-processing, PCA leads every neighbour embedding on both.”

### Limitations

**Original**

> “…the placement and separation step … does reach it…”

This is ambiguous after a sentence about the hybrid. The **hybrid does not reach PaCMAP’s MNIST ARI**: its post-processed three-seed mean is 0.649.

**Replace the relevant passage with**

> “Applied to the full-batch layout, the post-process brings mean MNIST \(k\)-means ARI to 0.916, comparable to the original panel’s PaCMAP value of 0.914; the post-processed hybrid remains lower. Placement directly fits sampled Spearman, and independently sampled evaluation does not remove benchmark-guided selection of the recipe. Both auxiliary global metrics increase in the cached MNIST example, but not on Fashion-MNIST. Resolution 0.3, contraction 0.7 and the guard settings are development choices, with only three runs on each image dataset and single runs elsewhere. The hard partition can create visual gaps in connected structure without removing graph edges; improved separability is not evidence of new input-space clusters.”

Also scope “PCA keeps the lead” to configurations **without** this post-process.

### Conclusion

**Original**

> “…a label-free placement and separation … puts every axis within 6% of the panel’s best on MNIST.”

**Replace with**

> “On MNIST, the optional metric-targeted community post-process improves full-batch Spearman and \(k\)-means ARI with a small 15-NN loss, reaching at least 94% of each frozen original-panel reference; similar results on Barnes–Hut \(t\)-SNE establish its method-agnostic scope.”

**Original**

> “With its settings frozen the recipe transfers to held-out tabular and text data…”

**Replace with**

> “The frozen-rule transfer results concern the staged embedding procedure; they do not establish frozen-rule generalisation of the subsequently developed community post-process.”

---

## 5. GO / NO-GO FOR arXiv

**NO-GO as pasted; GO after a factual and scope patch. No additional experiment is essential for a clearly labelled exploratory section.**

### Blockers only

1. **Correct the purported three-seed starting triple.**  
   Abstract and contribution: **0.348 / 0.342 / 0.80 → 0.347 / 0.350 / 0.787**. Keep the cached example separately identified.

2. **Resolve the unsupported Fashion row and partition counts.**  
   Trace **0.321 / 0.684 / 0.452** and its auxiliary metrics to one run, or replace the complete row consistently. Correct cached Fashion **10 communities → 11 total, 10 eligible**.

3. **Make composite references explicit and remove the resulting contradictions.**  
   “Within 6%” must say **original unprocessed panel**, not the expanded panel. Scope all “PCA leads every neighbour embedding” sentences to unprocessed configurations. Do not present the asymmetric aggregate leaderboard as method superiority.

4. **Remove the false invariance and causal claims.**  
   Replace “leaves its neighbourhoods unchanged,” “only sees,” “rigid contraction,” “the objectives are [in conflict],” and “plateau of the objective.” Attribute **0.342 → 0.460** to the combined transformation unless a placement-only evaluation is supplied. Replace “not an objective.”

5. **State the actual fitted protocol and its scope.**  
   Mark Spearman as directly targeted; reconcile independent versus disjoint sampling; disclose the **1% placement cutoff** and the guard’s limited scope. Restrict “both unoptimised metrics rise” to MNIST. Do not imply that the hybrid reaches PaCMAP’s ARI or that the earlier frozen-transfer study validates this new step. Add the direct prior-work citations identified above.

6. **Remove or substantiate “25 times the time.”**  
   It is not the post-processing ratio. Identify base-layout versus total-pipeline timings and account for graph construction when necessary.

### The one additional experiment

**A repeated, held-out \(2\times2\) component ablation on matched MAGNET full-batch and Barnes–Hut layouts:**

- identity;
- placement only;
- contraction only;
- placement followed by contraction.

Use MNIST and Fashion, the same graph protocol, frozen resolution 0.3 and \(g=0.7\), and repeated Leiden/search seeds. Exclude the evaluation points from the search sample and never use their scores for selection. Report paired changes in all five raw metrics, not primarily the composite.

This single experiment would resolve the missing placement-only attribution, quantify sampling/partition sensitivity, test the claimed preservation of local structure, and establish the method-agnostic benefit under an actually matched comparison.