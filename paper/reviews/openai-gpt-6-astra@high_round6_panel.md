## 1. Narrowing

**Use one competitive MAGNET row: hybrid + placement + separation.** Keep the unprocessed hybrid and standard full-batch mode in a separate diagnostic block or ablation table, not as additional entries competing for the podium.

First correct the inventory: the supplied 16-row tables contain **seven MAGNET configurations and nine comparator configurations**, not six and ten. Also, “PCA-2 + pass” is a control using MAGNET’s refinement, not an independent external competitor.

### Exact presentation

- **Main composite comparison:** hybrid + placement + separation, against the nine supplied comparator configurations. Name the post-process accurately: “contraction” alone conceals the placement optimisation.
- **Immediately adjacent diagnostic comparison:** unprocessed hybrid and standard full-batch, with the complete-system row repeated for reference.
  - Hybrid versus complete system exposes the post-process’s contribution.
  - Full-batch versus hybrid exposes whether the sampled start helps the composite before post-processing.
- **Appendix configuration table:** all seven current MAGNET configurations, raw axes, both composites and runtime, using exactly the same references as the main table. Include the baseline results unchanged, or reference a fixed baseline block. Do not recompute the normalisation when configurations are removed.
- Keep the quality preset in that appendix. It is a useful cost–quality operating point, not another independent method.

The appendix must preserve inconvenient results. For example, on letter the post-process lowers the hybrid’s geometric composite from **0.912 to 0.894** and minimum from **0.855 to 0.839**. On MNIST the quality full-batch post-processed configuration exceeds the complete hybrid: **0.937 versus 0.920** geometrically and **0.865 versus 0.791** on the minimum. Selecting one system configuration should not erase those facts.

### Freeze the references

**Yes: freeze them to a declared, versioned unprocessed calibration panel.** Using all unprocessed configurations in the supplied cache gives:

| Dataset | 15-NN reference | Spearman reference | ARI reference |
|---|---:|---:|---:|
| MNIST | 0.356 | 0.488 | 0.914 |
| Fashion-MNIST | 0.330 | 0.873 | 0.483 |
| letter | 0.642 | 0.716 | not used |
| covertype | 0.632 | 0.755 | not used |

The quality preset can remain part of this **fixed calibration panel** even though it leaves the main comparison. Publish the reference values and provenance. Use them for every configuration, post-processing control and seed; do not take new seed-wise maxima. Call the resulting quantities **ratios to fixed unprocessed references**, not shares of the current panel’s best. Permit values above one; do not clip them.

There is an important correction to the flooding argument. For a dataset with \(p\) axes,

\[
G_m=\left(\prod_{j=1}^{p}x_{mj}\right)^{1/p}
      \left(\prod_{j=1}^{p}r_j\right)^{-1/p}.
\]

Changing references multiplies **every geometric composite on that dataset by the same factor**. It cannot change its within-dataset ranking. It can change the across-dataset mean ranking by changing dataset weights. For the minimum, changing references can change within-dataset rankings as well.

Thus flooding affects the apparent podium, the opportunity to select a favourable configuration, and potentially the normalisation—not automatically the complete system’s pairwise advantage. That advantage survives removing its siblings: the supplied one-row panel gives **0.902 versus 0.794** for BH t-SNE on the mean geometric composite, and **0.822 versus 0.624** on the mean minimum. It wins three datasets under both composites; on letter PCA-2 + pass wins, **0.905 versus 0.894** and **0.881 versus 0.839**.

Do not carry those exact gaps into the frozen-reference table without recalculation. For example, freezing MNIST’s Spearman reference at **0.488**, rather than **0.531**, raises BH’s MNIST minimum from **0.637 to approximately 0.693**. Its mean minimum becomes approximately **0.638**, reducing the complete system’s mean advantage from **0.198 to approximately 0.184**.

**Reviewer rejection:** I would reject “our configurations occupy the top three places” as headline evidence of superiority from this 16-row panel. The exhaustive table itself is acceptable as an appendix ablation. I would also reject selecting a different MAGNET configuration per dataset, silently renormalising each table, or presenting a retrospectively selected configuration as preregistered.

## 2. The claim

### Abstract sentence supported now

> “In a single-seed comparison on four datasets, the complete MAGNET pipeline—sampled embedding, full-batch refinement and optional label-free community post-processing—ranks first against the evaluated comparison pipelines on three datasets and in the four-dataset mean under both specified composites; the comparison pipelines do not receive this post-processing.”

This is a benchmark result, not a statement of established expected performance over seeds or datasets. Replace “single-seed” only after replication supports the same result.

### Conclusion sentence supported now

> “The complete pipeline provides the strongest measured composite balance in this four-dataset comparison against pipelines without community post-processing, but these results neither establish metric-wise dominance nor attribute its advantage to the embedding stages rather than the transferable post-process.”

The current normalisation supports reporting the **0.108 geometric-mean advantage** and **0.198 minimum advantage** over BH, explicitly as single-run, panel-dependent quantities. After freezing references, report the regenerated numbers.

### One step too far

> “The complete system is better than every competitor almost everywhere.”

That is unsupported unless “better” and “almost everywhere” are immediately replaced by the actual scope: **both specified composites on three of four datasets, for one run and these comparator configurations**.

It is not dominance on the constituent metrics. On MNIST, complete-system ARI is **0.723**, below BH’s **0.829** and PaCMAP’s **0.914**, despite its Spearman of **0.531**. On Fashion-MNIST its 15-NN is **0.325**, below BH’s **0.327**, and ARI is **0.414**, below PaCMAP’s **0.483**. Two datasets omit ARI altogether. Four datasets also do not override the paper’s reported failures on other datasets and metrics.

### What the untreated baselines imply

It is legitimate to compare a shipped complete system against other shipped systems. A label-free optional module is part of the system, not an illicit advantage merely because competitors lack it.

Therefore, **“a configuration comparison, not a method comparison” is too categorical**. A more accurate formulation is:

> “A complete-pipeline comparison against configurations without the additional post-process, not a controlled comparison of embedding stages.”

The missing control limits **attribution and claims against enhanced competitors**. It does not invalidate the observed comparison against the configurations actually run. Conversely, “label-free” does not remove the asymmetry: the post-process explicitly optimises sampled Spearman, and its settings were selected on development benchmarks with their scores in view. State both facts.

## 3. The control experiment

### Primary protocol

**Rows.** Run each of the nine comparator configurations both without and with the identical post-process, including plain PCA-2 as well as PCA-2 + pass. Run MAGNET hybrid, standard full-batch and SGD both ways.

- Hybrid versus baseline, both post-processed, tests whether its embedding stage retains an advantage.
- Hybrid versus full-batch, both post-processed, tests the sampled start’s additional value.
- Hybrid versus SGD, both post-processed, tests the refinement’s additional value.
- Plain PCA-2 + post-process is a useful weak-layout control.
- Keep the quality preset secondary; do not introduce it selectively when the standard configuration loses.

**Seeds.** Use **five predeclared end-to-end seeds for every stochastic configuration on every main dataset**, not five post-processing seeds on one fixed baseline layout. Three is a minimum descriptive replication, not my recommended design for the central comparison.

Explicitly record graph-construction, embedding, community-detection, search and evaluation RNG streams. Within a replicate, share the post-processing graph, communities and sampling randomness across methods. Preserve the baseline layout when comparing its identity and post-processed versions. Use the same evaluation points across methods; fix and document the \(k\)-means evaluation protocol.

A supplementary crossed experiment on the strongest contenders—several layout seeds crossed with several community/search seeds—would identify whether the uncertainty comes from embedding or community detection. The current three searches on one BH layout do **not** estimate BH’s end-to-end variability.

**Graph and communities.** For the primary control, use **one common, frozen graph-construction recipe and the same communities for every method within each replicate**. Using MAGNET’s graph is appropriate for testing transfer of the *same post-processor*. Say explicitly that the common module includes access to that graph.

Do **not** use UMAP’s fuzzy set for UMAP, openTSNE’s affinities for openTSNE, and MAGNET’s graph for MAGNET in the primary experiment. That changes the partition as well as the layout, confounding the question. Resolution 0.3 also need not have comparable effects across differently weighted graphs.

Native-graph communities are a useful **secondary sensitivity experiment**, with a predeclared label-free construction and resolution rule. They test portability to native pipelines, not the primary matched-module comparison.

**Post-process.** Freeze all settings: resolution, \(g=0.7\), search candidates, sweep count, mixing guard and sampling sizes. Apply the same dimensionless scaling conventions. No method-specific retuning and no skipping a method’s harmful result. Search points must exclude the evaluation points. Account for graph construction and community detection in baseline-plus-post-process runtime; they are not free because MAGNET already has the graph.

**References and datasets.**

- Use the frozen references from Question 1 throughout.
- Run the complete factorial comparison on **MNIST 70k, Fashion-MNIST 70k, letter 20k and covertype 20k**.
- Extend the frozen protocol to the existing five transfer cases if claiming transfer beyond these four columns. Keep development and transfer results separate.
- A claim approaching “almost everywhere” must also confront the reported 20-newsgroups failure, rather than silently restricting attention to favourable datasets.
- Keep the two-axis definition on letter/covertype fixed and justified in advance; do not choose whether to include ARI after seeing results.

**Reporting.** For every dataset and configuration report:

1. Raw 15-NN, Spearman and \(k\)-means ARI, including ARI where it is excluded from the composite.
2. Both frozen-reference composites.
3. Mean ± sample SD over end-to-end runs.
4. Paired post-processing gains and paired differences between post-processed methods, with uncertainty intervals.
5. Embedding, graph/community, post-processing and total time separately.
6. Triplet accuracy and centroid Spearman as unfitted checks.

Compute each composite **per run**, then average. A composite of mean axes is not the mean composite.

### Decision rule

Predeclare a practically meaningful margin—**0.01 composite units** is a possible choice, not a universal standard—and an uncertainty procedure for paired differences. Treat the two composites as predeclared outcomes, not alternatives from which to select the favourable one. Account for multiple comparator claims if asserting superiority to all.

- **“The embedding stages contribute beyond the post-process”** is supported if hybrid + post-process retains a practically meaningful advantage over strong baseline + post-process configurations, with uncertainty excluding no meaningful advantage under the declared criterion. State which datasets and axes support it.
- **“Both stages are needed for that advantage”** additionally requires the SGD-plus-post-process and full-batch-plus-post-process ablations. Beating other libraries does not establish the necessity of each MAGNET stage.
- **“The post-process accounts for the observed quality advantage among the strong embeddings”** is supported if the formerly substantial gaps collapse and equivalence intervals lie within the declared margin.
- **Wide intervals or nonsignificant differences mean inconclusive**, not “the embedding method matters little.” Five seeds may be insufficient to establish equivalence; use a predeclared larger replication if that claim is important.

Seed uncertainty on four fixed datasets does not establish generalisation to a population of datasets.

### What k38 already predicts

The frozen-composite gains are:

- BH: **0.855 → 0.950**, gain **0.095**.
- openTSNE: **0.809 → 0.947**, gain **0.138**.
- UMAP: **0.555 → 0.653**, gain **0.098**.
- PaCMAP: **0.552 → 0.630**, gain **0.078**.

These already rule out attributing the adjustment’s benefit uniquely to MAGNET. BH and openTSNE end only **0.003** apart, making substantial convergence among strong local embeddings plausible. The current k39 results reinforce that: post-processed full-batch and BH have essentially identical 15-NN/Spearman, **0.346/0.464 versus 0.348/0.463**, with BH having higher mean ARI, **0.920 versus 0.887**.

But UMAP and PaCMAP remain far below the t-SNE layouts. The evidence does **not** predict that every embedding becomes interchangeable. Nor does it settle the hybrid’s residual advantage: its seed-0 Spearman of **0.531** leaves a concrete hypothesis to test under matched post-processing.

## 4. Seeds and variance

**A single-seed table is acceptable as an explicitly illustrative or supplementary table. It is not adequate as the final paper’s central evidence for complete-system superiority.** Existing repeated measurements of other configurations or other metrics do not supply the missing variance.

The cited k38 ARI of **0.860 ± 0.055** is material evidence of a discrete failure mode, not merely numerical noise. The supplied current k39 result, **0.887 ± 0.039**, is a different result and must not be substituted for it without identifying the experimental change.

Also reconcile the composite cache’s full-batch-plus-post-process seed-0 ARI of **0.787** with the placement table and its stated seed range. Different layout, community, search or evaluation seeds can explain different outcomes; an unidentified “seed 0” cannot serve as reproducible provenance across tables.

For a three-seed table:

- Report run-wise composites as **mean ± sample SD**, preferably with the individual runs in the supplement.
- Do not infer composite variability directly from ARI variability; propagate the actual joint outcomes through the composite.
- Distinguish variability conditional on one layout from end-to-end variability.
- Remove podium styling if it encourages an inferential reading unsupported by three runs.

If bold/italic are retained, define them strictly as **largest/second-largest sample mean, not statistically distinguishable best/second**. Overlapping SDs are not a significance test; paired methods can have overlapping marginal SDs but a precise difference. Conversely, non-overlap of SD bars is not the appropriate declared test. Inferential grouping requires uncertainty on paired differences and an explicit decision rule—not an “SD bars overlap” heuristic.

## 5. Timing

**An explicitly provisional arXiv posting does not require rerunning the entire seeds table on an idle machine. It does require removing controlled-benchmark interpretations of the mixed-condition timings.** Caption disclosure alone does not justify unqualified speed claims in the abstract or introduction.

The interference is substantial:

- Unchanged SGD: **3.6 → 4.5 s**, a **25%** increase.
- UMAP controls: **9.8–11.8 s**, versus **8.4 s** idle, approximately **17–40%** higher.
- Consequently, hybrid’s **8.4 s** under contention versus UMAP’s **8.4 s** idle is not a controlled demonstration of equal runtime.
- Likewise, **93.5/8.4 ≈ 11.1** is an observed mixed-condition ratio, not a cleanly measured elevenfold speedup.

The load probably penalises MAGNET here, but that does not turn uncontrolled measurements into a standardised benchmark. Timing SDs within one load condition do not capture the bias between conditions. Do not “correct” all times using the SGD slowdown factor.

For an immediate posting, label the timings provisional, identify measurement conditions per block, and replace precise comparative claims with appropriately qualified observations. The quality measurements need not be discarded merely because the machine was busy.

For the final performance claim, rerun **the timing benchmark** with the final implementation on a quiescent machine: identical thread limits and timing boundaries, audited warm-up, all compared pipelines, repeated and interleaved measurements, and complete-system costs including post-processing. That does not require repeating every fidelity experiment. If the authors want to retain clean “same runtime,” “fastest,” or exact speedup claims in the present posting, they should obtain those comparable timings before posting.