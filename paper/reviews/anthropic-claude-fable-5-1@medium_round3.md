**Preliminary.** The third draft is a different paper from the second: the premise is gone, the claims are scoped, and the five new experiments (2×2 with seeds, persistence, virial, projection, scaffold) are the right ones. What follows is where the text still outruns the tables, and where the tables disagree with each other.

---

## 1. THE THESIS

### (a) "Sampled pass, then a short full-batch pass under three hand-over conditions, leads both axes"

**Status: survives as "matches the full-batch 15-NN at k = 15 and keeps most of a spring-conditioned start's pairwise Spearman, on two image sets." The word "leads" is wrong on the local axis and incomplete on the global one.**

Local, with the paper's own numbers:
- Table `seeds`: hybrid 0.353 ± 0.001 vs BH 0.350 (deterministic) on MNIST; 0.328 ± 0.001 vs 0.327 on Fashion. The paper itself calls the Fashion margin "one standard deviation". That is a tie, and the abstract still says "highest 15-NN ... on both data sets".
- Table `kscale`: openTSNE 0.375 / 0.327 / 0.329 at k = 5 / 50 / 100 vs hybrid 0.351 / 0.319 / 0.313; hybrid has the lowest trust₅₀ (0.928) and cont₁₅ (0.945) of the six. The paper says this in §4.3 and Limitations; the abstract does not.
- The paper's own best explanation (§4.2) is correct and fatal to "leads": the 0.006 margin over openTSNE is what the pass adds to openTSNE's own layout (0.348 → 0.354, Table `staged`). The local column is a k = 15 graph-matching effect.
- Table `persist`: the "full-batch level" (0.347 at 600 iterations with exaggeration) is below what the pass itself reaches at 3000 iterations (0.367). So "raises to the full-batch level" means "to our 600-iteration schedule's level". Neither your full-batch mode nor openTSNE (750 it.) is converged on this metric.

Global:
- On pairwise Spearman among neighbour embeddings on MNIST/Fashion: yes, 0.449–0.455 / 0.750 over five runs, nothing else above 0.35 / 0.65. Real.
- But Table `staged_seeds` shows the shipped point is not the best of its own family: spring 1.0 → pass gives 0.456 ± 0.027 / 0.764 ± 0.003 with 15-NN 0.352 / 0.330; the "global preset" (scale 62, box 1.5) gives 0.468. The paper should say the shipped hybrid is a chosen operating point, not the maximum.
- Table `staged`, Fashion: PCA-2 → pass gives 0.315 / 0.799. That is +0.05 Spearman for −0.013 15-NN against the hybrid, and the text never confronts it as a Pareto alternative. (On MNIST PCA loses: 0.418 at 300, 0.377 at 1200.)
- News20: hybrid Spearman 0.073 is below UMAP 0.099, h-NNE 0.102, openTSNE 0.106, your own full-batch 0.110, densMAP 0.123, BH 0.232. The abstract's "trails Barnes–Hut" understates: it trails six methods including your own.
- Centroids on MNIST: last, 0.509 ± 0.075 (Table `seeds`) or 0.466 ± 0.158 (Table `staged_seeds`) — see §2 on why these two disagree.

The "three conditions":
- Scale: §3.5 shows the KL-optimal scale is not the best hand-over, 62 beats 31 on Spearman, and the sweep in §3.1 was done on one start. It is a dial, not a condition. The n^{0.42} / √n rule is fit on one data set at three sizes and never validated end-to-end at any n ≠ 70k.
- No exaggeration: by the paper's own admission (§3.1) never separated from the tiny scale. Untested as a separate condition.
- No attraction in the pass: §3.7 then adds an attraction (rest-length scaffold) to the pass and it helps. The condition is "no zero-rest-length spring", which is a statement about one functional form.

The paper no longer claims necessity, which is right. But contribution 1 still lists the three as the recipe; list them as "the configuration that worked, of those tried" and point to a table of what was tried (§