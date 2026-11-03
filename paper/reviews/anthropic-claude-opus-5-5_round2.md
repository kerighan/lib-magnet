# Second review: MAGNET, staged neighbour embedding (second draft)

## 1. The thesis

### (a) "A sampled pass then a short full-batch pass leads both the local and the global axis"

**Verdict: survives on MNIST and Fashion-MNIST for two columns, 15-NN and pairwise distance Spearman/triplets. Overstated as "both axes."**

The local lead is the size of the graph, not the staging:
- On Table `kscale` the hybrid leads at k = 5 and 15 (0.351 / 0.351). openTSNE leads at k = 50 and 100 (0.327 / 0.329 against 0.319 / 0.313).
- The hybrid has the lowest trust₅₀ of the six methods (0.928) and the lowest cont₁₅ (0.945).
- Trust₁₅ is below openTSNE and Barnes–Hut on every image row (0.960 against 0.968 on MNIST).
- The staged table shows that the pass applied to openTSNE's *converged* layout lifts it from 0.348 to 0.354. That is the hybrid's entire 15-NN margin over openTSNE (0.353 against 0.347). So the margin comes from refining on a 15-NN fuzzy graph and scoring on 15-NN, not from the sequence.

The global lead is one metric family:
- On centroid Spearman the hybrid is last among the neighbour embeddings on MNIST: 0.509 ± 0.075, against UMAP 0.728, PaCMAP 0.682, openTSNE 0.605 and your own SGD 0.635.
- It is also beaten inside your own family. SGD at stiffness 1.0 reaches Spearman 0.478, above the hybrid's 0.455, and PCA reaches 0.488.

It fails on the other two data sets:
- On news20 the hybrid's Spearman (0.073) is below UMAP 0.099, h-NNE 0.102, openTSNE 0.106, your full-batch mode 0.110, densMAP 0.123 and Barnes–Hut 0.232.
- On the blobs it leads neither column.

**Defensible form:** "on two image data sets, the sequence matches t-SNE on 15-NN and exceeds every neighbour embedding on pairwise distance Spearman, at the cost of class-level arrangement."

### (b) "The pass is generic across starts and optimisers"

**Verdict: generic across starts on 15-NN, yes. "Keeps the start's global structure" and "openTSNE does the same job" are both overstated.**

On starts:
- Every neighbour-embedding start lands at 0.353–0.356 / 0.327–0.330. That part is solid, though single-seed.
- The centroid column moves by −0.07 for UMAP (0.752 → 0.681) and −0.05 for PaCMAP. These are exactly the starts with the best arrangement.
- PCA loses 0.09 of Spearman.
- So "the global column after the pass is the global column of the start" is true for pairwise Spearman only.

On optimisers:
- openTSNE's optimiser gives 0.346 / 0.428 against your pass's 0.356 / 0.442. With your 15-NN SDs of ~0.001 that is not "the same job."
- The comparison also confounds affinities (perplexity 30 / 90-NN against a 15-NN fuzzy graph, scored at k = 15) and thread counts (4 threads, where openTSNE's FFT gradient was shown not to scale).
- On the PCA start openTSNE's optimiser *beats* yours on 15-NN (0.333 against 0.315). So the "pass" is not one object.

On novelty:
- The genericity is largely the Kobak & Linderman (2021) result: t-SNE keeps the global structure of an informative initialisation.
- What is new is the hand-over scale (σ ≈ 31, close to t-SNE's final scale, instead of their tiny initialisation plus exaggeration) and dropping early exaggeration. Say exactly that.

Persistence is untested:
- The hand-over conditions were tuned on the MAGNET start only. Scale 62 retains more, and the PCA start was never swept.
- You report 15-NN at 600 iterations (0.352) but not Spearman or centroids.
- Persistence over 300 iterations is not persistence. Without a 1000–3000-iteration run you cannot distinguish "the pass keeps the arrangement" from "the pass has not had time to forget it." I asked for this last time.

### (c) "The global gain is entirely the start's; MAGNET's SGD start with a stiff spring is the best start among neighbour embeddings"

**Verdict: the decomposition survives. "Best start" is an artefact of tuning one start and not the others. This is the claim I doubt most.**

The decomposition holds:
- Post-pass Spearman tracks pre-pass Spearman across all starts.
- "Entirely" is still too strong. Your start loses the most in the pass (−0.017 / −0.021, against roughly +0.003 to +0.035 for UMAP and PaCMAP).

"Best start" does not hold as stated:
- You compare your SGD at a spring stiffness tuned *on these data* (0.5) against UMAP and PaCMAP at defaults.
- PaCMAP has the same kind of term (mid-near pairs), with knobs for weight, ratio and phase length.
- By your own ablation the spring is "all of the SGD mode's global structure": 0.285 without it, below openTSNE's 0.330.
- So the specific asset is a stiff mid-near spring, not MAGNET's SGD. The claim "best start" also ignores that this start is the *worst* on MNIST centroids (0.524 against UMAP 0.752, PaCMAP 0.684).
- Your Sec. 4.2 sentence "Stiffness 0.5 is never below 0.1 on any column of any data set" is contradicted by the centroid column of your own Table `tab:staged`: 0.524 against 0.635 before the pass, 0.492 against 0.617 after.

**Smallest experiment that settles it** (half a day; I would want it before arXiv):
- Run a 2 × 2 design: {MAGNET SGD, PaCMAP} × {default mid-near strength, stiff mid-near strength}.
- For PaCMAP, the stiff setting means raising the mid-near weight and/or extending phases 1–2 so it reaches a Spearman comparable to your 0.459.
- Each cell goes to the same pass, 5 seeds, MNIST and Fashion-MNIST.
- Add one cell: MAGNET SGD at spring 0 → pass.

How to read it:
- If stiff PaCMAP → pass reaches ≈ 0.44, the contribution is "stiff mid-near attraction, then scale-matched full-batch refinement," a generic recipe. That is actually a better paper, but (c) as worded dies.
- If PaCMAP cannot reach it, you have earned "best start."

**The premise of the paper is contradicted by your own tables.** The abstract and introduction open with "negative sampling … gives global arrangement …, the all-pairs gradient … gives tight neighbourhoods and no arrangement." On your own distance Spearman, openTSNE beats UMAP on every data set:

| data set | openTSNE | UMAP |
|---|---|---|
| MNIST | 0.330 | 0.297 |
| Fashion-MNIST | 0.617 | 0.571 |
| news20 | 0.106 | 0.099 |
| blobs | 0.382 | 0.079 |

The premise holds only on MNIST centroids, which is the metric on which the hybrid is worst. Your SGD without the spring (0.285) is below openTSNE.

The global structure comes from the mid-range spring, not from sampling. The title ("a sampled pass for the global structure") and the conclusion ("the local and global regimes live in different optimisers") state a mechanism your ablation refutes. Reframe it: global structure from transient mid-range attraction, local structure from normalised full-batch refinement. The estimator story belongs to Böhm et al. and Damrich et al