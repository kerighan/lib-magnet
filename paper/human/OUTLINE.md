# MAGNET, the human-readable paper: outline (draft, 2026-11-04)

Ground truth: `paper/main.tex` (the exhaustive document). This version keeps the story, the decisive numbers and
one figure per idea. Every number is traceable to a table or a `k*` script of the long version.

## Working title
MAGNET: a sampled pass for the global structure, a full-batch pass for the local one, and a label-free
community post-process. (Shorter: "Two passes and a post-process".)

## Story in five claims (one section each after the intro)
1. Two objectives, two passes (Sec. 2). The sampled SGD pass arranges the clusters (global), the full-batch FFT
   pass refines the neighbourhoods (local); neither does both. Figure: six layouts (UMAP / openTSNE / PaCMAP /
   MAGNET SGD / full-batch / hybrid), the existing `layouts_mnist.png`.
2. What the exponents do (Sec. 3, short). Attraction exponent = robustness to contaminating neighbours (15-NN
   peaks at 0.5-0.7); the repulsion stays Cauchy. New: at small n the heavy tail tears the layout, the
   exponent follows n (or the graph-layout consistency). Figure: HDBSCAN clusters and 15-NN against n for
   b = 0.6 / 0.8 (k45).
3. The full-batch pass is accurate to its grid (Sec. 4, short). The FIt-SNE unit box leaves 3.5 % RMS force
   error and costs 0.004 of 15-NN; 0.7 is the knee (k40). Figure: 15-NN and time against the box width.
4. Community placement and separation (Sec. 5). Why modularity fails on kNN graphs (near-uniform strengths ->
   size^2 penalty -> field-of-view and resolution limits), the density-contrast merge with its reference
   (0.2 across a uniform cut, half of it at a real boundary), the placement by sampled Spearman under a mixing
   guard, the contraction. Figures: interface crossing shares (same-class splits vs real boundaries, k43);
   MNIST before / after the step (placement.png, regenerated with the final step).
5. Results (Sec. 6). (a) The seeds table (five runs, the decisive rows). (b) The composite against frozen
   references, one MAGNET configuration in the main table (round-6 advice), the rest in the appendix.
   (c) The matched post-process control (k41): every comparator with and without the identical step,
   five seeds, paired differences. (d) Transfer to held-out data (frozen rules). (e) Time. Figures: composite
   bars with error bars (k41), the time bar.
6. Limitations and what did not work (Sec. 7): the hybrid's k-means instability, Fashion's touching classes,
   the small-n regime (phonemes), text, HDBSCAN mixed, timings on a shared machine.

## Figures to produce (paper/human/make_figs.py -> paper/human/figures/)
- fig_box_width.png (k40): 15-NN (MNIST, blobs) and time against the box width; marks the knee at 0.7.
- fig_interfaces.png (k43): crossing-share ratio of every interface, same-class splits vs real boundaries, MNIST.
- fig_small_n.png (k45): HDBSCAN cluster count and 15-NN against n for b = 0.6 / 0.8, UMAP as reference.
- fig_composite.png (k41, when done): composite per configuration, with and without the step, mean +- sd.
- fig_placement.png: MNIST full-batch before / after (regenerate with the final step).
- fig_layouts.png: reuse layouts_mnist.png.

## Decisions (Maixent, 2026-11-04)
- Format: two-column conference style, 8 pages + references; appendix allowed.
- Language: English.
- Virial / scale identity and staged refinement: one paragraph each in the body, an appendix with the details.
