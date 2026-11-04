---
title: 'MAGNET: fast two-pass neighbour embeddings with a label-free community post-process'
tags:
  - Python
  - C++
  - dimensionality reduction
  - neighbour embedding
  - t-SNE
  - UMAP
  - visualisation
authors:
  - name: Maixent Chenebaux
    affiliation: 1
affiliations:
  - name: Trickstr, France
    index: 1
date: 4 November 2026
bibliography: paper.bib
---

# Summary

MAGNET is a Python library, with its numerical core in C++, that turns a data matrix, a sparse matrix or a graph
into a two-dimensional layout, in the family of $t$-SNE [@vandermaaten2008tsne] and UMAP [@mcinnes2018umap]. It
builds one fuzzy $k$-nearest-neighbour graph and offers three layouts on it: a sampled stochastic pass (UMAP's
negative-sampling optimiser with separate attraction and repulsion kernels and a transient mid-range attraction),
a full-batch pass (the $t$-SNE objective with FFT-interpolated repulsion, after FIt-SNE [@linderman2019fitsne]),
and the sequence of the two, in which the sampled pass provides the arrangement of the clusters and the full-batch
pass refines the neighbourhoods. An optional post-process, label-free in execution, detects the graph's
communities, translates them to increase a sampled rank correlation between input and layout distances, and
contracts each about its centroid. The library exposes a scikit-learn-style estimator (`MAGNET().fit_transform`,
`transform` for new points), accepts dense arrays, sparse CSR matrices (with the cosine metric, without
densification) and networkx graphs, and runs MNIST 70k in 3.6 to 9 seconds on a 20-thread CPU depending on the
mode.

# Statement of need

Neighbour embeddings are a standard tool for looking at high-dimensional data, and practitioners choose between
libraries that sit at different points of a local/global trade-off: $t$-SNE implementations keep neighbourhoods
and are slow at the scale of $10^5$ points, UMAP is fast and keeps fewer neighbourhoods, PaCMAP
[@wang2021pacmap] and TriMap [@amid2019trimap] add mid-range terms. MAGNET is meant for users who want both
quantities that are usually traded against each other, the overlap of 15-nearest-neighbour sets and the
Spearman correlation of pairwise distances, from one library, at the speed of UMAP. On MNIST 70k its full-batch
mode reaches the 15-NN overlap of scikit-learn's Barnes--Hut $t$-SNE [@vandermaaten2014bh] (0.349 against 0.350)
in 4.9 s against 175 s, and on MNIST and Fashion-MNIST the sequence has a higher pairwise Spearman than UMAP,
PaCMAP, openTSNE [@policar2019opentsne] and Barnes--Hut $t$-SNE at a 15-NN overlap comparable to the $t$-SNE
baselines (interleaved timings on one machine, five runs for the quality figures). The community post-process
applies to any layout and is offered as a separate function for that reason.

# State of the field

openTSNE [@policar2019opentsne] and FIt-SNE [@linderman2019fitsne] provide fast $t$-SNE; umap-learn
[@mcinnes2018umap] provides UMAP and densMAP [@narayan2021densmap]; PaCMAP, TriMap and h-NNE
[@sarfraz2022hnne] each ship one objective. The attraction--repulsion spectrum of @bohm2022spectrum and the
initialisation study of @kobak2021init explain much of the difference between them. MAGNET's contribution is a
recipe and its implementation rather than a new objective: two existing objectives in sequence, on one graph, with
a hand-over scale chosen empirically, plus a post-process. In a matched comparison in which every one of twelve
configurations receives the same post-process (five runs, four data sets), the sequence has the highest mean of a
composite of the two axes and of $k$-means ARI; its paired difference from a post-processed Barnes--Hut layout is
$0.023 \pm 0.047$ and reverses on MNIST, so the advantage is data-set dependent. The companion report documents
every experiment behind the defaults, including the negative results.

# Software design

The C++ extension (pybind11, header-only pocketfft, a persistent pthread pool) implements NN-descent
[@dong2011nndescent], the fuzzy-graph bandwidths, a Chebyshev-filtered spectral initialisation, the lock-free
SGD, the FFT repulsion with persistent buffers, and a strength-ordered Louvain [@blondel2008louvain] used by the
post-process. Python holds the estimator, the schedules and the post-process (`magnet.contraction`). Defaults
that were set by measurement carry their evidence in the docstrings: the attraction exponent follows the sample
size (a heavy tail fragments small data sets), the interpolation box is 0.7 (FIt-SNE's unit box leaves a 3.5%
force error that costs neighbourhood overlap), and the communities are merged across interfaces without a density
contrast because modularity on near-regular kNN graphs splits large geometric communities
[@fortunato2007resolution; @schaub2012markov]. A test suite runs on every push against the installed source
distribution, and `examples/quickstart.py` runs the three modes on scikit-learn's digits.

# Research impact statement

The library accompanies a short paper and an exhaustive companion report in the same repository
(`paper/human/main.tex`, `paper/main.tex`), whose every table and figure is regenerated by a script under
`benchmarks/`, with the result files committed. These provide a reusable evaluation protocol for neighbour
embeddings: raw axes over five runs, frozen composite references, and a matched control in which competitors
receive the same post-process.

# AI usage disclosure

Large parts of the implementation, the experiments and the writing of this version were produced with an AI
coding assistant (Anthropic's Claude) working under the author's direction; manuscripts were reviewed by an
external language model (OpenAI GPT) acting as a referee over several rounds. The author designed the research
directions, checked the results and is responsible for the content.

# Acknowledgements

The author thanks Trickstr for the computing time.

# References
