# Benchmark

Mean over seeds, defaults for every method, best value per column in bold. 20-core i9-9900X, 54 GB RAM.
Local structure: 15-NN preservation (fraction of each point's 15 nearest neighbours in the data that are among its 15 nearest in 2-D) and trustworthiness. Global structure: Spearman correlation of pairwise distances and random-triplet accuracy on a 2000-point sample. Labels: 10-NN accuracy in 2-D.
*Compactness is the silhouette of the labels: it rewards tight, well-separated clusters and can be raised at the expense of neighbourhood quality, so it describes the look rather than the fidelity of a layout.

## MNIST (70k x 784), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **5.4** | 0.128 | 0.953 | 0.324 | 0.616 | 0.968 | 0.517 |
| MAGNET (loss="tsne") | 12.6 | **0.352** | 0.967 | 0.335 | 0.621 | 0.967 | 0.355 |
| UMAP 0.5.11 | 18.2 | 0.116 | 0.951 | 0.300 | 0.607 | 0.967 | 0.520 |
| openTSNE 1.0.4 | 102.5 | 0.347 | 0.968 | 0.330 | 0.619 | 0.965 | 0.347 |
| PaCMAP | 24.8 | 0.093 | 0.946 | 0.307 | 0.614 | **0.976** | **0.579** |
| TriMap | 57.0 | 0.088 | 0.933 | 0.184 | 0.597 | 0.964 | 0.411 |
| sklearn t-SNE (Barnes-Hut) | 195.7 | 0.350 | **0.968** | **0.338** | **0.622** | 0.967 | 0.374 |

## Fashion-MNIST (70k x 784), 1 seed

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **4.5** | 0.150 | 0.976 | 0.571 | 0.736 | 0.787 | 0.193 |
| MAGNET (loss="tsne") | 12.4 | **0.328** | 0.981 | 0.621 | 0.733 | 0.815 | 0.114 |
| UMAP 0.5.11 | 15.0 | 0.131 | 0.972 | 0.581 | 0.730 | 0.772 | 0.195 |
| openTSNE 1.0.4 | 98.2 | 0.323 | **0.983** | 0.617 | 0.732 | 0.817 | 0.120 |
| PaCMAP | 24.2 | 0.099 | 0.971 | 0.581 | 0.729 | 0.773 | **0.209** |
| TriMap | 55.0 | 0.091 | 0.965 | **0.672** | **0.772** | 0.750 | 0.177 |
| sklearn t-SNE (Barnes-Hut) | 204.8 | 0.327 | 0.983 | 0.634 | 0.740 | **0.818** | 0.126 |

## 20 newsgroups (18.8k x 300 SVD, cosine), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **1.7** | 0.121 | 0.751 | -0.001 | 0.521 | 0.515 | -0.087 |
| MAGNET (loss="tsne") | 5.8 | 0.183 | **0.774** | 0.113 | 0.542 | 0.526 | -0.094 |
| UMAP 0.5.11 | 5.9 | 0.111 | 0.739 | 0.001 | 0.521 | 0.512 | -0.096 |
| openTSNE 1.0.4 | 38.8 | 0.187 | 0.772 | 0.106 | 0.542 | **0.535** | **-0.086** |
| PaCMAP | 6.6 | 0.057 | 0.683 | -0.086 | 0.499 | 0.440 | -0.101 |
| TriMap | 12.7 | 0.059 | 0.690 | 0.051 | 0.546 | 0.443 | -0.139 |
| sklearn t-SNE (Barnes-Hut) | 48.6 | **0.192** | 0.742 | **0.232** | **0.576** | 0.455 | -0.116 |

## 15 Gaussian blobs (10k x 200), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **1.1** | 0.047 | 0.971 | 0.522 | 0.684 |  |  |
| MAGNET (loss="tsne") | 3.2 | **0.128** | 0.974 | 0.347 | 0.616 |  |  |
| UMAP 0.5.11 | 2.3 | 0.045 | 0.971 | 0.229 | 0.578 |  |  |
| openTSNE 1.0.4 | 19.4 | 0.109 | 0.975 | 0.382 | 0.646 |  |  |
| PaCMAP | 4.1 | 0.033 | 0.971 | **0.666** | 0.720 |  |  |
| TriMap | 7.6 | 0.033 | 0.972 | 0.648 | **0.727** |  |  |
| sklearn t-SNE (Barnes-Hut) | 17.6 | 0.114 | **0.975** | 0.381 | 0.646 |  |  |
