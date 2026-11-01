# Benchmark

Mean over seeds, defaults for every method, best value per column in bold. 20-core i9-9900X, 54 GB RAM.
Local structure: 15-NN preservation (fraction of each point's 15 nearest neighbours in the data that are among its 15 nearest in 2-D) and trustworthiness. Global structure: Spearman correlation of pairwise distances and random-triplet accuracy on a 2000-point sample. Labels: 10-NN accuracy in 2-D.
*Compactness is the silhouette of the labels: it rewards tight, well-separated clusters and can be raised at the expense of neighbourhood quality, so it describes the look rather than the fidelity of a layout.

## MNIST (70k x 784), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **4.4** | 0.193 | 0.957 | 0.413 | 0.644 | 0.967 | 0.397 |
| MAGNET (loss="tsne") | 9.2 | 0.349 | 0.966 | 0.348 | 0.627 | 0.966 | 0.347 |
| MAGNET (loss="hybrid") | 12.5 | **0.353** | 0.961 | **0.444** | **0.656** | 0.960 | 0.203 |
| UMAP 0.5.12 | 11.2 | 0.117 | 0.951 | 0.297 | 0.606 | 0.967 | 0.511 |
| densMAP (umap-learn 0.5.12) | 37.2 | 0.068 | 0.921 | 0.364 | 0.623 | 0.945 | 0.412 |
| openTSNE 1.0.4 | 102.5 | 0.347 | 0.968 | 0.330 | 0.619 | 0.965 | 0.347 |
| PaCMAP | 24.8 | 0.093 | 0.946 | 0.307 | 0.614 | **0.976** | **0.579** |
| TriMap | 57.0 | 0.088 | 0.933 | 0.184 | 0.597 | 0.964 | 0.411 |
| PHATE 2.0 | 43.2 | 0.080 | 0.865 | 0.291 | 0.611 | 0.861 | 0.224 |
| h-NNE | 9.1 | 0.233 | 0.940 | 0.261 | 0.601 | 0.949 | 0.232 |
| sklearn t-SNE (Barnes-Hut) | 195.7 | 0.350 | **0.968** | 0.338 | 0.622 | 0.967 | 0.374 |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 13.8 | 0.169 | 0.967 | 0.169 | 0.567 | 0.957 | 0.341 |

## Fashion-MNIST (70k x 784), 1 seed

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **4.4** | 0.202 | 0.980 | 0.730 | 0.787 | 0.800 | 0.141 |
| MAGNET (loss="tsne") | 9.1 | 0.324 | 0.981 | 0.700 | 0.775 | 0.815 | 0.101 |
| MAGNET (loss="hybrid") | 10.5 | **0.328** | 0.979 | **0.752** | **0.799** | 0.814 | 0.106 |
| UMAP 0.5.12 | 10.9 | 0.133 | 0.975 | 0.571 | 0.728 | 0.778 | 0.195 |
| densMAP (umap-learn 0.5.12) | 38.9 | 0.071 | 0.955 | 0.583 | 0.729 | 0.755 | 0.190 |
| openTSNE 1.0.4 | 98.2 | 0.323 | **0.983** | 0.617 | 0.732 | 0.817 | 0.120 |
| PaCMAP | 24.2 | 0.099 | 0.971 | 0.581 | 0.729 | 0.773 | **0.209** |
| TriMap | 55.0 | 0.091 | 0.965 | 0.672 | 0.772 | 0.750 | 0.177 |
| PHATE 2.0 | 34.3 | 0.097 | 0.924 | 0.560 | 0.726 | 0.721 | 0.129 |
| h-NNE | 6.6 | 0.173 | 0.912 | 0.490 | 0.698 | 0.805 | 0.129 |
| sklearn t-SNE (Barnes-Hut) | 204.8 | 0.327 | 0.983 | 0.634 | 0.740 | **0.818** | 0.126 |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 13.9 | 0.235 | 0.975 | 0.269 | 0.614 | 0.811 | 0.004 |

## 20 newsgroups (18.8k x 300 SVD, cosine), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **1.3** | 0.134 | 0.760 | 0.059 | 0.529 | 0.508 | -0.111 |
| MAGNET (loss="tsne") | 4.4 | 0.184 | **0.776** | 0.110 | 0.541 | 0.526 | -0.090 |
| MAGNET (loss="hybrid") | 4.4 | 0.181 | 0.768 | 0.073 | 0.528 | 0.515 | -0.147 |
| UMAP 0.5.12 | 2.7 | 0.098 | 0.718 | 0.099 | 0.537 | 0.497 | -0.105 |
| densMAP (umap-learn 0.5.12) | 10.7 | 0.068 | 0.658 | 0.123 | 0.545 | 0.443 | -0.103 |
| openTSNE 1.0.4 | 38.8 | 0.187 | 0.772 | 0.106 | 0.542 | **0.535** | **-0.086** |
| PaCMAP | 6.6 | 0.057 | 0.683 | -0.086 | 0.499 | 0.440 | -0.101 |
| TriMap | 12.7 | 0.059 | 0.690 | 0.051 | 0.546 | 0.443 | -0.139 |
| PHATE 2.0 | 20.3 | 0.066 | 0.622 | -0.005 | 0.524 | 0.337 | -0.143 |
| h-NNE | 5.7 | 0.154 | 0.732 | 0.102 | 0.535 | 0.470 | -0.189 |
| sklearn t-SNE (Barnes-Hut) | 48.6 | **0.192** | 0.742 | **0.232** | **0.576** | 0.455 | -0.116 |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 8.5 | 0.148 | 0.651 | 0.039 | 0.517 | 0.333 | -0.146 |

## 15 Gaussian blobs (10k x 200), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **1.1** | 0.067 | 0.972 | **0.690** | **0.732** |  |  |
| MAGNET (loss="tsne") | 2.5 | **0.126** | 0.974 | 0.505 | 0.678 |  |  |
| MAGNET (loss="hybrid") | 2.5 | 0.124 | 0.974 | 0.684 | 0.730 |  |  |
| UMAP 0.5.12 | 2.4 | 0.047 | 0.971 | 0.079 | 0.530 |  |  |
| densMAP (umap-learn 0.5.12) | 7.5 | 0.032 | 0.968 | 0.127 | 0.543 |  |  |
| openTSNE 1.0.4 | 19.4 | 0.109 | 0.975 | 0.382 | 0.646 |  |  |
| PaCMAP | 4.1 | 0.033 | 0.971 | 0.666 | 0.720 |  |  |
| TriMap | 7.6 | 0.033 | 0.972 | 0.648 | 0.727 |  |  |
| PHATE 2.0 | 24.8 | 0.033 | 0.968 | 0.232 | 0.571 |  |  |
| h-NNE | 5.9 | 0.083 | 0.969 | 0.307 | 0.614 |  |  |
| sklearn t-SNE (Barnes-Hut) | 17.6 | 0.114 | **0.975** | 0.381 | 0.646 |  |  |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 7.7 | 0.074 | 0.974 | 0.127 | 0.540 |  |  |
