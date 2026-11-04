# Benchmark

Mean over seeds, defaults for every method, best value per column in bold, second underlined. 20-core i9-9900X, 54 GB RAM.
Local structure: 15-NN preservation (fraction of each point's 15 nearest neighbours in the data that are among its 15 nearest in 2-D) and trustworthiness. Global structure: Spearman correlation of pairwise distances and random-triplet accuracy on a 2000-point sample. Labels: 10-NN accuracy in 2-D.
*Compactness is the silhouette of the labels: it rewards tight, well-separated clusters and can be raised at the expense of neighbourhood quality, so it describes the look rather than the fidelity of a layout.

## MNIST (70k x 784), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **4.1** | 0.194 | 0.956 | <ins>0.414</ins> | <ins>0.644</ins> | 0.966 | 0.405 |
| MAGNET (loss="tsne") | <ins>7.0</ins> | 0.349 | 0.966 | 0.349 | 0.627 | <ins>0.967</ins> | 0.345 |
| MAGNET (loss="hybrid") | 8.6 | **0.355** | 0.960 | **0.445** | **0.652** | 0.962 | 0.217 |
| UMAP 0.5.12 | 11.2 | 0.117 | 0.951 | 0.297 | 0.606 | <ins>0.967</ins> | <ins>0.511</ins> |
| densMAP (umap-learn 0.5.12) | 37.2 | 0.068 | 0.921 | 0.364 | 0.623 | 0.945 | 0.412 |
| openTSNE 1.0.4 | 102.5 | 0.347 | **0.968** | 0.330 | 0.619 | 0.965 | 0.347 |
| PaCMAP | 24.8 | 0.093 | 0.946 | 0.307 | 0.614 | **0.976** | **0.579** |
| TriMap | 57.0 | 0.088 | 0.933 | 0.184 | 0.597 | 0.964 | 0.411 |
| PHATE 2.0 | 43.2 | 0.080 | 0.865 | 0.291 | 0.611 | 0.861 | 0.224 |
| h-NNE | 9.1 | 0.233 | 0.940 | 0.261 | 0.601 | 0.949 | 0.232 |
| sklearn t-SNE (Barnes-Hut) | 195.7 | <ins>0.350</ins> | **0.968** | 0.338 | 0.622 | <ins>0.967</ins> | 0.374 |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 13.8 | 0.169 | <ins>0.967</ins> | 0.169 | 0.567 | 0.957 | 0.341 |

## Fashion-MNIST (70k x 784), 1 seed

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **4.2** | 0.205 | 0.980 | <ins>0.729</ins> | <ins>0.786</ins> | 0.799 | 0.139 |
| MAGNET (loss="tsne") | 7.3 | 0.325 | <ins>0.981</ins> | 0.701 | 0.775 | 0.815 | 0.100 |
| MAGNET (loss="hybrid") | 8.4 | **0.328** | 0.980 | **0.753** | **0.798** | 0.812 | 0.100 |
| UMAP 0.5.12 | 10.9 | 0.133 | 0.975 | 0.571 | 0.728 | 0.778 | <ins>0.195</ins> |
| densMAP (umap-learn 0.5.12) | 38.9 | 0.071 | 0.955 | 0.583 | 0.729 | 0.755 | 0.190 |
| openTSNE 1.0.4 | 98.2 | 0.323 | **0.983** | 0.617 | 0.732 | <ins>0.817</ins> | 0.120 |
| PaCMAP | 24.2 | 0.099 | 0.971 | 0.581 | 0.729 | 0.773 | **0.209** |
| TriMap | 55.0 | 0.091 | 0.965 | 0.672 | 0.772 | 0.750 | 0.177 |
| PHATE 2.0 | 34.3 | 0.097 | 0.924 | 0.560 | 0.726 | 0.721 | 0.129 |
| h-NNE | <ins>6.6</ins> | 0.173 | 0.912 | 0.490 | 0.698 | 0.805 | 0.129 |
| sklearn t-SNE (Barnes-Hut) | 204.8 | <ins>0.327</ins> | **0.983** | 0.634 | 0.740 | **0.818** | 0.126 |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 13.9 | 0.235 | 0.975 | 0.269 | 0.614 | 0.811 | 0.004 |

## 20 newsgroups (18.8k x 300 SVD, cosine), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | **2.0** | 0.134 | 0.759 | 0.039 | 0.524 | 0.509 | -0.101 |
| MAGNET (loss="tsne") | 2.8 | 0.184 | **0.775** | 0.115 | 0.542 | <ins>0.525</ins> | <ins>-0.092</ins> |
| MAGNET (loss="hybrid") | 3.1 | 0.181 | 0.768 | 0.089 | 0.532 | 0.512 | -0.124 |
| UMAP 0.5.12 | <ins>2.7</ins> | 0.098 | 0.718 | 0.099 | 0.537 | 0.497 | -0.105 |
| densMAP (umap-learn 0.5.12) | 10.7 | 0.068 | 0.658 | <ins>0.123</ins> | 0.545 | 0.443 | -0.103 |
| openTSNE 1.0.4 | 38.8 | <ins>0.187</ins> | <ins>0.772</ins> | 0.106 | 0.542 | **0.535** | **-0.086** |
| PaCMAP | 6.6 | 0.057 | 0.683 | -0.086 | 0.499 | 0.440 | -0.101 |
| TriMap | 12.7 | 0.059 | 0.690 | 0.051 | <ins>0.546</ins> | 0.443 | -0.139 |
| PHATE 2.0 | 20.3 | 0.066 | 0.622 | -0.005 | 0.524 | 0.337 | -0.143 |
| h-NNE | 5.7 | 0.154 | 0.732 | 0.102 | 0.535 | 0.470 | -0.189 |
| sklearn t-SNE (Barnes-Hut) | 48.6 | **0.192** | 0.742 | **0.232** | **0.576** | 0.455 | -0.116 |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 8.5 | 0.148 | 0.651 | 0.039 | 0.517 | 0.333 | -0.146 |

## 15 Gaussian blobs (10k x 200), 2 seeds

| method | time (s) | 15-NN pres. | trust. | dist. Spearman | triplet acc. | 10-NN acc. | compactness* |
|---|---|---|---|---|---|---|---|
| MAGNET (SGD, default) | <ins>1.5</ins> | 0.053 | 0.971 | <ins>0.697</ins> | <ins>0.731</ins> |  |  |
| MAGNET (loss="tsne") | **1.4** | **0.133** | <ins>0.974</ins> | 0.460 | 0.666 |  |  |
| MAGNET (loss="hybrid") | 1.8 | <ins>0.128</ins> | <ins>0.974</ins> | **0.708** | **0.740** |  |  |
| UMAP 0.5.12 | 2.4 | 0.047 | 0.971 | 0.079 | 0.530 |  |  |
| densMAP (umap-learn 0.5.12) | 7.5 | 0.032 | 0.968 | 0.127 | 0.543 |  |  |
| openTSNE 1.0.4 | 19.4 | 0.109 | **0.975** | 0.382 | 0.646 |  |  |
| PaCMAP | 4.1 | 0.033 | 0.971 | 0.666 | 0.720 |  |  |
| TriMap | 7.6 | 0.033 | 0.972 | 0.648 | 0.727 |  |  |
| PHATE 2.0 | 24.8 | 0.033 | 0.968 | 0.232 | 0.571 |  |  |
| h-NNE | 5.9 | 0.083 | 0.969 | 0.307 | 0.614 |  |  |
| sklearn t-SNE (Barnes-Hut) | 17.6 | 0.114 | **0.975** | 0.381 | 0.646 |  |  |
| t-SNE-CUDA 3.0.1 (GPU, RTX 2070) | 7.7 | 0.074 | <ins>0.974</ins> | 0.127 | 0.540 |  |  |
