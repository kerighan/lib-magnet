#include <pybind11/pybind11.h>
#include "layout.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_magnet, m) {
    m.doc() = "MAGNET v3 native kernels";
    m.def("optimize_layout", &optimize_layout, "Hogwild SGD layout optimisation (in place)",
          py::arg("Y"), py::arg("head"), py::arg("tail"), py::arg("epochs_per_sample"),
          py::arg("epoch_of_next_sample"), py::arg("epoch_of_next_negative_sample"),
          py::arg("indptr"), py::arg("indices"),
          py::arg("epoch_start"), py::arg("epoch_end"), py::arg("alpha0"), py::arg("alpha1"),
          py::arg("mid_w0"), py::arg("mid_w1"),
          py::arg("a"), py::arg("b"), py::arg("rep_a"), py::arg("rep_b"), py::arg("negative_sample_rate"), py::arg("loss"),
          py::arg("exaggeration"), py::arg("repulsion_strength"), py::arg("walk_len"), py::arg("mid_rate"),
          py::arg("optimizer"), py::arg("clip_mode"), py::arg("n_fixed"),
          py::arg("adam_m"), py::arg("adam_v"), py::arg("adam_t"), py::arg("beta1"), py::arg("beta2"),
          py::arg("walk_lens"), py::arg("mid_pairs"), py::arg("jump_indptr"), py::arg("jump_indices"),
          py::arg("mid_jump"), py::arg("kernel"), py::arg("node_b"), py::arg("seed"), py::arg("n_threads"));
    m.def("subspace_iteration", &subspace_iteration, "Block power iteration for spectral init",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("X"),
          py::arg("inv_sqrt_deg"), py::arg("component"), py::arg("n_iter"), py::arg("tol"),
          py::arg("n_threads"));
    m.def("chebyshev_eigenvectors", &chebyshev_eigenvectors,
          "Chebyshev-filtered subspace iteration for the leading non-trivial eigenvectors",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("X"),
          py::arg("inv_sqrt_deg"), py::arg("component"), py::arg("n_wanted"), py::arg("degree"),
          py::arg("max_iter"), py::arg("tol"), py::arg("n_threads"));
    m.def("nn_descent", &nn_descent, "NN-descent approximate kNN graph (squared euclidean)",
          py::arg("X"), py::arg("k"), py::arg("max_candidates") = 50, py::arg("n_iters") = 10,
          py::arg("delta") = 0.001, py::arg("n_trees") = 8, py::arg("leaf_size") = 30,
          py::arg("seed") = 0, py::arg("n_threads") = -1,
          py::arg("verbose") = false);
    m.def("nn_descent_sparse", &nn_descent_sparse, "NN-descent on a CSR matrix (squared euclidean)",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n_features"), py::arg("k"),
          py::arg("max_candidates") = 50, py::arg("n_iters") = 10, py::arg("delta") = 0.001,
          py::arg("n_trees") = 8, py::arg("leaf_size") = 30, py::arg("seed") = 0, py::arg("n_threads") = -1,
          py::arg("verbose") = false);
    m.def("knn_search", &knn_search, "Beam search of queries over a neighbour graph (dense data)",
          py::arg("X"), py::arg("gptr"), py::arg("gidx"), py::arg("landmarks"), py::arg("Q"), py::arg("k"), py::arg("ef") = 64,
          py::arg("n_entry") = 10, py::arg("seed") = 0, py::arg("n_threads") = -1);
    m.def("knn_search_sparse", &knn_search_sparse, "Beam search of queries over a neighbour graph (CSR data)",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n_features"),
          py::arg("gptr"), py::arg("gidx"), py::arg("landmarks"), py::arg("qptr"), py::arg("qidx"), py::arg("qdata"),
          py::arg("k"), py::arg("ef") = 64, py::arg("n_entry") = 10, py::arg("seed") = 0, py::arg("n_threads") = -1);
    m.def("interp_scatter", &interp_scatter, "Spread point charges on an interpolation grid (2-D)",
          py::arg("Y"), py::arg("charges"), py::arg("xmin"), py::arg("ymin"), py::arg("width"),
          py::arg("n_boxes"), py::arg("n_interp"), py::arg("n_threads") = -1);
    m.def("interp_gather", &interp_gather, "Read grid potentials back at the points (2-D)",
          py::arg("Y"), py::arg("grid"), py::arg("xmin"), py::arg("ymin"), py::arg("width"),
          py::arg("n_boxes"), py::arg("n_interp"), py::arg("n_threads") = -1);
    m.def("tsne_attraction", &tsne_attraction, "sum_j p_ij (1 + d^2/dof)^-1 (y_i - y_j) over a CSR graph",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("Y"), py::arg("dof") = 1.0f, py::arg("n_threads") = -1);
    m.def("mid_near_pairs_dense", &mid_near_pairs_dense, "PaCMAP-style mid-near partners (dense rows)",
          py::arg("X"), py::arg("n_pairs") = 5, py::arg("n_cand") = 6, py::arg("seed") = 0, py::arg("n_threads") = -1);
    m.def("mid_near_pairs_sparse", &mid_near_pairs_sparse, "PaCMAP-style mid-near partners (CSR rows)",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n_features"),
          py::arg("n_pairs") = 5, py::arg("n_cand") = 6, py::arg("seed") = 0, py::arg("n_threads") = -1);
    m.def("mid_spring", &mid_spring, "grad += w (Y_i - mean of partner positions)",
          py::arg("Y"), py::arg("pairs"), py::arg("grad"), py::arg("w"), py::arg("n_threads") = -1);
    m.def("scatter_padded_2d", &scatter_padded_2d, "Charges (1, x, y) on a zero-padded M x M grid, no reduction",
          py::arg("Y"), py::arg("xmin"), py::arg("ymin"), py::arg("width"), py::arg("n_boxes"), py::arg("n_interp"), py::arg("M"), py::arg("n_threads") = -1);
    m.def("spectral_products", &spectral_products, "Kernel-spectrum products of the FFT repulsion",
          py::arg("K1h"), py::arg("K2h"), py::arg("Wh"), py::arg("n_threads") = -1);
    m.def("gather_forces_2d", &gather_forces_2d, "Potentials at the points and repulsive forces: (sumQ, F)",
          py::arg("Y"), py::arg("phi"), py::arg("xmin"), py::arg("ymin"), py::arg("width"), py::arg("n_boxes"), py::arg("n_interp"), py::arg("n_threads") = -1);
    m.def("bbox_2d", &bbox_2d, "Bounding box of a (n, 2) array", py::arg("Y"), py::arg("n_threads") = -1);
    m.def("umap_attraction", &umap_attraction, "exact UMAP attraction over a CSR graph",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("Y"), py::arg("a"), py::arg("b"), py::arg("n_threads") = -1);
    m.def("tsne_update", &tsne_update, "Fused gains / momentum / position update / recentring",
          py::arg("Y"), py::arg("update"), py::arg("gains"), py::arg("grad"), py::arg("lr"), py::arg("momentum"),
          py::arg("n_threads") = -1);
}
