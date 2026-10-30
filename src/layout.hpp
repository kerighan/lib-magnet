#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// NB: forcecast means a wrong dtype / non-contiguous input is silently copied. Kernels that update an
// array in place (optimize_layout, subspace_iteration, chebyshev_eigenvectors) therefore rely on the
// Python side passing C-contiguous arrays of the right dtype (magnet/__init__.py does).
template <typename T>
using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

constexpr int MAX_DIM = 64;  // stack buffers in the kernels

// Checks that (indptr, indices, data) is a valid CSR structure over n rows and n_cols columns
// (n_cols defaults to n: square matrix).
inline void validate_csr(const arr<int32_t>& indptr, const arr<int32_t>& indices, const arr<float>* data, int64_t n, int64_t n_cols = -1) {
    if (n_cols < 0) n_cols = n;
    if (indptr.ndim() != 1 || indptr.shape(0) != n + 1) throw std::runtime_error("indptr must have n + 1 entries");
    const int32_t* ip = indptr.data();
    if (ip[0] != 0) throw std::runtime_error("indptr[0] must be 0");
    for (int64_t i = 0; i < n; ++i) if (ip[i + 1] < ip[i]) throw std::runtime_error("indptr must be non-decreasing");
    const int64_t nnz = ip[n];
    if (indices.ndim() != 1 || indices.shape(0) != nnz) throw std::runtime_error("indices must have indptr[n] entries");
    if (data && (data->ndim() != 1 || data->shape(0) != nnz)) throw std::runtime_error("data must have indptr[n] entries");
    const int32_t* ix = indices.data();
    for (int64_t j = 0; j < nnz; ++j) if (ix[j] < 0 || ix[j] >= n_cols) throw std::runtime_error("column index out of range");
}

inline void require(bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); }

void optimize_layout(
    arr<float> Y,
    arr<int32_t> head, arr<int32_t> tail,
    arr<float> epochs_per_sample,
    arr<float> epoch_of_next_sample,
    arr<float> epoch_of_next_negative_sample,
    arr<int32_t> indptr, arr<int32_t> indices,
    int epoch_start, int epoch_end, float alpha0, float alpha1, float mid_w0, float mid_w1,
    float a, float b, float negative_sample_rate, int loss, float exaggeration,
    float repulsion_strength, int walk_len, float mid_rate,
    int optimizer, int clip_mode, int64_t n_fixed,
    arr<float> adam_m, arr<float> adam_v, arr<int32_t> adam_t, float beta1, float beta2,
    arr<int32_t> walk_lens, arr<int32_t> mid_pairs,
    arr<int32_t> jump_indptr, arr<int32_t> jump_indices, float mid_jump,
    int kernel, arr<float> node_b, uint64_t seed, int n_threads);

arr<float> subspace_iteration(
    arr<int32_t> indptr, arr<int32_t> indices, arr<float> data,
    arr<float> X, arr<float> inv_sqrt_deg, arr<int32_t> component,
    int n_iter, float tol, int n_threads);

arr<float> chebyshev_eigenvectors(
    arr<int32_t> indptr, arr<int32_t> indices, arr<float> data,
    arr<float> X, arr<float> inv_sqrt_deg, arr<int32_t> component,
    int n_wanted, int degree, int max_iter, float tol, int n_threads);

std::tuple<arr<int32_t>, arr<float>> nn_descent(
    arr<float> X, int k, int max_candidates, int n_iters, float delta, int n_trees, int leaf_size,
    uint64_t seed, int n_threads, bool verbose);

std::tuple<arr<int32_t>, arr<float>> nn_descent_sparse(
    arr<int32_t> indptr, arr<int32_t> indices, arr<float> data, int d,
    int k, int max_candidates, int n_iters, float delta, int n_trees, int leaf_size,
    uint64_t seed, int n_threads, bool verbose);

std::tuple<arr<int32_t>, arr<float>> knn_search(
    arr<float> X, arr<int32_t> gptr, arr<int32_t> gidx, arr<int32_t> landmarks, arr<float> Q,
    int k, int ef, int n_entry, uint64_t seed, int n_threads);

std::tuple<arr<int32_t>, arr<float>> knn_search_sparse(
    arr<int32_t> indptr, arr<int32_t> indices, arr<float> data, int d,
    arr<int32_t> gptr, arr<int32_t> gidx, arr<int32_t> landmarks,
    arr<int32_t> qptr, arr<int32_t> qidx, arr<float> qdata,
    int k, int ef, int n_entry, uint64_t seed, int n_threads);

arr<float> interp_scatter(arr<float> Y, arr<float> charges, float xmin, float ymin, float width,
                          int n_boxes, int n_interp, int n_threads);
arr<float> interp_gather(arr<float> Y, arr<float> grid, float xmin, float ymin, float width,
                         int n_boxes, int n_interp, int n_threads);
arr<float> tsne_attraction(arr<int32_t> indptr, arr<int32_t> indices, arr<float> data, arr<float> Y, float dof, int n_threads);

void tsne_update(arr<float> Y, arr<float> update, arr<float> gains, arr<float> grad, float lr, float momentum, int n_threads);

arr<int32_t> mid_near_pairs_dense(arr<float> X, int n_pairs, int n_cand, uint64_t seed, int n_threads);
arr<int32_t> mid_near_pairs_sparse(arr<int32_t> indptr, arr<int32_t> indices, arr<float> data, int d,
                                   int n_pairs, int n_cand, uint64_t seed, int n_threads);

void mid_spring(arr<float> Y, arr<int32_t> pairs, arr<float> grad, float w, int n_threads);

arr<float> scatter_padded_2d(arr<float> Y, float xmin, float ymin, float width, int n_boxes, int n_interp, int M, int n_threads);
arr<float> spectral_products(arr<float> K1h, arr<float> K2h, arr<float> Wh, int n_threads);
std::tuple<float, arr<float>> gather_forces_2d(arr<float> Y, arr<float> phi, float xmin, float ymin, float width, int n_boxes, int n_interp, int n_threads);

std::tuple<float, float, float, float> bbox_2d(arr<float> Y, int n_threads);

arr<float> umap_attraction(arr<int32_t> indptr, arr<int32_t> indices, arr<float> data, arr<float> Y, float a, float b, int n_threads);
