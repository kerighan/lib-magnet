// MAGNET v3 core: Hogwild stochastic gradient descent on a weighted neighbour graph.
//
// Positive (attractive) samples : edges of the graph, sampled proportionally to their weight
//                                 (UMAP-style "epochs_per_sample" scheduling).
// Negative (repulsive) samples  : uniformly random nodes.
// Mid-range samples             : nodes reached by a short random walk started from the tail of
//                                 the sampled edge. They pull the layout towards the diffusion
//                                 geometry of the graph (global structure) and are annealed to zero
//                                 during the first `mid_phase` fraction of the epochs.
#include "layout.hpp"
#include "rng.hpp"
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

static int resolve_threads(int n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 4 : (int)h;
}

namespace {

struct Adam {
    float* m; float* v; int32_t* t; float beta1, beta2; bool on;
};

// Everything the epoch worker needs; DIM is a compile-time dimension (2, 3) or 0 for a dynamic one.
struct Ctx {
    float* Y; int64_t n_nodes; int dim;
    const int32_t* head; const int32_t* tail; const float* eps; float* eons; float* eonns; int64_t n_edges;
    const int32_t* indptr; const int32_t* indices; const int32_t* walk_lens; bool use_mid;
    const int32_t* mid_pairs; int n_mid;
    const int32_t* jump_indptr; const int32_t* jump_indices; float mid_jump;  // two-layer walks
    float a, b, two_ab, two_b, negative_sample_rate, exaggeration, repulsion_strength, mid_rate;
    bool b_is_one; int m_neg, walk_len, loss, clip_mode; int64_t n_fixed;
    Adam ad; uint64_t seed; int nt;
};

template <int DIM>
inline void apply_step(const Ctx& c, float* yu, const float* g, float alpha, int64_t u) {
    constexpr int D = DIM ? DIM : MAX_DIM;
    const int dim = DIM ? DIM : c.dim;
    float step[D];
    if (c.ad.on) {
        const int32_t t = ++c.ad.t[u];
        const float c1 = 1.0f - powf(c.ad.beta1, (float)t), c2 = 1.0f - powf(c.ad.beta2, (float)t);
        float* m = c.ad.m + u * dim; float* v = c.ad.v + u * dim;
        for (int d = 0; d < dim; ++d) {
            m[d] = c.ad.beta1 * m[d] + (1.0f - c.ad.beta1) * g[d];
            v[d] = c.ad.beta2 * v[d] + (1.0f - c.ad.beta2) * g[d] * g[d];
            step[d] = alpha * (m[d] / c1) / (sqrtf(v[d] / c2) + 1e-7f);
        }
    } else {
        for (int d = 0; d < dim; ++d) step[d] = alpha * g[d];
    }
    const float cap = 4.0f * alpha;
    if (c.clip_mode == 1) {
        float n2 = 0.0f;
        for (int d = 0; d < dim; ++d) n2 += step[d] * step[d];
        if (n2 > cap * cap) { const float f = cap / sqrtf(n2); for (int d = 0; d < dim; ++d) step[d] *= f; }
    } else {
        for (int d = 0; d < dim; ++d) step[d] = step[d] > cap ? cap : (step[d] < -cap ? -cap : step[d]);
    }
    for (int d = 0; d < dim; ++d) yu[d] += step[d];
}

template <int DIM>
void epoch_worker(const Ctx& c, int epoch, float alpha, float mid_w, int tid) {
    constexpr int D = DIM ? DIM : MAX_DIM;
    const int dim = DIM ? DIM : c.dim;
    Rng rng(c.seed * 0x9E3779B1ULL + (uint64_t)epoch * 0x85EBCA6BULL + (uint64_t)tid * 0xC2B2AE35ULL);
    const int64_t e0 = c.n_edges * tid / c.nt;
    const int64_t e1 = c.n_edges * (tid + 1) / c.nt;
    const float fepoch = (float)epoch;
    float* Y = c.Y;
    float diff[D], gu[D], gk[D];
    int32_t negs[64]; float qk[64];

    for (int64_t i = e0; i < e1; ++i) {
        if (c.eons[i] > fepoch) continue;
        const int32_t u = c.head[i];
        const int32_t v = c.tail[i];
        float* yu = Y + (int64_t)u * dim;
        float* yv = Y + (int64_t)v * dim;
        const bool move_u = u >= c.n_fixed, move_v = v >= c.n_fixed;
        for (int d = 0; d < dim; ++d) gu[d] = 0.0f;

        float d2 = 0.0f;
        for (int d = 0; d < dim; ++d) { diff[d] = yu[d] - yv[d]; d2 += diff[d] * diff[d]; }
        const float pw_v = c.b_is_one ? d2 : (d2 > 0.0f ? powf(d2, c.b) : 0.0f);

        if (c.loss == 0) {
            // ---- attraction along the edge (gradient of log q)
            if (d2 > 0.0f) {
                const float coeff = -c.exaggeration * c.two_ab * pw_v / (d2 * (c.a * pw_v + 1.0f));
                for (int d = 0; d < dim; ++d) { const float g = coeff * diff[d]; gu[d] += g; gk[d] = -g; }
                if (move_v) apply_step<DIM>(c, yv, gk, alpha, v);
            }
            c.eons[i] += c.eps[i];
            // ---- repulsion from random nodes (gradient of log(1 - q))
            const float epn = c.eps[i] / c.negative_sample_rate;
            const int n_neg = (int)((fepoch - c.eonns[i]) / epn);
            for (int p = 0; p < n_neg; ++p) {
                const int32_t k = (int32_t)rng.below((uint32_t)c.n_nodes);
                if (k == u) continue;
                const float* yk = Y + (int64_t)k * dim;
                float dd = 0.0f;
                for (int d = 0; d < dim; ++d) { diff[d] = yu[d] - yk[d]; dd += diff[d] * diff[d]; }
                if (dd > 0.0f) {
                    const float pw = c.b_is_one ? dd : powf(dd, c.b);
                    const float coeff = c.two_b / ((0.001f + dd) * (c.a * pw + 1.0f));
                    for (int d = 0; d < dim; ++d) gu[d] += coeff * diff[d];
                } else {
                    for (int d = 0; d < dim; ++d) gu[d] += (rng.uniform() - 0.5f) * 1e-3f / std::max(alpha, 1e-6f);
                }
            }
            c.eonns[i] += n_neg * epn;
        } else {
            // ---- InfoNCE: softmax over {v, k_1..k_m} of the kernel q = 1 / (1 + a d^2b)
            const float q_v = 1.0f / (1.0f + c.a * pw_v);
            float Z = q_v;
            for (int p = 0; p < c.m_neg; ++p) {
                int32_t k = (int32_t)rng.below((uint32_t)c.n_nodes);
                if (k == u) k = (int32_t)((k + 1) % c.n_nodes);
                negs[p] = k;
                const float* yk = Y + (int64_t)k * dim;
                float dd = 0.0f;
                for (int d = 0; d < dim; ++d) { const float t = yu[d] - yk[d]; dd += t * t; }
                const float pw = c.b_is_one ? dd : (dd > 0.0f ? powf(dd, c.b) : 0.0f);
                qk[p] = 1.0f / (1.0f + c.a * pw);
                Z += qk[p];
            }
            const float w_v = q_v / Z;
            if (d2 > 0.0f) {
                const float coeff = -c.exaggeration * (1.0f - w_v) * c.two_ab * pw_v / (d2 * (c.a * pw_v + 1.0f));
                for (int d = 0; d < dim; ++d) { const float g = coeff * diff[d]; gu[d] += g; gk[d] = -g; }
                if (move_v) apply_step<DIM>(c, yv, gk, alpha, v);
            }
            for (int p = 0; p < c.m_neg; ++p) {
                const int32_t k = negs[p];
                float* yk = Y + (int64_t)k * dim;
                float dd = 0.0f;
                for (int d = 0; d < dim; ++d) { diff[d] = yu[d] - yk[d]; dd += diff[d] * diff[d]; }
                if (dd <= 0.0f) {
                    for (int d = 0; d < dim; ++d) gu[d] += (rng.uniform() - 0.5f) * 1e-3f / std::max(alpha, 1e-6f);
                    continue;
                }
                const float pw = c.b_is_one ? dd : powf(dd, c.b);
                const float w_k = qk[p] / Z;
                const float coeff = c.repulsion_strength * w_k * c.two_ab * pw / (dd * (c.a * pw + 1.0f));
                for (int d = 0; d < dim; ++d) { const float g = coeff * diff[d]; gu[d] += g; gk[d] = -g; }
                if (k >= c.n_fixed) apply_step<DIM>(c, yk, gk, alpha, k);
            }
            c.eons[i] += c.eps[i];
            c.eonns[i] = fepoch;
        }

        // ---- mid-range attraction: node reached by a short random walk from v
        if (mid_w > 0.0f && rng.uniform() < c.mid_rate) {
            int32_t w = v;
            if (c.jump_indptr) {  // walk on the two-layer graph: kNN edges, or a mid-range jump with probability mid_jump
                const int L = c.walk_lens ? c.walk_lens[v] : c.walk_len;
                for (int h = 0; h < L; ++h) {
                    const int32_t js = c.jump_indptr[w], je = c.jump_indptr[w + 1];
                    if (je > js && rng.uniform() < c.mid_jump) {
                        w = c.jump_indices[js + (int32_t)rng.below((uint32_t)(je - js))];
                    } else {
                        const int32_t s0 = c.indptr[w], e0w = c.indptr[w + 1];
                        if (e0w <= s0) break;
                        w = c.indices[s0 + (int32_t)rng.below((uint32_t)(e0w - s0))];
                    }
                }
            } else if (c.mid_pairs) {  // precomputed partners (PaCMAP-style mid-near pairs)
                w = c.mid_pairs[(int64_t)u * c.n_mid + rng.below((uint32_t)c.n_mid)];
            } else {            // short random walk on the graph from v
                const int L = c.walk_lens ? c.walk_lens[v] : c.walk_len;
                for (int h = 0; h < L; ++h) {
                    const int32_t s0 = c.indptr[w], e0w = c.indptr[w + 1];
                    if (e0w <= s0) break;
                    w = c.indices[s0 + (int32_t)rng.below((uint32_t)(e0w - s0))];
                }
            }
            if (w != u) {
                float* yw = Y + (int64_t)w * dim;
                for (int d = 0; d < dim; ++d) { const float g = -mid_w * (yu[d] - yw[d]); gu[d] += g; gk[d] = -g; }
                if (w >= c.n_fixed) apply_step<DIM>(c, yw, gk, alpha, w);
            }
        }
        if (move_u) apply_step<DIM>(c, yu, gu, alpha, u);
    }
}

template <int DIM>
void run_epoch(const Ctx& c, int epoch, float alpha, float mid_w) {
    if (c.nt == 1) { epoch_worker<DIM>(c, epoch, alpha, mid_w, 0); return; }
    std::vector<std::thread> pool;
    pool.reserve(c.nt);
    for (int t = 0; t < c.nt; ++t) pool.emplace_back([&c, epoch, alpha, mid_w, t] { epoch_worker<DIM>(c, epoch, alpha, mid_w, t); });
    for (auto& th : pool) th.join();
}

}  // namespace

void optimize_layout(
    arr<float> Y_,
    arr<int32_t> head_, arr<int32_t> tail_,
    arr<float> epochs_per_sample_,
    arr<float> epoch_of_next_sample_,
    arr<float> epoch_of_next_negative_sample_,
    arr<int32_t> indptr_, arr<int32_t> indices_,
    int epoch_start, int epoch_end, float alpha0, float alpha1, float mid_w0, float mid_w1,
    float a, float b, float negative_sample_rate, int loss, float exaggeration,
    float repulsion_strength, int walk_len, float mid_rate,
    int optimizer, int clip_mode, int64_t n_fixed,
    arr<float> adam_m_, arr<float> adam_v_, arr<int32_t> adam_t_, float beta1, float beta2,
    arr<int32_t> walk_lens_, arr<int32_t> mid_pairs_,
    arr<int32_t> jump_indptr_, arr<int32_t> jump_indices_, float mid_jump,
    uint64_t seed, int n_threads)
{
    auto Yb = Y_.request();
    require(Yb.ndim == 2, "Y must be 2-dimensional");
    const int64_t n_nodes = Yb.shape[0];
    const int dim = (int)Yb.shape[1];
    require(dim >= 1 && dim <= MAX_DIM, "n_components must be between 1 and 64");

    require(head_.ndim() == 1, "head must be 1-d");
    const int64_t n_edges = head_.shape(0);
    require(tail_.ndim() == 1 && tail_.shape(0) == n_edges && epochs_per_sample_.shape(0) == n_edges &&
            epoch_of_next_sample_.shape(0) == n_edges && epoch_of_next_negative_sample_.shape(0) == n_edges,
            "head, tail and epoch arrays must have the same length");
    require(epoch_start >= 0 && epoch_end >= epoch_start, "invalid epoch range");
    require(negative_sample_rate > 0.0f, "negative_sample_rate must be positive");
    require(loss == 0 || loss == 1, "loss must be 0 (umap) or 1 (infonce)");
    require(n_fixed >= 0 && n_fixed <= n_nodes, "n_fixed out of range");

    Ctx c;
    c.Y = static_cast<float*>(Yb.ptr); c.n_nodes = n_nodes; c.dim = dim;
    c.head = head_.data(); c.tail = tail_.data(); c.eps = epochs_per_sample_.data();
    c.eons = epoch_of_next_sample_.mutable_data(); c.eonns = epoch_of_next_negative_sample_.mutable_data();
    c.n_edges = n_edges;
    for (int64_t i = 0; i < n_edges; ++i) {
        require(c.head[i] >= 0 && c.head[i] < n_nodes && c.tail[i] >= 0 && c.tail[i] < n_nodes, "edge index out of range");
        require(c.eps[i] > 0.0f, "epochs_per_sample must be positive");
    }
    c.ad = Adam{nullptr, nullptr, nullptr, beta1, beta2, optimizer == 1};
    if (c.ad.on) {
        require(adam_m_.ndim() == 2 && adam_m_.shape(0) == n_nodes && adam_m_.shape(1) == dim &&
                adam_v_.ndim() == 2 && adam_v_.shape(0) == n_nodes && adam_v_.shape(1) == dim &&
                adam_t_.ndim() == 1 && adam_t_.shape(0) == n_nodes, "Adam state arrays have the wrong shape");
        c.ad.m = adam_m_.mutable_data(); c.ad.v = adam_v_.mutable_data(); c.ad.t = adam_t_.mutable_data();
    }
    c.mid_pairs = nullptr; c.n_mid = 0;
    const bool has_pairs = mid_pairs_.size() > 0;
    if (has_pairs) {
        require(mid_pairs_.ndim() == 2 && mid_pairs_.shape(0) == n_nodes && mid_pairs_.shape(1) >= 1, "mid_pairs must be (n, m)");
        for (int64_t i = 0; i < mid_pairs_.size(); ++i) require(mid_pairs_.data()[i] >= 0 && mid_pairs_.data()[i] < n_nodes, "mid_pairs index out of range");
        c.mid_pairs = mid_pairs_.data(); c.n_mid = (int)mid_pairs_.shape(1);
    }
    c.jump_indptr = nullptr; c.jump_indices = nullptr; c.mid_jump = mid_jump;
    const bool has_jumps = jump_indptr_.size() > 0 && mid_jump > 0.0f;
    if (has_jumps) {
        validate_csr(jump_indptr_, jump_indices_, nullptr, n_nodes);
        c.jump_indptr = jump_indptr_.data(); c.jump_indices = jump_indices_.data();
    }
    const bool walks = has_jumps || !has_pairs;
    c.use_mid = (mid_w0 > 0.0f || mid_w1 > 0.0f) && mid_rate > 0.0f && (has_pairs || walk_len > 0);
    if (c.use_mid && walks) validate_csr(indptr_, indices_, nullptr, n_nodes);
    c.indptr = (c.use_mid && walks) ? indptr_.data() : nullptr;
    c.indices = (c.use_mid && walks) ? indices_.data() : nullptr;
    if (has_jumps) c.mid_pairs = nullptr;
    c.walk_lens = nullptr;
    if (c.use_mid && walk_lens_.size() > 0) {
        require(walk_lens_.ndim() == 1 && walk_lens_.shape(0) == n_nodes, "walk_lens must have n entries");
        for (int64_t i = 0; i < n_nodes; ++i) require(walk_lens_.data()[i] >= 0 && walk_lens_.data()[i] <= 64, "walk_lens out of range");
        c.walk_lens = walk_lens_.data();
    }
    c.nt = (int)std::min<int64_t>(resolve_threads(n_threads), std::max<int64_t>(1, n_edges / 4096));
    c.a = a; c.b = b; c.two_ab = 2.0f * a * b; c.two_b = 2.0f * b * repulsion_strength;
    c.negative_sample_rate = negative_sample_rate; c.exaggeration = exaggeration;
    c.repulsion_strength = repulsion_strength; c.mid_rate = mid_rate;
    c.b_is_one = std::fabs(b - 1.0f) < 1e-6f;
    c.m_neg = std::min(64, std::max(1, (int)std::lround(negative_sample_rate)));
    c.walk_len = walk_len; c.loss = loss; c.clip_mode = clip_mode; c.n_fixed = n_fixed; c.seed = seed;
    const int n_span = std::max(1, epoch_end - epoch_start);

    py::gil_scoped_release release;

    for (int epoch = epoch_start; epoch < epoch_end; ++epoch) {
        const float frac = (float)(epoch - epoch_start) / (float)n_span;
        const float alpha = alpha0 + (alpha1 - alpha0) * frac;
        const float mid_w = c.use_mid ? mid_w0 + (mid_w1 - mid_w0) * frac : 0.0f;
        switch (dim) {
            case 2: run_epoch<2>(c, epoch, alpha, mid_w); break;
            case 3: run_epoch<3>(c, epoch, alpha, mid_w); break;
            default: run_epoch<0>(c, epoch, alpha, mid_w); break;
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Spectral initialisation by block power iteration on the lazy normalised adjacency
//     M = (I + D^-1/2 A D^-1/2) / 2
// X (n, c) is iterated in place: X <- M X, deflated of the trivial per-component eigenvector
// (deg^{1/2} restricted to each connected component), then orthonormalised (Gram-Schmidt, c small).
// Returns X. Stops early when the Rayleigh quotients stop moving (relative change < tol).
arr<float> subspace_iteration(
    arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_,
    arr<float> X_, arr<float> inv_sqrt_deg_, arr<int32_t> component_,
    int n_iter, float tol, int n_threads)
{
    auto Xb = X_.request();
    require(Xb.ndim == 2, "X must be 2-d");
    const int64_t n = Xb.shape[0];
    const int c = (int)Xb.shape[1];
    require(c >= 1 && c <= MAX_DIM, "X must have between 1 and 64 columns");
    validate_csr(indptr_, indices_, &data_, n);
    require(inv_sqrt_deg_.ndim() == 1 && inv_sqrt_deg_.shape(0) == n, "inv_sqrt_deg must have n entries");
    require(component_.ndim() == 1 && component_.shape(0) == n, "component must have n entries");
    float* X = static_cast<float*>(Xb.ptr);
    const int32_t* indptr = indptr_.data();
    const int32_t* indices = indices_.data();
    const float* data = data_.data();
    const float* sd = inv_sqrt_deg_.data();  // deg^{-1/2}
    const int32_t* comp = component_.data();
    int n_comp = 0;
    for (int64_t i = 0; i < n; ++i) { require(comp[i] >= 0, "component labels must be >= 0"); n_comp = std::max(n_comp, comp[i] + 1); }

    const int nt = std::min<int64_t>(resolve_threads(n_threads), std::max<int64_t>(1, n / 2048));
    std::vector<float> Z((size_t)n * c);
    std::vector<double> comp_norm(n_comp, 0.0);
    for (int64_t i = 0; i < n; ++i) comp_norm[comp[i]] += (double)sd[i] * sd[i];

    py::gil_scoped_release release;

    auto spmv = [&](const float* in, float* out) {
        auto work = [&](int tid) {
            const int64_t r0 = n * tid / nt, r1 = n * (tid + 1) / nt;
            float acc[MAX_DIM];
            for (int64_t i = r0; i < r1; ++i) {
                for (int k = 0; k < c; ++k) acc[k] = 0.0f;
                const float sdi = sd[i];
                for (int32_t j = indptr[i]; j < indptr[i + 1]; ++j) {
                    const int32_t col = indices[j];
                    const float w = data[j] * sd[col];           // D^-1/2 handled via sd = 1/sqrt(deg)
                    const float* xin = in + (int64_t)col * c;
                    for (int k = 0; k < c; ++k) acc[k] += w * xin[k];
                }
                const float* xi = in + i * c;
                float* o = out + i * c;
                for (int k = 0; k < c; ++k) o[k] = 0.5f * (xi[k] + sdi * acc[k]);
            }
        };
        if (nt == 1) { work(0); return; }
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; ++t) pool.emplace_back(work, t);
        for (auto& th : pool) th.join();
    };

    // sd holds deg^{-1/2}; the trivial eigenvector of M is deg^{1/2} = 1/sd.
    std::vector<double> proj((size_t)n_comp * c);
    std::vector<double> trivial_norm(n_comp, 0.0);
    for (int64_t i = 0; i < n; ++i) { const double t = sd[i] > 0 ? 1.0 / sd[i] : 0.0; trivial_norm[comp[i]] += t * t; }

    auto deflate_orthonormalize = [&](float* M) {
        // remove per-component trivial component
        std::fill(proj.begin(), proj.end(), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            const double t = sd[i] > 0 ? 1.0 / sd[i] : 0.0;
            double* p = proj.data() + (size_t)comp[i] * c;
            const float* m = M + i * c;
            for (int k = 0; k < c; ++k) p[k] += t * m[k];
        }
        for (int64_t i = 0; i < n; ++i) {
            const double t = sd[i] > 0 ? 1.0 / sd[i] : 0.0;
            const double* p = proj.data() + (size_t)comp[i] * c;
            const double nrm = trivial_norm[comp[i]];
            float* m = M + i * c;
            if (nrm > 0) for (int k = 0; k < c; ++k) m[k] -= (float)(t * p[k] / nrm);
        }
        // modified Gram-Schmidt over the c columns
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < k; ++j) {
                double dot = 0.0;
                for (int64_t i = 0; i < n; ++i) dot += (double)M[i * c + k] * M[i * c + j];
                for (int64_t i = 0; i < n; ++i) M[i * c + k] -= (float)(dot * M[i * c + j]);
            }
            double nrm = 0.0;
            for (int64_t i = 0; i < n; ++i) nrm += (double)M[i * c + k] * M[i * c + k];
            nrm = std::sqrt(nrm);
            const float inv = nrm > 1e-20 ? (float)(1.0 / nrm) : 0.0f;
            for (int64_t i = 0; i < n; ++i) M[i * c + k] *= inv;
        }
    };

    deflate_orthonormalize(X);
    std::vector<double> prev_ray(c, 0.0);
    for (int it = 0; it < n_iter; ++it) {
        spmv(X, Z.data());
        // Rayleigh quotients (X orthonormal): diag(X^T Z)
        double max_rel = 0.0;
        for (int k = 0; k < c; ++k) {
            double r = 0.0;
            for (int64_t i = 0; i < n; ++i) r += (double)X[i * c + k] * Z[i * c + k];
            max_rel = std::max(max_rel, std::fabs(r - prev_ray[k]) / std::max(1e-12, std::fabs(r)));
            prev_ray[k] = r;
        }
        std::copy(Z.begin(), Z.end(), X);
        deflate_orthonormalize(X);
        if (it > 10 && max_rel < tol) break;
    }
    return X_;
}
