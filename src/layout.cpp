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
#include <atomic>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>

static int resolve_threads(int n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 4 : (int)h;
}

namespace {

// Owner mode: the attraction moves the head only, scaled by OWNER_ATT. Hogwild moves both ends by the same gradient
// and the mirrored edge (v, u) is sampled at the same rate, so the expected pull per epoch is matched by a factor 2;
// 1.5 measured best on MNIST 70k (15-NN kept 0.189 / trustworthiness 0.948 / distance Spearman 0.451 against
// 0.193 / 0.957 / 0.415 for hogwild; factor 1: 0.192 / 0.917 / 0.485, factor 2: 0.183 / 0.948 / 0.435): part of the
// hogwild pull on v lands while v is being repelled, which a bigger single step on u does not reproduce. The
// mid-range spring keeps its coefficient and the pull hogwild applies to the partner is delivered through the
// reverse pairs (see the worker). A second pure-attraction pass over the node's sampled edges, emulating the
// mirrored pulls as separate clipped steps, was worse (0.158 / 0.951 / 0.388) and is not kept.
// MAGNET_PROFILE=1 in the environment prints the per-thread busy and barrier-wait times of each call.
static const float OWNER_ATT = 1.5f;
static const bool PROFILE = std::getenv("MAGNET_PROFILE") != nullptr;

// Reusable spin barrier (C++17 has none; atomics only, so the binary does not depend on a recent libstdc++):
// the persistent epoch threads meet here once per epoch, every 5-50 ms, so spinning with yield is the right tool.
struct Barrier {
    std::atomic<int> count{0}; std::atomic<int> gen{0}; const int n;
    explicit Barrier(int n_) : n(n_) {}
    void wait() {
        const int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            gen.store(g + 1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) std::this_thread::yield();
        }
    }
};

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
    float rep_a, rep_b;   // repulsion-side kernel (defaults to a, b)
    bool b_is_one, rep_b_is_one; int m_neg, walk_len, loss, clip_mode, kernel; int64_t n_fixed;
    const float* node_b;  // optional per-node kernel exponent (power kernel)
    bool rep_coupled;     // the per-node exponent also drives the repulsion only when the two exponents are coupled
    Adam ad; uint64_t seed; int nt;
    // parallel == 1 ("owner"): thread t owns the nodes [own[t], own[t+1]) and the edges [eown[t], eown[t+1]) whose head
    // lies there; it writes only its own nodes and reads every other node from Ysnap, a copy of Y frozen for the
    // epoch (double-buffered). No cache line is ever written by two threads, and the lines of the snapshot stay
    // shared across cores instead of bouncing. The attraction and the mid-range spring move the head only, with a
    // doubled coefficient: the mirrored edge (v, u) moves v, so the expected displacement per epoch is the hogwild one.
    int parallel; const int64_t* eown; int n_chunks;
    const int32_t* rev_indptr; const int32_t* rev_indices;   // owner mode: who chose u as a mid-range partner
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

template <int DIM, bool OWNER>
void epoch_worker(const Ctx& c, int epoch, float alpha, float mid_w, int tid, const float* Ysnap) {
    // tid is the thread in hogwild mode and the node chunk in owner mode (the RNG is seeded by it either way)
    constexpr int D = DIM ? DIM : MAX_DIM;
    const int dim = DIM ? DIM : c.dim;
    Rng rng(c.seed * 0x9E3779B1ULL + (uint64_t)epoch * 0x85EBCA6BULL + (uint64_t)tid * 0xC2B2AE35ULL);
    const int64_t e0 = OWNER ? c.eown[tid] : c.n_edges * tid / c.nt;
    const int64_t e1 = OWNER ? c.eown[tid + 1] : c.n_edges * (tid + 1) / c.nt;
    const float inv_n_mid = c.n_mid > 0 ? 1.0f / (float)c.n_mid : 0.0f;
    const float* Yr = OWNER ? Ysnap : c.Y;   // where the other nodes are read from
    const float pair_scale = OWNER ? OWNER_ATT : 1.0f;
    const float fepoch = (float)epoch;
    float* Y = c.Y;
    float diff[D], gu[D], gk[D];
    int32_t negs[64]; float qk[64];

    for (int64_t i = e0; i < e1; ++i) {
        if (c.eons[i] > fepoch) continue;
        const int32_t u = c.head[i];
        const int32_t v = c.tail[i];
        float* yu = Y + (int64_t)u * dim;
        float* yv = OWNER ? const_cast<float*>(Yr + (int64_t)v * dim) : Y + (int64_t)v * dim;
        const bool move_u = u >= c.n_fixed, move_v = !OWNER && v >= c.n_fixed;
        for (int d = 0; d < dim; ++d) gu[d] = 0.0f;

        float d2 = 0.0f;
        for (int d = 0; d < dim; ++d) { diff[d] = yu[d] - yv[d]; d2 += diff[d] * diff[d]; }
        const float b_u = c.node_b ? c.node_b[u] : c.b;
        const float b_uv = c.node_b ? 0.5f * (c.node_b[u] + c.node_b[v]) : c.b;
        const float pw_v = (c.b_is_one && !c.node_b) ? d2 : (d2 > 0.0f ? powf(d2, b_uv) : 0.0f);

        if (c.loss == 0) {
            // ---- attraction along the edge (gradient of log q)
            if (d2 > 0.0f) {
                float coeff;
                if (c.kernel == 0) coeff = -pair_scale * c.exaggeration * 2.0f * c.a * b_uv * pw_v / (d2 * (c.a * pw_v + 1.0f));
                else if (c.kernel == 1) coeff = -pair_scale * c.exaggeration * 2.0f * c.a;          // gaussian: q = exp(-a d^2)
                else if (c.kernel == 2) { const float dd = sqrtf(d2); coeff = -pair_scale * c.exaggeration * c.a * tanhf(c.a * dd) / dd; }  // sech: q = 1 / cosh(a d)
                else coeff = -pair_scale * c.exaggeration * 2.0f * c.a * (1.0f + tanhf(c.a * d2));  // tanh: q = 1 - tanh(a d^2)
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
                const float* yk = Yr + (int64_t)k * dim;
                float dd = 0.0f;
                for (int d = 0; d < dim; ++d) { diff[d] = yu[d] - yk[d]; dd += diff[d] * diff[d]; }
                if (dd > 0.0f) {
                    float coeff;
                    if (c.kernel == 0) {
                        if (c.node_b && c.rep_coupled) { const float pw = powf(dd, b_u); coeff = 2.0f * b_u * c.repulsion_strength / ((0.001f + dd) * (c.a * pw + 1.0f)); }
                        else { const float pw = c.rep_b_is_one ? dd : powf(dd, c.rep_b); coeff = c.two_b / ((0.001f + dd) * (c.rep_a * pw + 1.0f)); }
                    }
                    else if (c.kernel == 1) { const float q = expf(-c.a * dd); coeff = c.repulsion_strength * 2.0f * c.a * q / std::max(1.0f - q, 1e-3f); }
                    else if (c.kernel == 2) { const float r = sqrtf(dd); const float q = 1.0f / coshf(c.a * r); coeff = c.repulsion_strength * c.a * q * tanhf(c.a * r) / (r * std::max(1.0f - q, 1e-3f)); }
                    else { const float th = std::max(tanhf(c.a * dd), 1e-3f); coeff = c.repulsion_strength * 2.0f * c.a * (1.0f - th * th) / th; }
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
                float* yw = OWNER ? const_cast<float*>(Yr + (int64_t)w * dim) : Y + (int64_t)w * dim;
                for (int d = 0; d < dim; ++d) { const float g = -mid_w * (yu[d] - yw[d]); gu[d] += g; gk[d] = -g; }
                if (!OWNER && w >= c.n_fixed) apply_step<DIM>(c, yw, gk, alpha, w);
            }
        }
        if (move_u) apply_step<DIM>(c, yu, gu, alpha, u);
        // ---- owner mode: the pulls that hogwild applies to the partner w are delivered here, to u as the partner of
        // the nodes that chose it (reverse pairs), at the hogwild rate (mid_rate per edge sample of the chooser, one
        // chooser in n_mid), each as its own step so that the clipping matches
        if (OWNER && move_u && mid_w > 0.0f && c.rev_indptr) {
            const int32_t rs = c.rev_indptr[u], re = c.rev_indptr[u + 1];
            float p = c.mid_rate * (float)(re - rs) * inv_n_mid;
            while (p > 0.0f && re > rs) {
                if (rng.uniform() < p) {
                    const int32_t x = c.rev_indices[rs + (int32_t)rng.below((uint32_t)(re - rs))];
                    if (x != u) {
                        const float* yx = Yr + (int64_t)x * dim;
                        for (int d = 0; d < dim; ++d) gk[d] = -mid_w * (yu[d] - yx[d]);
                        apply_step<DIM>(c, yu, gk, alpha, u);
                    }
                }
                p -= 1.0f;
            }
        }
    }
}

// Runs epochs [e0, e1) with persistent threads. Hogwild: a barrier per epoch (the learning rate changes).
// Owner mode: thread t copies its own nodes into the next snapshot at the end of each epoch, then the barrier;
// the two snapshot buffers alternate so that one barrier per epoch suffices.
template <int DIM>
void run_epochs(Ctx& c, int e0, int e1, float alpha0, float alpha1, float mid_w0, float mid_w1, const int64_t* own,
                std::vector<float>* snapA, std::vector<float>* snapB) {
    const int n_span = std::max(1, e1 - e0);
    auto sched = [&](int epoch, float& alpha, float& mid_w) {
        const float frac = (float)(epoch - e0) / (float)n_span;
        alpha = alpha0 + (alpha1 - alpha0) * frac;
        mid_w = c.use_mid ? mid_w0 + (mid_w1 - mid_w0) * frac : 0.0f;
    };
    const bool owner = c.parallel == 1;
    if (c.nt == 1 && !owner) {
        for (int epoch = e0; epoch < e1; ++epoch) {
            float alpha, mid_w; sched(epoch, alpha, mid_w);
            epoch_worker<DIM, false>(c, epoch, alpha, mid_w, 0, nullptr);
        }
        return;
    }
    Barrier bar(c.nt);
    float* bufs[2] = {snapA ? snapA->data() : nullptr, snapB ? snapB->data() : nullptr};
    std::vector<double> busy(c.nt, 0.0), waited(c.nt, 0.0);
    const auto t_start = std::chrono::steady_clock::now();
    std::atomic<int64_t> next_chunk{0};   // owner mode: chunks are claimed dynamically (a preempted thread takes fewer)
    auto work = [&](int tid) {
        for (int epoch = e0; epoch < e1; ++epoch) {
            const auto t0 = std::chrono::steady_clock::now();
            float alpha, mid_w; sched(epoch, alpha, mid_w);
            if (owner) {
                const int cur = (epoch - e0) & 1;
                const int64_t base = (int64_t)(epoch - e0) * (c.n_chunks + c.nt);   // each thread adds exactly (chunks + 1)
                for (;;) {
                    const int64_t k = next_chunk.fetch_add(1, std::memory_order_relaxed) - base;
                    if (k >= c.n_chunks) break;
                    epoch_worker<DIM, true>(c, epoch, alpha, mid_w, (int)k, bufs[cur]);
                    const int64_t a = own[k] * c.dim, b = own[k + 1] * c.dim;
                    std::memcpy(bufs[cur ^ 1] + a, c.Y + a, (size_t)(b - a) * sizeof(float));
                }
            } else {
                epoch_worker<DIM, false>(c, epoch, alpha, mid_w, tid, nullptr);
            }
            const auto t1 = std::chrono::steady_clock::now();
            bar.wait();
            if (PROFILE) { busy[tid] += std::chrono::duration<double>(t1 - t0).count(); waited[tid] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(); }
        }
    };
    if (c.nt == 1) work(0);
    else {
        std::vector<std::thread> pool;
        pool.reserve(c.nt);
        for (int t = 0; t < c.nt; ++t) pool.emplace_back(work, t);
        for (auto& th : pool) th.join();
    }
    if (PROFILE) {
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        double bmax = 0, bsum = 0, wsum = 0;
        for (int t = 0; t < c.nt; ++t) { bmax = std::max(bmax, busy[t]); bsum += busy[t]; wsum += waited[t]; }
        std::fprintf(stderr, "[profile] %s nt=%d epochs=%d wall=%.3fs busy mean=%.3fs max=%.3fs wait mean=%.3fs\n",
                     owner ? "owner" : "hogwild", c.nt, e1 - e0, wall, bsum / c.nt, bmax, wsum / c.nt);
    }
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
    float a, float b, float rep_a, float rep_b, float negative_sample_rate, int loss, float exaggeration,
    float repulsion_strength, int walk_len, float mid_rate,
    int optimizer, int clip_mode, int64_t n_fixed,
    arr<float> adam_m_, arr<float> adam_v_, arr<int32_t> adam_t_, float beta1, float beta2,
    arr<int32_t> walk_lens_, arr<int32_t> mid_pairs_,
    arr<int32_t> jump_indptr_, arr<int32_t> jump_indices_, float mid_jump,
    int kernel, arr<float> node_b_, uint64_t seed, int n_threads, int parallel,
    arr<int32_t> rev_indptr_, arr<int32_t> rev_indices_)
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
    require(kernel >= 0 && kernel <= 3, "kernel must be 0 (power), 1 (gaussian), 2 (sech) or 3 (tanh)");
    require(loss == 0 || kernel == 0, "the InfoNCE loss supports the power kernel only");
    require(parallel == 0 || parallel == 1, "parallel must be 0 (hogwild) or 1 (owner)");
    require(parallel == 0 || loss == 0, "parallel='owner' supports the umap loss only (InfoNCE moves the negatives)");
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
    c.rep_coupled = rep_b <= 0.0f;
    if (rep_b <= 0.0f) { rep_a = a; rep_b = b; }
    require(rep_a > 0.0f, "repulsion kernel a must be positive");
    c.a = a; c.b = b; c.rep_a = rep_a; c.rep_b = rep_b; c.two_ab = 2.0f * a * b; c.two_b = 2.0f * rep_b * repulsion_strength;
    c.negative_sample_rate = negative_sample_rate; c.exaggeration = exaggeration;
    c.repulsion_strength = repulsion_strength; c.mid_rate = mid_rate;
    c.b_is_one = std::fabs(b - 1.0f) < 1e-6f;
    c.rep_b_is_one = std::fabs(rep_b - 1.0f) < 1e-6f;
    c.m_neg = std::min(64, std::max(1, (int)std::lround(negative_sample_rate)));
    c.walk_len = walk_len; c.loss = loss; c.clip_mode = clip_mode; c.kernel = kernel; c.n_fixed = n_fixed; c.seed = seed;
    c.node_b = nullptr;
    if (node_b_.size() > 0) {
        require(node_b_.ndim() == 1 && node_b_.shape(0) == n_nodes, "node_b must have n entries");
        for (int64_t i = 0; i < n_nodes; ++i) require(node_b_.data()[i] > 0.05f && node_b_.data()[i] <= 2.0f, "node_b out of range");
        c.node_b = node_b_.data();
    }
    c.parallel = parallel; c.eown = nullptr; c.n_chunks = 0; c.rev_indptr = nullptr; c.rev_indices = nullptr;
    std::vector<int64_t> own, eown;
    std::vector<float> snapA, snapB;
    if (parallel == 1) {
        for (int64_t i = 1; i < n_edges; ++i) require(c.head[i - 1] <= c.head[i], "parallel='owner' needs edges sorted by head");
        if (rev_indptr_.size() > 0) {
            validate_csr(rev_indptr_, rev_indices_, nullptr, n_nodes);
            c.rev_indptr = rev_indptr_.data(); c.rev_indices = rev_indices_.data();
        }
        // node chunks at the edge quantiles (about 4096 edges each), claimed dynamically by the threads. The chunking
        // and the per-chunk RNG streams do not depend on the thread count, so the result does not either.
        c.n_chunks = (int)std::min<int64_t>(4096, std::max<int64_t>(1, n_edges / 4096));
        own.assign(c.n_chunks + 1, 0); eown.assign(c.n_chunks + 1, 0);
        own[c.n_chunks] = n_nodes; eown[c.n_chunks] = n_edges;
        for (int t = 1; t < c.n_chunks; ++t) {
            const int64_t e = n_edges * t / c.n_chunks;
            int64_t node = std::max<int64_t>(c.head[e], own[t - 1]);
            own[t] = node;
            int64_t k = eown[t - 1];
            while (k < n_edges && c.head[k] < node) ++k;
            eown[t] = k;
        }
        c.eown = eown.data();
        snapA.assign(c.Y, c.Y + (size_t)n_nodes * dim);
        snapB.assign(c.Y, c.Y + (size_t)n_nodes * dim);
    }

    py::gil_scoped_release release;

    switch (dim) {
        case 2: run_epochs<2>(c, epoch_start, epoch_end, alpha0, alpha1, mid_w0, mid_w1, own.data(), &snapA, &snapB); break;
        case 3: run_epochs<3>(c, epoch_start, epoch_end, alpha0, alpha1, mid_w0, mid_w1, own.data(), &snapA, &snapB); break;
        default: run_epochs<0>(c, epoch_start, epoch_end, alpha0, alpha1, mid_w0, mid_w1, own.data(), &snapA, &snapB); break;
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
