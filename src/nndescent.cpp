// NN-descent (Dong, Moses & Li, 2011) for dense float32 data, multithreaded without locks:
// every phase that writes into a node's heap is executed by the thread that owns the node
// (node % n_threads). Distances are squared euclidean; cosine is handled upstream by normalising
// the rows. Initialisation by a forest of random projection trees built breadth-first for all
// trees at once (one pass over the data per level), then random fill.
#include "layout.hpp"
#include "rng.hpp"
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <chrono>
#include <cstdio>

namespace {

int resolve(int n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 4 : (int)h;
}

template <class F> void run_threads(int nt, F&& f) {
    if (nt <= 1) { f(0); return; }
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) pool.emplace_back([&, t] { f(t); });
    for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------- distance kernels
// On x86 with GCC/clang: AVX-512 / AVX2 versions selected at runtime from CPUID. Elsewhere (MSVC,
// ARM, ...) the plain loops below are used; they auto-vectorise reasonably with -O3 / -O2.
float sqeuclid_scalar(const float* a, const float* b, int d) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= d; i += 4) {
        const float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1], d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    }
    for (; i < d; ++i) { const float df = a[i] - b[i]; s0 += df * df; }
    return (s0 + s1) + (s2 + s3);
}
float dot_scalar(const float* a, const float* b, int d) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= d; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
    for (; i < d; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}
using Kernel = float (*)(const float*, const float*, int);

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define MAGNET_X86_DISPATCH 1
__attribute__((target("avx512f"))) float sqeuclid_avx512(const float* a, const float* b, int d) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m512 df = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        acc = _mm512_fmadd_ps(df, df, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < d; ++i) { const float df = a[i] - b[i]; s += df * df; }
    return s;
}
__attribute__((target("avx2,fma"))) float sqeuclid_avx2(const float* a, const float* b, int d) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 df = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(df, df, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
    for (; i < d; ++i) { const float df = a[i] - b[i]; s += df * df; }
    return s;
}
__attribute__((target("avx512f"))) float dot_avx512(const float* a, const float* b, int d) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}
__attribute__((target("avx2,fma"))) float dot_avx2(const float* a, const float* b, int d) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}
Kernel pick_sqeuclid() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return sqeuclid_avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return sqeuclid_avx2;
    return sqeuclid_scalar;
}
Kernel pick_dot() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return dot_avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return dot_avx2;
    return dot_scalar;
}
#else
Kernel pick_sqeuclid() { return sqeuclid_scalar; }
Kernel pick_dot() { return dot_scalar; }
#endif

// ---------------------------------------------------------------- data accessors
// A "plane" is the perpendicular bisector of two points a, b: x is on the left when
// x.(a - b) < (|a|^2 - |b|^2) / 2.
struct DenseData {
    const float* X; int64_t n; int d; Kernel sq; Kernel dot;
    struct Plane { std::vector<float> normal; float offset; };
    inline float dist(int32_t i, int32_t j) const { return sq(X + (int64_t)i * d, X + (int64_t)j * d, d); }
    inline float dist_to(const float* q, int32_t j) const { return sq(q, X + (int64_t)j * d, d); }
    // pinned-row API shared with SparseData (a no-op here): pin(a) then dist_pinned(b) == dist(a, b)
    struct Scratch { int32_t pinned = -1; };
    Scratch make_scratch() const { return Scratch{}; }
    inline void pin(Scratch& s, int32_t a) const { s.pinned = a; }
    inline float dist_pinned(const Scratch& s, int32_t b) const { return dist(s.pinned, b); }
    Plane make_plane(int32_t a, int32_t b) const {
        Plane p; p.normal.resize(d);
        const float* xa = X + (int64_t)a * d; const float* xb = X + (int64_t)b * d;
        float na = 0.f, nb = 0.f;
        for (int j = 0; j < d; ++j) { p.normal[j] = xa[j] - xb[j]; na += xa[j] * xa[j]; nb += xb[j] * xb[j]; }
        p.offset = 0.5f * (na - nb);
        return p;
    }
    Plane random_plane(Rng& rng, int32_t anchor) const {
        Plane p; p.normal.resize(d);
        for (int j = 0; j < d; ++j) p.normal[j] = rng.uniform() - 0.5f;
        p.offset = anchor >= 0 ? dot(X + (int64_t)anchor * d, p.normal.data(), d) : 0.0f;
        return p;
    }
    inline float project(int32_t i, const Plane& p) const { return dot(X + (int64_t)i * d, p.normal.data(), d) - p.offset; }
};

// CSR rows with sorted column indices. Squared euclidean through |a|^2 + |b|^2 - 2 a.b (merge join).
struct SparseData {
    const int32_t* indptr; const int32_t* indices; const float* vals; int64_t n; int d;
    std::vector<double> sqnorm;
    struct Plane { std::vector<int32_t> idx; std::vector<float> val; float offset; };
    void init() {
        sqnorm.resize(n);
        for (int64_t i = 0; i < n; ++i) { double s = 0.; for (int32_t j = indptr[i]; j < indptr[i + 1]; ++j) s += (double)vals[j] * vals[j]; sqnorm[i] = s; }
    }
    // double accumulation: |a|^2 + |b|^2 - 2 a.b cancels badly in float when a and b are close
    static inline double merge_dot(const int32_t* ia, const float* va, int32_t na, const int32_t* ib, const float* vb, int32_t nb) {
        double s = 0.; int32_t x = 0, y = 0;
        while (x < na && y < nb) {
            if (ia[x] == ib[y]) { s += (double)va[x] * vb[y]; ++x; ++y; }
            else if (ia[x] < ib[y]) ++x; else ++y;
        }
        return s;
    }
    inline double row_dot(int32_t i, int32_t j) const {
        return merge_dot(indices + indptr[i], vals + indptr[i], indptr[i + 1] - indptr[i],
                         indices + indptr[j], vals + indptr[j], indptr[j + 1] - indptr[j]);
    }
    inline float dist(int32_t i, int32_t j) const { return (float)std::max(0., sqnorm[i] + sqnorm[j] - 2. * row_dot(i, j)); }
    // Pinned-row distances: the row that is reused against many partners (the pivot of a clique or of the local
    // join) is scattered once into a dense per-thread buffer, and every partner's dot product is a gather over its
    // own non-zeros: no merge, no branch (4x the merge on 20 newsgroups TF-IDF). The buffer is d floats per thread;
    // above 2^22 columns the merge join is used instead.
    struct Scratch { std::vector<float> buf; int32_t pinned = -1; double pin_sq = 0.; };
    Scratch make_scratch() const { Scratch s; if (d <= (1 << 22)) s.buf.assign((size_t)d, 0.f); return s; }
    inline void pin(Scratch& s, int32_t a) const {
        if (s.pinned == a) return;
        if (!s.buf.empty()) {
            if (s.pinned >= 0) for (int32_t j = indptr[s.pinned]; j < indptr[s.pinned + 1]; ++j) s.buf[indices[j]] = 0.f;
            for (int32_t j = indptr[a]; j < indptr[a + 1]; ++j) s.buf[indices[j]] = vals[j];
        }
        s.pinned = a; s.pin_sq = sqnorm[a];
    }
    inline float dist_pinned(const Scratch& s, int32_t b) const {
        if (s.buf.empty()) return dist(s.pinned, b);
        double dot = 0.;
        const float* buf = s.buf.data();
        for (int32_t j = indptr[b]; j < indptr[b + 1]; ++j) dot += (double)vals[j] * buf[indices[j]];
        return (float)std::max(0., s.pin_sq + sqnorm[b] - 2. * dot);
    }
    // query given as a sparse row (sorted) with its squared norm
    inline float dist_to(const int32_t* qi, const float* qv, int32_t qn, double qsq, int32_t j) const {
        return (float)std::max(0., qsq + sqnorm[j] - 2. * merge_dot(qi, qv, qn, indices + indptr[j], vals + indptr[j], indptr[j + 1] - indptr[j]));
    }
    Plane make_plane(int32_t a, int32_t b) const {
        Plane p;
        const int32_t *ia = indices + indptr[a], *ib = indices + indptr[b];
        const float *va = vals + indptr[a], *vb = vals + indptr[b];
        int32_t na = indptr[a + 1] - indptr[a], nb = indptr[b + 1] - indptr[b], x = 0, y = 0;
        p.idx.reserve(na + nb); p.val.reserve(na + nb);
        while (x < na || y < nb) {
            if (y >= nb || (x < na && ia[x] < ib[y])) { p.idx.push_back(ia[x]); p.val.push_back(va[x]); ++x; }
            else if (x >= na || ib[y] < ia[x]) { p.idx.push_back(ib[y]); p.val.push_back(-vb[y]); ++y; }
            else { p.idx.push_back(ia[x]); p.val.push_back(va[x] - vb[y]); ++x; ++y; }
        }
        p.offset = (float)(0.5 * (sqnorm[a] - sqnorm[b]));
        return p;
    }
    Plane random_plane(Rng& rng, int32_t anchor) const {
        // random sparse direction supported on the anchor's coordinates (or a few random ones)
        Plane p;
        if (anchor >= 0 && indptr[anchor + 1] > indptr[anchor]) {
            for (int32_t j = indptr[anchor]; j < indptr[anchor + 1]; ++j) { p.idx.push_back(indices[j]); p.val.push_back(rng.uniform() - 0.5f); }
        } else {
            for (int t = 0; t < 8; ++t) p.idx.push_back((int32_t)rng.below((uint32_t)d));
            std::sort(p.idx.begin(), p.idx.end()); p.idx.erase(std::unique(p.idx.begin(), p.idx.end()), p.idx.end());
            for (size_t t = 0; t < p.idx.size(); ++t) p.val.push_back(rng.uniform() - 0.5f);
        }
        p.offset = anchor >= 0 ? (float)merge_dot(p.idx.data(), p.val.data(), (int32_t)p.idx.size(), indices + indptr[anchor], vals + indptr[anchor], indptr[anchor + 1] - indptr[anchor]) : 0.f;
        return p;
    }
    inline float project(int32_t i, const Plane& p) const {
        return (float)merge_dot(indices + indptr[i], vals + indptr[i], indptr[i + 1] - indptr[i], p.idx.data(), p.val.data(), (int32_t)p.idx.size()) - p.offset;
    }
};

// ---------------------------------------------------------------- random projection forest
// All trees are grown together, level by level: each level is one pass over the rows of X, during
// which every row is projected on the hyperplane of its current node in every tree.
struct Forest {
    int n_trees; int64_t n;
    std::vector<int32_t> leaf_of;                 // [tree * n + i] -> leaf id (per tree)
    std::vector<std::vector<int32_t>> leaf_ptr;   // per tree: CSR offsets of the leaves
    std::vector<std::vector<int32_t>> leaf_members;
};

template <class Data>
Forest build_forest(const Data& D, int n_trees, int leaf_size, uint64_t seed, int nt) {
    using Plane = typename Data::Plane;
    const int64_t n = D.n;
    Forest F;
    F.n_trees = n_trees; F.n = n;
    const int max_depth = (int)std::ceil(std::log2(std::max<double>(2.0, (double)n / leaf_size))) + 8;
    std::vector<int32_t> node(n_trees * (size_t)n, 0);   // current node of each point in each tree
    std::vector<int32_t> n_nodes(n_trees, 1);
    const int ntt = std::min(nt, n_trees);
    for (int depth = 0; depth < max_depth; ++depth) {
        // 1. sizes and two random members per node (reservoir sampling), parallel over trees
        std::vector<std::vector<int64_t>> size(n_trees);
        std::vector<std::vector<int32_t>> s1(n_trees), s2(n_trees);
        run_threads(ntt, [&](int t) {
            for (int tr = t; tr < n_trees; tr += ntt) {
                Rng rng(seed + 977ULL * (uint64_t)(tr + 1) + 7919ULL * (uint64_t)(depth + 1));
                const int m = n_nodes[tr];
                size[tr].assign(m, 0); s1[tr].assign(m, -1); s2[tr].assign(m, -1);
                const int32_t* nd = node.data() + (size_t)tr * n;
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t v = nd[i];
                    const int64_t c = ++size[tr][v];
                    if (rng.below((uint32_t)std::min<int64_t>(c, 0x7fffffff)) == 0) s1[tr][v] = (int32_t)i;
                    if (rng.below((uint32_t)std::min<int64_t>(c, 0x7fffffff)) == 0) s2[tr][v] = (int32_t)i;
                }
            }
        });
        // 2. frontier: nodes larger than leaf_size get a hyperplane and two children
        std::vector<std::vector<int32_t>> child(n_trees);   // per node: id of the left child or -1
        std::vector<std::vector<int32_t>> plane(n_trees);   // per node: index into planes or -1
        std::vector<std::vector<Plane>> planes(n_trees);
        int64_t frontier_total = 0;
        for (int tr = 0; tr < n_trees; ++tr) {
            const int m = n_nodes[tr];
            child[tr].assign(m, -1); plane[tr].assign(m, -1);
            int next = m, np = 0;
            for (int v = 0; v < m; ++v)
                if (size[tr][v] > leaf_size && depth < max_depth - 1) { child[tr][v] = next; next += 2; plane[tr][v] = np++; }
            planes[tr].resize(np);
            frontier_total += np;
            n_nodes[tr] = next;
        }
        if (frontier_total == 0) break;
        run_threads(ntt, [&](int t) {
            for (int tr = t; tr < n_trees; tr += ntt) {
                Rng rng(seed + 13ULL * (uint64_t)(tr + 1) + 101ULL * (uint64_t)(depth + 1));
                for (int v = 0; v < (int)child[tr].size(); ++v) {
                    const int pidx = plane[tr][v];
                    if (pidx < 0) continue;
                    const int32_t a = s1[tr][v], b = s2[tr][v];
                    if (a == b || a < 0 || b < 0) planes[tr][pidx] = D.random_plane(rng, a);  // duplicates / tiny node
                    else planes[tr][pidx] = D.make_plane(a, b);
                }
            }
        });
        // 3. one pass over the rows: route every point to a child in every tree
        run_threads(nt, [&](int t) {
            const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
            Rng rng(seed + 31ULL * (uint64_t)(t + 1) + 17ULL * (uint64_t)(depth + 1));
            for (int64_t i = r0; i < r1; ++i) {
                for (int tr = 0; tr < n_trees; ++tr) {
                    int32_t& v = node[(size_t)tr * n + i];
                    const int c = child[tr][v];
                    if (c < 0) continue;
                    const float pr = D.project((int32_t)i, planes[tr][plane[tr][v]]);
                    const bool left = pr < 0.f || (pr == 0.f && rng.below(2));
                    v = left ? c : c + 1;
                }
            }
        });
    }
    // leaves: compress node ids per tree into CSR member lists
    F.leaf_of.resize(n_trees * (size_t)n);
    F.leaf_ptr.resize(n_trees); F.leaf_members.resize(n_trees);
    run_threads(ntt, [&](int t) {
        for (int tr = t; tr < n_trees; tr += ntt) {
            const int m = n_nodes[tr];
            std::vector<int32_t> remap(m, -1), count;
            int nl = 0;
            const int32_t* nd = node.data() + (size_t)tr * n;
            for (int64_t i = 0; i < n; ++i) if (remap[nd[i]] < 0) remap[nd[i]] = nl++;
            count.assign(nl + 1, 0);
            for (int64_t i = 0; i < n; ++i) ++count[remap[nd[i]] + 1];
            for (int l = 0; l < nl; ++l) count[l + 1] += count[l];
            F.leaf_ptr[tr] = count;
            F.leaf_members[tr].resize(n);
            std::vector<int32_t> fill(count.begin(), count.end() - 1);
            for (int64_t i = 0; i < n; ++i) { const int l = remap[nd[i]]; F.leaf_of[(size_t)tr * n + i] = l; F.leaf_members[tr][fill[l]++] = (int32_t)i; }
        }
    });
    return F;
}

// ---------------------------------------------------------------- heaps
// Max-heap of size k per node on (dist, idx, flag). Root = worst neighbour.
struct Heaps {
    int64_t n; int k;
    std::vector<int32_t> idx; std::vector<float> dist; std::vector<uint8_t> flag;
    Heaps(int64_t n_, int k_) : n(n_), k(k_), idx((size_t)n_ * k_, -1), dist((size_t)n_ * k_, INFINITY), flag((size_t)n_ * k_, 0) {}
    inline int push(int64_t row, float d, int32_t j, uint8_t fl) {
        int32_t* I = idx.data() + row * k; float* D = dist.data() + row * k; uint8_t* F = flag.data() + row * k;
        if (d >= D[0]) return 0;
        for (int c = 0; c < k; ++c) if (I[c] == j) return 0;
        D[0] = d; I[0] = j; F[0] = fl;
        int i = 0;
        while (true) {
            const int l = 2 * i + 1, r = l + 1;
            int swap = i;
            if (l < k && D[l] > D[swap]) swap = l;
            if (r < k && D[r] > D[swap]) swap = r;
            if (swap == i) break;
            std::swap(D[i], D[swap]); std::swap(I[i], I[swap]); std::swap(F[i], F[swap]);
            i = swap;
        }
        return 1;
    }
    inline float worst(int64_t row) const { return dist[row * k]; }
};

// candidate lists sampled with random priorities (max-heap on priority -> uniform sample)
struct Candidates {
    int64_t n; int m;
    std::vector<int32_t> idx; std::vector<float> prio;
    Candidates(int64_t n_, int m_) : n(n_), m(m_), idx((size_t)n_ * m_, -1), prio((size_t)n_ * m_, INFINITY) {}
    inline void push(int64_t row, float p, int32_t j) {
        int32_t* I = idx.data() + row * m; float* P = prio.data() + row * m;
        if (p >= P[0]) return;
        for (int c = 0; c < m; ++c) if (I[c] == j) return;
        P[0] = p; I[0] = j;
        int i = 0;
        while (true) {
            const int l = 2 * i + 1, r = l + 1;
            int swap = i;
            if (l < m && P[l] > P[swap]) swap = l;
            if (r < m && P[r] > P[swap]) swap = r;
            if (swap == i) break;
            std::swap(P[i], P[swap]); std::swap(I[i], I[swap]);
            i = swap;
        }
    }
    inline bool contains(int64_t row, int32_t j) const {
        const int32_t* I = idx.data() + row * m;
        for (int c = 0; c < m; ++c) if (I[c] == j) return true;
        return false;
    }
};

struct Update { int32_t p, q; float d; };
struct CandRec { int32_t target, src; float p; };

template <class Data>
std::tuple<arr<int32_t>, arr<float>> nn_descent_core(
    const Data& D, int k, int max_candidates, int n_iters, float delta, int n_trees, int leaf_size,
    uint64_t seed, int n_threads, bool verbose)
{
    const int64_t n = D.n;
    require(k >= 1 && k < n, "k must be between 1 and n - 1");
    require(max_candidates >= 1 && n_iters >= 0 && leaf_size >= 2 && n_trees >= 0, "invalid NN-descent parameters");
    require(n < (int64_t)1 << 31, "at most 2^31 - 1 points");
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 1024));
    const int m = std::min(max_candidates, k);

    arr<int32_t> out_idx({n, (int64_t)k});
    arr<float> out_dist({n, (int64_t)k});
    int32_t* OI = out_idx.mutable_data();
    float* OD = out_dist.mutable_data();

    py::gil_scoped_release release;

    Heaps H(n, k);
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t0 = now();
    auto lap = [&](const char* what) { if (verbose) { fprintf(stderr, "  %-14s %.3fs\n", what, now() - t0); t0 = now(); } };
    auto dist = [&](int32_t a, int32_t b) { return D.dist(a, b); };

    // ---- initialisation 1: random projection forest. Leaves are processed in parallel (their rows
    // stay in cache for the whole clique) and the resulting pairs are applied by the owner threads.
    std::vector<std::vector<std::vector<Update>>> buckets(nt, std::vector<std::vector<Update>>(nt));
    if (n_trees > 0) {
        Forest F = build_forest(D, n_trees, leaf_size, seed, nt);
        lap("rp-forest");
        for (int tr = 0; tr < n_trees; ++tr) {
            const int n_leaves = (int)F.leaf_ptr[tr].size() - 1;
            const int32_t* mem = F.leaf_members[tr].data();
            const int32_t* ptr = F.leaf_ptr[tr].data();
            run_threads(nt, [&](int t) {
                for (auto& b : buckets[t]) b.clear();
                Rng rng(seed + 4099ULL * (uint64_t)(t + 1) + 7ULL * tr);
                auto S = D.make_scratch();
                for (int l = t; l < n_leaves; l += nt) {
                    const int32_t b = ptr[l], e = ptr[l + 1];
                    if (e - b <= 4 * leaf_size) {
                        for (int32_t a = b; a < e; ++a) {
                            const int32_t p = mem[a];
                            const float wp = H.worst(p);
                            D.pin(S, p);
                            for (int32_t c = a + 1; c < e; ++c) {
                                const int32_t q = mem[c];
                                const float dd = D.dist_pinned(S, q);
                                if (dd < wp) buckets[t][p % nt].push_back({p, q, dd});
                                if (dd < H.worst(q)) buckets[t][q % nt].push_back({q, p, dd});
                            }
                        }
                    } else {  // oversized leaf (duplicates): sample instead of the full clique
                        for (int32_t a = b; a < e; ++a) {
                            const int32_t p = mem[a];
                            D.pin(S, p);
                            for (int s2 = 0; s2 < leaf_size; ++s2) {
                                const int32_t q = mem[b + rng.below((uint32_t)(e - b))];
                                if (q == p) continue;
                                const float dd = D.dist_pinned(S, q);
                                if (dd < H.worst(p)) buckets[t][p % nt].push_back({p, q, dd});
                            }
                        }
                    }
                }
            });
            run_threads(nt, [&](int t) {
                for (int s2 = 0; s2 < nt; ++s2)
                    for (const Update& u : buckets[s2][t]) H.push(u.p, u.d, u.q, 1);
            });
        }
        lap("leaf-cliques");
    }
    // ---- initialisation 2: random neighbours to fill the remaining slots
    run_threads(nt, [&](int t) {
        Rng rng(seed + 1000003ULL * (uint64_t)t);
        auto S = D.make_scratch();
        for (int64_t i = t; i < n; i += nt) {
            int filled = 0, tries = 0;
            for (int c = 0; c < k; ++c) filled += H.idx[i * k + c] >= 0;
            if (filled < k) D.pin(S, (int32_t)i);
            while (filled < k && tries < 10 * k) {
                const int32_t j = (int32_t)rng.below((uint32_t)n);
                ++tries;
                if (j == i) continue;
                filled += H.push(i, D.dist_pinned(S, j), j, 1);
            }
        }
    });
    // ---- safety net: a row left incomplete (tiny n, many duplicates) is completed by brute force
    run_threads(nt, [&](int t) {
        auto S = D.make_scratch();
        for (int64_t i = t; i < n; i += nt) {
            bool full = true;
            for (int c = 0; c < k; ++c) full &= H.idx[i * k + c] >= 0;
            if (full) continue;
            D.pin(S, (int32_t)i);
            for (int64_t j = 0; j < n; ++j) if (j != i) H.push(i, D.dist_pinned(S, (int32_t)j), (int32_t)j, 1);
        }
    });
    lap("random-init");

    std::vector<int64_t> counts(nt, 0);
    std::vector<std::vector<std::vector<CandRec>>> rev_new(nt, std::vector<std::vector<CandRec>>(nt)),
                                                    rev_old(nt, std::vector<std::vector<CandRec>>(nt));
    for (int it = 0; it < n_iters; ++it) {
        // ---- 1. candidate sampling: new (flagged) and old neighbours, forward + reverse. Each thread
        // scans its own rows; reverse pushes (i into the list of j) are buffered per owner of j.
        Candidates newc(n, m), oldc(n, m);
        run_threads(nt, [&](int t) {
            for (auto& b : rev_new[t]) b.clear();
            for (auto& b : rev_old[t]) b.clear();
            Rng rng(seed + 7919ULL * (uint64_t)(it + 1) + 104729ULL * (uint64_t)t);
            for (int64_t i = t; i < n; i += nt) {
                for (int c = 0; c < k; ++c) {
                    const int32_t j = H.idx[i * k + c];
                    if (j < 0) continue;
                    const float p = rng.uniform();
                    if (H.flag[i * k + c]) { newc.push(i, p, j); rev_new[t][j % nt].push_back({j, (int32_t)i, p}); }
                    else { oldc.push(i, p, j); rev_old[t][j % nt].push_back({j, (int32_t)i, p}); }
                }
            }
        });
        run_threads(nt, [&](int t) {
            for (int s2 = 0; s2 < nt; ++s2) {
                for (const CandRec& r : rev_new[s2][t]) newc.push(r.target, r.p, r.src);
                for (const CandRec& r : rev_old[s2][t]) oldc.push(r.target, r.p, r.src);
            }
        });
        // ---- 2. un-flag the neighbours that were sampled as new candidates
        run_threads(nt, [&](int t) {
            for (int64_t i = t; i < n; i += nt)
                for (int c = 0; c < k; ++c)
                    if (H.flag[i * k + c] && newc.contains(i, H.idx[i * k + c])) H.flag[i * k + c] = 0;
        });
        lap("candidates");
        // ---- 3. local join: distances between candidates, buffered per owner thread
        run_threads(nt, [&](int t) {
            for (auto& b : buckets[t]) b.clear();
            std::vector<int32_t> nw, od;
            auto S = D.make_scratch();
            for (int64_t i = t; i < n; i += nt) {
                nw.clear(); od.clear();
                for (int c = 0; c < m; ++c) { if (newc.idx[i * m + c] >= 0) nw.push_back(newc.idx[i * m + c]); }
                for (int c = 0; c < m; ++c) { if (oldc.idx[i * m + c] >= 0) od.push_back(oldc.idx[i * m + c]); }
                for (size_t a = 0; a < nw.size(); ++a) {
                    const int32_t p = nw[a];
                    const float wp = H.worst(p);
                    if (a + 1 < nw.size() || !od.empty()) D.pin(S, p);
                    for (size_t b = a + 1; b < nw.size(); ++b) {
                        const int32_t q = nw[b];
                        if (p == q) continue;
                        const float wq = H.worst(q);
                        const float dd = D.dist_pinned(S, q);
                        if (dd < wp) buckets[t][p % nt].push_back({p, q, dd});
                        if (dd < wq) buckets[t][q % nt].push_back({q, p, dd});
                    }
                    for (size_t b = 0; b < od.size(); ++b) {
                        const int32_t q = od[b];
                        if (p == q) continue;
                        const float wq = H.worst(q);
                        const float dd = D.dist_pinned(S, q);
                        if (dd < wp) buckets[t][p % nt].push_back({p, q, dd});
                        if (dd < wq) buckets[t][q % nt].push_back({q, p, dd});
                    }
                }
            }
        });
        lap("local-join");
        // ---- 4. apply updates: thread t only touches the heaps of the nodes it owns
        run_threads(nt, [&](int t) {
            int64_t c = 0;
            for (int s2 = 0; s2 < nt; ++s2)
                for (const Update& u : buckets[s2][t]) c += H.push(u.p, u.d, u.q, 1);
            counts[t] = c;
        });
        lap("apply");
        const int64_t total = std::accumulate(counts.begin(), counts.end(), (int64_t)0);
        if (verbose) fprintf(stderr, "nn-descent iteration %d: %lld updates\n", it + 1, (long long)total);
        if (total <= (int64_t)(delta * (double)n * k)) break;
    }

    // ---- sort each heap by increasing distance
    run_threads(nt, [&](int t) {
        std::vector<int> order(k);
        for (int64_t i = t; i < n; i += nt) {
            const int32_t* I = H.idx.data() + i * k; const float* Dd = H.dist.data() + i * k;
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return Dd[a] < Dd[b]; });
            for (int c = 0; c < k; ++c) { OI[i * k + c] = I[order[c]]; OD[i * k + c] = Dd[order[c]]; }
        }
    });
    return std::make_tuple(out_idx, out_dist);
}

// ---------------------------------------------------------------- graph search (queries)
// Greedy beam search over a neighbour graph (CSR, symmetric) from random entry points.
template <class Data, class DistTo>
void search_one(const Data& D, const int32_t* gptr, const int32_t* gidx, int64_t n, DistTo&& dist_to,
                const int32_t* landmarks, int64_t n_landmarks,
                int k, int ef, int n_entry, Rng& rng, std::vector<uint32_t>& stamp, uint32_t& cur,
                int32_t* out_i, float* out_d)
{
    ++cur;
    if (cur == 0) { std::fill(stamp.begin(), stamp.end(), 0); cur = 1; }
    // results: max-heap on distance (size <= ef); candidates: min-heap on distance
    std::vector<std::pair<float, int32_t>> res, cand;
    auto res_cmp = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) { return a.first < b.first; };
    auto cand_cmp = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) { return a.first > b.first; };
    auto consider = [&](int32_t j) {
        if (stamp[j] == cur) return;
        stamp[j] = cur;
        const float dd = dist_to(j);
        if ((int)res.size() < ef || dd < res.front().first) {
            res.emplace_back(dd, j); std::push_heap(res.begin(), res.end(), res_cmp);
            if ((int)res.size() > ef) { std::pop_heap(res.begin(), res.end(), res_cmp); res.pop_back(); }
            cand.emplace_back(dd, j); std::push_heap(cand.begin(), cand.end(), cand_cmp);
        }
    };
    if (n_landmarks > 0) {
        // entry points: the n_entry closest landmarks (handles disconnected neighbour graphs)
        std::vector<std::pair<float, int32_t>> best;
        best.reserve(n_landmarks);
        for (int64_t l = 0; l < n_landmarks; ++l) best.emplace_back(dist_to(landmarks[l]), landmarks[l]);
        const int take = (int)std::min<int64_t>(n_entry, n_landmarks);
        std::partial_sort(best.begin(), best.begin() + take, best.end());
        for (int e = 0; e < take; ++e) consider(best[e].second);
    } else {
        for (int e = 0; e < n_entry; ++e) consider((int32_t)rng.below((uint32_t)n));
    }
    while (!cand.empty()) {
        std::pop_heap(cand.begin(), cand.end(), cand_cmp);
        const auto c = cand.back(); cand.pop_back();
        if ((int)res.size() >= ef && c.first > res.front().first) break;
        for (int32_t j = gptr[c.second]; j < gptr[c.second + 1]; ++j) consider(gidx[j]);
    }
    std::sort(res.begin(), res.end());
    for (int c = 0; c < k; ++c) {
        if (c < (int)res.size()) { out_i[c] = res[c].second; out_d[c] = res[c].first; }
        else { out_i[c] = -1; out_d[c] = INFINITY; }
    }
}

}  // namespace


std::tuple<arr<int32_t>, arr<float>> nn_descent(
    arr<float> X_, int k, int max_candidates, int n_iters, float delta, int n_trees, int leaf_size,
    uint64_t seed, int n_threads, bool verbose)
{
    auto Xb = X_.request();
    require(Xb.ndim == 2, "X must be 2-d");
    require(Xb.shape[1] >= 1, "X must have at least one column");
    DenseData D{static_cast<const float*>(Xb.ptr), Xb.shape[0], (int)Xb.shape[1], pick_sqeuclid(), pick_dot()};
    return nn_descent_core(D, k, max_candidates, n_iters, delta, n_trees, leaf_size, seed, n_threads, verbose);
}

std::tuple<arr<int32_t>, arr<float>> nn_descent_sparse(
    arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_, int d,
    int k, int max_candidates, int n_iters, float delta, int n_trees, int leaf_size,
    uint64_t seed, int n_threads, bool verbose)
{
    require(indptr_.ndim() == 1 && indptr_.shape(0) >= 2, "indptr must have n + 1 entries");
    const int64_t n = indptr_.shape(0) - 1;
    validate_csr(indptr_, indices_, &data_, n, d);
    SparseData D{indptr_.data(), indices_.data(), data_.data(), n, d, {}};
    D.init();
    return nn_descent_core(D, k, max_candidates, n_iters, delta, n_trees, leaf_size, seed, n_threads, verbose);
}

std::tuple<arr<int32_t>, arr<float>> knn_search(
    arr<float> X_, arr<int32_t> gptr_, arr<int32_t> gidx_, arr<int32_t> landmarks_, arr<float> Q_,
    int k, int ef, int n_entry, uint64_t seed, int n_threads)
{
    auto Xb = X_.request(); auto Qb = Q_.request();
    require(Xb.ndim == 2 && Qb.ndim == 2 && Xb.shape[1] == Qb.shape[1], "X and Q must be 2-d with the same width");
    const int64_t n = Xb.shape[0], m = Qb.shape[0];
    const int d = (int)Xb.shape[1];
    require(k >= 1 && k <= n && ef >= k, "need 1 <= k <= n and ef >= k");
    validate_csr(gptr_, gidx_, nullptr, n);
    const int64_t n_land = landmarks_.shape(0);
    for (int64_t l = 0; l < n_land; ++l) require(landmarks_.data()[l] >= 0 && landmarks_.data()[l] < n, "landmark out of range");
    const int32_t* land = landmarks_.data();
    DenseData D{static_cast<const float*>(Xb.ptr), n, d, pick_sqeuclid(), pick_dot()};
    const float* Q = static_cast<const float*>(Qb.ptr);
    arr<int32_t> out_i({m, (int64_t)k}); arr<float> out_d({m, (int64_t)k});
    int32_t* OI = out_i.mutable_data(); float* OD = out_d.mutable_data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, m / 64));
    const int32_t* gptr = gptr_.data(); const int32_t* gidx = gidx_.data();
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        Rng rng(seed + 7ULL * (uint64_t)(t + 1));
        std::vector<uint32_t> stamp(n, 0); uint32_t cur = 0;
        for (int64_t q = t; q < m; q += nt) {
            const float* qv = Q + q * d;
            search_one(D, gptr, gidx, n, [&](int32_t j) { return D.dist_to(qv, j); }, land, n_land, k, ef, n_entry,
                       rng, stamp, cur, OI + q * k, OD + q * k);
        }
    });
    return std::make_tuple(out_i, out_d);
}

std::tuple<arr<int32_t>, arr<float>> knn_search_sparse(
    arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_, int d,
    arr<int32_t> gptr_, arr<int32_t> gidx_, arr<int32_t> landmarks_,
    arr<int32_t> qptr_, arr<int32_t> qidx_, arr<float> qdata_,
    int k, int ef, int n_entry, uint64_t seed, int n_threads)
{
    const int64_t n = indptr_.shape(0) - 1, m = qptr_.shape(0) - 1;
    require(n >= 1 && m >= 0, "empty input");
    validate_csr(indptr_, indices_, &data_, n, d);
    validate_csr(qptr_, qidx_, &qdata_, m, d);
    require(k >= 1 && k <= n && ef >= k, "need 1 <= k <= n and ef >= k");
    validate_csr(gptr_, gidx_, nullptr, n);
    const int64_t n_land = landmarks_.shape(0);
    for (int64_t l = 0; l < n_land; ++l) require(landmarks_.data()[l] >= 0 && landmarks_.data()[l] < n, "landmark out of range");
    const int32_t* land = landmarks_.data();
    SparseData D{indptr_.data(), indices_.data(), data_.data(), n, d, {}};
    D.init();
    const int32_t* qp = qptr_.data(); const int32_t* qi = qidx_.data(); const float* qv = qdata_.data();
    arr<int32_t> out_i({m, (int64_t)k}); arr<float> out_d({m, (int64_t)k});
    int32_t* OI = out_i.mutable_data(); float* OD = out_d.mutable_data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, m / 64));
    const int32_t* gptr = gptr_.data(); const int32_t* gidx = gidx_.data();
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        Rng rng(seed + 7ULL * (uint64_t)(t + 1));
        std::vector<uint32_t> stamp(n, 0); uint32_t cur = 0;
        for (int64_t q = t; q < m; q += nt) {
            const int32_t nq = qp[q + 1] - qp[q];
            double qsq = 0.;
            for (int32_t j = qp[q]; j < qp[q + 1]; ++j) qsq += (double)qv[j] * qv[j];
            search_one(D, gptr, gidx, n, [&](int32_t j) { return D.dist_to(qi + qp[q], qv + qp[q], nq, qsq, j); },
                       land, n_land, k, ef, n_entry, rng, stamp, cur, OI + q * k, OD + q * k);
        }
    });
    return std::make_tuple(out_i, out_d);
}


// PaCMAP-style mid-near partners: for every row, the second closest of n_cand random rows, n_pairs times.
template <class Data>
arr<int32_t> mid_near_core(const Data& D, int n_pairs, int n_cand, uint64_t seed, int n_threads)
{
    const int64_t n = D.n;
    require(n_pairs >= 1 && n_cand >= 2 && n >= n_cand + 1, "need n_pairs >= 1, n_cand >= 2 and n > n_cand");
    arr<int32_t> out({n, (int64_t)n_pairs});
    int32_t* O = out.mutable_data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 2048));
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        Rng rng(seed + 48271ULL * (uint64_t)(t + 1));
        std::vector<std::pair<float, int32_t>> cand(n_cand);
        auto S = D.make_scratch();
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        for (int64_t i = r0; i < r1; ++i) {
            D.pin(S, (int32_t)i);
            for (int p = 0; p < n_pairs; ++p) {
                for (int c = 0; c < n_cand; ++c) {
                    int32_t j = (int32_t)rng.below((uint32_t)n);
                    if (j == i) j = (int32_t)((j + 1) % n);
                    cand[c] = {D.dist_pinned(S, j), j};
                }
                std::partial_sort(cand.begin(), cand.begin() + 2, cand.end());
                O[i * n_pairs + p] = cand[1].second;
            }
        }
    });
    return out;
}

arr<int32_t> mid_near_pairs_dense(arr<float> X_, int n_pairs, int n_cand, uint64_t seed, int n_threads)
{
    auto Xb = X_.request();
    require(Xb.ndim == 2 && Xb.shape[1] >= 1, "X must be 2-d");
    DenseData D{static_cast<const float*>(Xb.ptr), Xb.shape[0], (int)Xb.shape[1], pick_sqeuclid(), pick_dot()};
    return mid_near_core(D, n_pairs, n_cand, seed, n_threads);
}

arr<int32_t> mid_near_pairs_sparse(arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_, int d,
                                   int n_pairs, int n_cand, uint64_t seed, int n_threads)
{
    const int64_t n = indptr_.shape(0) - 1;
    validate_csr(indptr_, indices_, &data_, n, d);
    SparseData D{indptr_.data(), indices_.data(), data_.data(), n, d, {}};
    D.init();
    return mid_near_core(D, n_pairs, n_cand, seed, n_threads);
}
