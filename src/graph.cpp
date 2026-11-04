// Graph-side kernels:
//  * smooth_knn : UMAP's per-row bandwidth search (binary search of sigma_i so that sum_j exp(-(d_ij - rho_i) / sigma_i)
//                 = log2(k)), one row per thread slot; replaces the vectorised numpy version (64 passes over the
//                 whole (n, k) array)
//  * louvain    : Louvain community detection for the generalised modularity Q = sum_c [A_cc / 2m - gamma (K_c / 2m)^2]
//                 on a symmetric weighted CSR graph (local moving in random node order, aggregation, repeat), used
//                 by the placement / separation step instead of python-igraph's Leiden (1.9 s on MNIST 70k with the
//                 edge-list conversion; this runs in ~0.2 s)
#include "layout.hpp"
#include "pool.hpp"
#include "rng.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
int resolve_threads(int n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 4 : (int)h;
}
}

// dist (n, k): sorted distances to the neighbours (self excluded). Returns memberships (n, k) in (0, 1].
arr<float> smooth_knn(arr<float> dist_, int n_neighbors, float bandwidth, int n_iter, int n_threads)
{
    require(dist_.ndim() == 2, "dist must be (n, k)");
    const int64_t n = dist_.shape(0); const int k = (int)dist_.shape(1);
    const float* D = dist_.data();
    arr<float> out({n, (int64_t)k});
    float* O = out.mutable_data();
    const float target = std::log2((float)n_neighbors) * bandwidth;
    const int nt = (int)std::min<int64_t>(resolve_threads(n_threads), std::max<int64_t>(1, n / 2048));
    // mean distance over the whole array (for the floor of sigma on rows with rho = 0)
    std::vector<double> part(nt, 0.0);
    std::vector<float> sigma(n), rho(n);
    py::gil_scoped_release release;
    pool_run(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        double s = 0.0;
        for (int64_t i = r0; i < r1; ++i) {
            const float* d = D + i * k;
            double rowsum = 0.0; float r = 0.f; bool has = false;
            for (int j = 0; j < k; ++j) { rowsum += d[j]; if (!has && d[j] > 0.f) { r = d[j]; has = true; } }
            s += rowsum;
            rho[i] = r;
            float lo = 0.f, hi = INFINITY, mid = 1.f, sg = 1.f; bool done = false;
            for (int it = 0; it < n_iter; ++it) {
                float psum = 0.f;
                for (int j = 0; j < k; ++j) { const float dmr = d[j] - r; psum += dmr > 0.f ? std::exp(-dmr / mid) : 1.f; }
                if (std::fabs(psum - target) < 1e-5f) { sg = mid; done = true; break; }
                if (psum > target) { hi = mid; mid = (lo + hi) * 0.5f; }
                else { lo = mid; mid = std::isinf(hi) ? mid * 2.f : (lo + hi) * 0.5f; }
            }
            if (!done) sg = mid;
            const float mean_row = (float)(rowsum / k);
            if (r > 0.f) sg = std::max(sg, 1e-3f * mean_row);
            sigma[i] = sg;
        }
        part[t] = s;
    });
    double tot = 0.0; for (double v : part) tot += v;
    const float mean_all = (float)(tot / (double)(n * k));
    pool_run(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        for (int64_t i = r0; i < r1; ++i) {
            const float* d = D + i * k; float* o = O + i * k;
            float sg = sigma[i];
            if (!(rho[i] > 0.f)) sg = std::max(sg, 1e-3f * mean_all);
            for (int j = 0; j < k; ++j) { const float dmr = d[j] - rho[i]; o[j] = dmr > 0.f ? std::exp(-dmr / sg) : 1.f; }
        }
    });
    return out;
}


// Louvain on a symmetric weighted CSR graph; returns community labels (n,), contiguous from 0, sorted by size (0 = largest).
arr<int32_t> louvain(arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_, float resolution, uint64_t seed, int max_levels, int max_sweeps, int order_mode)
{
    const int64_t n0 = indptr_.shape(0) - 1;
    validate_csr(indptr_, indices_, &data_, n0);
    // working graph (copied; aggregated in place level after level)
    std::vector<int64_t> ip(indptr_.data(), indptr_.data() + n0 + 1);
    std::vector<int32_t> ix(indices_.data(), indices_.data() + ip[n0]);
    std::vector<double> w(ip[n0]);
    for (int64_t e = 0; e < ip[n0]; ++e) w[e] = data_.data()[e];
    std::vector<double> self(n0, 0.0);   // self-loop weight of a node (internal weight after aggregation)
    std::vector<int32_t> membership(n0); std::iota(membership.begin(), membership.end(), 0);   // original node -> current community
    double two_m = 0.0; for (double v : w) two_m += v;
    if (two_m <= 0) { arr<int32_t> out(std::vector<py::ssize_t>{(py::ssize_t)n0}); std::fill(out.mutable_data(), out.mutable_data() + n0, 0); return out; }
    Rng rng(seed);
    std::vector<int32_t> final_labels(n0);
    {
    py::gil_scoped_release release;   // numpy allocations must not happen in this block
    const bool dbg = getenv("MAGNET_DEBUG") != nullptr;
    auto tick = std::chrono::steady_clock::now();
    auto ms = [&]() { auto now = std::chrono::steady_clock::now(); double v = std::chrono::duration<double, std::milli>(now - tick).count(); tick = now; return v; };
    if (dbg) fprintf(stderr, "louvain: copy %.1f ms\n", ms());
    for (int level = 0; level < max_levels; ++level) {
        const int64_t n = (int64_t)ip.size() - 1;
        int sweeps_done = 0;
        std::vector<double> deg(n);
        for (int64_t i = 0; i < n; ++i) { double s = self[i]; for (int64_t e = ip[i]; e < ip[i + 1]; ++e) s += w[e]; deg[i] = s; }
        std::vector<int32_t> comm(n); std::iota(comm.begin(), comm.end(), 0);
        std::vector<double> tot(deg);               // total degree of each community
        std::vector<double> link(n, 0.0); std::vector<int32_t> touched; touched.reserve(64);
        std::vector<int32_t> order(n); std::iota(order.begin(), order.end(), 0);
        // deterministic visiting orders: by strength (descending or ascending, ties by index) or natural index order
        if (order_mode == 1) std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return deg[a] > deg[b]; });
        else if (order_mode == 2) std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return deg[a] < deg[b]; });
        bool moved_any = false;
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            if (order_mode == 0) for (int64_t i = n - 1; i > 0; --i) std::swap(order[i], order[rng.next() % (uint64_t)(i + 1)]);
            int64_t moved = 0;
            for (int64_t oi = 0; oi < n; ++oi) {
                const int32_t i = order[oi];
                const int32_t ci = comm[i];
                touched.clear();
                for (int64_t e = ip[i]; e < ip[i + 1]; ++e) {
                    const int32_t c = comm[ix[e]];
                    if (link[c] == 0.0) touched.push_back(c);
                    link[c] += w[e];
                }
                const double ki = deg[i];
                tot[ci] -= ki;
                double best_gain = link[ci] - resolution * ki * tot[ci] / two_m;   // staying (community without i)
                int32_t best = ci;
                for (int32_t c : touched) {
                    if (c == ci) continue;
                    const double gain = link[c] - resolution * ki * tot[c] / two_m;
                    if (gain > best_gain + 1e-12) { best_gain = gain; best = c; }
                }
                tot[best] += ki;
                if (best != ci) { comm[i] = best; ++moved; }
                for (int32_t c : touched) link[c] = 0.0;
                if (link[ci] != 0.0) link[ci] = 0.0;
            }
            if (moved > 0) moved_any = true;
            ++sweeps_done;
            if (dbg) fprintf(stderr, "  level %d sweep %d: moved %ld (%.1f ms)\n", level, sweep, (long)moved, ms());
            if (moved == 0 || (sweep >= 1 && moved * 100 < n)) break;   // converged: fewer than 1 % of the nodes moved (the next level moves the rest)
        }
        // renumber communities contiguously
        std::vector<int32_t> newid(n, -1); int32_t nc = 0;
        for (int64_t i = 0; i < n; ++i) if (newid[comm[i]] < 0) newid[comm[i]] = nc++;
        for (int64_t i = 0; i < n; ++i) comm[i] = newid[comm[i]];
        for (int64_t v = 0; v < n0; ++v) membership[v] = comm[membership[v]];
        if (!moved_any || nc == (int32_t)n) break;
        // aggregate: community graph with self loops
        std::vector<std::vector<std::pair<int32_t, double>>> agg(nc);
        std::vector<double> nself(nc, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            const int32_t ci = comm[i];
            nself[ci] += self[i];
            for (int64_t e = ip[i]; e < ip[i + 1]; ++e) {
                const int32_t cj = comm[ix[e]];
                if (cj == ci) nself[ci] += w[e]; else agg[ci].push_back({cj, w[e]});
            }
        }
        std::vector<int64_t> nip(nc + 1, 0); std::vector<int32_t> nix; std::vector<double> nw;
        for (int32_t c = 0; c < nc; ++c) {
            auto& a = agg[c]; std::sort(a.begin(), a.end());
            for (size_t k = 0; k < a.size(); ++k) {
                if (k > 0 && a[k].first == a[k - 1].first) nw.back() += a[k].second;
                else { nix.push_back(a[k].first); nw.push_back(a[k].second); }
            }
            nip[c + 1] = (int64_t)nix.size();
        }
        ip.swap(nip); ix.swap(nix); w.swap(nw); self.swap(nself);
        if (dbg) fprintf(stderr, "  level %d: %ld -> %d communities after %d sweeps, aggregation %.1f ms\n", level, (long)n, nc, sweeps_done, ms());
    }
    // relabel by decreasing size
    int32_t nc = 0; for (int64_t v = 0; v < n0; ++v) nc = std::max(nc, membership[v] + 1);
    std::vector<int64_t> size(nc, 0); for (int64_t v = 0; v < n0; ++v) ++size[membership[v]];
    std::vector<int32_t> ord(nc); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return size[a] > size[b]; });
    std::vector<int32_t> rank(nc); for (int32_t r = 0; r < nc; ++r) rank[ord[r]] = r;
    for (int64_t v = 0; v < n0; ++v) final_labels[v] = rank[membership[v]];
    }
    arr<int32_t> out(std::vector<py::ssize_t>{(py::ssize_t)n0}); int32_t* O = out.mutable_data();
    std::copy(final_labels.begin(), final_labels.end(), O);
    return out;
}
