// Helpers for the exact-repulsion (t-SNE) optimiser in 2-D, in the spirit of FIt-SNE:
//  * interp_scatter : spreads per-point charges onto an equispaced grid with Lagrange weights
//  * interp_gather  : reads the grid potentials back at the points with the same weights
//  * tsne_attraction: sum_j p_ij q_ij (y_i - y_j) over the CSR graph, per row (no race)
// The convolution with the kernel between grid nodes is done by FFT on the Python side.
#include "layout.hpp"
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>

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

// Lagrange weights of the n_interp equispaced nodes of a box for a point at local coordinate u in [0,1)
inline void lagrange(float u, int n_interp, float* w) {
    for (int k = 0; k < n_interp; ++k) {
        const float uk = (k + 0.5f) / n_interp;
        float num = 1.f, den = 1.f;
        for (int m = 0; m < n_interp; ++m) {
            if (m == k) continue;
            const float um = (m + 0.5f) / n_interp;
            num *= (u - um); den *= (uk - um);
        }
        w[k] = num / den;
    }
}

struct Box {
    float xmin, ymin, width; int n_boxes, n_interp, N;
    inline void locate(float x, float y, int& bx, int& by, float* wx, float* wy) const {
        float ux = (x - xmin) / width, uy = (y - ymin) / width;
        bx = std::min(n_boxes - 1, std::max(0, (int)ux));
        by = std::min(n_boxes - 1, std::max(0, (int)uy));
        lagrange(std::min(1.f, std::max(0.f, ux - bx)), n_interp, wx);
        lagrange(std::min(1.f, std::max(0.f, uy - by)), n_interp, wy);
    }
};

}  // namespace


// Y (n, 2), charges (n, T) -> grid (T, N, N) with N = n_boxes * n_interp
arr<float> interp_scatter(arr<float> Y_, arr<float> charges_, float xmin, float ymin, float width,
                          int n_boxes, int n_interp, int n_threads)
{
    require(Y_.ndim() == 2 && Y_.shape(1) == 2, "Y must be (n, 2)");
    const int64_t n = Y_.shape(0);
    require(charges_.ndim() == 2 && charges_.shape(0) == n, "charges must be (n, T)");
    const int T = (int)charges_.shape(1);
    require(n_boxes >= 1 && n_interp >= 1 && n_interp <= 8 && width > 0, "bad grid parameters");
    const Box B{xmin, ymin, width, n_boxes, n_interp, n_boxes * n_interp};
    const int N = B.N;
    arr<float> out({(int64_t)T, (int64_t)N, (int64_t)N});
    float* O = out.mutable_data();
    std::fill(O, O + (size_t)T * N * N, 0.f);
    const float* Y = Y_.data(); const float* C = charges_.data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 20000));
    py::gil_scoped_release release;
    std::vector<std::vector<float>> local(nt);
    run_threads(nt, [&](int t) {
        std::vector<float>& G = local[t];
        G.assign((size_t)T * N * N, 0.f);
        float wx[8], wy[8]; int bx, by;
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        for (int64_t i = r0; i < r1; ++i) {
            B.locate(Y[2 * i], Y[2 * i + 1], bx, by, wx, wy);
            for (int a = 0; a < n_interp; ++a) {
                const int gx = bx * n_interp + a;
                for (int b = 0; b < n_interp; ++b) {
                    const int gy = by * n_interp + b;
                    const float w = wx[a] * wy[b];
                    for (int c = 0; c < T; ++c) G[((size_t)c * N + gx) * N + gy] += w * C[i * T + c];
                }
            }
        }
    });
    run_threads(nt, [&](int t) {  // reduce, each thread owns a slice of the grid
        const size_t total = (size_t)T * N * N, s0 = total * t / nt, s1 = total * (t + 1) / nt;
        for (int src = 0; src < nt; ++src) for (size_t g = s0; g < s1; ++g) O[g] += local[src][g];
    });
    return out;
}

// grid potentials (T, N, N) -> values at the points (n, T)
arr<float> interp_gather(arr<float> Y_, arr<float> grid_, float xmin, float ymin, float width,
                         int n_boxes, int n_interp, int n_threads)
{
    require(Y_.ndim() == 2 && Y_.shape(1) == 2, "Y must be (n, 2)");
    const int64_t n = Y_.shape(0);
    require(grid_.ndim() == 3 && grid_.shape(1) == n_boxes * n_interp && grid_.shape(2) == n_boxes * n_interp, "grid must be (T, N, N)");
    const int T = (int)grid_.shape(0);
    const Box B{xmin, ymin, width, n_boxes, n_interp, n_boxes * n_interp};
    const int N = B.N;
    arr<float> out({n, (int64_t)T});
    float* O = out.mutable_data();
    const float* Y = Y_.data(); const float* G = grid_.data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 20000));
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        float wx[8], wy[8]; int bx, by;
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        for (int64_t i = r0; i < r1; ++i) {
            B.locate(Y[2 * i], Y[2 * i + 1], bx, by, wx, wy);
            for (int c = 0; c < T; ++c) O[i * T + c] = 0.f;
            for (int a = 0; a < n_interp; ++a) {
                const int gx = bx * n_interp + a;
                for (int b = 0; b < n_interp; ++b) {
                    const int gy = by * n_interp + b;
                    const float w = wx[a] * wy[b];
                    for (int c = 0; c < T; ++c) O[i * T + c] += w * G[((size_t)c * N + gx) * N + gy];
                }
            }
        }
    });
    return out;
}

// attraction term of t-SNE: out_i = sum_j p_ij q_ij (y_i - y_j), q = 1 / (1 + |y_i - y_j|^2)
arr<float> tsne_attraction(arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_, arr<float> Y_, int n_threads)
{
    require(Y_.ndim() == 2, "Y must be 2-d");
    const int64_t n = Y_.shape(0); const int dim = (int)Y_.shape(1);
    require(dim >= 1 && dim <= MAX_DIM, "dim out of range");
    validate_csr(indptr_, indices_, &data_, n);
    arr<float> out({n, (int64_t)dim});
    float* O = out.mutable_data();
    const float* Y = Y_.data(); const int32_t* ip = indptr_.data(); const int32_t* ix = indices_.data(); const float* P = data_.data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 4096));
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        float acc[MAX_DIM], diff[MAX_DIM];
        for (int64_t i = r0; i < r1; ++i) {
            for (int d = 0; d < dim; ++d) acc[d] = 0.f;
            const float* yi = Y + i * dim;
            for (int32_t e = ip[i]; e < ip[i + 1]; ++e) {
                const float* yj = Y + (int64_t)ix[e] * dim;
                float d2 = 0.f;
                for (int d = 0; d < dim; ++d) { diff[d] = yi[d] - yj[d]; d2 += diff[d] * diff[d]; }
                const float w = P[e] / (1.f + d2);
                for (int d = 0; d < dim; ++d) acc[d] += w * diff[d];
            }
            for (int d = 0; d < dim; ++d) O[i * dim + d] = acc[d];
        }
    });
    return out;
}


// Fused t-SNE step: delta-bar-delta gains, momentum, position update and recentring (all in place).
void tsne_update(arr<float> Y_, arr<float> update_, arr<float> gains_, arr<float> grad_, float lr, float momentum, int n_threads)
{
    require(Y_.ndim() == 2, "Y must be 2-d");
    const int64_t n = Y_.shape(0); const int dim = (int)Y_.shape(1);
    const int64_t total = n * dim;
    require(update_.size() == total && gains_.size() == total && grad_.size() == total, "arrays must match Y");
    float* Y = Y_.mutable_data(); float* U = update_.mutable_data(); float* G = gains_.mutable_data();
    const float* D = grad_.data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 8192));
    py::gil_scoped_release release;
    std::vector<double> sums((size_t)nt * dim, 0.0);
    run_threads(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        double* s = sums.data() + (size_t)t * dim;
        for (int64_t i = r0 * dim; i < r1 * dim; ++i) {
            const bool same = (D[i] > 0.f) == (U[i] > 0.f);
            float g = same ? G[i] * 0.8f : G[i] + 0.2f;
            if (g < 0.01f) g = 0.01f;
            G[i] = g;
            U[i] = momentum * U[i] - lr * g * D[i];
            Y[i] += U[i];
            s[i % dim] += Y[i];
        }
    });
    std::vector<float> mean(dim, 0.f);
    for (int d = 0; d < dim; ++d) { double m = 0; for (int t = 0; t < nt; ++t) m += sums[(size_t)t * dim + d]; mean[d] = (float)(m / (double)n); }
    run_threads(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        for (int64_t i = r0; i < r1; ++i) for (int d = 0; d < dim; ++d) Y[i * dim + d] -= mean[d];
    });
}


// grad += w * (Y_i - mean of the partners' positions), partners given as (n, m) indices
void mid_spring(arr<float> Y_, arr<int32_t> pairs_, arr<float> grad_, float w, int n_threads)
{
    require(Y_.ndim() == 2 && grad_.ndim() == 2 && pairs_.ndim() == 2, "bad shapes");
    const int64_t n = Y_.shape(0); const int dim = (int)Y_.shape(1); const int m = (int)pairs_.shape(1);
    require(pairs_.shape(0) == n && grad_.shape(0) == n && grad_.shape(1) == dim && m >= 1, "bad shapes");
    const float* Y = Y_.data(); const int32_t* Pp = pairs_.data(); float* G = grad_.mutable_data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 8192));
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        float acc[MAX_DIM];
        for (int64_t i = r0; i < r1; ++i) {
            for (int d = 0; d < dim; ++d) acc[d] = 0.f;
            for (int k = 0; k < m; ++k) { const float* yp = Y + (int64_t)Pp[i * m + k] * dim; for (int d = 0; d < dim; ++d) acc[d] += yp[d]; }
            for (int d = 0; d < dim; ++d) G[i * dim + d] += w * (Y[i * dim + d] - acc[d] / m);
        }
    });
}


// ---------------------------------------------------------------------------------------------
// Padded, reduction-free scatter: points are bucketed by their x-box, threads own disjoint ranges of
// x-boxes (hence disjoint rows of the grid) so no private grids and no reduction are needed.
// Output (3, M, M) zero-padded charge grids for the charges (1, x, y).
arr<float> scatter_padded_2d(arr<float> Y_, float xmin, float ymin, float width, int n_boxes, int n_interp, int M, int n_threads)
{
    require(Y_.ndim() == 2 && Y_.shape(1) == 2, "Y must be (n, 2)");
    const int64_t n = Y_.shape(0);
    const int N = n_boxes * n_interp;
    require(n_interp >= 1 && n_interp <= 8 && M >= N && width > 0, "bad grid parameters");
    const float* Y = Y_.data();
    arr<float> out({(int64_t)3, (int64_t)M, (int64_t)M});
    float* O = out.mutable_data();
    std::fill(O, O + (size_t)3 * M * M, 0.f);
    const Box B{xmin, ymin, width, n_boxes, n_interp, N};
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 20000));
    py::gil_scoped_release release;
    // counting sort of the points by x-box
    std::vector<int32_t> bxs(n);
    std::vector<int64_t> count(n_boxes + 1, 0);
    for (int64_t i = 0; i < n; ++i) {
        const float ux = (Y[2 * i] - xmin) / width;
        const int bx = std::min(n_boxes - 1, std::max(0, (int)ux));
        bxs[i] = bx; ++count[bx + 1];
    }
    for (int b = 0; b < n_boxes; ++b) count[b + 1] += count[b];
    std::vector<int32_t> order(n);
    { std::vector<int64_t> fill(count.begin(), count.end() - 1); for (int64_t i = 0; i < n; ++i) order[fill[bxs[i]]++] = (int32_t)i; }
    run_threads(nt, [&](int t) {
        const int b0 = n_boxes * t / nt, b1 = n_boxes * (t + 1) / nt;   // this thread's x-boxes
        float wx[8], wy[8]; int bx, by;
        for (int64_t k = count[b0]; k < count[b1]; ++k) {
            const int64_t i = order[k];
            B.locate(Y[2 * i], Y[2 * i + 1], bx, by, wx, wy);
            const float ch[3] = {1.f, Y[2 * i], Y[2 * i + 1]};
            for (int a = 0; a < n_interp; ++a) {
                const int gx = bx * n_interp + a;
                for (int b = 0; b < n_interp; ++b) {
                    const int gy = by * n_interp + b;
                    const float w = wx[a] * wy[b];
                    for (int c = 0; c < 3; ++c) O[((size_t)c * M + gx) * M + gy] += w * ch[c];
                }
            }
        }
    });
    return out;
}

// Products in the spectral domain: Ph[0] = K1h * Wh[0], Ph[1..3] = K2h * Wh[0..2]; arrays are complex64
// viewed as float pairs: K (M, Mh, 2), W (3, M, Mh, 2) -> (4, M, Mh, 2)
arr<float> spectral_products(arr<float> K1h_, arr<float> K2h_, arr<float> Wh_, int n_threads)
{
    require(K1h_.ndim() == 3 && K1h_.shape(2) == 2 && Wh_.ndim() == 4 && Wh_.shape(0) == 3 && Wh_.shape(3) == 2, "bad shapes");
    const int64_t M = K1h_.shape(0), Mh = K1h_.shape(1);
    require(K2h_.shape(0) == M && K2h_.shape(1) == Mh && Wh_.shape(1) == M && Wh_.shape(2) == Mh, "bad shapes");
    const float* K1 = K1h_.data(); const float* K2 = K2h_.data(); const float* W = Wh_.data();
    arr<float> out({(int64_t)4, M, Mh, (int64_t)2});
    float* O = out.mutable_data();
    const int64_t L = M * Mh;
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, L / 200000));
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        const int64_t k0 = L * t / nt, k1 = L * (t + 1) / nt;
        for (int64_t k = k0; k < k1; ++k) {
            const float ar = K1[2 * k], ai = K1[2 * k + 1], br = W[2 * k], bi = W[2 * k + 1];
            O[2 * k] = ar * br - ai * bi; O[2 * k + 1] = ar * bi + ai * br;
            const float cr = K2[2 * k], ci = K2[2 * k + 1];
            for (int c = 0; c < 3; ++c) {
                const float wr = W[2 * (c * L + k)], wi = W[2 * (c * L + k) + 1];
                O[2 * ((c + 1) * L + k)] = cr * wr - ci * wi; O[2 * ((c + 1) * L + k) + 1] = cr * wi + ci * wr;
            }
        }
    });
    return out;
}

// Gather from the padded potentials (4, M, M) and compute the forces: returns (sumQ, F (n, 2))
std::tuple<float, arr<float>> gather_forces_2d(arr<float> Y_, arr<float> phi_, float xmin, float ymin, float width, int n_boxes, int n_interp, int n_threads)
{
    require(Y_.ndim() == 2 && Y_.shape(1) == 2 && phi_.ndim() == 3 && phi_.shape(0) == 4, "bad shapes");
    const int64_t n = Y_.shape(0); const int64_t M = phi_.shape(1);
    const int N = n_boxes * n_interp;
    require(phi_.shape(2) == M && M >= N, "phi must be (4, M, M) with M >= N");
    const float* Y = Y_.data(); const float* phi = phi_.data();
    const Box B{xmin, ymin, width, n_boxes, n_interp, N};
    arr<float> F_({n, (int64_t)2});
    float* F = F_.mutable_data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 20000));
    std::vector<double> partial(nt, 0.0);
    {
        py::gil_scoped_release release;
        run_threads(nt, [&](int t) {
            float wx[8], wy[8]; int bx, by; double s = 0.0;
            const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
            for (int64_t i = r0; i < r1; ++i) {
                B.locate(Y[2 * i], Y[2 * i + 1], bx, by, wx, wy);
                float pot[4] = {0.f, 0.f, 0.f, 0.f};
                for (int a = 0; a < n_interp; ++a) {
                    const int64_t gx = bx * n_interp + a;
                    for (int b = 0; b < n_interp; ++b) {
                        const int64_t gy = by * n_interp + b;
                        const float w = wx[a] * wy[b];
                        for (int c = 0; c < 4; ++c) pot[c] += w * phi[((size_t)c * M + gx) * M + gy];
                    }
                }
                s += pot[0];
                F[2 * i] = pot[1] * Y[2 * i] - pot[2];
                F[2 * i + 1] = pot[1] * Y[2 * i + 1] - pot[3];
            }
            partial[t] = s;
        });
    }
    double tot = 0.0;
    for (double v : partial) tot += v;
    return std::make_tuple((float)(tot - (double)n), F_);
}


// bounding box of Y (n, 2): (xmin, ymin, xmax, ymax)
std::tuple<float, float, float, float> bbox_2d(arr<float> Y_, int n_threads)
{
    require(Y_.ndim() == 2 && Y_.shape(1) == 2 && Y_.shape(0) >= 1, "Y must be (n, 2)");
    const int64_t n = Y_.shape(0); const float* Y = Y_.data();
    const int nt = (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 100000));
    std::vector<float> lo(nt * 2), hi(nt * 2);
    py::gil_scoped_release release;
    run_threads(nt, [&](int t) {
        const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
        float x0 = Y[2 * r0], x1 = x0, y0 = Y[2 * r0 + 1], y1 = y0;
        for (int64_t i = r0; i < r1; ++i) {
            x0 = std::min(x0, Y[2 * i]); x1 = std::max(x1, Y[2 * i]);
            y0 = std::min(y0, Y[2 * i + 1]); y1 = std::max(y1, Y[2 * i + 1]);
        }
        lo[2 * t] = x0; lo[2 * t + 1] = y0; hi[2 * t] = x1; hi[2 * t + 1] = y1;
    });
    float xmin = lo[0], ymin = lo[1], xmax = hi[0], ymax = hi[1];
    for (int t = 1; t < nt; ++t) { xmin = std::min(xmin, lo[2 * t]); ymin = std::min(ymin, lo[2 * t + 1]); xmax = std::max(xmax, hi[2 * t]); ymax = std::max(ymax, hi[2 * t + 1]); }
    return std::make_tuple(xmin, ymin, xmax, ymax);
}
