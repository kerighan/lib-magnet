// Chebyshev-filtered subspace iteration (Zhou-Saad style filter + Rayleigh-Ritz) for the leading
// non-trivial eigenvectors of the lazy normalised adjacency M = (I + D^-1/2 A D^-1/2) / 2.
// Spectrum of M lies in [0, 1]; the trivial eigenvector (deg^1/2 on each connected component) is
// deflated explicitly. Converges orders of magnitude faster than plain power iteration when the
// eigengap is tiny (large graphs, long cycles, many clusters), which is the usual case.
#include "layout.hpp"
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {

struct Graph {
    const int32_t* indptr; const int32_t* indices; const float* data; const float* sd;
    const int32_t* comp; int64_t n; int nt;
};

int resolve(int n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 4 : (int)h;
}

template <class F> void run_parallel(int nt, int64_t n, F&& f) {
    if (nt <= 1) { f(0, n); return; }
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) pool.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
    for (auto& th : pool) th.join();
}

// out = M in  (both n x p, row major, double precision)
void spmv(const Graph& g, const double* in, double* out, int p) {
    run_parallel(g.nt, g.n, [&](int64_t r0, int64_t r1) {
        std::vector<double> acc(p);
        for (int64_t i = r0; i < r1; ++i) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int32_t j = g.indptr[i]; j < g.indptr[i + 1]; ++j) {
                const int32_t col = g.indices[j];
                const double w = (double)g.data[j] * g.sd[col];
                const double* x = in + (int64_t)col * p;
                for (int k = 0; k < p; ++k) acc[k] += w * x[k];
            }
            const double* xi = in + i * p; double* o = out + i * p;
            for (int k = 0; k < p; ++k) o[k] = 0.5 * (xi[k] + g.sd[i] * acc[k]);
        }
    });
}

// remove, per connected component, the projection on the trivial eigenvector t_i = 1 / sd_i
void deflate(const Graph& g, double* X, int p, int n_comp, const std::vector<double>& tnorm) {
    std::vector<double> proj((size_t)n_comp * p, 0.0);
    for (int64_t i = 0; i < g.n; ++i) {
        const double t = g.sd[i] > 0 ? 1.0 / g.sd[i] : 0.0;
        double* pr = proj.data() + (size_t)g.comp[i] * p; const double* x = X + i * p;
        for (int k = 0; k < p; ++k) pr[k] += t * x[k];
    }
    for (int c = 0; c < n_comp; ++c)
        for (int k = 0; k < p; ++k) proj[(size_t)c * p + k] = tnorm[c] > 0 ? proj[(size_t)c * p + k] / tnorm[c] : 0.0;
    run_parallel(g.nt, g.n, [&](int64_t r0, int64_t r1) {
        for (int64_t i = r0; i < r1; ++i) {
            const double t = g.sd[i] > 0 ? 1.0 / g.sd[i] : 0.0;
            const double* pr = proj.data() + (size_t)g.comp[i] * p; double* x = X + i * p;
            for (int k = 0; k < p; ++k) x[k] -= t * pr[k];
        }
    });
}

// modified Gram-Schmidt on the p columns of X (n x p)
void orthonormalize(int64_t n, double* X, int p) {
    for (int pass = 0; pass < 2; ++pass)  // two passes of modified Gram-Schmidt: numerically safe
        for (int k = 0; k < p; ++k) {
            for (int j = 0; j < k; ++j) {
                double dot = 0.0;
                for (int64_t i = 0; i < n; ++i) dot += X[i * p + k] * X[i * p + j];
                for (int64_t i = 0; i < n; ++i) X[i * p + k] -= dot * X[i * p + j];
            }
            double nrm = 0.0;
            for (int64_t i = 0; i < n; ++i) nrm += X[i * p + k] * X[i * p + k];
            const double inv = nrm > 1e-300 ? 1.0 / std::sqrt(nrm) : 0.0;
            for (int64_t i = 0; i < n; ++i) X[i * p + k] *= inv;
        }
}

// Jacobi eigen-decomposition of a small symmetric matrix H (p x p). Eigenvalues descending.
void sym_eig(std::vector<double> H, int p, std::vector<double>& vals, std::vector<double>& vecs) {
    vecs.assign((size_t)p * p, 0.0);
    for (int i = 0; i < p; ++i) vecs[(size_t)i * p + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < p; ++i) for (int j = i + 1; j < p; ++j) off += H[(size_t)i * p + j] * H[(size_t)i * p + j];
        if (off < 1e-22) break;
        for (int i = 0; i < p; ++i) for (int j = i + 1; j < p; ++j) {
            const double h_ij = H[(size_t)i * p + j];
            if (std::fabs(h_ij) < 1e-300) continue;
            const double theta = (H[(size_t)j * p + j] - H[(size_t)i * p + i]) / (2.0 * h_ij);
            const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < p; ++k) {
                const double hik = H[(size_t)k * p + i], hjk = H[(size_t)k * p + j];
                H[(size_t)k * p + i] = c * hik - s * hjk; H[(size_t)k * p + j] = s * hik + c * hjk;
            }
            for (int k = 0; k < p; ++k) {
                const double hik = H[(size_t)i * p + k], hjk = H[(size_t)j * p + k];
                H[(size_t)i * p + k] = c * hik - s * hjk; H[(size_t)j * p + k] = s * hik + c * hjk;
            }
            for (int k = 0; k < p; ++k) {
                const double vki = vecs[(size_t)k * p + i], vkj = vecs[(size_t)k * p + j];
                vecs[(size_t)k * p + i] = c * vki - s * vkj; vecs[(size_t)k * p + j] = s * vki + c * vkj;
            }
        }
    }
    vals.resize(p);
    for (int i = 0; i < p; ++i) vals[i] = H[(size_t)i * p + i];
    std::vector<int> order(p);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return vals[a] > vals[b]; });
    std::vector<double> v2(vals), e2(vecs);
    for (int c = 0; c < p; ++c) {
        vals[c] = v2[order[c]];
        for (int k = 0; k < p; ++k) vecs[(size_t)k * p + c] = e2[(size_t)k * p + order[c]];
    }
}

// X <- X V   (n x p times p x p)
void rotate(int64_t n, double* X, int p, const std::vector<double>& V, int nt) {
    run_parallel(nt, n, [&](int64_t r0, int64_t r1) {
        std::vector<double> tmp(p);
        for (int64_t i = r0; i < r1; ++i) {
            double* x = X + i * p;
            for (int c = 0; c < p; ++c) { double s = 0; for (int k = 0; k < p; ++k) s += x[k] * V[(size_t)k * p + c]; tmp[c] = s; }
            for (int c = 0; c < p; ++c) x[c] = tmp[c];
        }
    });
}

}  // namespace


arr<float> chebyshev_eigenvectors(
    arr<int32_t> indptr_, arr<int32_t> indices_, arr<float> data_,
    arr<float> X_, arr<float> inv_sqrt_deg_, arr<int32_t> component_,
    int n_wanted, int degree, int max_iter, float tol, int n_threads)
{
    auto Xb = X_.request();
    require(Xb.ndim == 2, "X must be 2-d");
    const int64_t n = Xb.shape[0];
    const int p = (int)Xb.shape[1];
    require(n_wanted >= 1 && n_wanted < p, "X needs at least n_wanted + 1 columns");
    require(degree >= 2 && max_iter >= 0, "degree must be >= 2");
    validate_csr(indptr_, indices_, &data_, n);
    require(inv_sqrt_deg_.ndim() == 1 && inv_sqrt_deg_.shape(0) == n, "inv_sqrt_deg must have n entries");
    require(component_.ndim() == 1 && component_.shape(0) == n, "component must have n entries");
    for (int64_t i = 0; i < n; ++i) require(component_.data()[i] >= 0, "component labels must be >= 0");
    float* Xout = static_cast<float*>(Xb.ptr);
    Graph g{indptr_.data(), indices_.data(), data_.data(), inv_sqrt_deg_.data(), component_.data(), n,
            (int)std::min<int64_t>(resolve(n_threads), std::max<int64_t>(1, n / 4096))};
    int n_comp = 0;
    for (int64_t i = 0; i < n; ++i) n_comp = std::max(n_comp, g.comp[i] + 1);
    std::vector<double> tnorm(n_comp, 0.0);
    for (int64_t i = 0; i < n; ++i) { const double t = g.sd[i] > 0 ? 1.0 / g.sd[i] : 0.0; tnorm[g.comp[i]] += t * t; }

    py::gil_scoped_release release;

    std::vector<double> Xd((size_t)n * p), Z((size_t)n * p), Y((size_t)n * p), Ynew((size_t)n * p);
    for (size_t i = 0; i < Xd.size(); ++i) Xd[i] = Xout[i];
    double* X = Xd.data();
    std::vector<double> H((size_t)p * p), theta, V;

    auto rayleigh_ritz = [&]() {
        spmv(g, X, Z.data(), p);
        deflate(g, Z.data(), p, n_comp, tnorm);
        std::fill(H.begin(), H.end(), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            const double* x = X + i * p; const double* z = Z.data() + i * p;
            for (int a = 0; a < p; ++a) for (int b = a; b < p; ++b) H[(size_t)a * p + b] += x[a] * z[b];
        }
        for (int a = 0; a < p; ++a) for (int b = a + 1; b < p; ++b) H[(size_t)b * p + a] = H[(size_t)a * p + b];
        sym_eig(H, p, theta, V);
        rotate(n, X, p, V, g.nt);
        rotate(n, Z.data(), p, V, g.nt);
        double worst = 0.0;
        for (int c = 0; c < n_wanted; ++c) {
            double r = 0.0;
            for (int64_t i = 0; i < n; ++i) { const double d = Z[i * p + c] - theta[c] * X[i * p + c]; r += d * d; }
            worst = std::max(worst, std::sqrt(r));
        }
        return worst;
    };

    deflate(g, X, p, n_comp, tnorm);
    orthonormalize(n, X, p);
    double res = rayleigh_ritz();
    double prev_sum = -1.0;
    for (int it = 0; it < max_iter; ++it) {
        // convergence: residual small compared to the gap between the last wanted and the first
        // unwanted Ritz value (angle bound), or Ritz values no longer moving. The gap is floored:
        // when the leading eigenvalues are (nearly) degenerate, e.g. a graph with many clusters,
        // any vector of their span is as good an initialisation as another.
        const double gap = std::max(theta[n_wanted - 1] - theta[n_wanted], 1e-3);
        double sum = 0.0;
        for (int c = 0; c < n_wanted; ++c) sum += theta[c];
        const bool stagnant = prev_sum >= 0 && std::fabs(sum - prev_sum) < 1e-9 * (double)n_wanted;
        if (res < tol * gap || res < 1e-9 || stagnant) break;
        prev_sum = sum;

        // Chebyshev filter damping [0, alpha] (Zhou-Saad scaled three-term recurrence)
        double alpha = theta[p - 1];
        alpha = std::min(alpha, theta[n_wanted - 1] - 0.5 * gap);
        alpha = std::max(alpha, 1e-3);
        const double a0 = std::min(1.0, theta[0] + 1e-9);
        const double e = 0.5 * alpha, cc = 0.5 * alpha;
        const double sigma1 = e / (a0 - cc);
        double sigma = sigma1;
        spmv(g, X, Y.data(), p);
        deflate(g, Y.data(), p, n_comp, tnorm);
        {
            const double s1 = sigma1 / e;
            run_parallel(g.nt, n * p, [&](int64_t i0, int64_t i1) { for (int64_t i = i0; i < i1; ++i) Y[i] = (Y[i] - cc * X[i]) * s1; });
        }
        for (int k = 2; k <= degree; ++k) {
            const double sigma_new = 1.0 / (2.0 / sigma1 - sigma);
            spmv(g, Y.data(), Ynew.data(), p);
            deflate(g, Ynew.data(), p, n_comp, tnorm);
            const double s2 = 2.0 * sigma_new / e, ss = sigma * sigma_new;
            run_parallel(g.nt, n * p, [&](int64_t i0, int64_t i1) {
                for (int64_t i = i0; i < i1; ++i) Ynew[i] = (Ynew[i] - cc * Y[i]) * s2 - ss * X[i];
            });
            std::copy(Y.begin(), Y.end(), X);
            std::swap(Y, Ynew);
            sigma = sigma_new;
        }
        std::copy(Y.begin(), Y.end(), X);
        deflate(g, X, p, n_comp, tnorm);
        orthonormalize(n, X, p);
        res = rayleigh_ritz();
    }
    for (size_t i = 0; i < Xd.size(); ++i) Xout[i] = (float)Xd[i];
    return X_;
}
