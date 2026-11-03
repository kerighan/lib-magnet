// The whole FFT repulsion of the full-batch optimiser in one C++ object with persistent buffers:
// scatter of the charges (1, x, y) on the padded grid, forward real 2-D FFTs (pocketfft, header-only, its own
// threading disabled: rows / columns are split over MAGNET's thread pool), products with the two kernel
// spectra, inverse FFTs, gather of the potentials and the forces. Replaces the scipy.fft path, which allocated
// ~30 MB of temporaries per iteration and crossed Python five times.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"
#include "layout.hpp"
#include "pool.hpp"
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
int resolve_nt(int n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 4 : (int)h;
}
inline void lagrange_w(float u, int n_interp, float* w) {
    if (n_interp == 3) {   // nodes at 1/6, 1/2, 5/6: constant denominators 2/9, -1/9, 2/9
        const float a = u - 1.f / 6.f, b = u - 0.5f, c = u - 5.f / 6.f;
        w[0] = b * c * 4.5f; w[1] = -a * c * 9.f; w[2] = a * b * 4.5f;
        return;
    }
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
}

class FFTConv2D {
public:
    FFTConv2D(int M, int n_interp) : M_(M), Mh_(M / 2 + 1), n_interp_(n_interp),
        W_((size_t)3 * M * M, 0.f), phi_((size_t)4 * M * M, 0.f), Wh_((size_t)3 * M * Mh_), Ph_((size_t)4 * M * Mh_),
        K1h_((size_t)M * Mh_), K2h_((size_t)M * Mh_), tmp_((size_t)4 * M * Mh_) {}

    // spectra of the two kernels on the padded torus, as complex64 (M, Mh) arrays
    void set_kernels(arr<float> K1_, arr<float> K2_) {
        require(K1_.ndim() == 3 && K1_.shape(0) == M_ && K1_.shape(1) == Mh_ && K1_.shape(2) == 2, "K1h must be (M, M/2+1, 2)");
        require(K2_.ndim() == 3 && K2_.shape(0) == M_ && K2_.shape(1) == Mh_ && K2_.shape(2) == 2, "K2h must be (M, M/2+1, 2)");
        const float* a = K1_.data(); const float* b = K2_.data();
        for (size_t k = 0; k < (size_t)M_ * Mh_; ++k) { K1h_[k] = {a[2 * k], a[2 * k + 1]}; K2h_[k] = {b[2 * k], b[2 * k + 1]}; }
    }

    // returns (sumQ, F (n, 2)); the live grid is N = n_boxes * n_interp nodes per side (N <= M)
    std::tuple<float, arr<float>> compute(arr<float> Y_, float xmin, float ymin, float width, int n_boxes, int n_threads) {
        require(Y_.ndim() == 2 && Y_.shape(1) == 2, "Y must be (n, 2)");
        const int64_t n = Y_.shape(0);
        const int N = n_boxes * n_interp_;
        require(N <= M_ && width > 0, "grid larger than the buffer");
        const float* Y = Y_.data();
        arr<float> F_({n, (int64_t)2});
        float* F = F_.mutable_data();
        const int nt = std::min(resolve_nt(n_threads), ThreadPool::get().size());
        const int M = M_, Mh = Mh_, ni = n_interp_;
        double tot = 0.0;
        const bool dbg = getenv("MAGNET_DEBUG") != nullptr; auto tick = std::chrono::steady_clock::now();
        auto lap = [&](const char* name) { if (!dbg) return; auto now = std::chrono::steady_clock::now(); fprintf(stderr, "  %s %.2f ms\n", name, std::chrono::duration<double, std::milli>(now - tick).count()); tick = now; };
        {
            py::gil_scoped_release release;
            // ---- scatter (counting sort by x-box, one thread per x-box range: no write races)
            float* W = W_.data();
            {
                const int ntz = std::min(nt, std::max(1, N / 16));
                pool_run(ntz, [&](int t) {   // zero the live block only (channels interleaved: row gx holds N x 3 live floats)
                    const int g0 = N * t / ntz, g1 = N * (t + 1) / ntz;
                    for (int gx = g0; gx < g1; ++gx) std::fill(W + (size_t)gx * M * 3, W + (size_t)gx * M * 3 + (size_t)N * 3, 0.f);
                });
            }
            lap("zero");
            const int nts = std::min(nt, std::max(1, (int)(n / 4000)));
            std::vector<int32_t> bxs(n); std::vector<int64_t> hist((size_t)nts * (n_boxes + 1), 0);
            const float inv_w = 1.f / width;
            pool_run(nts, [&](int t) {   // box of every point, per-thread histograms
                const int64_t r0 = n * t / nts, r1 = n * (t + 1) / nts; int64_t* h = hist.data() + (size_t)t * (n_boxes + 1);
                for (int64_t i = r0; i < r1; ++i) { const int bx = std::min(n_boxes - 1, std::max(0, (int)((Y[2 * i] - xmin) * inv_w))); bxs[i] = bx; ++h[bx + 1]; }
            });
            std::vector<int64_t> count(n_boxes + 1, 0);
            for (int b = 0; b < n_boxes; ++b) { int64_t s = 0; for (int t = 0; t < nts; ++t) { const int64_t c = hist[(size_t)t * (n_boxes + 1) + b + 1]; hist[(size_t)t * (n_boxes + 1) + b + 1] = count[b] + s; s += c; } count[b + 1] = count[b] + s; }
            std::vector<int32_t> order(n);
            pool_run(nts, [&](int t) {   // counting-sort placement, per-thread offsets
                const int64_t r0 = n * t / nts, r1 = n * (t + 1) / nts; int64_t* off = hist.data() + (size_t)t * (n_boxes + 1) + 1;
                for (int64_t i = r0; i < r1; ++i) order[off[bxs[i]]++] = (int32_t)i;
            });
            lap("sort");
            const int nranges = std::min(n_boxes, 8 * nts);   // striped x-box ranges: the dense centre of a layout would otherwise load one thread
            pool_run(nts, [&](int t) {
                float wx[8], wy[8];
                for (int rg = t; rg < nranges; rg += nts) {
                const int b0 = n_boxes * rg / nranges, b1 = n_boxes * (rg + 1) / nranges;
                for (int64_t k = count[b0]; k < count[b1]; ++k) {
                    const int64_t i = order[k];
                    const float x = Y[2 * i], y = Y[2 * i + 1];
                    const float ux = (x - xmin) / width, uy = (y - ymin) / width;
                    const int bx = std::min(n_boxes - 1, std::max(0, (int)ux)), by = std::min(n_boxes - 1, std::max(0, (int)uy));
                    lagrange_w(std::min(1.f, std::max(0.f, ux - bx)), ni, wx); lagrange_w(std::min(1.f, std::max(0.f, uy - by)), ni, wy);
                    const float ch[3] = {1.f, x, y};
                    for (int a = 0; a < ni; ++a) {
                        const int gx = bx * ni + a;
                        for (int b = 0; b < ni; ++b) {
                            const int gy = by * ni + b; const float w = wx[a] * wy[b];
                            float* cell = W + ((size_t)gx * M + gy) * 3;
                            cell[0] += w; cell[1] += w * x; cell[2] += w * y;
                        }
                    }
                }
                }
            });
            lap("scatter");
            // ---- forward: r2c along the last axis (rows independent), then c2c along the first axis (columns independent)
            using namespace pocketfft;
            std::complex<float>* Wh = Wh_.data();
            const size_t L = (size_t)M * Mh;
            const int ntf = nt;
            pool_run(ntf, [&](int t) {   // rows of the 3 channels (only the N live rows carry data), one multi-line call per block
                const int64_t rows = (int64_t)3 * N, r0 = rows * t / ntf, r1 = rows * (t + 1) / ntf;
                int64_t r = r0;
                while (r < r1) {
                    const int c = (int)(r / N); const int64_t gx0 = r % N, cnt = std::min<int64_t>(r1 - r, N - gx0);
                    r2c<float>({(size_t)cnt, (size_t)M}, {(ptrdiff_t)(M * 3 * sizeof(float)), (ptrdiff_t)(3 * sizeof(float))}, {(ptrdiff_t)(Mh * sizeof(std::complex<float>)), sizeof(std::complex<float>)}, {1}, FORWARD,
                               W + (size_t)gx0 * M * 3 + c, Wh + (size_t)c * L + (size_t)gx0 * Mh, 1.f, 1);
                    r += cnt;
                }
            });
            // zero the spectra rows of the padded (never written) grid rows
            pool_run(ntf, [&](int t) {
                const int64_t rows = (int64_t)3 * (M - N), r0 = rows * t / ntf, r1 = rows * (t + 1) / ntf;
                for (int64_t r = r0; r < r1; ++r) { const int c = (int)(r / (M - N)), gx = N + (int)(r % (M - N)); std::fill(Wh + (size_t)c * L + (size_t)gx * Mh, Wh + (size_t)c * L + (size_t)gx * Mh + Mh, std::complex<float>(0.f, 0.f)); }
            });
            // ---- forward c2c along the first axis (columns independent), one multi-line call per block
            const ptrdiff_t st = (ptrdiff_t)(Mh * sizeof(std::complex<float>));
            pool_run(ntf, [&](int t) {
                const int64_t cols = (int64_t)3 * Mh, c0 = cols * t / ntf, c1 = cols * (t + 1) / ntf;
                int64_t q = c0;
                while (q < c1) {
                    const int c = (int)(q / Mh); const int64_t col0 = q % Mh, cnt = std::min<int64_t>(c1 - q, Mh - col0);
                    c2c<float>({(size_t)M, (size_t)cnt}, {st, sizeof(std::complex<float>)}, {st, sizeof(std::complex<float>)}, {0}, FORWARD,
                               Wh + (size_t)c * L + col0, Wh + (size_t)c * L + col0, 1.f, 1);
                    q += cnt;
                }
            });
            lap("forward columns");
            // ---- products: Ph0 = K1h W0, Ph1..3 = K2h W0..2 (one linear sweep; memory-bound)
            std::complex<float>* Ph = Ph_.data();
            {   // explicit real / imaginary arithmetic: std::complex's operator* calls __mulsc3 (inf / nan handling) and does not vectorise
                const float* wh = reinterpret_cast<const float*>(Wh); const float* k1 = reinterpret_cast<const float*>(K1h_.data());
                const float* k2 = reinterpret_cast<const float*>(K2h_.data()); float* ph = reinterpret_cast<float*>(Ph);
                pool_run(nt, [&](int t) {
                    const size_t k0 = L * t / nt, k1_ = L * (t + 1) / nt;
                    for (size_t k = k0; k < k1_; ++k) {
                        const float ar = k1[2 * k], ai = k1[2 * k + 1], br = k2[2 * k], bi = k2[2 * k + 1];
                        const float w0r = wh[2 * k], w0i = wh[2 * k + 1];
                        ph[2 * k] = ar * w0r - ai * w0i; ph[2 * k + 1] = ar * w0i + ai * w0r;
                        ph[2 * (L + k)] = br * w0r - bi * w0i; ph[2 * (L + k) + 1] = br * w0i + bi * w0r;
                        const float w1r = wh[2 * (L + k)], w1i = wh[2 * (L + k) + 1];
                        ph[2 * (2 * L + k)] = br * w1r - bi * w1i; ph[2 * (2 * L + k) + 1] = br * w1i + bi * w1r;
                        const float w2r = wh[2 * (2 * L + k)], w2i = wh[2 * (2 * L + k) + 1];
                        ph[2 * (3 * L + k)] = br * w2r - bi * w2i; ph[2 * (3 * L + k) + 1] = br * w2i + bi * w2r;
                    }
                });
            }
            lap("products");
            // ---- inverse c2c along the first axis
            const float scale = 1.f / ((float)M * (float)M);
            pool_run(nt, [&](int t) {
                const int64_t cols = (int64_t)4 * Mh, c0 = cols * t / nt, c1 = cols * (t + 1) / nt;
                int64_t q = c0;
                while (q < c1) {
                    const int c = (int)(q / Mh); const int64_t col0 = q % Mh, cnt = std::min<int64_t>(c1 - q, Mh - col0);
                    c2c<float>({(size_t)M, (size_t)cnt}, {st, sizeof(std::complex<float>)}, {st, sizeof(std::complex<float>)}, {0}, BACKWARD,
                               Ph + (size_t)c * L + col0, Ph + (size_t)c * L + col0, 1.f, 1);
                    q += cnt;
                }
            });
            lap("inverse columns");
            float* phi = phi_.data();
            pool_run(nt, [&](int t) {
                const int64_t rows = (int64_t)4 * N, r0 = rows * t / nt, r1 = rows * (t + 1) / nt;
                int64_t r = r0;
                while (r < r1) {
                    const int c = (int)(r / N); const int64_t gx0 = r % N, cnt = std::min<int64_t>(r1 - r, N - gx0);
                    c2r<float>({(size_t)cnt, (size_t)M}, {(ptrdiff_t)(Mh * sizeof(std::complex<float>)), sizeof(std::complex<float>)}, {(ptrdiff_t)(M * 4 * sizeof(float)), (ptrdiff_t)(4 * sizeof(float))}, {1}, BACKWARD,
                               Ph + (size_t)c * L + (size_t)gx0 * Mh, phi + (size_t)gx0 * M * 4 + c, scale, 1);
                    r += cnt;
                }
            });
            lap("inverse rows");
            // ---- gather
            std::vector<double> partial(nt, 0.0);
            pool_run(nt, [&](int t) {
                float wx[8], wy[8]; double s = 0.0;
                const int64_t r0 = n * t / nt, r1 = n * (t + 1) / nt;
                for (int64_t kk = r0; kk < r1; ++kk) {
                    const int64_t i = order[kk];   // points in x-box order: the potentials are read with locality
                    const float x = Y[2 * i], y = Y[2 * i + 1];
                    const float ux = (x - xmin) / width, uy = (y - ymin) / width;
                    const int bx = std::min(n_boxes - 1, std::max(0, (int)ux)), by = std::min(n_boxes - 1, std::max(0, (int)uy));
                    lagrange_w(std::min(1.f, std::max(0.f, ux - bx)), ni, wx); lagrange_w(std::min(1.f, std::max(0.f, uy - by)), ni, wy);
                    float pot[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int a = 0; a < ni; ++a) {
                        const size_t gx = (size_t)bx * ni + a;
                        for (int b = 0; b < ni; ++b) {
                            const size_t gy = (size_t)by * ni + b; const float w = wx[a] * wy[b];
                            const float* cell = phi + (gx * M + gy) * 4;
                            pot[0] += w * cell[0]; pot[1] += w * cell[1]; pot[2] += w * cell[2]; pot[3] += w * cell[3];
                        }
                    }
                    s += pot[0];
                    F[2 * i] = pot[1] * x - pot[2];
                    F[2 * i + 1] = pot[1] * y - pot[3];
                }
                partial[t] = s;
            });
            for (double v : partial) tot += v;
            lap("gather");
        }
        return std::make_tuple((float)(tot - (double)n), F_);
    }

private:
    int M_, Mh_, n_interp_;
    std::vector<float> W_, phi_;
    std::vector<std::complex<float>> Wh_, Ph_, K1h_, K2h_, tmp_;
};

void bind_fftconv(py::module_& m) {
    py::class_<FFTConv2D>(m, "FFTConv2D", "FFT repulsion with persistent buffers: scatter, FFTs, kernel products, gather")
        .def(py::init<int, int>(), py::arg("M"), py::arg("n_interp"))
        .def("set_kernels", &FFTConv2D::set_kernels, py::arg("K1h"), py::arg("K2h"))
        .def("compute", &FFTConv2D::compute, py::arg("Y"), py::arg("xmin"), py::arg("ymin"), py::arg("width"), py::arg("n_boxes"), py::arg("n_threads") = -1);
}
