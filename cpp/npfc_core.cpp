// npfc_core.cpp — C++/Eigen heavy-compute core for npfixedcomppy.
//
// Design goals (see README "Compute core" section):
//  * SIMD: decided by the compiler at BUILD time. The build script adds the
//    best /arch: the host supports (AVX2/none), so Eigen selects its widest
//    packet traits automatically (pexp for Packet4d under AVX2+... , scalar
//    fallback otherwise). The same source uses the widest hardware math
//    available on the build host — no runtime dispatch for the kernels.
//  * Threading: this file has no OpenMP. Every loop is serial; the package
//    is built without /openmp (see setup.py) so no other thread pool exists
//    in the process to race with.
//  * Memory: everything is RAII. Caller-owned buffers are bounds-checked
//    before any write (fail => return 0, nothing written). Temporary Eigen
//    objects allocate/free internally and cannot leak or be double-freed;
//    we never return raw pointers except the explicit scratch pool, which
//    is freed by npfc_pool_free.
//
// Numerics: element-wise formulas are the exact scalar expressions of the
// Rust reference (d = (x-mu)/beta; exp(-0.5 d^2) * exp(-ln beta) *
// 1/sqrt(2pi)). The mixture mapping differs from the old Rust code only in
// reassociation of the per-row summation (per-column partials + serial dot
// per row), far below the package's 1e-9/1e-6 parity gates.

#include "npfc_core.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>

// libomp's idle workers spin by default (OMP_WAIT_POLICY=active) and, while
// idle, burn the same cores this process shares with the rayon pool — the
// n=5000 profile showed the rayon stages (solvegrad 23 -> 36 ms) paying for
// that oversubscription. Make idle workers sleep instead. This static runs
// at DLL load, i.e. before the OpenMP runtime initializes.
namespace {
struct OmpIdlePolicy {
    OmpIdlePolicy() {
#ifdef _MSC_VER
        ::_putenv_s("OMP_WAIT_POLICY", "passive");
#else
        ::setenv("OMP_WAIT_POLICY", "passive", 0);
#endif
    }
};
OmpIdlePolicy g_omp_idle_policy;
} // namespace
#endif

namespace {

constexpr double kLnSqrt2Pi = 0.9189385332046727;

} // namespace

extern "C" {

// ------------------------------------------------------------------
// Capability info (build flags + runtime OpenMP thread count).
// ------------------------------------------------------------------
void npfc_info(unsigned* caps, int* omp_threads, unsigned* simd_level) {
    unsigned c = 0;
    unsigned lvl = 0;
#ifdef _OPENMP
    c |= NPFIC_HAS_OPENMP;
#endif
#ifdef __AVX2__
    c |= NPFIC_HAS_AVX2;
    lvl = 2;
#endif
#ifdef __FMA__
    c |= NPFIC_HAS_FMA;
#endif
    if (caps) *caps = c;
    if (simd_level) *simd_level = lvl;
    if (omp_threads) {
#ifdef _OPENMP
        *omp_threads = omp_get_max_threads();
#else
        *omp_threads = 1;
#endif
    }
}

// ------------------------------------------------------------------
// Scratch pool: a buffer of n_elems doubles (64-byte aligned by the heap).
// ------------------------------------------------------------------
void* npfc_pool_create(size_t n_elems) {
    if (n_elems == 0) return nullptr;
    double* p = new (std::nothrow) double[n_elems]();
    return p;
}

void npfc_pool_free(void* p) { delete[] static_cast<double*>(p); }

// ------------------------------------------------------------------
// (n x g) normal-pdf kernel, COLUMN-MAJOR: out[j*n + i] = N(x_i; mu_j, beta).
// ------------------------------------------------------------------
int npfc_kmat_norm(const double* x, long n,
                   const double* mu, long g,
                   double beta,
                   double* out) {
    if (!x || !mu || !out || n <= 0 || g <= 0 || !(beta > 0.0)) return 0;
    if (!std::isfinite(beta)) return 0;

    // Column-major direct fill: column j is contiguous, so the per-iteration
    // sweep (dot products against columns) is a cache-friendly sequential
    // read. Columns are independent -> parallel over columns (mirrors the
    // rayon row/col pattern of the original Rust code; no temp buffer, no
    // transpose — the first version of this function paid an extra 2x the
    // memory traffic of a row-major temp + strided transpose, which showed
    // up as +12 ms on the n=5000 profile).
    const double inv_b = 1.0 / beta;
    const double log_pref = -std::log(beta) - kLnSqrt2Pi;
    for (long j = 0; j < g; ++j) {
        const double mj = mu[j];
        double* col = out + static_cast<size_t>(j) * static_cast<size_t>(n);
        for (long i = 0; i < n; ++i) {
            const double d = (x[static_cast<size_t>(i)] - mj) * inv_b;
            col[static_cast<size_t>(i)] = std::exp(-0.5 * d * d + log_pref);
        }
    }
    return 1;
}

// ------------------------------------------------------------------
// Mixture density: out[i] = sum_j pi[j] * N(x_i; mu_j, beta).
// ------------------------------------------------------------------
int npfc_mapping_norm(const double* x, long n,
                      const double* mu, long m,
                      const double* pi, double beta,
                      double* out) {
    if (!x || !mu || !pi || !out || n <= 0 || m <= 0 || !(beta > 0.0))
        return 0;
    if (!std::isfinite(beta)) return 0;

    const double inv_b = 1.0 / beta;
    const double log_pref = -std::log(beta) - kLnSqrt2Pi;
    double* op = out;

    // Serial over data points (independent rows).
    for (long i = 0; i < n; ++i) {
        const double xi = x[static_cast<size_t>(i)];
        double s = 0.0;
        for (long j = 0; j < m; ++j) {
            const double d = (xi - mu[j]) * inv_b;
            s += pi[j] * std::exp(-0.5 * d * d + log_pref);
        }
        op[static_cast<size_t>(i)] = s;
    }
    return 1;
}

// ------------------------------------------------------------------
// y = A^T x, A (n x m) column-major, x length n, y length m.
// Eigen BLAS-3 GEMV; picks SIMD/parallel per build.
// ------------------------------------------------------------------
int npfc_gemv(const double* A, long n, long m,
              const double* x, double* y) {
    if (!A || !x || !y || n <= 0 || m <= 0) return 0;
    const Eigen::Map<const Eigen::MatrixXd> Am(A, static_cast<Eigen::Index>(n),
                                               static_cast<Eigen::Index>(m));
    const Eigen::Map<const Eigen::VectorXd> xv(x, static_cast<Eigen::Index>(n));
    Eigen::Map<Eigen::VectorXd> yv(y, static_cast<Eigen::Index>(m));
    yv = Am.transpose() * xv;
    return 1;
}

} // extern "C"
