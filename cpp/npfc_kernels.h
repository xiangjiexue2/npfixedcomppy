// npfc_kernels.h — vectorised mixture-density kernels shared by the families
// (the non-log `dnpnorm_` / `dnpt_` / `pnpnorm_` / `dnpnormc_` / `dnppois_`
// family of functions from the R package's miscfuns.h, ported from
// npfixedcomppy/src/families.rs).
//
// Conventions: `x` has length n, `mu0`/`pi0` have length m.
//
// Threading: these kernels are serial. All multi-threading in the package
// comes from Eigen's own compile-time-gated parallel GEMM/GEMV in the family
// hot paths — there are no hand-written OpenMP loops here. The j-outer/i-inner
// accumulation order below is the bit-identical reference order.
#ifndef NPFIC_KERNELS_H
#define NPFIC_KERNELS_H

#include "npfc_stats.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace npfc {
namespace kern {

// ---------------------------------------------------------------------------
// Kernel column cache
// ---------------------------------------------------------------------------
//
// A kernel column K[:, mu] (the n-vector of kernel values at one support
// value `mu`) depends ONLY on (data, beta, mu) — never on the current
// weights, density, or iteration. The mixing engine re-evaluates the same
// support columns on every pass of an outer iteration (mapping, the weights
// fill, collapse's repeated remapping, the end-of-iteration mapping), which
// is the dominant cost for the expensive kernels (non-central-t, the
// one-parameter normal, the normal CDF).
//
// This cache evaluates each distinct `mu` column exactly once per solver run
// (in the exact reference per-row order, so a cached column is BIT-IDENTICAL
// to a fresh fill — only the redundant re-evaluation is removed, never the
// arithmetic or its accumulation order) and serves every consumer from the
// shared column. After the first iteration all (n x m) fills degrade to a
// memcpy + scale, and the grid sweep can be seeded once from `prepare`.
class KernelColumnCache {
public:
    using Eval = std::function<double(double /*x*/, double /*mu*/)>;

    KernelColumnCache() = default;

    // Bind the data and the kernel; drops any previously cached columns.
    void init(const std::vector<double>& x, Eval eval) {
        x_ = &x;
        n_ = x.size();
        eval_ = std::move(eval);
        cols_.clear();
        fresh_ = 0;
        fresh_ms_ = 0.0;
    }

    // The n-vector column K[:, mu]; computed on first use, then O(1). The
    // map is `mutable`, so this const accessor may populate it.
    const std::vector<double>& column(double mu) const {
        if (!x_)
            return empty();
        auto it = cols_.find(mu);
        if (it != cols_.end())
            return it->second;
        std::vector<double> col(n_);
        const double* x = x_->data();
        const auto t0 =
            prof_ ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
        for (std::size_t i = 0; i < n_; ++i)
            col[i] = eval_(x[i], mu);
        if (prof_) {
            fresh_ += 1;
            fresh_ms_ +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count() *
                1e3;
        }
        auto r = cols_.emplace(mu, std::move(col));
        return r.first->second;
    }

    // Seed a column that was already produced (e.g. the eager grid matrix).
    void insert(double mu, std::vector<double> col) {
        cols_.emplace(mu, std::move(col));
    }

    // Returned when `column` is called before `init` (the constructor runs
    // before the engine exists, so the cache is default-built then bound).
    const std::vector<double>& empty() const {
        static const std::vector<double> kEmpty;
        return kEmpty;
    }

    std::size_t n() const { return n_; }
    std::size_t size() const { return cols_.size(); }
    void clear() { cols_.clear(); }

    // Diagnostics (NPFIXEDCOMPY_PROFILE=1): number of distinct columns
    // freshly evaluated and the wall time they took.
    std::size_t fresh_count() const { return fresh_; }
    double fresh_ms() const { return fresh_ms_; }

private:
    const std::vector<double>* x_ = nullptr;
    std::size_t n_ = 0;
    Eval eval_ = [](double, double) { return 0.0; };
    mutable std::unordered_map<double, std::vector<double>> cols_;
    bool prof_ = std::getenv("NPFIXEDCOMPY_PROFILE") != nullptr;
    mutable std::size_t fresh_ = 0;
    mutable double fresh_ms_ = 0.0;
};

// ---------------------------------------------------------------------------
// normal
// ---------------------------------------------------------------------------

// `N(x; mu, beta)`, the non-log version.
inline double dnormv(double x, double mu, double beta) {
    const double d = x - mu;
    return std::exp(d * d / (-2.0 * beta * beta) -
                    (stats::LN_SQRT_2PI + std::log(beta)));
}

// `dnpnorm_(x, mu0, pi0, beta)` (non-log): the normal mixture density
// sum_j pi0[j] N(x[i]; mu0[j], beta) at all data points.
inline std::vector<double> dnpnorm(const std::vector<double>& x,
                                   const std::vector<double>& mu0,
                                   const std::vector<double>& pi0,
                                   double beta) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        const double pj = pi0[j];
        for (std::size_t i = 0; i < n; ++i)
            out[i] += dnormv(x[i], mu, beta) * pj;
    }
    return out;
}

// Fill the (n x m) column-major normal-pdf kernel:
// `out[j*n + i] = N(x[i]; mu0[j], beta)`.
inline void kmat_norm(const std::vector<double>& x,
                      const std::vector<double>& mu0, double beta,
                      std::vector<double>& out) {
    const std::size_t n = x.size();
    const std::size_t m = mu0.size();
    out.assign(n * m, 0.0);
    const double b2 = beta * beta;
    const double base = stats::LN_SQRT_2PI + std::log(beta);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        double* col = out.data() + j * n;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = x[i] - mu;
            col[i] = std::exp(d * d / (-2.0 * b2) - base);
        }
    }
}

// `pnpnorm_(x, mu0, pi0, beta)` (non-log): the normal-cdf mixture
// `sum_j pi0[j] Phi(x[i]; mu0[j], beta)`.
inline std::vector<double> pnpnorm(const std::vector<double>& x,
                                   const std::vector<double>& mu0,
                                   const std::vector<double>& pi0,
                                   double beta) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        const double pj = pi0[j];
        for (std::size_t i = 0; i < n; ++i)
            out[i] += stats::pnorm(x[i], mu, beta, true) * pj;
    }
    return out;
}

// ---------------------------------------------------------------------------
// one-parameter normal (sample correlations)
// ---------------------------------------------------------------------------

// `dnormcarray(x, mu, n)` (non-log) for a single support point.
inline double dnpnormc_single(double x, double mu, double n) {
    const double stdev = (1.0 - mu * mu) / std::sqrt(n);
    const double d = (x - mu) / stdev;
    return std::exp(-0.5 * d * d - stats::LN_SQRT_2PI - std::log(stdev));
}

// `dnpnormc_(x, mu0, pi0, n)` (non-log): the one-parameter normal pdf
// mixture at all data points.
inline std::vector<double> dnpnormc(const std::vector<double>& x,
                                    const std::vector<double>& mu0,
                                    const std::vector<double>& pi0,
                                    double n) {
    const std::size_t len = x.size();
    if (mu0.empty())
        return std::vector<double>(len, 0.0);
    const std::size_t m = mu0.size();
    std::vector<double> out(len, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double pj = pi0[j];
        const double mu = mu0[j];
        for (std::size_t i = 0; i < len; ++i)
            out[i] += dnpnormc_single(x[i], mu, n) * pj;
    }
    return out;
}

// Fill the (n x m) column-major one-parameter normal pdf kernel:
// `out[j*n + i] = dnpnormc_single(x[i]; mu0[j], n)`.
inline void kmat_normc(const std::vector<double>& x,
                       const std::vector<double>& mu0, double n,
                       std::vector<double>& out) {
    const std::size_t len = x.size();
    const std::size_t m = mu0.size();
    out.assign(len * m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        double* col = out.data() + j * len;
        for (std::size_t i = 0; i < len; ++i)
            col[i] = dnpnormc_single(x[i], mu, n);
    }
}

// ---------------------------------------------------------------------------
// t (non-central)
// ---------------------------------------------------------------------------

// `dnpt_(x, mu0, pi0, beta)` (non-log): the non-central-t mixture density
// sum_j pi0[j] dt(x[i], df = beta, ncp = mu0[j]) (the `nptll` kernel; the
// support point is the non-centrality, and beta = Inf collapses to the
// normal with mean mu0[j]).
inline std::vector<double> dnpt(const std::vector<double>& x,
                                const std::vector<double>& mu0,
                                const std::vector<double>& pi0,
                                double beta) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        const double pj = pi0[j];
        for (std::size_t i = 0; i < n; ++i)
            out[i] += stats::dnt(x[i], beta, mu) * pj;
    }
    return out;
}

// Fill the (n x g) column-major non-central-t kernel:
// `out[j*n + i] = dt(x[i], df = beta, ncp = mu0[j])`.
inline void kmat_t(const std::vector<double>& x,
                   const std::vector<double>& mu0, double beta,
                   std::vector<double>& out) {
    const std::size_t n = x.size();
    const std::size_t m = mu0.size();
    out.assign(n * m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        double* col = out.data() + j * n;
        for (std::size_t i = 0; i < n; ++i)
            col[i] = stats::dnt(x[i], beta, mu);
    }
}

// ---------------------------------------------------------------------------
// poisson
// ---------------------------------------------------------------------------

// C++ `dpoisarray` pmf, exactly replicated: the log-density is
// `(x + mu > 0) ? x*ln(mu) - mu - lgamma(x+1) : 0`, then the whole matrix is
// `.exp()`-ed — so the non-log else-branch is `exp(0) = 1`, not 0 (in
// particular `dpois(0; 0) = 1`, the Poisson limit as mu -> 0). Matching this
// matters: the estpi0 sp0 ratio divides by this, and a zero at mu = 0
// corrupts the whole bisection.
inline double pois_pmf_c(double x, double mu) {
    if (x + mu > 0.0)
        return std::exp(x * std::log(mu) - mu - stats::gammln(x + 1.0));
    return 1.0;
}

// `dnppois_(x, mu0, pi0)` (non-log): the Poisson pmf mixture at all data
// points.
inline std::vector<double> dnppois(const std::vector<double>& x,
                                   const std::vector<double>& mu0,
                                   const std::vector<double>& pi0) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double pj = pi0[j];
        const double mu = mu0[j];
        for (std::size_t i = 0; i < n; ++i)
            out[i] += pois_pmf_c(x[i], mu) * pj;
    }
    return out;
}

// Fill the (n x m) column-major Poisson pmf kernel:
// `out[j*n + i] = pois_pmf_c(x[i], mu0[j])`.
inline void kmat_pois(const std::vector<double>& x,
                      const std::vector<double>& mu0,
                      std::vector<double>& out) {
    const std::size_t n = x.size();
    const std::size_t m = mu0.size();
    out.assign(n * m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        double* col = out.data() + j * n;
        for (std::size_t i = 0; i < n; ++i)
            col[i] = pois_pmf_c(x[i], mu);
    }
}

}  // namespace kern
}  // namespace npfc

#endif  // NPFIC_KERNELS_H
