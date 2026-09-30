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

#include <Eigen/Dense>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace npfc {
namespace kern {

// Scalar helpers from `npfc::stats` used unqualified in the log-branch CDFs
// and the binned kernels below (this header includes `npfc_stats.h`).
using stats::expm1_tiny;
using stats::nanv;

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
        pinned_.clear();
        total_bytes_ = 0;
        evictions_ = 0;
        fresh_ = 0;
        fresh_ms_ = 0.0;
    }

    // Mark a column pinned (never evicted by the byte cap). The grid
    // columns are pinned: the per-iteration grid sweep and the solver's
    // endpoint checks (`gp[0]`, `gp[last]`) read them by exact match on
    // every run phase, and their memory is already paid for by the eager
    // grid matrix the families build anyway.
    void pin(double mu) { pinned_.insert(mu); }

    // Byte cap on the EVICTABLE (non-pinned) columns (default 512 MiB;
    // 0 = unbounded). `estpi0`'s bisection drives ~15 full solver runs and
    // each accumulates fresh interior columns (brmin/dfmin points), so an
    // unbounded cache would grow with the data size x the run count. An
    // evicted column is recomputable: `column` re-evaluates the dropped
    // `mu` with the identical reference expression, so eviction trades
    // wall time for memory and never changes a result.
    void set_evictable_cap(std::size_t bytes) { evictable_cap_ = bytes; }

    // The n-vector column K[:, mu]; computed on first use, then O(1). The
    // map is `mutable`, so this const accessor may populate it.
    //
    // The returned reference must not be held across a later `column`
    // call: a later call may evict another entry (callers in this package
    // consume each column within its own expression).
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
        const std::size_t bytes = n_ * sizeof(double);
        // Trim BEFORE inserting so the just-computed column can never be
        // its own victim.
        if (evictable_cap_)
            trim_for(bytes);
        auto r = cols_.emplace(mu, std::move(col));
        if (!pinned_.count(mu))
            total_bytes_ += bytes;
        return r.first->second;
    }

    // Seed a column that was already produced (e.g. the eager grid matrix).
    void insert(double mu, std::vector<double> col) {
        const std::size_t bytes = col.size() * sizeof(double);
        if (evictable_cap_)
            trim_for(bytes);
        cols_.emplace(mu, std::move(col));
        if (!pinned_.count(mu))
            total_bytes_ += bytes;
    }

    // Drop evictable (non-pinned) columns until a `bytes`-sized new
    // column fits under the cap. Pinned columns are never touched; if
    // only pinned columns remain the cap is effectively waived.
    void trim_for(std::size_t bytes) const {
        while (total_bytes_ + bytes > evictable_cap_) {
            auto victim = cols_.end();
            for (auto it = cols_.begin(); it != cols_.end(); ++it)
                if (!pinned_.count(it->first)) {
                    victim = it;
                    break;
                }
            if (victim == cols_.end())
                return;
            total_bytes_ -= n_ * sizeof(double);
            evictions_ += 1;
            cols_.erase(victim);
        }
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

    // Byte-cap diagnostics: current evictable-column footprint and the
    // number of evictions so far.
    std::size_t bytes_used() const { return total_bytes_; }
    std::size_t eviction_count() const { return evictions_; }

private:
    const std::vector<double>* x_ = nullptr;
    std::size_t n_ = 0;
    Eval eval_ = [](double, double) { return 0.0; };
    mutable std::unordered_map<double, std::vector<double>> cols_;
    mutable std::unordered_set<double> pinned_;
    mutable std::size_t total_bytes_ = 0;
    mutable std::size_t evictions_ = 0;
    std::size_t evictable_cap_ = 512 * 1024 * 1024;  // 512 MiB
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

// ---------------------------------------------------------------------------
// binned (discrete) kernels — ports of npfixedcomp2/inst/include/miscfuns.h
// ---------------------------------------------------------------------------
//
// The binned families (npnormllw / npnormcvmw / npnormadw / nptllw) replace
// the continuous densities with binned (trapezoid / cdf-difference) versions
// on a grid of step `h`. These are their kernels; every expression mirrors
// the R package's C++ exactly, including the log-space `logspacesub` NaN
// handling (NaN -> -100) the t branch relies on.

// Fill an (n x m) column-major matrix from a per-row closure (serial; rows
// are independent). Local copy of the families' `detail::mat_fill` (this
// header is included before `npfc_families.h`).
namespace bfill {
template <typename RowFill>
inline Eigen::MatrixXd fill(std::size_t n, std::size_t m, RowFill f) {
    std::vector<double> raw(n * m, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        f(i, raw.data() + i * m);
    return Eigen::Map<Eigen::MatrixXd>(raw.data(), m, n).transpose();
}
}  // namespace bfill

// R's `pnorm` with `log.p = TRUE` (the Cody `pnorm_both` log branch, ported
// from src/nmath/pnorm.c). The plain `pnorm` here is the gamma route; the
// two agree to ~1e-16, which is the parity band for these kernels — the
// log branch is the exact R route for the binned CDFs, so this is the one
// that mirrors R bit-for-bit in structure.
inline double pnorm_log(double x, double mu, double sd, bool lower_tail) {
    if (std::isnan(x) || std::isnan(mu) || std::isnan(sd))
        return nanv();
    if (sd <= 0.0)
        return (x < mu) ? (lower_tail ? 0.0 : 1.0) : (lower_tail ? 1.0 : 0.0);
    double p = (x - mu) / sd;
    if (!isfinite(p))
        return (x < mu) ? (lower_tail ? 0.0 : 1.0) : (lower_tail ? 1.0 : 0.0);
    const double y = std::fabs(p);
    const double eps = DBL_EPSILON * 0.5;
    constexpr double M_SQRT_32 = 5.6568542494923802;  // sqrt(32)
    constexpr double M_1_SQRT_2PI = 0.7978845608028654;
    constexpr double a[5] = {2.2352520354606839287, 161.02823106855587881,
                             1067.6894854603709582, 18154.981253343561249,
                             0.065682337918207449113};
    constexpr double b[4] = {47.20258190468824187, 976.09855173777669322,
                             10260.932208618978205, 45507.789335026729956};
    constexpr double c[9] = {0.39894151208813466764, 8.8831497943883759412,
                             93.506656132177855979, 597.27027639480026226,
                             2494.5375852903726711, 6848.1904505362823326,
                             11602.651437647350124, 9842.7148383839780218,
                             1.0765576773720192317e-8};
    constexpr double d[8] = {22.266688044328115691, 235.38790178262499861,
                             1519.377599407554805, 6485.558298266760755,
                             18615.571640885098091, 34900.952721145977266,
                             38912.003286093271411, 19685.429676859990727};
    constexpr double pp[6] = {0.21589853405795699, 0.1274011611602473639,
                              0.022235277870649807, 0.001421619193227893466,
                              2.9112874951168792e-5, 0.02307344176494017303};
    constexpr double qq[5] = {1.28426009614491121, 0.468238212480865118,
                              0.0659881378689285515, 0.00378239633202758244,
                              7.29751555083966205e-5};
    double cum = 0.0, ccum = 0.0, xnum = 0.0, xden = 0.0, temp = 0.0;
    double xsq = 0.0, del = 0.0;
    const bool lower = lower_tail;
    const bool upper = !lower_tail;
    if (y <= 0.67448975) {
        if (y > eps) {
            xsq = p * p;
            xnum = a[4] * xsq;
            xden = xsq;
            for (int i = 0; i < 3; ++i) {
                xnum = (xnum + a[i]) * xsq;
                xden = (xden + b[i]) * xsq;
            }
        }
        temp = p * (xnum + a[3]) / (xden + b[3]);
        if (lower)
            cum = 0.5 + temp;
        if (upper)
            ccum = 0.5 - temp;
        if (lower)
            cum = std::log(cum);
        if (upper)
            ccum = std::log(ccum);
    } else if (y <= M_SQRT_32) {
        xnum = c[8] * y;
        xden = y;
        for (int i = 0; i < 7; ++i) {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        temp = (xnum + c[7]) / (xden + d[7]);
        const double sixteen = 16.0;
        xsq = std::trunc(y * sixteen) / sixteen;
        del = (y - xsq) * (y + xsq);
        cum = (-xsq * xsq * 0.5) + (-del * 0.5) + std::log(temp);
        if ((lower && p > 0.0) || (upper && p <= 0.0))
            ccum = std::log1p(-std::exp(-xsq * xsq * 0.5) *
                              std::exp(-del * 0.5) * temp);
        if (p > 0.0)
            std::swap(cum, ccum);  // R's `swap_tail`
    } else if (y < 1e170 || (lower && -37.5193 < p && p < 8.2924) ||
               (upper && -8.2924 < p && p < 37.5193)) {
        xsq = 1.0 / (p * p);
        xnum = pp[5] * xsq;
        xden = xsq;
        for (int i = 0; i < 4; ++i) {
            xnum = (xnum + pp[i]) * xsq;
            xden = (xden + qq[i]) * xsq;
        }
        temp = xsq * (xnum + pp[4]) / (xden + qq[4]);
        temp = (M_1_SQRT_2PI - temp) / y;
        const double sixteen = 16.0;
        xsq = std::trunc(p * sixteen) / sixteen;
        del = (p - xsq) * (p + xsq);
        cum = (-xsq * xsq * 0.5) + (-del * 0.5) + std::log(temp);
        if ((lower && p > 0.0) || (upper && p <= 0.0))
            ccum = std::log1p(-std::exp(-xsq * xsq * 0.5) *
                              std::exp(-del * 0.5) * temp);
        if (p > 0.0)
            std::swap(cum, ccum);  // R's `swap_tail`
    } else {
        if (p > 0.0) {
            cum = 0.0;
            ccum = -std::numeric_limits<double>::infinity();
        } else {
            cum = -std::numeric_limits<double>::infinity();
            ccum = 0.0;
        }
    }
    return lower_tail ? cum : ccum;
}

// R's `pt` (central) with `log.p = TRUE` (src/nmath/pt.c log branch).
inline double pt_log(double x, double df, bool lower_tail) {
    if (std::isnan(x) || std::isnan(df))
        return nanv();
    if (df <= 0.0)
        return nanv();
    if (!isfinite(x))
        return (x < 0.0) ? (lower_tail ? 0.0 : 1.0) : (lower_tail ? 1.0 : 0.0);
    if (!isfinite(df))
        return pnorm_log(x, 0.0, 1.0, lower_tail);
    const double nx = 1.0 + (x / df) * x;
    double lval;
    if (nx > 1e100) {
        // R's underflow-guard branch (A&S 26.5.4): log pbeta via the
        // asymptotic `z^a / (a B(a, b))`.
        lval = -0.5 * df * (2.0 * std::log(std::fabs(x)) - std::log(df)) -
               (stats::gammln(0.5 * df) + stats::gammln(0.5) -
                stats::gammln(0.5 * df + 0.5)) -
               std::log(0.5 * df);
    } else {
        // R's `pbeta(q, 0.5, df/2, lower = FALSE, log_p = TRUE)` =
        // log(1 - I_q(0.5, df/2)) in the `df > x*x` branch, and
        // `pbeta(1/nx, df/2, 0.5, lower = TRUE, log_p = TRUE)` =
        // log(I_{1/nx}(df/2, 0.5)) otherwise.
        const double pb = (df > x * x)
                              ? std::log1p(-stats::incbeta(0.5, df / 2.0,
                                                           x * x /
                                                               (df + x * x)))
                              : std::log(stats::incbeta(df / 2.0, 0.5, 1.0 / nx));
        lval = pb;
    }
    if (x <= 0.0)
        lower_tail = !lower_tail;
    if (lower_tail)
        return std::log1p(-0.5 * std::exp(lval));
    return lval - stats::LN2_C;
}

// R's `pnt` (non-central t, AS 243) with `log.p = TRUE`: the twin-series
// accumulation is identical to `stats::pnt`, but the result is returned in
// log space (R's `R_DT_val` log branch: `log1p(-cc)`/`log(c)`/0/-inf).
inline double pnt_log(double t, double df, double ncp, bool lower_tail) {
    constexpr int ITRMAX = 1000;
    constexpr double ERRMAX = 1e-12;
    constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
    constexpr double M_LN_SQRT_PI = 0.5723649429247001;
    // R: `del*del > 2*M_LN2*(-(DBL_MIN_EXP))`; `DBL_MIN_EXP` = -1022. Must
    // match the parity-verified non-log `pnt` in npfc_stats.h exactly (same
    // AS 243 algorithm, just log-space accumulation).
    constexpr double DEL2_MAX = 2.0 * 0.6931471805599453 * 1022.0;

    if (df <= 0.0)
        return nanv();
    if (ncp == 0.0)
        return pt_log(t, df, lower_tail);
    if (!isfinite(t))
        return (t < 0.0) ? (lower_tail ? 0.0 : 1.0) : (lower_tail ? 1.0 : 0.0);

    bool negdel;
    double tt, del;
    if (t >= 0.0) {
        negdel = false;
        tt = t;
        del = ncp;
    } else {
        if (ncp > 40.0 && (!lower_tail))
            return 0.0;
        negdel = true;
        tt = -t;
        del = -ncp;
    }

    if (df > 4e5 || del * del > DEL2_MAX) {
        const double s = 1.0 / (4.0 * df);
        return pnorm_log(tt * (1.0 - s), del, std::sqrt(1.0 + tt * tt * 2.0 * s),
                         lower_tail != negdel);
    }

    double tnc;
    double p = 0.0;

    double x = t * t;
    const double rxb0 = df / (x + df);
    x = x / (x + df);
    if (x > 0.0) {
        const double lambda = del * del;
        p = 0.5 * std::exp(-0.5 * lambda);
        if (p == 0.0)  // R: underflow warning, R_DT_0 = -inf in log space
            return -std::numeric_limits<double>::infinity();
        double xodd, xeven, a, b;
        double q = SQRT_2_OVER_PI * p * del;
        double s = 0.5 - p;
        if (s < 1e-7)
            s = -0.5 * expm1_tiny(-0.5 * lambda);
        a = 0.5;
        b = 0.5 * df;
        const double rxb = std::pow(rxb0, b);
        const double albeta = M_LN_SQRT_PI + stats::gammln(b) -
                              stats::gammln(0.5 + b);
        xodd = stats::incbeta(a, b, x);
        double godd = 2.0 * rxb * std::exp(a * std::log(x) - albeta);
        const double tnc0 = b * x;
        xeven = (tnc0 < DBL_EPSILON) ? tnc0 : (1.0 - rxb);
        double geven = tnc0 * rxb;
        tnc = p * xodd + q * xeven;

        int it = 0;
        while (it < ITRMAX) {
            it += 1;
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * it);
            q *= lambda / (2.0 * it + 1.0);
            tnc += p * xodd + q * xeven;
            s -= p;
            if (s < -1e-10)
                break;
            if (s <= 0.0 && it > 1)
                break;
            const double errbd = 2.0 * s * (xodd - godd);
            if (std::abs(errbd) < ERRMAX)
                break;
        }
    } else {
        tnc = 0.0;
    }

    tnc += stats::pnorm(-del, 0.0, 1.0, true);
    const double capped = std::min(tnc, 1.0);
    const double pval = (lower_tail != negdel) ? capped : (1.0 - capped);
    if (pval <= 0.0)
        return -std::numeric_limits<double>::infinity();
    if (pval >= 1.0)
        return 0.0;
    return std::log(pval);
}

// `pdiscnormarray(x, mu0, stdev, h)`: the R CDF-difference form of the
// binned normal cdf, `Phi(x; mu0 - h, beta) - Phi(x; mu0, beta)`. R applies
// the shift in the mu direction when `n > m` and in the x direction
// (`x + h`, mu untouched) otherwise — mathematically identical, but the two
// orderings are not bit-identical, so the size branch is mirrored exactly.
// Returns the (n x m) column-major CDF matrix.
inline Eigen::MatrixXd pnorm_disc_m(const std::vector<double>& x,
                                    const std::vector<double>& mu0,
                                    double beta, double h) {
    const std::size_t m = mu0.size();
    const bool shift_mu = x.size() > m;
    return bfill::fill(x.size(), m,
                       [&x, &mu0, m, beta, h,
                        shift_mu](std::size_t i, double* row) {
                           for (std::size_t j = 0; j < m; ++j)
                               row[j] = shift_mu
                                           ? stats::pnorm(x[i], mu0[j] - h, beta,
                                                          true)
                                           : stats::pnorm(x[i] + h, mu0[j], beta,
                                                          true);
                       });
}

// `pnpdiscnorm_(x, mu0, pi0, stdev, h)` (non-log): the binned normal-cdf
// mixture, `pdiscnormarray(x, mu0, beta, h) * pi0` (R's matrix x vector
// route; i-outer/j-inner accumulation mirrors the per-row dot product).
inline std::vector<double> pnpdiscnorm(const std::vector<double>& x,
                                       const std::vector<double>& mu0,
                                       const std::vector<double>& pi0,
                                       double beta, double h) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    const Eigen::MatrixXd M = pnorm_disc_m(x, mu0, beta, h);
    std::vector<double> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < m; ++j)
            s += M(i, j) * pi0[j];
        out[i] = s;
    }
    return out;
}

// `ddiscnormarray(x, mu0, stdev, h)` (non-log, (n x m) column-major): the
// trapezoid-rule binned normal density
// `((N(mu) + N(mu - h)) / 2 + sum_{k=1}^{N-1} N(mu - k*h/N)) * h/N` with
// `N = max(ceil(range * 1e3 * h^1.5), 5)`. R's scalar-mu overload uses
// `range = max_i |x_i - mu0|`; its vectorised overload uses
// `max(xmax - mumin, mumax - xmin)`; the two coincide for `m == 1`, so one
// formula covers both.
inline Eigen::MatrixXd ddiscnorm_m(const std::vector<double>& x,
                                   const std::vector<double>& mu0,
                                   double beta, double h) {
    const std::size_t n = x.size();
    const std::size_t m = mu0.size();
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double mumin = std::numeric_limits<double>::infinity();
    double mumax = -std::numeric_limits<double>::infinity();
    for (double v : x) {
        xmin = std::min(xmin, v);
        xmax = std::max(xmax, v);
    }
    for (double v : mu0) {
        mumin = std::min(mumin, v);
        mumax = std::max(mumax, v);
    }
    int N = static_cast<int>(std::max(
        std::ceil(std::max(xmax - mumin, mumax - xmin) * 1e3 *
                         std::pow(h, 1.5)),
        5.0));
    const double delta = h / N;
    return bfill::fill(n, m, [&x, &mu0, m, N, delta, beta, h](
                                       std::size_t i, double* row) {
        const double xi = x[i];
        for (std::size_t j = 0; j < m; ++j) {
            const double mu = mu0[j];
            double ans = (kern::dnormv(xi, mu, beta) +
                          kern::dnormv(xi, mu - h, beta)) *
                         0.5;
            for (int k = 1; k < N; ++k)
                ans += kern::dnormv(xi, mu - delta * k, beta);
            row[j] = ans * delta;
        }
    });
}

// `dnpdiscnorm_(x, mu0, pi0, stdev, h)` (non-log): the binned normal-density
// mixture, `ddiscnormarray(x, mu0, beta, h) * pi0` (R's matrix x vector
// route; i-outer/j-inner accumulation).
inline std::vector<double> dnpdiscnorm(const std::vector<double>& x,
                                       const std::vector<double>& mu0,
                                       const std::vector<double>& pi0,
                                       double beta, double h) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    const Eigen::MatrixXd M = ddiscnorm_m(x, mu0, beta, h);
    std::vector<double> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < m; ++j)
            s += M(i, j) * pi0[j];
        out[i] = s;
    }
    return out;
}

// `ddisctarray(x, mu, df, h)` (non-log, single support point): the binned
// non-central-t density `exp(logspacesub(pt(x + h, df, mu), pt(x, df, mu)))`
// with the R NaN -> -100 guard (log space) before the exp.
inline double ddisct_v(double x, double df, double mu, double h) {
    const double lx = pnt_log(x + h, df, mu, true);
    const double ly = pnt_log(x, df, mu, true);
    double l = stats::logspacesub(lx, ly);
    if (std::isnan(l))
        l = -100.0;
    return std::exp(l);
}

// `ddisctarray(x, mu0, df, h)` (non-log, (n x m) column-major).
inline Eigen::MatrixXd ddisct_m(const std::vector<double>& x,
                                const std::vector<double>& mu0, double df,
                                double h) {
    const std::size_t m = mu0.size();
    return bfill::fill(x.size(), m, [&x, &mu0, m, df, h](std::size_t i, double* row) {
        for (std::size_t j = 0; j < m; ++j)
            row[j] = ddisct_v(x[i], df, mu0[j], h);
    });
}

// `dnpdisct_(x, mu0, pi0, df, h)` (non-log): the binned non-central-t
// mixture `sum_j pi0[j] Ddisc_t(x[i]; mu0[j], df, h)`.
inline std::vector<double> dnpdisct(const std::vector<double>& x,
                                    const std::vector<double>& mu0,
                                    const std::vector<double>& pi0,
                                    double df, double h) {
    const std::size_t n = x.size();
    if (mu0.empty())
        return std::vector<double>(n, 0.0);
    const std::size_t m = mu0.size();
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        const double mu = mu0[j];
        const double pj = pi0[j];
        for (std::size_t i = 0; i < n; ++i)
            out[i] += ddisct_v(x[i], df, mu, h) * pj;
    }
    return out;
}

// `ptarray(x, mu, df, log.p = TRUE)` (n x m column-major): the non-central-t
// CDF matrix in log space, the R `pnt_functor` route.
inline Eigen::MatrixXd ptarray_log(const std::vector<double>& x,
                                   const std::vector<double>& mu0, double df) {
    const std::size_t m = mu0.size();
    return bfill::fill(x.size(), m,
                            [&x, &mu0, m, df](std::size_t i, double* row) {
                                for (std::size_t j = 0; j < m; ++j)
                                    row[j] = pnt_log(x[i], df, mu0[j], true);
                            });
}

}  // namespace kern
}  // namespace npfc

#endif  // NPFIC_KERNELS_H
