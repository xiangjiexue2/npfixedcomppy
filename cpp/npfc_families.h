// npfc_families.h — the six mixing-distribution families (a faithful port of
// npfixedcomppy/src/families.rs, which ports the C++ `npfixedcomp2` family
// classes):
//   NpNormLL  (flag d1)  normal mixture, maximum likelihood
//   NpTLL     (flag d0)  non-central-t mixture, maximum likelihood
//   NpNormCVM (flag d1)  normal mixture, Cramer-von Mises distance
//   NpNormAD  (flag d1)  normal mixture, Anderson-Darling distance
//   NpNormCLL (flag d0)  one-parameter normal mixture (sample correlations)
//   NpPoisLL  (flag d0)  Poisson mixture, weighted maximum likelihood
//
// Threading: the per-observation loops here are serial (the bit-identical
// reference order). All multi-threading in the package comes from Eigen's
// own compile-time-gated parallel GEMM/GEMV (EIGEN_HAS_OPENMP, enabled by
// the build's /openmp flag) in the mapping/computeweights hot paths — there
// are no hand-written OpenMP loops in this package.
#ifndef NPFIC_FAMILIES_H
#define NPFIC_FAMILIES_H

#include "npfc_engine.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

namespace npfc {
namespace fam {

namespace detail {

// `par3`: sum of 3 per-index contributions (serial).
inline std::array<double, 3>
par3(std::size_t n,
     const std::function<std::array<double, 3>(std::size_t)>& f) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto b = f(i);
        a0 += b[0];
        a1 += b[1];
        a2 += b[2];
    }
    return {a0, a1, a2};
}

// `par_sum1`: plain sum of `f(i)` (serial).
template <typename F>
inline double par_sum1(std::size_t n, F f) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += f(i);
    return s;
}

// `par_acc`: one pass over `i in 0..n` accumulating `k` sums.
inline std::vector<double>
par_acc(std::size_t n, std::size_t k,
        const std::function<void(std::size_t, double*)>& f) {
    std::vector<double> acc(k, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        f(i, acc.data());
    return acc;
}

// Fill an (n x m) column-major matrix from a per-row closure (serial; rows
// are independent).
template <typename RowFill>
inline Eigen::MatrixXd mat_fill(std::size_t n, std::size_t m, RowFill f) {
    std::vector<double> raw(n * m, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        f(i, raw.data() + i * m);
    return Eigen::Map<Eigen::MatrixXd>(raw.data(), m, n).transpose();
}

// `dnorm_m` — the (n x m) column-major normal-pdf matrix
// `out[i + j*m] = N(x[i]; mu0[j], beta)`.
inline Eigen::MatrixXd dnorm_m(const std::vector<double>& x,
                               const std::vector<double>& mu0, double beta) {
    const std::size_t m = mu0.size();
    const double b2 = beta * beta;
    const double base = stats::LN_SQRT_2PI + std::log(beta);
    return mat_fill(x.size(), m, [&x, &mu0, m, b2, base](std::size_t i, double* row) {
        const double xi = x[i];
        for (std::size_t j = 0; j < m; ++j) {
            const double d = xi - mu0[j];
            row[j] = std::exp(d * d / (-2.0 * b2) - base);
        }
    });
}

// `pnorm_m` — the (n x m) column-major normal-cdf matrix.
inline Eigen::MatrixXd pnorm_m(const std::vector<double>& x,
                               const std::vector<double>& mu0, double beta) {
    const std::size_t m = mu0.size();
    return mat_fill(x.size(), m, [&x, &mu0, m, beta](std::size_t i, double* row) {
        const double xi = x[i];
        for (std::size_t j = 0; j < m; ++j)
            row[j] = stats::pnorm(xi, mu0[j], beta, true);
    });
}

// Build the (n x m) normal-pdf kernel `K[i, j] = dnormv(x[i], mu[j], beta)`.
// This is the single, bounds-checked materialisation seam for every normal
// family hot path: the matrix is resized by Eigen (so no index can stray off
// the buffer), each element is written exactly once, and the per-column
// `exp` is a pure SIMD expression. The column loop is serial; the per-column
// SIMD exp is the whole of the vectorisation.
inline Eigen::MatrixXd norm_kernel(const std::vector<double>& x,
                                   const std::vector<double>& mu, double beta) {
    const Eigen::Index n = static_cast<Eigen::Index>(x.size());
    const Eigen::Index m = static_cast<Eigen::Index>(mu.size());
    Eigen::MatrixXd K(n, m);
    if (n == 0 || m == 0)
        return K;
    const double b2 = beta * beta;
    const double base = stats::LN_SQRT_2PI + std::log(beta);
    const Eigen::Map<const Eigen::VectorXd> dx(x.data(), n);
    for (Eigen::Index j = 0; j < m; ++j) {
        // Element-wise exp lives in the ARRAY domain (`.array().exp()`);
        // assigning through `K.col(j).array()` keeps the whole expression in
        // the array domain and performs the single bounds-checked write of
        // column j (Eigen SIMD-ises the exp).
        K.col(j).array() = (-0.5 * (dx.array() - mu[j]).square() / b2 - base)
                               .array()
                               .exp();
    }
    return K;
}

// `norm_kernel_scaled` — like `norm_kernel`, but every row i is additionally
// divided by `row_scale[i]`. One column-parallel pass fills
// K[i, j] = dnorm(x[i]; mu[j], beta) / row_scale[i]; the division is part of
// the fill, so there is no second materialisation of the (n x m) matrix.
inline Eigen::MatrixXd norm_kernel_scaled(const std::vector<double>& x,
                                          const std::vector<double>& mu,
                                          double beta,
                                          const std::vector<double>& row_scale) {
    const Eigen::Index n = static_cast<Eigen::Index>(x.size());
    const Eigen::Index m = static_cast<Eigen::Index>(mu.size());
    Eigen::MatrixXd K(n, m);
    if (n == 0 || m == 0)
        return K;
    const double b2 = beta * beta;
    const double base = stats::LN_SQRT_2PI + std::log(beta);
    const Eigen::Map<const Eigen::VectorXd> dx(x.data(), n);
    // Plain reciprocal pass (the vendored Eigen build lacks the scalar/vector
    // `operator/`); O(n), trivially SIMD, run once per call.
    std::vector<double> inv_rs(n);
    for (Eigen::Index i = 0; i < n; ++i)
        inv_rs[i] = 1.0 / row_scale[i];
    const Eigen::Map<const Eigen::VectorXd> inv_rsv(inv_rs.data(), n);
    for (Eigen::Index j = 0; j < m; ++j) {
        // The row scale is folded into the same array-domain pass.
        K.col(j).array() =
            (-0.5 * (dx.array() - mu[j]).square() / b2 - base).array().exp() *
            inv_rsv.array();
    }
    return K;
}

// Materialise the (n x m) column-major kernel matrix from a column cache:
// column j is the cached (reference-ordered) column at mu[j]. BIT-IDENTICAL
// to a fresh fill — every element is exactly the reference per-row
// evaluation, shared across all consumers of the same support value; only
// the redundant re-evaluation is removed, never the arithmetic or its
// accumulation order.
inline Eigen::MatrixXd kmat_cached(const kern::KernelColumnCache& kc,
                                   const std::vector<double>& mu) {
    const Eigen::Index n = static_cast<Eigen::Index>(kc.n());
    const Eigen::Index m = static_cast<Eigen::Index>(mu.size());
    Eigen::MatrixXd K(n, m);
    for (Eigen::Index j = 0; j < m; ++j)
        std::memcpy(K.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n),
                    kc.column(mu[j]).data(),
                    static_cast<std::size_t>(n) * sizeof(double));
    return K;
}

// Like `kmat_cached`, but row i is additionally divided by `row_scale[i]`
// (the weights `tp = sp / fp` fill, computed once per column instead of
// re-evaluating the kernel).
inline Eigen::MatrixXd kmat_cached_scaled(const kern::KernelColumnCache& kc,
                                          const std::vector<double>& mu,
                                          const std::vector<double>& row_scale) {
    const std::size_t n = kc.n();
    const std::size_t m = mu.size();
    Eigen::MatrixXd K(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(m));
    for (std::size_t j = 0; j < m; ++j) {
        const std::vector<double>& col = kc.column(mu[j]);
        double* dst = K.data() + j * n;
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = col[i] / row_scale[i];
    }
    return K;
}

// Like `kmat_cached`, but row i is additionally multiplied by the
// precomputed reciprocal `1.0 / row_scale[i]` (the `tp = sp / fp` fill in
// its reciprocal-multiply form, matching `norm_kernel_scaled`: the same
// reciprocal pass over the rows, then the same single-rounded
// `column * reciprocal` multiply per element). Each cached column is a
// bit-identical copy of the reference per-row kernel evaluation, so the
// filled matrix is bit-identical to a fresh `norm_kernel_scaled` build —
// only the redundant re-evaluation of the kernel is removed.
inline Eigen::MatrixXd kmat_cached_recip(const kern::KernelColumnCache& kc,
                                         const std::vector<double>& mu,
                                         const std::vector<double>& row_scale) {
    const std::size_t n = kc.n();
    const std::size_t m = mu.size();
    Eigen::MatrixXd K(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(m));
    // The reciprocal pass, exactly as `norm_kernel_scaled` computes it
    // (once per call, O(n)).
    std::vector<double> inv_rs(n);
    for (std::size_t i = 0; i < n; ++i)
        inv_rs[i] = 1.0 / row_scale[i];
    for (std::size_t j = 0; j < m; ++j) {
        const std::vector<double>& col = kc.column(mu[j]);
        double* dst = K.data() + j * n;
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = col[i] * inv_rs[i];
    }
    return K;
}

// A (n x m) kernel-matrix fill keyed on the exact `mu` vector. The (sorted)
// grid matrix is pinned once by the family's `prepare`; every other fill is
// memoised in a single slot. `get` returns a matrix BIT-IDENTICAL to
// `fill(mu)` — a stored matrix is served only when `mu` is element-exact to
// one already computed. That is the only safe key for the binned normal
// kernels: the trapezoid `N` and the CDF-shift direction both depend on the
// whole support set, so no weaker key may be used. The outer loop re-requests
// exactly a small set of repeating vectors (the grid, the current support
// set, a few solver-interior points), so after the first iteration nearly
// every call is a hit and only redundant re-evaluation is removed — never
// arithmetic.
struct KernelMemo {
    template <typename Fill>
    const Eigen::MatrixXd& get(const std::vector<double>& mu, Fill fill) const {
        if (!grid.empty() && mu == grid)
            return grid_m;
        if (last_mu == mu)
            return last_m;
        last_mu = mu;
        last_m = fill(mu);
        return last_m;
    }
    mutable std::vector<double> grid, last_mu;
    mutable Eigen::MatrixXd grid_m, last_m;
};

// The shared empty (0 x 0) matrix for lvalue-bound ternaries
// (`cond ? pmat(mu) : detail::empty_m()`): a prvalue operand would dangle
// the bound reference.
inline const Eigen::MatrixXd& empty_m() {
    static const Eigen::MatrixXd k(0, 0);
    return k;
}

// Copy a dense Eigen vector into an owning std::vector (the seam for results
// crossing back into the std::vector-based family API).
inline std::vector<double> to_vec(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

// Zero-copy read-only Eigen view over a std::vector (valid while the vector
// is alive; use only inside expressions that evaluate immediately).
inline Eigen::Map<const Eigen::VectorXd> to_eigen(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(
        v.data(), static_cast<Eigen::Index>(v.size()));
}

}  // namespace detail

// ===========================================================================
// NpNormLL — normal mixing distribution, maximum likelihood
// ===========================================================================

class NpNormLL : public Family {
public:
    NpNormLL(std::vector<double> data, std::vector<double> mu0fixed,
             std::vector<double> pi0fixed, double beta)
        : data_(std::move(data)),
          len_(data_.size()),
          beta_(beta),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          precompute_(kern::dnpnorm(data_, mu0fixed_, pi0fixed_, beta_)) {}

    void prepare(const std::vector<double>& grid) override {
        // Bind the run-wide kernel cache: the column K[:, mu] (the data
        // evaluated at mu under N(., mu, beta)) depends only on (data,
        // beta, mu) — never on the weights — so it stays valid for the
        // whole run, including `estpi0`'s bisection. The per-row eval is
        // the EXACT expression of `norm_kernel` (same `b2`/`base`
        // constants, same two-rounding form `-0.5*d*d/b2 - base` inside
        // one `exp`) in scalar form, so a cached column is bit-identical
        // to the SIMD column fill; only the redundant re-evaluation is
        // removed.
        const double b2 = beta_ * beta_;
        const double base = stats::LN_SQRT_2PI + std::log(beta_);
        kc_.init(data_, [b2, base](double x, double mu) {
            return std::exp(-0.5 * (x - mu) * (x - mu) / b2 - base);
        });
        if (grid.empty()) {
            kgrid_.clear();
            kmat_.resize(0, 0);
            return;
        }
        kgrid_ = grid;
        // Seed the cache with every grid column (the eager materialisation
        // seam: each is computed ONCE, in the reference per-row order),
        // then build the grid matrix as a plain memcpy of the cached
        // columns — bit-identical to the old per-iteration `norm_kernel`
        // fill, but every later consumer of a grid point is an O(1) lookup.
        for (double mu : grid) {
            kc_.pin(mu);
            kc_.column(mu);
        }
        kmat_ = detail::kmat_cached(kc_, grid);
    }

    const std::vector<double>& precompute() const override { return precompute_; }

    // Build the per-solve invariants `fl`/`dens_dot_fl_` from `dens` (the
    // engine calls this once per `solvegrad` with the outer loop's
    // persistent `dens`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        dens_dot_fl_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = 1.0 / (dens[i] + precompute_[i]);
            fl_[i] = f;
            s += dens[i] * f;
        }
        dens_dot_fl_ = s;
        cached_fl_dens_ = &dens;
    }

    // Same arithmetic, but only when the cache is not already built for
    // `dens` (the `finish()` / verbose paths call gradfun* without a
    // prepare_solve; a mismatch rebuilds on the spot).
    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormLL*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        // -sum_i log(maps[i] + pre[i]): a pure element-wise expression the
        // compiler SIMD-ises; no parallel reduction needed (the log is cheap
        // and the sweep is bandwidth-bound).
        const Eigen::VectorXd mv = Eigen::Map<const Eigen::VectorXd>(
            maps.data(), static_cast<Eigen::Index>(len_));
        return -(mv + pvec()).array().log().sum();
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // density(x) = sum_j pi0[j] N(x; mu0[j], beta) = K(x, mu0) * pi0:
        // the kernel columns come from the (once-per-point) cache — a
        // bit-identical memcpy of the reference per-row evaluation — plus a
        // single GEMV.
        if (mu0.empty())
            return std::vector<double>(len_, 0.0);
        const Eigen::MatrixXd K = detail::kmat_cached(kc_, mu0);
        return detail::to_vec(K * detail::to_eigen(pi0));
    }

    // Unified gradient sweep over support points `mu` (single support point
    // for `gradfun`, the grid for `solvegrad`, solver-interior points for the
    // d1 second pass). All work is one (or two) bounds-checked GEMVs plus
    // element-wise SIMD reductions:
    //   kfl      = K * fl              (kfl[j]  = sum_i K[i,j] fl[i])
    //   kflx     = K * (fl o x)        (kflx[j] = sum_i K[i,j] fl[i] x[i])
    //   a0[j]    = sum(dens o fl) - kfl[j] * scale
    //   a1[j]    = (kfl[j] * mu[j] - kflx[j]) * scale / beta^2
    // On the full-grid sweep (`mu == kgrid_`) the precomputed grid kernel is
    // reused instead of rebuilding it — the dominant cost of every iteration.
    void gradfunvec_impl(const std::vector<double>& mu,
                         const std::vector<double>& dens, bool d0, bool d1,
                         std::vector<double>& a0,
                         std::vector<double>& a1) const {
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0 || (!d0 && !d1))
            return;
        const Eigen::Index n = static_cast<Eigen::Index>(len_);
        // fl[i] = 1 / (dens[i] + pre[i]); dens_dot_fl = sum_i dens[i]*fl[i].
        // Computed ONCE per `solvegrad` (the outer loop's `dens` is fixed
        // for the whole support-point search; see `prepare_solve`) —
        // exactly the same values and accumulation order as the former
        // per-call evaluation, so a hit is bit-identical; a different
        // `dens` rebuilds on the spot (only the `finish()` path).
        ensure_fl(dens);
        const Eigen::Map<const Eigen::VectorXd> fl(fl_.data(), n);
        // Reuse the precomputed grid kernel when the sweep points are the
        // (sorted) grid itself (the per-iteration grid sweep); otherwise
        // materialise the (n x m) kernel for these points from the
        // run-wide column cache — a bit-identical memcpy of the reference
        // per-row evaluation for every column. The reference (not a copy)
        // keeps the per-iteration grid sweep allocation-free.
        const bool cached =
            m == kgrid_.size() &&
            (m == 0 || std::equal(mu.begin(), mu.end(), kgrid_.begin()));
        // Two lvalue branches (NOT `cached ? kmat_ : kmat_cached(...)`,
        // whose prvalue operand would copy the grid matrix every sweep and
        // dangle the reference).
        Eigen::MatrixXd ktmp;
        if (!cached)
            ktmp = detail::kmat_cached(kc_, mu);
        const Eigen::MatrixXd& K = cached ? kmat_ : ktmp;
        // kfl[j]  = sum_i K[i, j] fl[i];  kflx[j] = sum_i K[i, j] fl[i] x[i].
        const Eigen::VectorXd kfl = K.transpose() * fl;
        const double scale = 1.0 - sum_pi0fixed();
        if (d0) {
            const Eigen::VectorXd g0 =
                Eigen::VectorXd::Constant(m, dens_dot_fl_) - kfl * scale;
            a0 = detail::to_vec(g0);
        }
        if (d1) {
            const Eigen::VectorXd kflx =
                K.transpose() * (fl.cwiseProduct(dvec())).eval();
            const Eigen::VectorXd muv = detail::to_eigen(mu);
            const Eigen::VectorXd g1 =
                (kfl.cwiseProduct(muv) - kflx) * (scale / (beta_ * beta_));
            a1 = detail::to_vec(g1);
        }
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        std::vector<double> v0(1, 0.0), v1(1, 0.0);
        gradfunvec_impl({mu}, dens, d0, d1, v0, v1);
        a0 = v0[0];
        a1 = v1[0];
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        gradfunvec_impl(mu, dens, d0, d1, a0, a1);
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const Eigen::Index n = static_cast<Eigen::Index>(len_);
        const double sum = 1.0 - sum_pi0fixed();
        const Eigen::VectorXd densv =
            Eigen::Map<const Eigen::VectorXd>(dens.data(), n);
        // fp[i] = dens[i] + pre[i]; tp = sp / fp (row i divided by fp[i]),
        // sp = the (n x m) normal-pdf kernel. The row scale is folded into
        // the (single) column-parallel fill, so the matrix is written once
        // and materialised column-major — the layout pnnlssum / pnnqp read.
        std::vector<double> fp(len_);
        for (std::size_t i = 0; i < len_; ++i)
            fp[i] = dens[i] + precompute_[i];
        // The same reciprocal-multiply form as `norm_kernel_scaled` (the
        // kernel columns are served from the run-wide cache, bit-identical
        // to a fresh fill), so `tp` is unchanged at the bit level.
        const Eigen::MatrixXd tp =
            detail::kmat_cached_recip(kc_, mu0, fp);
        const Eigen::Map<const Eigen::VectorXd> fpv(fp.data(), n);
        // The constrained weight subproblem (column-major GEMM/GEMV feed the
        // NNLS solver).
        std::vector<double> nw;
        if (n > 1000) {
            const Eigen::MatrixXd q = (tp.transpose() * tp).eval();
            // p = tp^T (pre o fp^-1 - 1)
            const Eigen::VectorXd pv =
                (pvec().array() / fpv.array() - 2.0).eval();
            const Eigen::VectorXd p = (tp.transpose() * pv).eval();
            nw = nnls::pnnqp(q.data(), m, p.data(), sum);
        } else {
            // b = 1 - pre o fp^-1  (element-wise: array form)
            const Eigen::VectorXd b = (2.0 - pvec().array() / fpv.array()).eval();
            nw = nnls::pnnlssum(tp.data(), n, m, b.data(), sum);
        }
        // diff = sp * nw - dens, where sp[i, j] = tp[i, j] * fp[i].
        const Eigen::VectorXd nwv = detail::to_eigen(nw);
        const Eigen::VectorXd diff =
            ((tp * nwv).cwiseProduct(fpv) - densv).eval();
        const Eigen::VectorXd etav = (nwv - detail::to_eigen(pi0)).eval();
        // p = the column sums of tp (tp^T 1).
        const Eigen::VectorXd pcol =
            (tp.transpose() * Eigen::VectorXd::Ones(n)).eval();
        checklossfun2(detail::to_vec(diff), pi0, detail::to_vec(etav),
                      detail::to_vec(pcol), dens);
    }

    double extrafun() const override { return 0.0; }
    double hypofun(double ll, double minloss) const override { return ll - minloss; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        // Cold path (plots / diagnostics): a plain index-bounded dot product.
        if (mu0.empty())
            return 0.0;
        if (mu0.size() == 1)
            return kern::dnormv(x, mu0[0], beta_) * pi0[0];
        double s = 0.0;
        for (std::size_t j = 0; j < mu0.size(); ++j)
            s += kern::dnormv(x, mu0[j], beta_) * pi0[j];
        return s;
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        precompute_ = kern::dnpnorm(data_, mu0fixed_, pi0fixed_, beta_);
    }

    const char* family_name() const override { return "npnorm"; }
    const char* flag() const override { return "d1"; }
    double beta_value() const override { return beta_; }
    std::pair<std::size_t, double> kernel_fresh() const override {
        return {kc_.fresh_count(), kc_.fresh_ms()};
    }

private:
    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    // Zero-copy read-only Eigen views over the member buffers (the
    // "lazy materialisation" seam: no copy until an expression needs one).
    const Eigen::VectorXd dvec() const {
        return Eigen::Map<const Eigen::VectorXd>(
            data_.data(), static_cast<Eigen::Index>(len_));
    }
    const Eigen::VectorXd pvec() const {
        return Eigen::Map<const Eigen::VectorXd>(
            precompute_.data(), static_cast<Eigen::Index>(len_));
    }

    std::vector<double> data_;
    std::size_t len_;
    double beta_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    std::vector<double> kgrid_;
    Eigen::MatrixXd kmat_;
    kern::KernelColumnCache kc_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double dens_dot_fl_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ===========================================================================
// NpTLL — t mixing distribution, maximum likelihood (non-central t kernel)
// ===========================================================================

class NpTLL : public Family {
public:
    NpTLL(std::vector<double> data, std::vector<double> mu0fixed,
          std::vector<double> pi0fixed, double beta)
        : data_(std::move(data)),
          len_(data_.size()),
          beta_(beta),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          precompute_(kern::dnpt(data_, mu0fixed_, pi0fixed_, beta_)),
          dfc_(stats::DntConst::make(beta)) {}

    void prepare(const std::vector<double>& grid) override {
        kc_.init(data_, [this](double x, double mu) {
            return stats::dnt_c(x, mu, dfc_);
        });
        if (grid.empty()) {
            kgrid_.clear();
            kmat_.clear();
            return;
        }
        kgrid_ = grid;
        const std::size_t g = grid.size();
        const std::size_t n = len_;
        // Column-major: column j is contiguous, so the per-iteration sweep
        // is a cache-friendly sequential read. The columns also seed the
        // run-wide kernel cache, so every later consumer (mapping, weights,
        // collapse remaps) of a grid point is an O(1) lookup.
        std::vector<double> raw(n * g, 0.0);
        for (std::size_t j = 0; j < g; ++j) {
            const double mj = grid[j];
            double* col = raw.data() + j * n;
            for (std::size_t i = 0; i < n; ++i)
                col[i] = stats::dnt_c(data_[i], mj, dfc_);
            kc_.pin(mj);
            kc_.insert(mj, std::vector<double>(col, col + n));
        }
        kmat_ = std::move(raw);
    }

    const std::vector<double>& precompute() const override { return precompute_; }

    // Build the per-solve invariants from `dens` (called once per
    // `solvegrad` by the engine; same loop and accumulation order as the
    // former per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        dens_dot_fl_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = 1.0 / (dens[i] + precompute_[i]);
            fl_[i] = f;
            s += dens[i] * f;
        }
        dens_dot_fl_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpTLL*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i)
            s += std::log(maps[i] + precompute_[i]);
        return -s;
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // sum_j pi0[j] K[:, mu0[j]] from the (once-per-point) column cache:
        // the per-element accumulation order over i matches the reference
        // `dnpt` (j-outer, i-inner, `out[i] += K* pi`), so the result is
        // bit-identical to a fresh evaluation.
        if (mu0.empty())
            return std::vector<double>(len_, 0.0);
        const Eigen::MatrixXd K = detail::kmat_cached(kc_, mu0);
        return detail::to_vec(K * detail::to_eigen(pi0));
    }

    // Only `ansd0` is defined (the C++ class leaves `ansd1` uninitialised;
    // the `d0` solver never reads it).
    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        const std::vector<double>& col = kc_.column(mu);
        ensure_fl(dens);
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            s += dens[i] * fl_[i] - col[i] * fl_[i] * scale;
        a0 = s;
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0 || !d0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        // Per-solve invariants (see `prepare_solve`), same loop + order as
        // the former per-call evaluation.
        ensure_fl(dens);
        // The full-grid sweep reads the precomputed data x grid kernel
        // columns; anything else falls back to direct dnt.
        const std::size_t g = kgrid_.size();
        const bool cached = g == m && kmat_.size() == n * g &&
                            std::equal(mu.begin(), mu.end(), kgrid_.begin(),
                                       [](double a, double b) {
                                           return std::isfinite(a) && a == b;
                                       });
        // Both paths read per-column: the grid sweep from the eager matrix,
        // anything else from the once-per-point cache (bit-identical values,
        // ascending-i accumulation as before).
        const double* kdata = kmat_.data();
        for (std::size_t j = 0; j < m; ++j) {
            const std::vector<double>* colp = cached ? nullptr : &kc_.column(mu[j]);
            const double* col = colp ? colp->data() : kdata + j * n;
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                s += col[i] * fl_[i];
            a0[j] = dens_dot_fl_ - s * scale;
        }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double sum = 1.0 - sum_pi0fixed();
        std::vector<double> fp(n);
        for (std::size_t i = 0; i < n; ++i)
            fp[i] = dens[i] + precompute_[i];
        // tp = sp columnwise / fp, where sp = the (n x m) non-central-t
        // pdf matrix (columns from the once-per-point cache).
        const Eigen::MatrixXd tp =
            detail::kmat_cached_scaled(kc_, mu0, fp);
        std::vector<double> nw;
        if (n > 1000) {
            const Eigen::MatrixXd tptp = tp.transpose() * tp;
            Eigen::VectorXd pv(n);
            for (std::size_t i = 0; i < n; ++i)
                pv[i] = precompute_[i] / fp[i] - 2.0;
            const Eigen::VectorXd ttpv = tp.transpose() * pv;
            nw = nnls::pnnqp(tptp.data(), m, ttpv.data(), sum);
        } else {
            Eigen::VectorXd bv(n);
            for (std::size_t i = 0; i < n; ++i)
                bv[i] = 2.0 - precompute_[i] / fp[i];
            nw = nnls::pnnlssum(tp.data(), n, m, bv.data(), sum);
        }
        // diff = sp * nw - dens, where sp[i, j] = tp[i, j] * fp[i]
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                s += tp(i, j) * nw[j];
            diff[i] = fp[i] * s - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        std::vector<double> pcol(m, 0.0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                pcol[j] += tp(i, j);
        checklossfun2(diff, pi0, eta, pcol, dens);
    }

    double extrafun() const override { return 0.0; }
    double hypofun(double ll, double minloss) const override { return ll - minloss; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnpt({x}, mu0, pi0, beta_)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        precompute_ = kern::dnpt(data_, mu0fixed_, pi0fixed_, beta_);
    }

    const char* family_name() const override { return "npt"; }
    const char* flag() const override { return "d0"; }
    double beta_value() const override { return beta_; }
    std::pair<std::size_t, double> kernel_fresh() const override {
        return {kc_.fresh_count(), kc_.fresh_ms()};
    }

private:
    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    std::vector<double> data_;
    std::size_t len_;
    double beta_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    std::vector<double> kgrid_, kmat_;
    kern::KernelColumnCache kc_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double dens_dot_fl_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
    stats::DntConst dfc_;
};

// ===========================================================================
// NpNormCVM — normal mixing, Cramer-von Mises distance
// ===========================================================================

class NpNormCVM : public Family {
public:
    NpNormCVM(std::vector<double> data, std::vector<double> mu0fixed,
              std::vector<double> pi0fixed, double beta)
        : data_(std::move(data)),
          len_(data_.size()),
          beta_(beta),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)) {
        recompute_pre();
    }

    void prepare(const std::vector<double>& grid) override {
        kc_.init(data_, [this](double x, double mu) {
            return stats::pnorm(x, mu, beta_, true);
        });
        for (double mu : grid) {
            kc_.pin(mu);
            kc_.column(mu);
        }
    }

    const std::vector<double>& precompute() const override { return precompute_; }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        fd_dot_dens_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = dens[i] - precompute_[i];
            fl_[i] = f;
            s += f * dens[i];
        }
        fd_dot_dens_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormCVM*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        // CVM: a squared distance that is minimised (not maximised).
        return detail::par_sum1(len_, [this, &maps](std::size_t i) {
            const double d = maps[i] - precompute_[i];
            return d * d;
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // sum_j pi0[j] Phi(x; mu0[j], beta) from the once-per-point cache
        // (ascending-i accumulation matches the reference `pnpnorm`).
        if (mu0.empty())
            return std::vector<double>(len_, 0.0);
        const Eigen::MatrixXd K = detail::kmat_cached(kc_, mu0);
        return detail::to_vec(K * detail::to_eigen(pi0));
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        if (!d0 && !d1) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::vector<double>& pcol = kc_.column(mu);
        double sum_new = 0.0, sum_d1 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = fl_[i];
            if (d0)
                sum_new += (pcol[i] * scale - dens[i]) * fl;
            if (d1)
                sum_d1 += stats::dnorm(data_[i], mu, beta_) * fl;
        }
        a0 = d0 ? sum_new * 2.0 : 0.0;
        a1 = d1 ? sum_d1 * (-2.0 * scale) : 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::size_t k0 = d0 ? 1 : 0;
        const std::size_t k1 = d1 ? 1 : 0;
        const std::size_t k = (k0 + k1) * m;
        if (k == 0)
            return;
        // Per-column: the CDF part from the once-per-point cache, the cheap
        // pdf part direct (both ascending-i, as the reference par_acc).
        for (std::size_t j = 0; j < m; ++j) {
            if (d0) {
                const std::vector<double>& col = kc_.column(mu[j]);
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += col[i] * fl_[i];
                a0[j] = s * 2.0 * scale - fd_dot_dens_ * 2.0;
            }
            if (d1) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += stats::dnorm(data_[i], mu[j], beta_) * fl_[i];
                a1[j] = s * (-2.0 * scale);
            }
        }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& /*dens*/) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double scale = 1.0 - sum_pi0fixed();
        const Eigen::MatrixXd fp = detail::kmat_cached(kc_, mu0);
        std::vector<double> nw;
        if (n > 1000) {
            const Eigen::MatrixXd tptp = fp.transpose() * fp;
            std::vector<double> pv(n);
            for (std::size_t i = 0; i < n; ++i)
                pv[i] = -precompute_[i];
            const Eigen::VectorXd ttpv =
                fp.transpose() * Eigen::Map<const Eigen::VectorXd>(pv.data(), n);
            nw = nnls::pnnqp(tptp.data(), m, ttpv.data(), scale);
        } else {
            nw = nnls::pnnlssum(fp.data(), n, m, precompute_.data(), scale);
        }
        pi0 = std::move(nw);
    }

    double extrafun() const override {
        // C++: `1 / 12 / this->len` — integer division, so always 0.
        return 0.0;
    }
    double hypofun(double ll, double /*minloss*/) const override { return ll; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        if (mu0.empty())
            return 0.0;
        if (mu0.size() == 1)
            return kern::dnormv(x, mu0[0], beta_) * pi0[0];
        double s = 0.0;
        for (std::size_t j = 0; j < mu0.size(); ++j)
            s += kern::dnormv(x, mu0[j], beta_) * pi0[j];
        return s;
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        recompute_pre();
    }

    const char* family_name() const override { return "npnorm"; }
    const char* flag() const override { return "d1"; }
    double beta_value() const override { return beta_; }
    std::pair<std::size_t, double> kernel_fresh() const override {
        return {kc_.fresh_count(), kc_.fresh_ms()};
    }

private:
    // CVM precompute: the empirical midpoints `(i + 0.5)/n` minus the
    // fixed-component CDF.
    void recompute_pre() {
        const std::vector<double> fixed =
            kern::pnpnorm(data_, mu0fixed_, pi0fixed_, beta_);
        const double nf = static_cast<double>(len_);
        precompute_.resize(len_);
        for (std::size_t i = 0; i < len_; ++i)
            precompute_[i] = (static_cast<double>(i) + 0.5) / nf - fixed[i];
    }

    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    std::vector<double> data_;
    std::size_t len_;
    double beta_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    kern::KernelColumnCache kc_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double fd_dot_dens_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ===========================================================================
// NpNormAD — normal mixing, Anderson-Darling distance
// ===========================================================================

class NpNormAD : public Family {
public:
    NpNormAD(std::vector<double> data, std::vector<double> mu0fixed,
             std::vector<double> pi0fixed, double beta)
        : data_(std::move(data)),
          len_(data_.size()),
          beta_(beta),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)) {
        const double nf = static_cast<double>(len_);
        w1_.resize(len_);
        for (std::size_t i = 0; i < len_; ++i)
            w1_[i] = (2.0 * static_cast<double>(i) + 1.0) / nf;
        w2_ = w1_;
        std::reverse(w2_.begin(), w2_.end());
        recompute_pre();
    }

    const std::vector<double>& precompute() const override { return precompute_; }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        s1_.clear();
        sum_w2_term_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        s1_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double fl = dens[i] + precompute_[i];
            s1_[i] = w1_[i] / fl - w2_[i] / (1.0 - fl);
            s += w2_[i] / (1.0 - fl);
        }
        sum_w2_term_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (s1_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormAD*>(this)->prepare_solve(dens);
    }

    void prepare(const std::vector<double>& grid) override {
        kc_.init(data_, [this](double x, double mu) {
            return stats::pnorm(x, mu, beta_, true);
        });
        for (double mu : grid) {
            kc_.pin(mu);
            kc_.column(mu);
        }
    }

    double lossfunction(const std::vector<double>& maps) const override {
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            const double t = maps[i] + precompute_[i];
            return w1_[i] * std::log(t) + w2_[i] * std::log1p(-t);
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // sum_j pi0[j] Phi(x; mu0[j], beta) from the once-per-point cache
        // (ascending accumulation matches the reference `pnpnorm`).
        if (mu0.empty())
            return std::vector<double>(len_, 0.0);
        const Eigen::MatrixXd K = detail::kmat_cached(kc_, mu0);
        return detail::to_vec(K * detail::to_eigen(pi0));
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        if (!d0 && !d1) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        // The CDF part from the once-per-point cache (the cheap pdf part is
        // direct); ascending-i accumulation as the reference par_sum1.
        const std::vector<double>& pcol = d0 ? kc_.column(mu) : kc_.empty();
        double s1_dot_new = 0.0;
        if (d0)
            for (std::size_t i = 0; i < n; ++i)
                s1_dot_new += s1_[i] * (pcol[i] * scale + precompute_[i]);
        double s1_dot_d1 = 0.0;
        if (d1)
            for (std::size_t i = 0; i < n; ++i)
                s1_dot_d1 += s1_[i] * stats::dnorm(data_[i], mu, beta_) * scale;
        // C++: (s1.dot(new) + sum(w2/(1-fl))) * -1 + 2n
        a0 = d0 ? (s1_dot_new + sum_w2_term_) * -1.0 +
                      2.0 * static_cast<double>(len_)
                : 0.0;
        a1 = d1 ? s1_dot_d1 : 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const double sum_w2_term = d0 ? sum_w2_term_ : 0.0;
        const std::size_t k0 = d0 ? 1 : 0;
        const std::size_t k1 = d1 ? 1 : 0;
        if ((k0 + k1) * m == 0)
            return;
        // Per-column: the CDF part from the once-per-point cache, the cheap
        // pdf part direct (ascending-i accumulation as the reference).
        for (std::size_t j = 0; j < m; ++j) {
            if (k0) {
                const std::vector<double>& col = kc_.column(mu[j]);
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += (col[i] * scale + precompute_[i]) * s1_[i];
                a0[j] = s * -1.0 + 2.0 * static_cast<double>(len_) -
                        sum_w2_term;
            }
            if (k1) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += stats::dnorm(data_[i], mu[j], beta_) * s1_[i];
                a1[j] = s * scale;
            }
        }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double scale = 1.0 - sum_pi0fixed();
        const Eigen::MatrixXd sf = detail::kmat_cached(kc_, mu0);
        std::vector<double> sp(n);
        for (std::size_t i = 0; i < n; ++i)
            sp[i] = dens[i] + precompute_[i];
        // S = sf / sp, U = sf / (sp - 1) — both (n x m): rows are data
        // points, columns mixture components (R's `colwise() / sp`).
        Eigen::MatrixXd S(n, m), U(n, m);
        for (std::size_t i = 0; i < n; ++i) {
            const double spi = sp[i];
            const double spm1 = sp[i] - 1.0;
            for (std::size_t j = 0; j < m; ++j) {
                S(i, j) = sf(i, j) / spi;
                U(i, j) = sf(i, j) / spm1;
            }
        }
        Eigen::VectorXd s2 =
            S.transpose() * Eigen::Map<const Eigen::VectorXd>(w1_.data(), n) +
            U.transpose() * Eigen::Map<const Eigen::VectorXd>(w2_.data(), n);
        // q = S^T diag(w1) S + U^T diag(w2) U (m x m) — a SINGLE power of
        // the weights (row-scale by sqrt(w), not w).
        Eigen::MatrixXd sr(n, m), ur(n, m);
        for (std::size_t i = 0; i < n; ++i) {
            const double sw1 = std::sqrt(w1_[i]);
            const double sw2 = std::sqrt(w2_[i]);
            for (std::size_t j = 0; j < m; ++j) {
                sr(i, j) = S(i, j) * sw1;
                ur(i, j) = U(i, j) * sw2;
            }
        }
        const Eigen::MatrixXd q = sr.transpose() * sr + ur.transpose() * ur;
        // p = -2*S2 + S^T (precompute/sp o w1) + U^T ((1-precompute)/(1-sp) o w2)
        Eigen::VectorXd p = -2.0 * s2;
        for (std::size_t j = 0; j < m; ++j) {
            p[j] += detail::par_sum1(n, [this, &S, &U, &sp, j](std::size_t i) {
                return S(i, j) * precompute_[i] / sp[i] * w1_[i] +
                       U(i, j) * (1.0 - precompute_[i]) / (1.0 - sp[i]) * w2_[i];
            });
        }
        const std::vector<double> nw =
            nnls::pnnqp(q.data(), m, p.data(), scale);
        // diff = sf * nw - dens
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double v = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                v += sf(i, j) * nw[j];
            diff[i] = v - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        // C++ passes S2 (= S^T w1 + U^T w2) as the Armijo `p` argument.
        std::vector<double> s2v(m);
        for (std::size_t j = 0; j < m; ++j)
            s2v[j] = s2[j];
        checklossfun2(diff, pi0, eta, s2v, dens);
    }

    double extrafun() const override {
        return -static_cast<double>(len_);
    }
    double hypofun(double ll, double /*minloss*/) const override { return ll; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        if (mu0.empty())
            return 0.0;
        if (mu0.size() == 1)
            return kern::dnormv(x, mu0[0], beta_) * pi0[0];
        double s = 0.0;
        for (std::size_t j = 0; j < mu0.size(); ++j)
            s += kern::dnormv(x, mu0[j], beta_) * pi0[j];
        return s;
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        recompute_pre();
    }

    const char* family_name() const override { return "npnorm"; }
    const char* flag() const override { return "d1"; }
    double beta_value() const override { return beta_; }
    std::pair<std::size_t, double> kernel_fresh() const override {
        return {kc_.fresh_count(), kc_.fresh_ms()};
    }

private:
    void recompute_pre() {
        precompute_ = kern::pnpnorm(data_, mu0fixed_, pi0fixed_, beta_);
    }

    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    std::vector<double> data_;
    std::size_t len_;
    double beta_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    std::vector<double> w1_, w2_;
    kern::KernelColumnCache kc_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> s1_;
    double sum_w2_term_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ===========================================================================
// NpNormCLL — one-parameter normal mixing (sample correlations), MLE
// ===========================================================================

class NpNormCLL : public Family {
public:
    NpNormCLL(std::vector<double> data, std::vector<double> mu0fixed,
              std::vector<double> pi0fixed, double beta)
        : data_(std::move(data)),
          len_(data_.size()),
          beta_(beta),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          precompute_(kern::dnpnormc(data_, mu0fixed_, pi0fixed_, beta_)) {}

    void prepare(const std::vector<double>& grid) override {
        kc_.init(data_, [this](double x, double mu) {
            return kern::dnpnormc_single(x, mu, beta_);
        });
        // Seed the cache with the full grid sweep (each column evaluated
        // once, in the reference row order); every later consumer of a grid
        // point is an O(1) lookup.
        for (double mu : grid) {
            kc_.pin(mu);
            kc_.column(mu);
        }
    }

    const std::vector<double>& precompute() const override { return precompute_; }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        dens_dot_fl_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = 1.0 / (dens[i] + precompute_[i]);
            fl_[i] = f;
            s += dens[i] * f;
        }
        dens_dot_fl_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormCLL*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            return std::log(maps[i] + precompute_[i]);
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // sum_j pi0[j] K[:, mu0[j]] from the once-per-point cache (the
        // ascending-i accumulation matches the reference `dnpnormc`).
        if (mu0.empty())
            return std::vector<double>(len_, 0.0);
        const Eigen::MatrixXd K = detail::kmat_cached(kc_, mu0);
        return detail::to_vec(K * detail::to_eigen(pi0));
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::vector<double>& col = kc_.column(mu);
        double sum_temp = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            sum_temp += col[i] * scale * fl_[i];
        a0 = dens_dot_fl_ - sum_temp;
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::size_t k = d0 ? m : 0;
        if (k == 0)
            return;
        for (std::size_t j = 0; j < m; ++j)
            if (d0) {
                const std::vector<double>& col = kc_.column(mu[j]);
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += col[i] * fl_[i];
                a0[j] = dens_dot_fl_ - s * scale;
            }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double sum = 1.0 - sum_pi0fixed();
        std::vector<double> fp(n);
        for (std::size_t i = 0; i < n; ++i)
            fp[i] = dens[i] + precompute_[i];
        // tp = sp / fp, where sp = the (n x m) one-parameter normal pdf
        // matrix (columns from the once-per-point cache).
        const Eigen::MatrixXd tp = detail::kmat_cached_scaled(kc_, mu0, fp);
        std::vector<double> nw;
        if (n > 1000) {
            const Eigen::MatrixXd tptp = tp.transpose() * tp;
            Eigen::VectorXd pv(n);
            for (std::size_t i = 0; i < n; ++i)
                pv[i] = precompute_[i] / fp[i] - 2.0;
            const Eigen::VectorXd ttpv = tp.transpose() * pv;
            nw = nnls::pnnqp(tptp.data(), m, ttpv.data(), sum);
        } else {
            Eigen::VectorXd bv(n);
            for (std::size_t i = 0; i < n; ++i)
                bv[i] = 2.0 - precompute_[i] / fp[i];
            nw = nnls::pnnlssum(tp.data(), n, m, bv.data(), sum);
        }
        // diff = sp * nw - dens, where sp[i, j] = tp[i, j] * fp[i]
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                s += tp(i, j) * nw[j];
            diff[i] = fp[i] * s - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        std::vector<double> pcol(m, 0.0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                pcol[j] += tp(i, j);
        checklossfun2(diff, pi0, eta, pcol, dens);
    }

    double extrafun() const override { return 0.0; }
    double hypofun(double ll, double minloss) const override { return ll - minloss; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnpnormc({x}, mu0, pi0, beta_)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        precompute_ = kern::dnpnormc(data_, mu0fixed_, pi0fixed_, beta_);
    }

    const char* family_name() const override { return "npnormc"; }
    const char* flag() const override { return "d0"; }
    double beta_value() const override { return beta_; }
    std::pair<std::size_t, double> kernel_fresh() const override {
        return {kc_.fresh_count(), kc_.fresh_ms()};
    }

private:
    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    std::vector<double> data_;
    std::size_t len_;
    double beta_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    kern::KernelColumnCache kc_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double dens_dot_fl_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ===========================================================================
// NpPoisLL — Poisson mixing, weighted MLE
// ===========================================================================

class NpPoisLL : public Family {
public:
    NpPoisLL(std::vector<double> data, std::vector<double> weights,
             std::vector<double> mu0fixed, std::vector<double> pi0fixed,
             double /*beta*/)
        : data_(std::move(data)),
          len_(data_.size()),
          weights_(std::move(weights)),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          precompute_(kern::dnppois(data_, mu0fixed_, pi0fixed_)) {}

    const std::vector<double>& precompute() const override { return precompute_; }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        dens_dot_fl_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = weights_[i] / (dens[i] + precompute_[i]);
            fl_[i] = f;
            s += dens[i] * f;
        }
        dens_dot_fl_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpPoisLL*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            return std::log(maps[i] + precompute_[i]) * weights_[i];
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        return kern::dnppois(data_, mu0, pi0);
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        a0 = detail::par_sum1(n, [this, mu, &dens, scale](std::size_t i) {
            const double base = fl_[i];
            return (dens[i] - kern::pois_pmf_c(data_[i], mu) * scale) * base;
        });
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::size_t k = d0 ? m : 0;
        if (k == 0)
            return;
        const std::vector<double> acc =
            detail::par_acc(n, k, [this, &mu, m](std::size_t i, double* a) {
                const double base = fl_[i];
                const double xi = data_[i];
                for (std::size_t j = 0; j < m; ++j)
                    a[j] += kern::pois_pmf_c(xi, mu[j]) * base;
            });
        for (std::size_t j = 0; j < m; ++j)
            if (d0)
                a0[j] = dens_dot_fl_ - acc[j] * scale;
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double scale = 1.0 - sum_pi0fixed();
        std::vector<double> fp(n);
        for (std::size_t i = 0; i < n; ++i)
            fp[i] = dens[i] + precompute_[i];
        // tp = sp / fp, sp = the (n x m) Poisson pmf matrix
        const Eigen::MatrixXd tp =
            detail::mat_fill(n, m, [this, &mu0, m, &fp](std::size_t i, double* row) {
                const double d = data_[i];
                const double fi = fp[i];
                for (std::size_t j = 0; j < m; ++j)
                    row[j] = kern::pois_pmf_c(d, mu0[j]) / fi;
            });
        // nw = pnnlssum(tp * sqrt(w) row-scaled, (2 - precompute/fp) * sqrt(w),
        // scale)
        std::vector<double> wsq(n);
        for (std::size_t i = 0; i < n; ++i)
            wsq[i] = std::sqrt(weights_[i]);
        const Eigen::MatrixXd tw =
            detail::mat_fill(n, m, [&tp, &wsq, m](std::size_t i, double* row) {
                const double w = wsq[i];
                for (std::size_t j = 0; j < m; ++j)
                    row[j] = tp(i, j) * w;
            });
        Eigen::VectorXd bv(n);
        for (std::size_t i = 0; i < n; ++i)
            bv[i] = (2.0 - precompute_[i] / fp[i]) * wsq[i];
        const std::vector<double> nw =
            nnls::pnnlssum(tw.data(), n, m, bv.data(), scale);
        // diff = sp * nw - dens, where sp[i, j] = tp[i, j] * fp[i]
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double v = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                v += tp(i, j) * nw[j];
            diff[i] = fp[i] * v - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        // p = tp^T * weights
        std::vector<double> pcol(m, 0.0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                pcol[j] += tp(i, j) * weights_[i];
        checklossfun2(diff, pi0, eta, pcol, dens);
    }

    double extrafun() const override { return 0.0; }
    double hypofun(double ll, double minloss) const override { return ll - minloss; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnppois({x}, mu0, pi0)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        precompute_ = kern::dnppois(data_, mu0fixed_, pi0fixed_);
    }

    const char* family_name() const override { return "nppois"; }
    const char* flag() const override { return "d0"; }
    double beta_value() const override { return 1.0; }

private:
    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    std::vector<double> data_;
    std::size_t len_;
    std::vector<double> weights_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double dens_dot_fl_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ===========================================================================
// Binned ("...w") families — the R `npnormllw` / `npnormcvmw` / `npnormadw` /
// `nptllw` large-scale variants. The caller pre-bins the observations into
// `(bin centres, counts)` and passes `h = 10^order`; the kernels are the
// binned ones in npfc_kernels.h (trapezoid-rule binned normal density,
// CDF-difference binned normal cdf, CDF-difference binned non-central-t
// density). The loss is the weighted form: each bin `i` contributes `count_i`
// copies of its centre, so every loss/gradient/weight sweep is `count`-
// weighted.
//
// Cache note (faithful to R): the trapezoid binned-density kernel
// `ddiscnorm_m` fixes its subdivision count `N` from the RANGE of the whole
// support set, and the CDF-difference kernel `pnorm_disc_m` flips its shift
// direction on the `n > m` size test — so neither is a function of `(x, mu)`
// alone and CANNOT be served from the per-point column cache. These three
// normal-binned families therefore rebuild the (n x m) kernel matrix on each
// mapping/gradient/weight call, exactly as the R package does. The binned
// non-central-t density `ddisct_v` IS a pure function of `(x, mu)`, so
// `NpTLLW` reuses the run-wide column cache (the expensive kernel is cached;
// a cached column is bit-identical to a fresh per-row fill).
// ===========================================================================

// `weights_sum`: the total count (sum of the bin weights).
namespace detail {
template <typename T>
inline double weights_sum_of(const T& v) {
    double s = 0.0;
    for (double x : v)
        s += x;
    return s;
}
}  // namespace detail

// ---------------------------------------------------------------------------
// NpNormLLW — binned normal mixing, maximum likelihood (flag d0)
// ---------------------------------------------------------------------------

class NpNormLLW : public Family {
public:
    NpNormLLW(std::vector<double> data, std::vector<double> weights,
              std::vector<double> mu0fixed, std::vector<double> pi0fixed,
              double beta, double h)
        : data_(std::move(data)),
          len_(data_.size()),
          weights_(std::move(weights)),
          beta_(beta),
          h_(h),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          precompute_(kern::dnpdiscnorm(data_, mu0fixed_, pi0fixed_, beta_,
                                        h_)) {}

    void prepare(const std::vector<double>& grid) override {
        // The binned trapezoid-density matrix depends only on
        // (data, mu, beta, h) — never on pi0/dens — and the outer loop
        // re-requests it with repeating mu vectors: the per-iteration
        // grid sweep (solvegrad) with the (sorted) grid itself, plus the
        // mapping/collapse/computeweights passes over the current support
        // set. Pin the grid matrix here (bit-identical to the fresh
        // fill with the same arguments); `dmat` serves it on every later
        // grid sweep and memoises the small support-set fills. (The
        // run-wide COLUMN cache does not apply: the trapezoid
        // subdivision count N depends on the whole mu range, see the
        // `ddiscnorm_m` note.)
        if (!grid.empty()) {
            dmemo_.grid = grid;
            dmemo_.grid_m = kern::ddiscnorm_m(data_, grid, beta_, h_);
        }
    }

    const std::vector<double>& precompute() const override {
        return precompute_;
    }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        dens_dot_fl_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = weights_[i] / (dens[i] + precompute_[i]);
            fl_[i] = f;
            s += dens[i] * f;
        }
        dens_dot_fl_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormLLW*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        // -sum_i log(maps[i] + pre[i]) * count_i (weighted log-likelihood).
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            return std::log(maps[i] + precompute_[i]) * weights_[i];
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        const std::size_t n = len_;
        if (mu0.empty())
            return std::vector<double>(n, 0.0);
        const Eigen::MatrixXd& D = dmat(mu0);
        std::vector<double> out(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < mu0.size(); ++j)
                s += D(i, j) * pi0[j];
            out[i] = s;
        }
        return out;
    }

    // Only the probability-direction gradient `a0` is defined (the R class
    // leaves `a1` uninitialised; the `d0` solver never reads it).
    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        ensure_fl(dens);
        // temp = dnpdiscnorm_(data, mu, scale, beta, h): the single support
        // point `mu` carrying the remaining mass `scale` (column j * pi0[j],
        // single column — the i-outer/j-inner accumulation of `dnpdiscnorm`).
        const Eigen::MatrixXd& D = dmat({mu});
        std::vector<double> temp(len_);
        for (std::size_t i = 0; i < len_; ++i)
            temp[i] = D(i, 0) * scale;
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i)
            // Keep the per-point division inline (the historical expression):
            // the pre-cached reciprocal would round differently under FMA
            // contraction and drift the trajectory by ~1 ulp (see the golden
            // gate). The scalar path is not the hot loop — the vector path is.
            s += (dens[i] - temp[i]) * weights_[i] / (dens[i] + precompute_[i]);
        a0 = s;
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0 || !d0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const Eigen::MatrixXd& D = dmat(mu);
        const Eigen::Map<const Eigen::VectorXd> fv(fl_.data(),
                                                   static_cast<Eigen::Index>(n));
        const Eigen::VectorXd kfl = D.transpose() * fv;
        for (std::size_t j = 0; j < m; ++j)
            a0[j] = dens_dot_fl_ -
                    kfl(static_cast<Eigen::Index>(j)) * scale;
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double sum = 1.0 - sum_pi0fixed();
        std::vector<double> fp(n);
        for (std::size_t i = 0; i < n; ++i)
            fp[i] = dens[i] + precompute_[i];
        const Eigen::MatrixXd& sp = dmat(mu0);
        std::vector<double> wsq(n);
        for (std::size_t i = 0; i < n; ++i)
            wsq[i] = std::sqrt(weights_[i]);
        // tw = (sp / fp) * sqrt(count) row-scaled (the weighted `tp`).
        Eigen::MatrixXd tw(n, static_cast<Eigen::Index>(m));
        for (std::size_t i = 0; i < n; ++i) {
            const double w = wsq[i];
            const double fi = fp[i];
            for (std::size_t j = 0; j < m; ++j)
                tw(i, j) = sp(i, j) / fi * w;
        }
        Eigen::VectorXd bv(n);
        for (std::size_t i = 0; i < n; ++i)
            bv(i) = (2.0 - precompute_[i] / fp[i]) * wsq[i];
        const std::vector<double> nw =
            nnls::pnnlssum(tw.data(), n, m, bv.data(), sum);
        // diff = sp * nw - dens
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double v = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                v += sp(i, j) * nw[j];
            diff[i] = v - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        // p = tp^T * weights, where tp[i, j] = sp[i, j] / fp[i].
        std::vector<double> pcol(m, 0.0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                pcol[j] += sp(i, j) / fp[i] * weights_[i];
        checklossfun2(diff, pi0, eta, pcol, dens);
    }

    double extrafun() const override {
        return detail::weights_sum_of(weights_) * std::log(h_);
    }
    double hypofun(double ll, double minloss) const override {
        return ll - minloss;
    }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnpdiscnorm({x}, mu0, pi0, beta_, h_)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        precompute_ = kern::dnpdiscnorm(data_, mu0fixed_, pi0fixed_, beta_, h_);
    }

    const char* family_name() const override { return "npnorm"; }
    const char* flag() const override { return "d0"; }
    double beta_value() const override { return beta_; }

private:
    // The (n x m) trapezoid-density fill for `mu` through the memo
    // (`detail::KernelMemo`): a served matrix is bit-identical to a fresh
    // `ddiscnorm_m(data, mu, beta, h)` (exact mu-vector key; see the note
    // above the binned families).
    const Eigen::MatrixXd& dmat(const std::vector<double>& mu) const {
        return dmemo_.get(mu, [this](const std::vector<double>& v) {
            return kern::ddiscnorm_m(data_, v, beta_, h_);
        });
    }

    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    mutable detail::KernelMemo dmemo_;

    std::vector<double> data_;
    std::size_t len_;
    std::vector<double> weights_;
    double beta_;
    double h_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double dens_dot_fl_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ---------------------------------------------------------------------------
// NpNormCVMW — binned normal mixing, Cramer-von Mises distance (flag d1)
// ---------------------------------------------------------------------------

class NpNormCVMW : public Family {
public:
    NpNormCVMW(std::vector<double> data, std::vector<double> weights,
               std::vector<double> mu0fixed, std::vector<double> pi0fixed,
               double beta, double h)
        : data_(std::move(data)),
          len_(data_.size()),
          weights_(std::move(weights)),
          beta_(beta),
          h_(h),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)) {
        recompute_pre();
    }

    void prepare(const std::vector<double>& grid) override {
        // Pin the binned cdf AND density grid matrices (see `detail::
        // KernelMemo`): the per-iteration `solvegradd1` requests each of
        // them for the (sorted) grid, and the mapping/collapse/
        // computeweights passes reuse the last small support-set fill.
        if (!grid.empty()) {
            pmemo_.grid = grid;
            pmemo_.grid_m = kern::pnorm_disc_m(data_, grid, beta_, h_);
            dmemo_.grid = grid;
            dmemo_.grid_m = kern::ddiscnorm_m(data_, grid, beta_, h_);
        }
    }

    const std::vector<double>& precompute() const override {
        return precompute_;
    }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        fd_dot_dens_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        for (std::size_t i = 0; i < len_; ++i)
            fl_[i] = (dens[i] - precompute_[i]) * weights_[i];
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormCVMW*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        // CVM: a squared distance, minimised, count-weighted:
        // sum_i (maps[i] - pre[i])^2 * count_i.
        return detail::par_sum1(len_, [this, &maps](std::size_t i) {
            const double d = maps[i] - precompute_[i];
            return d * d * weights_[i];
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // `pnpdiscnorm`: P * pi0 as the i-outer/j-inner accumulation
        // (bit-identical; the cdf columns come from the memo).
        const std::size_t n = len_;
        if (mu0.empty())
            return std::vector<double>(n, 0.0);
        const Eigen::MatrixXd& P = pmat(mu0);
        std::vector<double> out(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < mu0.size(); ++j)
                s += P(i, j) * pi0[j];
            out[i] = s;
        }
        return out;
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        if (!d0 && !d1) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        // fl_[i] = (dens[i] - pre[i]) * count_i (per-solve invariant, see
        // `prepare_solve`).
        ensure_fl(dens);
        // d0: 2 * sum_i (Phi_disc(mu; scale) - dens[i]) * fl_[i], where
        // Phi_disc(mu; scale) is the single-point binned-cdf mixture.
        double sum_new = 0.0;
        if (d0) {
            // pcol = P({mu}) * scale (single column, as `pnpdiscnorm`).
            const Eigen::MatrixXd& P = pmat({mu});
            for (std::size_t i = 0; i < n; ++i)
                sum_new += (P(i, 0) * scale - dens[i]) * fl_[i];
        }
        // d1: -2*scale * sum_i N_disc(x_i; mu) * fl_[i] (binned pdf,
        // pi0 = 1.0 in the `dnpdiscnorm` single column).
        double sum_d1 = 0.0;
        if (d1) {
            const Eigen::MatrixXd& D = dmat({mu});
            for (std::size_t i = 0; i < n; ++i)
                sum_d1 += D(i, 0) * 1.0 * fl_[i];
        }
        a0 = d0 ? sum_new * 2.0 : 0.0;
        a1 = d1 ? sum_d1 * (-2.0 * scale) : 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const bool do0 = d0, do1 = d1;
        if (!do0 && !do1)
            return;
        const Eigen::MatrixXd& P = do0 ? pmat(mu) : detail::empty_m();
        const Eigen::MatrixXd& D = do1 ? dmat(mu) : detail::empty_m();
        for (std::size_t j = 0; j < m; ++j) {
            if (do0) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += (P(i, j) * scale - dens[i]) * fl_[i];
                a0[j] = s * 2.0;
            }
            if (do1) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += D(i, j) * fl_[i];
                a1[j] = s * (-2.0 * scale);
            }
        }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& /*dens*/) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double scale = 1.0 - sum_pi0fixed();
        const Eigen::MatrixXd& P = pmat(mu0);
        std::vector<double> wsq(n);
        for (std::size_t i = 0; i < n; ++i)
            wsq[i] = std::sqrt(weights_[i]);
        Eigen::MatrixXd tw(n, static_cast<Eigen::Index>(m));
        for (std::size_t i = 0; i < n; ++i) {
            const double w = wsq[i];
            for (std::size_t j = 0; j < m; ++j)
                tw(i, j) = P(i, j) * w;
        }
        Eigen::VectorXd bv(n);
        for (std::size_t i = 0; i < n; ++i)
            bv(i) = precompute_[i] * wsq[i];
        // The weighted CVM weight subproblem assigns the NNLS solution
        // directly (the R class has no line search here).
        pi0 = nnls::pnnlssum(tw.data(), n, m, bv.data(), scale);
    }

    double extrafun() const override {
        // count.sum()/3 - sum_i ((ecdf_i - count_i/2)/count.sum())^2 * count_i.
        const double sumw = detail::weights_sum_of(weights_);
        double cum = 0.0, s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            cum += weights_[i];
            const double d = (cum - weights_[i] / 2.0) / sumw;
            s += d * d * weights_[i];
        }
        return sumw / 3.0 - s;
    }
    double hypofun(double ll, double /*minloss*/) const override { return ll; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnpdiscnorm({x}, mu0, pi0, beta_, h_)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        recompute_pre();
    }

    const char* family_name() const override { return "npnorm"; }
    const char* flag() const override { return "d1"; }
    double beta_value() const override { return beta_; }

private:
    // Binned-CVM precompute: empirical midpoints (ecdf_i - count_i/2)/count.sum()
    // minus the fixed-component binned-cdf.
    void recompute_pre() {
        const std::vector<double> fixed =
            kern::pnpdiscnorm(data_, mu0fixed_, pi0fixed_, beta_, h_);
        const double sumw = detail::weights_sum_of(weights_);
        double cum = 0.0;
        precompute_.resize(len_);
        for (std::size_t i = 0; i < len_; ++i) {
            cum += weights_[i];
            precompute_[i] = (cum - weights_[i] / 2.0) / sumw - fixed[i];
        }
    }

    // The (n x m) binned-cdf / trapezoid-density fills for `mu` through the
    // memo (`detail::KernelMemo`): a served matrix is bit-identical to a
    // fresh fill with the same arguments (exact mu-vector key).
    const Eigen::MatrixXd& pmat(const std::vector<double>& mu) const {
        return pmemo_.get(mu, [this](const std::vector<double>& v) {
            return kern::pnorm_disc_m(data_, v, beta_, h_);
        });
    }
    const Eigen::MatrixXd& dmat(const std::vector<double>& mu) const {
        return dmemo_.get(mu, [this](const std::vector<double>& v) {
            return kern::ddiscnorm_m(data_, v, beta_, h_);
        });
    }

    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    // Both the binned-cdf memo (d0 direction, `pmat`) and the
    // trapezoid-density memo (d1 direction, `dmat`) are pinned to the grid in
    // `prepare` and reused on every small support-set fill.
    mutable detail::KernelMemo pmemo_;
    mutable detail::KernelMemo dmemo_;

    std::vector<double> data_;
    std::size_t len_;
    std::vector<double> weights_;
    double beta_;
    double h_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double fd_dot_dens_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ---------------------------------------------------------------------------
// NpNormADW — binned normal mixing, Anderson-Darling distance (flag d1)
// ---------------------------------------------------------------------------

class NpNormADW : public Family {
public:
    NpNormADW(std::vector<double> data, std::vector<double> weights,
              std::vector<double> mu0fixed, std::vector<double> pi0fixed,
              double beta, double h)
        : data_(std::move(data)),
          len_(data_.size()),
          weights_(std::move(weights)),
          beta_(beta),
          h_(h),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)) {
        // Weighted AD endpoints from the binned empirical cdf:
        // w1 = (2*ecdf - count) * count / count.sum(); w2 = 2*count - w1.
        const double sumw = detail::weights_sum_of(weights_);
        double cum = 0.0;
        w1_.resize(len_);
        w2_.resize(len_);
        for (std::size_t i = 0; i < len_; ++i) {
            cum += weights_[i];
            w1_[i] = (2.0 * cum - weights_[i]) * weights_[i] / sumw;
            w2_[i] = 2.0 * weights_[i] - w1_[i];
        }
        recompute_pre();
    }

    void prepare(const std::vector<double>& grid) override {
        // Pin the binned-cdf grid matrix (see `detail::KernelMemo`): the
        // per-iteration `solvegradd1` d0 pass requests it for the (sorted)
        // grid; mapping/collapse/computeweights reuse the last small
        // support-set fill. (ADW's d1 direction uses the UNBINNED normal
        // pdf pointwise — no matrix to pin.)
        if (!grid.empty()) {
            pmemo_.grid = grid;
            pmemo_.grid_m = kern::pnorm_disc_m(data_, grid, beta_, h_);
        }
    }

    const std::vector<double>& precompute() const override {
        return precompute_;
    }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        s1_.clear();
        sum_w2_term_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        s1_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double fl = dens[i] + precompute_[i];
            s1_[i] = w1_[i] / fl - w2_[i] / (1.0 - fl);
            s += w2_[i] / (1.0 - fl);
        }
        sum_w2_term_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (s1_.empty() || &dens != cached_fl_dens_)
            const_cast<NpNormADW*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            const double t = maps[i] + precompute_[i];
            return w1_[i] * std::log(t) + w2_[i] * std::log1p(-t);
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        // `pnpdiscnorm`: P * pi0 as the i-outer/j-inner accumulation
        // (bit-identical; the cdf columns come from the memo).
        const std::size_t n = len_;
        if (mu0.empty())
            return std::vector<double>(n, 0.0);
        const Eigen::MatrixXd& P = pmat(mu0);
        std::vector<double> out(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < mu0.size(); ++j)
                s += P(i, j) * pi0[j];
            out[i] = s;
        }
        return out;
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        if (!d0 && !d1) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        // s1_[i] = w1/fl - w2/(1-fl) (per-solve invariant, see
        // `prepare_solve`).
        ensure_fl(dens);
        double s1_dot_new = 0.0;
        if (d0) {
            // pcol[i] = P(i, 0) * scale (single column, as `pnpdiscnorm`).
            const Eigen::MatrixXd& P = pmat({mu});
            for (std::size_t i = 0; i < n; ++i)
                s1_dot_new += s1_[i] * (P(i, 0) * scale + precompute_[i]);
        }
        // d1 uses the UNBINNED normal pdf shifted by -h (R's `dnpnorm_(data,
        // mu - h, scale, beta)`).
        double s1_dot_d1 = 0.0;
        if (d1)
            for (std::size_t i = 0; i < n; ++i)
                s1_dot_d1 += s1_[i] * stats::dnorm(data_[i], mu - h_, beta_) * scale;
        a0 = d0 ? (s1_dot_new + sum_w2_term_) * -1.0 +
                      2.0 * detail::weights_sum_of(weights_)
                : 0.0;
        a1 = d1 ? s1_dot_d1 : 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const double sum_w2_term = d0 ? sum_w2_term_ : 0.0;
        if (!d0 && !d1)
            return;
        const Eigen::MatrixXd& P = d0 ? pmat(mu) : detail::empty_m();
        for (std::size_t j = 0; j < m; ++j) {
            if (d0) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += (P(i, j) * scale + precompute_[i]) * s1_[i];
                a0[j] = s * -1.0 +
                        2.0 * detail::weights_sum_of(weights_) - sum_w2_term;
            }
            if (d1) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += stats::dnorm(data_[i], mu[j] - h_, beta_) * s1_[i];
                a1[j] = s * scale;
            }
        }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double scale = 1.0 - sum_pi0fixed();
        const Eigen::MatrixXd& sf = pmat(mu0);
        std::vector<double> sp(n);
        for (std::size_t i = 0; i < n; ++i)
            sp[i] = dens[i] + precompute_[i];
        Eigen::MatrixXd S(n, static_cast<Eigen::Index>(m)),
            U(n, static_cast<Eigen::Index>(m));
        for (std::size_t i = 0; i < n; ++i) {
            const double spi = sp[i];
            const double spm1 = sp[i] - 1.0;
            for (std::size_t j = 0; j < m; ++j) {
                S(i, j) = sf(i, j) / spi;
                U(i, j) = sf(i, j) / spm1;
            }
        }
        Eigen::VectorXd s2 =
            S.transpose() * Eigen::Map<const Eigen::VectorXd>(w1_.data(), n) +
            U.transpose() * Eigen::Map<const Eigen::VectorXd>(w2_.data(), n);
        Eigen::MatrixXd sr(n, static_cast<Eigen::Index>(m)),
            ur(n, static_cast<Eigen::Index>(m));
        for (std::size_t i = 0; i < n; ++i) {
            const double sw1 = std::sqrt(w1_[i]);
            const double sw2 = std::sqrt(w2_[i]);
            for (std::size_t j = 0; j < m; ++j) {
                sr(i, j) = S(i, j) * sw1;
                ur(i, j) = U(i, j) * sw2;
            }
        }
        const Eigen::MatrixXd q = sr.transpose() * sr + ur.transpose() * ur;
        Eigen::VectorXd p = -2.0 * s2;
        for (std::size_t j = 0; j < m; ++j) {
            p[j] += detail::par_sum1(n, [this, &S, &U, &sp, j](std::size_t i) {
                return S(i, j) * precompute_[i] / sp[i] * w1_[i] +
                       U(i, j) * (1.0 - precompute_[i]) / (1.0 - sp[i]) * w2_[i];
            });
        }
        const std::vector<double> nw =
            nnls::pnnqp(q.data(), m, p.data(), scale);
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double v = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                v += sf(i, j) * nw[j];
            diff[i] = v - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        std::vector<double> s2v(m);
        for (std::size_t j = 0; j < m; ++j)
            s2v[j] = s2[j];
        checklossfun2(diff, pi0, eta, s2v, dens);
    }

    double extrafun() const override {
        return -detail::weights_sum_of(weights_);
    }
    double hypofun(double ll, double /*minloss*/) const override { return ll; }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnpdiscnorm({x}, mu0, pi0, beta_, h_)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        recompute_pre();
    }

    const char* family_name() const override { return "npnorm"; }
    const char* flag() const override { return "d1"; }
    double beta_value() const override { return beta_; }

private:
    void recompute_pre() {
        precompute_ = kern::pnpdiscnorm(data_, mu0fixed_, pi0fixed_, beta_, h_);
    }

    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    const Eigen::MatrixXd& pmat(const std::vector<double>& mu) const {
        return pmemo_.get(mu, [this](const std::vector<double>& v) {
            return kern::pnorm_disc_m(data_, v, beta_, h_);
        });
    }

    std::vector<double> data_;
    std::size_t len_;
    std::vector<double> weights_;
    double beta_;
    double h_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    std::vector<double> w1_, w2_;
    mutable detail::KernelMemo pmemo_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> s1_;
    double sum_w2_term_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

// ---------------------------------------------------------------------------
// NpTLLW — binned non-central-t mixing, maximum likelihood (flag d0)
// ---------------------------------------------------------------------------

class NpTLLW : public Family {
public:
    NpTLLW(std::vector<double> data, std::vector<double> weights,
           std::vector<double> mu0fixed, std::vector<double> pi0fixed,
           double beta, double h)
        : data_(std::move(data)),
          len_(data_.size()),
          weights_(std::move(weights)),
          beta_(beta),
          h_(h),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          precompute_(kern::dnpdisct(data_, mu0fixed_, pi0fixed_, beta_, h_)) {}

    void prepare(const std::vector<double>& grid) override {
        // The binned non-central-t density is a pure function of (x, mu), so
        // it IS served from the run-wide column cache (the expensive kernel;
        // a cached column is bit-identical to a fresh per-row fill).
        kc_.init(data_, [this](double x, double mu) {
            return kern::ddisct_v(x, beta_, mu, h_);
        });
        if (grid.empty()) {
            kgrid_.clear();
            kmat_.clear();
            return;
        }
        kgrid_ = grid;
        const std::size_t g = grid.size();
        const std::size_t n = len_;
        std::vector<double> raw(n * g, 0.0);
        for (std::size_t j = 0; j < g; ++j) {
            const double mj = grid[j];
            double* col = raw.data() + j * n;
            for (std::size_t i = 0; i < n; ++i)
                col[i] = kern::ddisct_v(data_[i], beta_, mj, h_);
            kc_.pin(mj);
            kc_.insert(mj, std::vector<double>(col, col + n));
        }
        kmat_ = std::move(raw);
    }

    const std::vector<double>& precompute() const override {
        return precompute_;
    }

    // Build the per-solve invariants from `dens` (the engine calls this
    // once per `solvegrad`; same loop and accumulation order as the former
    // per-call evaluation, so a hit is bit-identical).
    void prepare_solve(const std::vector<double>& dens) override {
        fl_.clear();
        dens_dot_fl_ = 0.0;
        cached_fl_dens_ = nullptr;
        if (dens.size() != len_)
            return;
        fl_.resize(len_);
        double s = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = weights_[i] / (dens[i] + precompute_[i]);
            fl_[i] = f;
            s += dens[i] * f;
        }
        dens_dot_fl_ = s;
        cached_fl_dens_ = &dens;
    }

    void ensure_fl(const std::vector<double>& dens) const {
        if (fl_.empty() || &dens != cached_fl_dens_)
            const_cast<NpTLLW*>(this)->prepare_solve(dens);
    }

    double lossfunction(const std::vector<double>& maps) const override {
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            return std::log(maps[i] + precompute_[i]) * weights_[i];
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        if (mu0.empty())
            return std::vector<double>(len_, 0.0);
        const Eigen::MatrixXd K = detail::kmat_cached(kc_, mu0);
        return detail::to_vec(K * detail::to_eigen(pi0));
    }

    void gradfun(double mu, const std::vector<double>& dens,
                 SolveCtx& ctx, bool d0, bool d1, double& a0,
                 double& a1) const override {
        (void)ctx;
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::vector<double>& col = kc_.column(mu);
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            s += (dens[i] - col[i] * scale) * fl_[i];
        a0 = s;
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu,
                    const std::vector<double>& dens, SolveCtx& ctx, bool d0,
                    bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)ctx;
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0 || !d0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        ensure_fl(dens);
        const std::size_t g = kgrid_.size();
        const bool cached =
            g == m && kmat_.size() == n * g &&
            std::equal(mu.begin(), mu.end(), kgrid_.begin(),
                       [](double a, double b) {
                           return std::isfinite(a) && a == b;
                       });
        const double* kdata = kmat_.data();
        for (std::size_t j = 0; j < m; ++j) {
            const std::vector<double>* colp =
                cached ? nullptr : &kc_.column(mu[j]);
            const double* col = colp ? colp->data() : kdata + j * n;
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                s += col[i] * fl_[i];
            a0[j] = dens_dot_fl_ - s * scale;
        }
    }

    void computeweights(const std::vector<double>& mu0, std::vector<double>& pi0,
                        const std::vector<double>& dens) const override {
        const std::size_t m = mu0.size();
        if (m == 0 || pi0.size() != m)
            return;
        const std::size_t n = len_;
        const double sum = 1.0 - sum_pi0fixed();
        std::vector<double> fp(n);
        for (std::size_t i = 0; i < n; ++i)
            fp[i] = dens[i] + precompute_[i];
        // tp = sp / fp, where sp is the (n x m) binned non-central-t pdf
        // matrix (columns from the run-wide cache).
        const Eigen::MatrixXd tp = detail::kmat_cached_scaled(kc_, mu0, fp);
        std::vector<double> wsq(n);
        for (std::size_t i = 0; i < n; ++i)
            wsq[i] = std::sqrt(weights_[i]);
        Eigen::MatrixXd tw(n, static_cast<Eigen::Index>(m));
        for (std::size_t i = 0; i < n; ++i) {
            const double w = wsq[i];
            for (std::size_t j = 0; j < m; ++j)
                tw(i, j) = tp(i, j) * w;
        }
        Eigen::VectorXd bv(n);
        for (std::size_t i = 0; i < n; ++i)
            bv(i) = (2.0 - precompute_[i] / fp[i]) * wsq[i];
        // R's nptllw weight subproblem is a weighted NNLS (no pnnqp branch).
        const std::vector<double> nw =
            nnls::pnnlssum(tw.data(), n, m, bv.data(), sum);
        std::vector<double> diff(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double v = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                v += tp(i, j) * nw[j];
            diff[i] = fp[i] * v - dens[i];
        }
        std::vector<double> eta(m);
        for (std::size_t j = 0; j < m; ++j)
            eta[j] = nw[j] - pi0[j];
        // p = tp^T * weights.
        std::vector<double> pcol(m, 0.0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                pcol[j] += tp(i, j) * weights_[i];
        checklossfun2(diff, pi0, eta, pcol, dens);
    }

    double extrafun() const override {
        return detail::weights_sum_of(weights_) * std::log(h_);
    }
    double hypofun(double ll, double minloss) const override {
        return ll - minloss;
    }

    double familydensity(double x, const std::vector<double>& mu0,
                         const std::vector<double>& pi0) const override {
        return kern::dnpdisct({x}, mu0, pi0, beta_, h_)[0];
    }

    void set_fixed(const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed) override {
        mu0fixed_ = mu0fixed;
        pi0fixed_ = pi0fixed;
        precompute_ = kern::dnpdisct(data_, mu0fixed_, pi0fixed_, beta_, h_);
    }

    const char* family_name() const override { return "npt"; }
    const char* flag() const override { return "d0"; }
    double beta_value() const override { return beta_; }
    std::pair<std::size_t, double> kernel_fresh() const override {
        return {kc_.fresh_count(), kc_.fresh_ms()};
    }

private:
    double sum_pi0fixed() const {
        double s = 0.0;
        for (double v : pi0fixed_)
            s += v;
        return s;
    }

    std::vector<double> data_;
    std::size_t len_;
    std::vector<double> weights_;
    double beta_;
    double h_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> precompute_;
    std::vector<double> kgrid_, kmat_;
    kern::KernelColumnCache kc_;
    // Per-solve invariants (see `prepare_solve`); the `dens` pointer the
    // cache was built for (a different `dens` triggers a rebuild).
    std::vector<double> fl_;
    double dens_dot_fl_ = 0.0;
    const std::vector<double>* cached_fl_dens_ = nullptr;
};

}  // namespace fam
}  // namespace npfc

#endif  // NPFIC_FAMILIES_H
