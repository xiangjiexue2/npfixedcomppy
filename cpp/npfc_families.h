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
        kgrid_ = grid;
        // The n x g grid kernel is materialised ONCE here (the eager
        // materialisation seam) and reused at every iteration's grid sweep.
        // `norm_kernel` resizes the matrix via Eigen, so no index can stray
        // off the buffer, and each column's `exp` is a pure SIMD expression.
        kmat_ = detail::norm_kernel(data_, grid, beta_);
    }

    const std::vector<double>& precompute() const override { return precompute_; }

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
        // one bounds-checked SIMD kernel fill plus a single GEMV.
        return detail::to_vec(
            (detail::norm_kernel(data_, mu0, beta_) * detail::to_eigen(pi0))
                .eval());
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
        const Eigen::VectorXd densv =
            Eigen::Map<const Eigen::VectorXd>(dens.data(), n);
        // fl[i] = 1 / (dens[i] + pre[i]); dens_dot_fl = sum_i dens[i]*fl[i].
        // Explicit element-wise reciprocal (`.inverse()` on a non-square
        // vector expression is not what this Eigen build computes).
        std::vector<double> flv(len_);
        double dens_dot_fl = 0.0;
        for (std::size_t i = 0; i < len_; ++i) {
            const double f = 1.0 / (dens[i] + precompute_[i]);
            flv[i] = f;
            dens_dot_fl += dens[i] * f;
        }
        const Eigen::Map<const Eigen::VectorXd> fl(flv.data(), n);
        // Reuse the precomputed grid kernel when the sweep points are the
        // (sorted) grid itself; otherwise build the (n x m) kernel for these
        // points. The reference (not a copy) keeps the per-iteration grid
        // sweep allocation-free.
        const bool cached =
            m == kgrid_.size() &&
            (m == 0 || std::equal(mu.begin(), mu.end(), kgrid_.begin()));
        Eigen::MatrixXd ktmp;
        if (!cached)
            ktmp = detail::norm_kernel(data_, mu, beta_);
        const Eigen::MatrixXd& K = cached ? kmat_ : ktmp;
        // kfl[j]  = sum_i K[i, j] fl[i];  kflx[j] = sum_i K[i, j] fl[i] x[i].
        const Eigen::VectorXd kfl = K.transpose() * fl;
        const double scale = 1.0 - sum_pi0fixed();
        if (d0) {
            const Eigen::VectorXd g0 =
                Eigen::VectorXd::Constant(m, dens_dot_fl) - kfl * scale;
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

    void gradfun(double mu, const std::vector<double>& dens, bool d0, bool d1,
                 double& a0, double& a1) const override {
        std::vector<double> v0(1, 0.0), v1(1, 0.0);
        gradfunvec_impl({mu}, dens, d0, d1, v0, v1);
        a0 = v0[0];
        a1 = v1[0];
    }

    void gradfunvec(const std::vector<double>& mu, const std::vector<double>& dens,
                    bool d0, bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
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
        const Eigen::MatrixXd tp =
            detail::norm_kernel_scaled(data_, mu0, beta_, fp);
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
          precompute_(kern::dnpt(data_, mu0fixed_, pi0fixed_, beta_)) {}

    void prepare(const std::vector<double>& grid) override {
        kc_.init(data_, [this](double x, double mu) {
            return stats::dnt(x, beta_, mu);
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
                col[i] = stats::dnt(data_[i], beta_, mj);
            kc_.insert(mj, std::vector<double>(col, col + n));
        }
        kmat_ = std::move(raw);
    }

    const std::vector<double>& precompute() const override { return precompute_; }

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
    void gradfun(double mu, const std::vector<double>& dens, bool d0, bool d1,
                 double& a0, double& a1) const override {
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        const std::vector<double>& col = kc_.column(mu);
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = 1.0 / (dens[i] + precompute_[i]);
            s += dens[i] * fl - col[i] * fl * scale;
        }
        a0 = s;
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu, const std::vector<double>& dens,
                    bool d0, bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0 || !d0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        std::vector<double> fullden(n);
        double dens_dot_fullden = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = 1.0 / (dens[i] + precompute_[i]);
            fullden[i] = fl;
            dens_dot_fullden += dens[i] * fl;
        }
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
                s += col[i] * fullden[i];
            a0[j] = dens_dot_fullden - s * scale;
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
        for (double mu : grid)
            kc_.column(mu);
    }

    const std::vector<double>& precompute() const override { return precompute_; }

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

    void gradfun(double mu, const std::vector<double>& dens, bool d0, bool d1,
                 double& a0, double& a1) const override {
        if (!d0 && !d1) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        const std::vector<double>& pcol = kc_.column(mu);
        double sum_new = 0.0, sum_d1 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = dens[i] - precompute_[i];
            if (d0)
                sum_new += (pcol[i] * scale - dens[i]) * fl;
            if (d1)
                sum_d1 += stats::dnorm(data_[i], mu, beta_) * fl;
        }
        a0 = d0 ? sum_new * 2.0 : 0.0;
        a1 = d1 ? sum_d1 * (-2.0 * scale) : 0.0;
    }

    void gradfunvec(const std::vector<double>& mu, const std::vector<double>& dens,
                    bool d0, bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        std::vector<double> fullden(n);
        double fd_dot_dens = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = dens[i] - precompute_[i];
            fullden[i] = fl;
            fd_dot_dens += fl * dens[i];
        }
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
                    s += col[i] * fullden[i];
                a0[j] = s * 2.0 * scale - fd_dot_dens * 2.0;
            }
            if (d1) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += stats::dnorm(data_[i], mu[j], beta_) * fullden[i];
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

    void prepare(const std::vector<double>& grid) override {
        kc_.init(data_, [this](double x, double mu) {
            return stats::pnorm(x, mu, beta_, true);
        });
        for (double mu : grid)
            kc_.column(mu);
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

    void gradfun(double mu, const std::vector<double>& dens, bool d0, bool d1,
                 double& a0, double& a1) const override {
        if (!d0 && !d1) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        std::vector<double> s1(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = dens[i] + precompute_[i];
            s1[i] = w1_[i] / fl - w2_[i] / (1.0 - fl);
        }
        // The CDF part from the once-per-point cache (the cheap pdf part is
        // direct); ascending-i accumulation as the reference par_sum1.
        const std::vector<double>& pcol = d0 ? kc_.column(mu) : kc_.empty();
        double s1_dot_new = 0.0;
        if (d0)
            for (std::size_t i = 0; i < n; ++i)
                s1_dot_new += s1[i] * (pcol[i] * scale + precompute_[i]);
        double s1_dot_d1 = 0.0;
        if (d1)
            for (std::size_t i = 0; i < n; ++i)
                s1_dot_d1 += s1[i] * stats::dnorm(data_[i], mu, beta_) * scale;
        // C++: (s1.dot(new) + sum(w2/(1-fl))) * -1 + 2n
        double sum_w2_term = 0.0;
        if (d0)
            for (std::size_t i = 0; i < n; ++i)
                sum_w2_term += w2_[i] / (1.0 - (dens[i] + precompute_[i]));
        a0 = d0 ? (s1_dot_new + sum_w2_term) * -1.0 + 2.0 * static_cast<double>(len_)
                : 0.0;
        a1 = d1 ? s1_dot_d1 : 0.0;
    }

    void gradfunvec(const std::vector<double>& mu, const std::vector<double>& dens,
                    bool d0, bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        const double sum_w2_term =
            d0 ? detail::par_sum1(n, [this, &dens](std::size_t i) {
                    return w2_[i] / (1.0 - (dens[i] + precompute_[i]));
                })
               : 0.0;
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
                for (std::size_t i = 0; i < n; ++i) {
                    const double fl = dens[i] + precompute_[i];
                    s += (col[i] * scale + precompute_[i]) *
                         (w1_[i] / fl - w2_[i] / (1.0 - fl));
                }
                a0[j] = s * -1.0 + 2.0 * static_cast<double>(len_) -
                        sum_w2_term;
            }
            if (k1) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double fl = dens[i] + precompute_[i];
                    s += stats::dnorm(data_[i], mu[j], beta_) *
                         (w1_[i] / fl - w2_[i] / (1.0 - fl));
                }
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
        for (double mu : grid)
            kc_.column(mu);
    }

    const std::vector<double>& precompute() const override { return precompute_; }

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

    void gradfun(double mu, const std::vector<double>& dens, bool d0, bool d1,
                 double& a0, double& a1) const override {
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        const std::vector<double>& col = kc_.column(mu);
        double sum_temp = 0.0, dens_dot_fl = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = 1.0 / (dens[i] + precompute_[i]);
            sum_temp += col[i] * scale * fl;
            dens_dot_fl += dens[i] * fl;
        }
        a0 = dens_dot_fl - sum_temp;
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu, const std::vector<double>& dens,
                    bool d0, bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        std::vector<double> fullden(n);
        double dens_dot_fl = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double fl = 1.0 / (dens[i] + precompute_[i]);
            fullden[i] = fl;
            dens_dot_fl += dens[i] * fl;
        }
        const std::size_t k = d0 ? m : 0;
        if (k == 0)
            return;
        for (std::size_t j = 0; j < m; ++j)
            if (d0) {
                const std::vector<double>& col = kc_.column(mu[j]);
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    s += col[i] * fullden[i];
                a0[j] = dens_dot_fl - s * scale;
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

    double lossfunction(const std::vector<double>& maps) const override {
        return -detail::par_sum1(len_, [this, &maps](std::size_t i) {
            return std::log(maps[i] + precompute_[i]) * weights_[i];
        });
    }

    std::vector<double> mapping(const std::vector<double>& mu0,
                                const std::vector<double>& pi0) const override {
        return kern::dnppois(data_, mu0, pi0);
    }

    void gradfun(double mu, const std::vector<double>& dens, bool d0, bool d1,
                 double& a0, double& a1) const override {
        (void)d1;
        if (!d0) {
            a0 = a1 = 0.0;
            return;
        }
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        a0 = detail::par_sum1(n, [this, mu, &dens, scale](std::size_t i) {
            const double d = dens[i] + precompute_[i];
            const double fl = weights_[i] / d;
            return (dens[i] - kern::pois_pmf_c(data_[i], mu) * scale) * fl;
        });
        a1 = 0.0;
    }

    void gradfunvec(const std::vector<double>& mu, const std::vector<double>& dens,
                    bool d0, bool d1, std::vector<double>& a0,
                    std::vector<double>& a1) const override {
        (void)d1;
        const std::size_t m = mu.size();
        a0.assign(m, 0.0);
        a1.assign(m, 0.0);
        if (m == 0)
            return;
        const double scale = 1.0 - sum_pi0fixed();
        const std::size_t n = len_;
        std::vector<double> fullden(n);
        double dens_dot = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = dens[i] + precompute_[i];
            fullden[i] = weights_[i] / d;
            dens_dot += dens[i] * fullden[i];
        }
        const std::size_t k = d0 ? m : 0;
        if (k == 0)
            return;
        const std::vector<double> acc =
            detail::par_acc(n, k, [this, &mu, m, &fullden](std::size_t i, double* a) {
                const double base = fullden[i];
                const double xi = data_[i];
                for (std::size_t j = 0; j < m; ++j)
                    a[j] += kern::pois_pmf_c(xi, mu[j]) * base;
            });
        for (std::size_t j = 0; j < m; ++j)
            if (d0)
                a0[j] = dens_dot - acc[j] * scale;
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
};

}  // namespace fam
}  // namespace npfc

#endif  // NPFIC_FAMILIES_H
