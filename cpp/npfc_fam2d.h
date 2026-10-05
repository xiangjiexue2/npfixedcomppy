// npfc_fam2d.h — the bivariate-normal mixing family `npnorm2Dll`
// (maximum likelihood, experimental). A faithful port of the R package's
// `src/npnorm2Dll.cpp` + `inst/include/npfixedcomp.h` 2D collapse/simplify
// + `inst/include/densityND.h`.
//
// Unlike the one-dimensional families, the support points are 2-vectors and
// the new-support-point search is a box-constrained L-BFGS-B run over every
// cell of the (tensor-product) grid, so this family does NOT go through the
// 1D `MixSolver`/`Family` machinery. It is a standalone solver mirroring the
// R `npnorm2Dll` class, including its per-cell LBFGS-B `solvegrad`, its
// 2D `collapse`, and its `computeweights` (the `pnnlssum_`/`pnnqp_` NNLS
// subproblem). The vendored LBFGS-B solver (LBFGSB.h + LBFGSpp/) is the same
// source the R build uses, minus the `<RcppEigen.h>` include.
#ifndef NPFIC_FAM2D_H
#define NPFIC_FAM2D_H

#include "npfc_kernels.h"
#include "npfc_nnls.h"

#include "LBFGSB.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace npfc {
namespace fam {

// ---------------------------------------------------------------------------
// 2D mixing utilities (port of `simplifymix2D` / `collapsemix2D`)
// ---------------------------------------------------------------------------

// Drop points with |pi| <= 1e-14 (port of `simplifymix2D`).
inline void simplifymix2d(Eigen::MatrixXd& mu0, Eigen::VectorXd& pi0) {
    if (mu0.rows() != 1) {
        std::vector<Eigen::Index> keep;
        for (Eigen::Index i = 0; i < pi0.size(); ++i)
            if (std::abs(pi0[i]) > 1e-14)
                keep.push_back(i);
        if (static_cast<std::size_t>(keep.size()) !=
            static_cast<std::size_t>(pi0.size())) {
            Eigen::MatrixXd mu0new(static_cast<Eigen::Index>(keep.size()), 2);
            for (std::size_t r = 0; r < keep.size(); ++r)
                mu0new.row(r) = mu0.row(keep[r]);
            Eigen::VectorXd pi0new(static_cast<Eigen::Index>(keep.size()));
            for (std::size_t r = 0; r < keep.size(); ++r)
                pi0new[r] = pi0[keep[r]];
            mu0 = mu0new;
            pi0 = pi0new;
        }
    }
}

// Merge the closest pair of points (Euclidean) while the minimum pairwise
// distance is within `prec` (port of `collapsemix2D`).
inline void collapsemix2d(Eigen::MatrixXd& mu0, Eigen::VectorXd& pi0,
                          double prec) {
    bool foo = false;
    if (mu0.rows() > 1) {
        const Eigen::Index n = mu0.rows();
        Eigen::MatrixXd distmat(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                distmat(i, j) = (mu0.row(i) - mu0.row(j)).norm();
        for (Eigen::Index i = 0; i < n; ++i)
            distmat(i, i) = std::numeric_limits<double>::infinity();
        foo = (distmat.array() <= prec).any();
        double temp;
        while (foo) {
            Eigen::Index mini, minj;
            distmat.minCoeff(&mini, &minj);
            temp = pi0[mini] + pi0[minj];
            mu0.row(mini) =
                (mu0.row(mini) * pi0[mini] + mu0.row(minj) * pi0[minj]) / temp;
            pi0[minj] = 0.0;
            pi0[mini] = temp;
            simplifymix2d(mu0, pi0);
            if (mu0.rows() <= 1) {
                foo = false;
            } else {
                const Eigen::Index n2 = mu0.rows();
                Eigen::MatrixXd dm(n2, n2);
                for (Eigen::Index i = 0; i < n2; ++i)
                    for (Eigen::Index j = 0; j < n2; ++j)
                        dm(i, j) = (mu0.row(i) - mu0.row(j)).norm();
                for (Eigen::Index i = 0; i < n2; ++i)
                    dm(i, i) = std::numeric_limits<double>::infinity();
                distmat = dm;
                foo = (distmat.array() <= prec).any();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the bivariate-normal family
// ---------------------------------------------------------------------------

class NpNorm2D {
public:
    NpNorm2D(Eigen::MatrixXd data, Eigen::MatrixXd mu0fixed,
             Eigen::VectorXd pi0fixed, Eigen::MatrixXd beta,
             Eigen::MatrixXd initpt, Eigen::VectorXd initpr,
             Eigen::MatrixXd gridpoints, int verbose)
        : data_(std::move(data)),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          beta_(std::move(beta)),
          initpt_(std::move(initpt)),
          initpr_(std::move(initpr)),
          gridpoints_(std::move(gridpoints)),
          len_(data_.rows()),
          iter_(0),
          convergence_(0),
          verbose_(verbose),
          exact2d_(std::getenv("NPFIC_2D_EXACT") != nullptr),
          beta_inv_(beta_.inverse()) {
        precompute_ = mapping(mu0fixed_, pi0fixed_);
        // Explicit 2x2 inverse of `beta_` (once): the per-evaluation
        // quadratic form becomes `inv00*d0^2 + 2*inv01*d0*d1 +
        // inv11*d1^2` — 0 divisions per point, where the Cholesky-solve
        // kernel needs 4 (see the fast path in `dnormNDarray`).
        const double detb =
            beta_(0, 0) * beta_(1, 1) - beta_(0, 1) * beta_(1, 0);
        inv00_ = beta_(1, 1) / detb;
        inv01_ = -beta_(0, 1) / detb;
        inv10_ = -beta_(1, 0) / detb;
        inv11_ = beta_(0, 0) / detb;
        // Same normalizing constant as `dnormNDarray` (R's grouping).
        base_ = stats::LN_SQRT_2PI * 2 + 0.5 * std::log(beta_.determinant());
        buf_q_.resize(len_);
        buf_t_.resize(len_);
    }

    // Loss `-sum log(maps + precompute)`.
    double lossfunction(const Eigen::VectorXd& maps) const {
        return -(maps + precompute_).array().log().sum();
    }

    // Mixture density `sum_j pi0[j] N(data[i]; mu0[j], beta)`.
    Eigen::VectorXd mapping(const Eigen::MatrixXd& mu0,
                            const Eigen::VectorXd& pi0) const {
        return kern::dnpnormND(data_, mu0, pi0, beta_, false).reshaped();
    }

    // Gradient of the loss at a single support point `mu` (2-vector):
    //   d0 = (dens - temp) . (1/(dens + precompute))      (probability dir)
    //   d1 = temp^T (mu 1^T - data) beta^-1               (2 support dirs)
    void gradfun(const Eigen::VectorXd& mu, const Eigen::VectorXd& dens,
                 double& ansd0, Eigen::RowVectorXd& ansd1, bool d0, bool d1)
        const {
        // `fullden`/`scale` are per-`solvegrad` invariants cached by
        // `setdens` (see there for the bit-exactness argument).
        const Eigen::VectorXd& fullden = fullden_;
        const double scale = scale_;
        Eigen::MatrixXd mu1x2(1, 2);
        mu1x2(0, 0) = mu[0];
        mu1x2(0, 1) = mu[1];
        Eigen::VectorXd scalev(1);
        scalev[0] = scale;
        const Eigen::VectorXd temp =
            kern::dnpnormND(data_, mu1x2, scalev, beta_, false).reshaped();
        if (d0)
            ansd0 = (dens - temp).dot(fullden);
        if (d1) {
            const Eigen::MatrixXd murep = mu1x2.replicate(len_, 1);
            ansd1 = temp.transpose() * (murep - data_) * beta_inv_;
        }
    }

    void gradfunvec(const Eigen::MatrixXd& mu, const Eigen::VectorXd& dens,
                    Eigen::VectorXd& ansd0, Eigen::MatrixXd& ansd1, bool d0,
                    bool d1) const {
        const Eigen::Index k = mu.rows();
        ansd0.resize(k);
        ansd1.resize(k, 2);
        for (Eigen::Index i = 0; i < k; ++i) {
            const Eigen::VectorXd mui = mu.row(i).transpose();
            Eigen::RowVectorXd g1(2);
            gradfun(mui, dens, ansd0[i], g1, d0, d1);
            ansd1.row(i) = g1;
        }
    }

    // Armijo backtracking line search on the weight step (port of
    // `checklossfun2`).
    void checklossfun2(const Eigen::VectorXd& diff, Eigen::VectorXd& pi0,
                       const Eigen::VectorXd& eta, const Eigen::VectorXd& p,
                       const Eigen::VectorXd& dens) const {
        const double llorigin = lossfunction(dens);
        double sigma = 2.0, alpha = 0.3333;
        const double con = -p.dot(eta);
        Eigen::VectorXd ans = pi0;
        while (true) {
            sigma *= 0.5;
            const double lhs = lossfunction(dens + sigma * diff);
            const double rhs = llorigin + alpha * sigma * con;
            if (lhs < rhs) {
                ans = pi0 + sigma * eta;
                break;
            }
            if (sigma < 0.001)
                break;
        }
        pi0 = ans;
    }

    // 2D collapse (port of the R `collapse`).
    void collapse(Eigen::MatrixXd& mu0, Eigen::VectorXd& pi0,
                  double tol = 1e-6) const {
        const double ll = lossfunction(mapping(mu0, pi0));
        const double ntol = std::max(tol * 0.1, ll * 1e-16);
        Eigen::MatrixXd mu0new = mu0;
        Eigen::VectorXd pi0new = pi0;
        double prec;
        while (true) {
            if (mu0.rows() <= 1)
                break;
            const Eigen::Index n = mu0.rows();
            Eigen::MatrixXd distmat(n, n);
            for (Eigen::Index i = 0; i < n; ++i)
                for (Eigen::Index j = 0; j < n; ++j)
                    distmat(i, j) = (mu0.row(i) - mu0.row(j)).norm();
            for (Eigen::Index i = 0; i < n; ++i)
                distmat(i, i) = std::numeric_limits<double>::infinity();
            prec = 10.0 * distmat.minCoeff();
            collapsemix2d(mu0new, pi0new, prec);
            const double nll = lossfunction(mapping(mu0new, pi0new));
            if (nll <= ll + ntol) {
                pi0 = pi0new;
                mu0 = mu0new;
            } else {
                break;
            }
        }
        simplifymix2d(mu0, pi0);
    }

    // Weight subproblem (port of the R `computeweights`).
    void computeweights(const Eigen::MatrixXd& mu0, Eigen::VectorXd& pi0,
                        const Eigen::VectorXd& dens) const {
        const Eigen::Index m = mu0.rows();
        if (m == 0)
            return;
        const Eigen::VectorXd fp = dens + precompute_;
        const Eigen::MatrixXd sp = kern::dnormNDarray(data_, mu0, beta_, false);
        const Eigen::MatrixXd tp = sp.array().colwise() / fp.array();
        const double sum = 1.0 - pi0fixed_.sum();
        std::vector<double> nw;
        if (len_ > 1000) {
            const Eigen::MatrixXd q = (tp.transpose() * tp).eval();
            const Eigen::VectorXd p =
                (tp.transpose() *
                 (precompute_.cwiseQuotient(fp) -
                  Eigen::VectorXd::Constant(len_, 2.0)))
                    .eval();
            nw = nnls::pnnqp(q.data(), static_cast<std::size_t>(m), p.data(),
                             sum);
        } else {
            const Eigen::VectorXd b =
                (Eigen::VectorXd::Constant(len_, 2.0) -
                 precompute_.cwiseQuotient(fp))
                    .eval();
            nw = nnls::pnnlssum(tp.data(), static_cast<std::size_t>(len_),
                                static_cast<std::size_t>(m), b.data(), sum);
        }
        // (The vendored Eigen 5.0.0 range-constructor SFINAE rejects
        // std::vector iterators, so wrap the `nw` buffer in a Map.)
        const Eigen::Map<const Eigen::VectorXd> nwv(nw.data(),
                                                    static_cast<Eigen::Index>(nw.size()));
        const Eigen::VectorXd diffv = sp * nwv - dens;
        const Eigen::VectorXd pv = tp.colwise().sum().reshaped();
        checklossfun2(diffv, pi0, nwv - pi0, pv, dens);
    }

    // L-BFGS-B objective + gradient (the `operator()` the solver calls).
    //
    // Hot-path specialization (this is the only caller of the
    // probability+support gradient pair, called ~7.7e5 times on the n=300
    // benchmark): the per-evaluation work is (1) the single-component
    // mixture density `temp = N(data; mu, beta) * scale`, (2)
    // `ansd0 = (dens - temp) . (1/(dens + precompute))`, and (3) the two
    // support directions `ansd1 = temp^T (mu - data^T) beta^-1`. The
    // kernel below does all three in one pass over the n rows with:
    //
    //   * the explicit-inverse quadratic form
    //     `inv00*d0^2 + 2*inv01*d0*d1 + inv11*d1^2` (0 divisions per
    //     point; the Cholesky-solve kernel needs 4),
    //   * no per-evaluation heap temporaries (the two `len_`-vectors in
    //     `buf_q_`/`buf_t_` are allocated once in the constructor),
    //   * the SAME Eigen SIMD `exp` as the `dnormNDarray` fast path —
    //     the log-densities are written to a buffer and exponentiated
    //     with `Map::array().exp()`, the identical instruction sequence
    //     the kernel path uses (an elementwise scalar `std::exp` drifts
    //     by 1 ulp and was rejected during development).
    //
    // The returned gradient is the TRUE directional derivative of the
    // returned objective. The objective is the linearised loss increment
    //   g(mu) = sum_i (dens_i - t_i(mu)) / F_i,  F_i = dens_i + pre_i,
    // so  dg/dmu_k = sum_i (t_i / F_i)(mu_k - x_ik), beta^-1 applied at
    // the end — each point's contribution is weighted by 1/F_i. The
    // R-ported `gradfun` (and the `NPFIC_2D_EXACT=1` path) returns the
    // UNWEIGHTED `sum_i t_i (mu - x_i) beta^-1` instead: that is
    // -grad(K), K = scale * sum_i N(x_i; mu, beta) — the gradient of the
    // unnormalised KDE, which only aligns with g's gradient when 1/F_i
    // is ~constant over the data. 1-D's `d1` families weight by 1/F_i
    // (their `a1` IS dg/dmu, which is what makes `brmin`/`dfmin` true
    // derivative methods); the 2-D port did not, so L-BFGS-B was
    // following KDE directions of the gain objective. Weighting costs
    // one multiply per point in pass 3; the OBJECTIVE arithmetic above
    // (ansd0) is left bit-identical, so every value-based decision
    // (accept / f_stop / grid_gain) is unchanged.
    //
    // `gradfun` keeps the original Cholesky-based kernel for the remaining
    // (non-hot) callers (`gradfunvec` at `get_ans`, the `min_gradient`
    // certificate, and the exact path); the two are numerically
    // identical to ~1e-13 relative on the VALUE.
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
        if (std::getenv("NPFIXEDCOMPY_PROFILE") != nullptr)
            ++objevals_;
        if (exact2d_) {
            // R-identical path (NPFIC_2D_EXACT=1): the original
            // Cholesky-based kernel via `gradfun` — the bit-exact 0.2.1
            // trajectory (see the fast path below for why the two
            // differ).
            double ansd0;
            Eigen::RowVectorXd grad1(2);
            gradfun(x, dens_, ansd0, grad1, true, true);
            grad = grad1.transpose();
            return ansd0;
        }
        const double m0 = x[0], m1 = x[1];
        // COLUMN-major (n x 2): the x coordinate is the first column and
        // the y coordinate the second — NOT interleaved. The previous
        // `data_.data()[2*i]` / `[2*i+1]` read assumed row-major layout,
        // which pairs consecutive x's (and later consecutive y's) as
        // (x, y) coordinates: the fast path was minimizing the objective
        // of a GARBAGE dataset (measured: certificate gain -518 at the
        // final fit, while the correct-data gain is -8.8, and the whole
        // mixture was a fit to the corrupted data). The R port reads via
        // `x(i, 0)` / `x(i, 1)`; the two column pointers do the same.
        const double* xd0 = data_.col(0).data();
        const double* xd1 = data_.col(1).data();
        const double* dd = dens_.data();
        const double* fd = fullden_.data();
        double* qb = buf_q_.data();
        // Pass 1: log single-component densities, `N(data; mu, beta)`.
        // The quadratic form uses the explicit inverse (0 divisions):
        // d^T beta^-1 d = inv00*d0^2 + (inv01+inv10)*d0*d1 + inv11*d1^2.
        for (Eigen::Index i = 0; i < len_; ++i) {
            const double d0 = xd0[i] - m0;
            const double d1 = xd1[i] - m1;
            const double quad =
                inv00_ * d0 * d0 + (inv01_ + inv10_) * d0 * d1 +
                inv11_ * d1 * d1;
            qb[i] = -0.5 * quad - base_;
        }
        // Pass 2: the SAME Eigen SIMD exp as the `dnormNDarray` fast path.
        Eigen::Map<Eigen::VectorXd> tm(buf_t_.data(), len_);
        tm = Eigen::Map<const Eigen::VectorXd>(qb, len_).array().exp() *
             scale_;
        // Pass 3: the three scalar reductions, one fused loop, no heap.
        // The gradient accumulates the WEIGHTED sums q_i = t_i * fd[i]
        // (see the true-gradient note above); ansd0 itself keeps the
        // original unweighted product (dens - t) * fd, bit-identical.
        const double* tb = buf_t_.data();
        double s0 = 0.0, s1 = 0.0, ansd0 = 0.0;
        for (Eigen::Index i = 0; i < len_; ++i) {
            const double t = tb[i];
            const double q = t * fd[i];
            s0 += (m0 - xd0[i]) * q;
            s1 += (m1 - xd1[i]) * q;
            ansd0 += (dd[i] - t) * fd[i];
        }
        // ansd1 = [s0 s1] * beta^-1 (beta^-1 from the explicit inverse;
        // see the ctor).
        grad[0] = s0 * inv00_ + s1 * inv01_;
        grad[1] = s0 * inv10_ + s1 * inv11_;
        return ansd0;
    }

    // New-support-point search (2-D port of the 1-D 0.2.3 relaxed search,
    // docs/PERF.md §4a.3). For every grid cell the L-BFGS-B run starts
    // from the PREVIOUS call's refined point for that cell (the CNM
    // working-set hot start, the 2-D analog of the 1-D
    // `warm_root_d1_`/`warm_root_d0_` arrays) when it is still inside the
    // cell, else from the cell midpoint (the original R/0.2.1 start).
    // `minimize` evaluates its start point FIRST, and the objective-value
    // early stop `f_stop = 0` (the 2-D analog of the 1-D negative-gain
    // early stop in `brmin`/`dfmin`) makes it return IMMEDIATELY when
    // that gain — or any later evaluated gain — is negative: a negative
    // grid point costs exactly ONE evaluation and is accepted as-is (the
    // weights are re-solved by `computeweights` anyway), and a cell whose
    // refined point re-verifies negative skips the search entirely. No
    // cell ever costs more evaluations than the original loop, because
    // the start point is either the original midpoint or a hot start.
    // WARM-ROOT POLICY: a run terminated by the early stop returns the
    // first NEGATIVE point, not a minimum — storing it as the next call's
    // start lets the per-cell local search converge to a positive local
    // minimum and MISS a negative region (measured: grid_gain = -518 at
    // the final fit vs +2.7e-7 for the original loop). So early-stopped
    // runs DROP their entry (the next call restarts from the midpoint —
    // the original behavior), and only a CONVERGED non-negative minimum
    // is stored: the next call's start is then always either the original
    // midpoint or a verified minimum, and no cell ever costs more
    // evaluations than the original loop.
    // `NPFIC_2D_EXACT=1` runs the ORIGINAL loop (L-BFGS-B from the fixed
    // midpoint, no hot start, no early stop) — the bit-exact 0.2.1
    // trajectory.
    Eigen::MatrixXd solvegrad(const Eigen::VectorXd& dens, double tol) const {
        setdens(dens);
        const Eigen::Index G = gridpoints_.rows();
        if (exact2d_) {
            Eigen::MatrixXd ans(0, 2);
            double fval;
            Eigen::VectorXd xval(2), lb(2), ub(2);
            LBFGSpp::LBFGSBParam<double> param;
            param.epsilon = tol;
            param.max_linesearch = 100;
            param.max_iterations = 100;
            LBFGSpp::LBFGSBSolver<double> solver(param);
            for (Eigen::Index i = 0; i < G - 1; ++i) {
                for (Eigen::Index j = 0; j < G - 1; ++j) {
                    lb[0] = gridpoints_(i, 0);
                    lb[1] = gridpoints_(j, 1);
                    ub[0] = gridpoints_(i + 1, 0);
                    ub[1] = gridpoints_(j + 1, 1);
                    xval = (lb + ub) * 0.5;
                    solver.minimize(*this, xval, fval, lb, ub);
                    if (fval < 0.0) {
                        ans.conservativeResize(ans.rows() + 1, 2);
                        ans.bottomRows(1) = xval.transpose();
                    }
                }
            }
            return ans;
        }
        // Fast path (see the comment above): hot start + grid eval with
        // the negative-gain early stop `f_stop = 0`.
        Eigen::MatrixXd ans(0, 2);
        double fval;
        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = tol;
        param.max_linesearch = 100;
        param.max_iterations = 100;
        param.f_stop = 0.0;
        LBFGSpp::LBFGSBSolver<double> solver(param);
        const std::size_t Gm = static_cast<std::size_t>(G - 1);
        // `warm_2d_` PERSISTS across calls (no clear): every cell is
        // visited on every sweep, so each entry is overwritten in place —
        // the 1-D arrays are updated the same way (a root that
        // re-verifies negative is accepted at one evaluation, a stale
        // one just serves as a cheaper L-BFGS-B start).
        for (Eigen::Index i = 0; i < G - 1; ++i) {
            for (Eigen::Index j = 0; j < G - 1; ++j) {
                const std::size_t key =
                    static_cast<std::size_t>(i) * Gm +
                    static_cast<std::size_t>(j);
                Eigen::VectorXd lb(2), ub(2);
                lb[0] = gridpoints_(i, 0);
                lb[1] = gridpoints_(j, 1);
                ub[0] = gridpoints_(i + 1, 0);
                ub[1] = gridpoints_(j + 1, 1);
                // Start point: the previous call's refined point for this
                // cell when it is still inside it (the hot start / warm
                // re-verification), else the cell midpoint (the original
                // R/0.2.1 start). `minimize` evaluates this point FIRST,
                // and the `f_stop = 0` hook returns immediately when the
                // gain is already negative — the grid eval and the
                // zero-cost acceptance collapse into that one evaluation.
                Eigen::VectorXd xval((Eigen::VectorXd(2) <<
                                         0.5 * (lb[0] + ub[0]),
                                     0.5 * (lb[1] + ub[1]))
                                         .finished());
                const auto it = warm_2d_.find(key);
                if (it != warm_2d_.end()) {
                    const Eigen::VectorXd& r = it->second;
                    if (r(0) > lb[0] && r(0) < ub[0] && r(1) > lb[1] &&
                        r(1) < ub[1])
                        xval = r;
                }
                solver.minimize(*this, xval, fval, lb, ub);
                if (fval < 0.0) {
                    // Early-stopped (negative) run: the point is not a
                    // minimum — drop the warm entry (see the WARM-ROOT
                    // POLICY above); next call restarts from the midpoint.
                    warm_2d_.erase(key);
                    ans.conservativeResize(ans.rows() + 1, 2);
                    ans.bottomRows(1) = xval.transpose();
                } else {
                    // Converged non-negative minimum: a safe hot start
                    // for the next call (re-verified at one evaluation).
                    warm_2d_[key] = xval;
                }
            }
        }
        return ans;
    }

    // Outer iteration loop (port of the R `computemixdist`).
    void computemixdist(double tol, long maxit) {
        Eigen::MatrixXd mu0 = initpt_;
        Eigen::VectorXd pi0 = initpr_ * (1.0 - pi0fixed_.sum());
        iter_ = 0;
        Eigen::MatrixXd newpoints;
        Eigen::VectorXd dens = mapping(mu0, pi0);
        const double closs0 = lossfunction(dens);
        double closs = closs0, nloss = std::numeric_limits<double>::quiet_NaN();
        while (true) {
            newpoints = solvegrad(dens, tol);
            mu0.conservativeResize(mu0.rows() + newpoints.rows(), 2);
            pi0.conservativeResize(pi0.size() + newpoints.rows());
            mu0.bottomRows(newpoints.rows()) = newpoints;
            pi0.tail(newpoints.rows()) = Eigen::VectorXd::Zero(newpoints.rows());

            if (verbose_ >= 1) {
                std::fprintf(stderr, "Iteration: %ld with loss %g\n", iter_,
                             nloss);
                fprintf_pts(mu0, pi0);
            }
            computeweights(mu0, pi0, dens);
            collapse(mu0, pi0);
            iter_ += 1;
            dens = mapping(mu0, pi0);
            nloss = lossfunction(dens);

            if (closs - nloss < tol) {
                convergence_ = 0;
                break;
            }
            if (iter_ > maxit) {
                convergence_ = 1;
                break;
            }
            closs = nloss;
        }
        resultpt_ = mu0;
        resultpr_ = pi0;
        if (std::getenv("NPFIXEDCOMPY_PROFILE") != nullptr)
            std::fprintf(stderr,
                         "[npnorm2Dll] iters=%ld objevals=%ld "
                         "n=%ld grid=%ld support=%ld\n",
                         iter_, objevals_, (long)len_,
                         (long)gridpoints_.rows(), (long)resultpt_.rows());
    }

    // Final answer (port of the R `get_ans`): the non-fixed points stacked
    // over the fixed ones, plus the minimum d0 gradient and the likelihood.
    struct Ans {
        std::vector<std::vector<double>> pt;  // k x 2 (row-major)
        std::vector<double> pr;
        double ll;
        long iter;
        int convergence;
        double min_gradient;
        // Grid-level certificate: the minimum gain over all grid-cell
        // midpoints at the final mixture (computed in `get_ans`; the 2-D
        // analog of the 1-D `grid_gain`, docs/PERF.md §4a.3).
        double grid_gain = std::numeric_limits<double>::quiet_NaN();
        std::vector<std::vector<double>> beta;
        std::string family;
        std::string flag;
    };
    Ans get_ans() const {
        const Eigen::Index nres = resultpt_.rows();
        const Eigen::Index nfix = mu0fixed_.rows();
        Eigen::MatrixXd mu0new(nres + nfix, 2);
        mu0new.topRows(nres) = resultpt_;
        mu0new.bottomRows(nfix) = mu0fixed_;
        Eigen::VectorXd pi0new(nres + nfix);
        pi0new.head(nres) = resultpr_;
        pi0new.tail(nfix) = pi0fixed_;

        // The final-mixture density, once (both certificates below use
        // it); `setdens` refreshes the `fullden_`/`scale_` cache that
        // `gradfun` reads, so the min_gradient sweep below evaluates
        // against the FINAL mixture rather than the stale cache left by
        // the last `solvegrad`.
        const Eigen::VectorXd fdens = mapping(resultpt_, resultpr_);
        setdens(fdens);
        Eigen::VectorXd maxgrad;
        Eigen::MatrixXd _g2;
        gradfunvec(resultpt_, fdens, maxgrad, _g2, true, false);
        double ming = 0.0;
        if (maxgrad.size() > 0)
            ming = maxgrad.minCoeff();
        else
            ming = 0.0;

        Ans a;
        const Eigen::Index k = mu0new.rows();
        a.pt.resize(k);
        for (Eigen::Index i = 0; i < k; ++i) {
            a.pt[i].push_back(mu0new(i, 0));
            a.pt[i].push_back(mu0new(i, 1));
        }
        a.pr.assign(pi0new.data(), pi0new.data() + pi0new.size());
        // Grid-level certificate (2-D): the gains over ALL grid-cell
        // midpoints at the FINAL mixture — exactly the zero-cost candidate
        // set the search itself starts from (each cell's L-BFGS-B run
        // evaluates its midpoint first), so a negative value here means a
        // direction outside both the support and the search's own hot
        // starts. Informational only (grid resolution dependent), like the
        // 1-D `grid_gain`; it does not gate anything.
        {
            Eigen::VectorXd gg(2);
            double gmin = std::numeric_limits<double>::infinity();
            const Eigen::Index G = gridpoints_.rows();
            for (Eigen::Index i = 0; i < G - 1; ++i) {
                for (Eigen::Index j = 0; j < G - 1; ++j) {
                    const double m0 =
                        0.5 * (gridpoints_(i, 0) + gridpoints_(i + 1, 0));
                    const double m1 =
                        0.5 * (gridpoints_(j, 1) + gridpoints_(j + 1, 1));
                    const Eigen::VectorXd mid(
                        (Eigen::VectorXd(2) << m0, m1).finished());
                    const double gm = (*this)(mid, gg);
                    if (gm < gmin)
                        gmin = gm;
                }
            }
            a.grid_gain = (gmin == std::numeric_limits<double>::infinity())
                              ? std::numeric_limits<double>::quiet_NaN()
                              : gmin;
        }
        a.ll = lossfunction(fdens);
        a.iter = iter_;
        a.convergence = convergence_;
        a.min_gradient = ming;
        a.beta.resize(2);
        for (int i = 0; i < 2; ++i) {
            a.beta[i].push_back(beta_(i, 0));
            a.beta[i].push_back(beta_(i, 1));
        }
        a.family = "npnorm2D";
        a.flag = "d1";
        return a;
    }

    void setdens(const Eigen::VectorXd& d) const {
        dens_ = d;
        // Per-call invariants of `gradfun`, cached once per `solvegrad`:
        // `fullden = 1/(dens + precompute)` and the single-component
        // scaling `scale = 1 - pi0fixed.sum()`. Both are constant for all
        // objective/gradient evaluations inside one per-cell search pass
        // (dens does not change until the next `setdens`), so computing
        // them once instead of ~7.7e5 times saves ~10^6 divisions and
        // one `pi0fixed.sum()` per evaluation. The arithmetic applied to
        // the cached values is bit-identical to the old per-call form
        // (`fullden` is still elementwise `1/(dens+precompute)`).
        fullden_ = (d + precompute_).cwiseInverse();
        scale_ = 1.0 - pi0fixed_.sum();
    }

private:
    void fprintf_pts(const Eigen::MatrixXd& mu0, const Eigen::VectorXd& pi0) {
        std::fprintf(stderr, "support points:\n");
        for (Eigen::Index i = 0; i < mu0.rows(); ++i)
            std::fprintf(stderr, "  %g %g\n", mu0(i, 0), mu0(i, 1));
        std::fprintf(stderr, "probabilities:\n");
        for (Eigen::Index i = 0; i < pi0.size(); ++i)
            std::fprintf(stderr, "  %g\n", pi0[i]);
    }

    Eigen::MatrixXd data_;
    Eigen::MatrixXd mu0fixed_;
    Eigen::VectorXd pi0fixed_;
    Eigen::MatrixXd beta_;
    // Inverse of `beta_`, precomputed once: it is a fixed argument of the
    // family, and `gradfun` re-computed it on every objective evaluation.
    Eigen::MatrixXd beta_inv_;
    Eigen::MatrixXd initpt_;
    Eigen::VectorXd initpr_;
    Eigen::MatrixXd gridpoints_;
    const Eigen::Index len_;
    mutable Eigen::VectorXd precompute_;
    mutable Eigen::VectorXd dens_;
    // Cached invariants of the per-cell objective (see `setdens`).
    mutable Eigen::VectorXd fullden_;
    mutable double scale_ = 1.0;
    // Explicit 2x2 inverse of `beta_` and the normalizing constant
    // (see the ctor; used by the hot path in `operator()`).
    double inv00_ = 0.0, inv01_ = 0.0, inv10_ = 0.0, inv11_ = 0.0;
    double base_ = 0.0;
    // Per-evaluation scratch buffers (sized once in the ctor): `buf_q_`
    // holds the log densities, `buf_t_` the exponentiated ones.
    mutable Eigen::VectorXd buf_q_, buf_t_;
    mutable long iter_;
    // NPFIXEDCOMPY_PROFILE=1 diagnostic: L-BFGS-B objective evaluations.
    mutable long objevals_ = 0;
    // Per-cell refined roots from PREVIOUS `solvegrad` calls (fast path;
    // the CNM working-set hot start, the 2-D analog of the 1-D
    // `warm_root_d1_`/`warm_root_d0_` arrays), keyed by
    // `i * (G-1) + j` over grid cells. Persists across calls (every cell
    // is visited on every sweep, so entries are overwritten in place);
    // a root that re-verifies negative is accepted at one evaluation, a
    // stale one just serves as a cheaper L-BFGS-B start.
    mutable std::map<std::size_t, Eigen::VectorXd> warm_2d_;
    mutable int convergence_;
    const int verbose_;
    // NPFIC_2D_EXACT=1 selects the bit-exact R-identical objective path
    // (see `operator()`); the default is the fast path.
    const bool exact2d_;
    Eigen::MatrixXd resultpt_;
    Eigen::VectorXd resultpr_;
};

}  // namespace fam
}  // namespace npfc

#endif  // NPFIC_FAM2D_H
