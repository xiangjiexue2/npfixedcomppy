// npfc_famnd.h — the multivariate-normal (N-d) mixing family `npnormND`
// (maximum likelihood, experimental).
//
// The N-d generalization of the R package's bivariate family
// `npnorm2Dll` (R's `src/npnorm2Dll.cpp` + the 2D collapse/simplify
// utilities + `densityND.h`), generalized from 2-D to arbitrary
// dimension `k >= 2`:
//
//   * the support points are k-vectors;
//   * the new-support-point search is a box-constrained L-BFGS-B run
//     over every cell of the (tensor-product) grid — each axis has its
//     own 1-D grid and the cells are the Cartesian product;
//   * the objective is the linearised loss increment with the TRUE
//     directional derivative (each data point's gradient contribution
//     weighted by 1/(dens + pre)), the same per-evaluation
//     explicit-inverse / preallocated-buffer hot path, and the same
//     CNM working-set warm start + negative-gain early stop the 2-D
//     version had;
//   * `k == 2` keeps the fully unrolled hot path bit-identical to the
//     established 2-D golden trajectory (same arithmetic, same
//     operation order).
//
// Unlike the one-dimensional families this does NOT go through the
// `MixSolver`/`Family` machinery — it is a standalone solver mirroring
// the R class, with the vendored LBFGS-B (LBFGSB.h + LBFGSpp/).
#ifndef NPFIC_FAMND_H
#define NPFIC_FAMND_H

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
// N-d mixing utilities (port of `simplifymix2D` / `collapsemix2D`,
// generalized to k columns)
// ---------------------------------------------------------------------------

// Drop points with |pi| <= 1e-14 (port of `simplifymix2D`).
inline void simplifymix(Eigen::MatrixXd& mu0, Eigen::VectorXd& pi0) {
    if (mu0.rows() != 1) {
        const Eigen::Index k = mu0.cols();
        std::vector<Eigen::Index> keep;
        for (Eigen::Index i = 0; i < pi0.size(); ++i)
            if (std::abs(pi0[i]) > 1e-14)
                keep.push_back(i);
        if (static_cast<std::size_t>(keep.size()) !=
            static_cast<std::size_t>(pi0.size())) {
            Eigen::MatrixXd mu0new(static_cast<Eigen::Index>(keep.size()), k);
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
inline void collapsemix(Eigen::MatrixXd& mu0, Eigen::VectorXd& pi0,
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
            simplifymix(mu0, pi0);
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
// the multivariate-normal family
// ---------------------------------------------------------------------------

class NpNormND {
public:
    NpNormND(Eigen::MatrixXd data, Eigen::MatrixXd mu0fixed,
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
          dim_(data_.cols()),
          iter_(0),
          convergence_(0),
          verbose_(verbose),
          beta_inv_(beta_.inverse()) {
        precompute_ = mapping(mu0fixed_, pi0fixed_);
        // Explicit inverse of `beta_` (once, row-major in `bi_`): the
        // per-evaluation quadratic form becomes a division-free
        // `d^T beta^-1 d` over the explicit entries (0 divisions per
        // point, where the Cholesky-solve kernel needs several).
        bi_.resize(static_cast<std::size_t>(dim_) * dim_);
        for (Eigen::Index r = 0; r < dim_; ++r)
            for (Eigen::Index c = 0; c < dim_; ++c)
                bi_[static_cast<std::size_t>(r) * dim_ + c] = beta_(r, c);
        if (dim_ == 2) {
            // The four explicit-inverse scalars the unrolled 2-D hot path
            // reads directly (same values as `bi_`).
            const double detb =
                beta_(0, 0) * beta_(1, 1) - beta_(0, 1) * beta_(1, 0);
            inv00_ = beta_(1, 1) / detb;
            inv01_ = -beta_(0, 1) / detb;
            inv10_ = -beta_(1, 0) / detb;
            inv11_ = beta_(0, 0) / detb;
        }
        // Same normalizing constant as `dnormNDarray` (R's grouping).
        base_ = stats::LN_SQRT_2PI * dim_ + 0.5 * std::log(beta_.determinant());
        buf_q_.resize(len_);
        buf_t_.resize(len_);
        buf_d_.resize(dim_);
        buf_s_.resize(dim_);
        // COLUMN-major (n x k): each axis is a contiguous column — the
        // per-evaluation hot path reads these column pointers, NOT an
        // interleaved row-major layout (see the 2-D data-read lesson).
        xcol_.resize(dim_);
        for (Eigen::Index c = 0; c < dim_; ++c)
            xcol_[c] = data_.col(c).data();
        // Tensor-product grid: `G_` points per axis, `(G_ - 1)` cells per
        // axis, `ncells_` cells in total.
        G_ = gridpoints_.rows();
        ncells_ = 1;
        for (Eigen::Index c = 0; c < dim_; ++c)
            ncells_ *= (G_ - 1);
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

    // Gradient of the loss at a single support point `mu` (k-vector):
    //   d0 = (dens - temp) . (1/(dens + precompute))      (probability dir)
    //   d1 = temp^T (mu 1^T - data) beta^-1               (k support dirs)
    void gradfun(const Eigen::VectorXd& mu, const Eigen::VectorXd& dens,
                 double& ansd0, Eigen::RowVectorXd& ansd1, bool d0, bool d1)
        const {
        // `fullden`/`scale` are per-`solvegrad` invariants cached by
        // `setdens` (see there for the bit-exactness argument).
        const Eigen::VectorXd& fullden = fullden_;
        const double scale = scale_;
        Eigen::MatrixXd mu1xk(1, dim_);
        mu1xk.row(0) = mu.transpose();
        Eigen::VectorXd scalev(1);
        scalev[0] = scale;
        const Eigen::VectorXd temp =
            kern::dnpnormND(data_, mu1xk, scalev, beta_, false).reshaped();
        if (d0)
            ansd0 = (dens - temp).dot(fullden);
        if (d1) {
            const Eigen::MatrixXd murep = mu1xk.replicate(len_, 1);
            ansd1 = temp.transpose() * (murep - data_) * beta_inv_;
        }
    }

    void gradfunvec(const Eigen::MatrixXd& mu, const Eigen::VectorXd& dens,
                    Eigen::VectorXd& ansd0, Eigen::MatrixXd& ansd1, bool d0,
                    bool d1) const {
        const Eigen::Index k = mu.rows();
        ansd0.resize(k);
        ansd1.resize(k, dim_);
        for (Eigen::Index i = 0; i < k; ++i) {
            const Eigen::VectorXd mui = mu.row(i).transpose();
            Eigen::RowVectorXd g1(dim_);
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

    // N-d collapse (port of the R `collapse`).
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
            collapsemix(mu0new, pi0new, prec);
            const double nll = lossfunction(mapping(mu0new, pi0new));
            if (nll <= ll + ntol) {
                pi0 = pi0new;
                mu0 = mu0new;
            } else {
                break;
            }
        }
        simplifymix(mu0, pi0);
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
    // The per-evaluation work is (1) the single-component mixture
    // density `temp = N(data; mu, beta) * scale`, (2)
    // `ansd0 = (dens - temp) . (1/(dens + precompute))`, and (3) the k
    // support directions `ansd1 = temp^T (mu - data^T) beta^-1`. The
    // kernels do all three in one pass over the n rows with:
    //
    //   * the explicit-inverse quadratic form (0 divisions per point;
    //     the Cholesky-solve kernel needs several),
    //   * no per-evaluation heap temporaries (the `len_`-vectors in
    //     `buf_q_`/`buf_t_` are allocated once in the constructor),
    //   * the SAME Eigen SIMD `exp` as the `dnormNDarray` fast path —
    //     the log-densities are written to a buffer and exponentiated
    //     with `Map::array().exp()`, the identical instruction sequence
    //     the kernel path uses.
    //
    // The returned gradient is the TRUE directional derivative of the
    // returned objective. The objective is the linearised loss
    // increment
    //   g(mu) = sum_i (dens_i - t_i(mu)) / F_i,  F_i = dens_i + pre_i,
    // so  dg/dmu_a = sum_i (t_i / F_i)(mu_a - x_ia), beta^-1 applied at
    // the end — each point's contribution is weighted by 1/F_i. 1-D's
    // `d1` families weight by 1/F_i (their `a1` IS dg/dmu, which is
    // what makes `brmin`/`dfmin` true derivative methods); the R 2-D
    // port returned the unweighted `sum_i t_i (mu - x_i) beta^-1`
    // (-grad of the unnormalised KDE, which only aligns with g's
    // gradient when 1/F_i is ~constant over the data). Weighting costs
    // one multiply per point in pass 3; the OBJECTIVE arithmetic
    // (ansd0) is left bit-identical, so every value-based decision
    // (accept / f_stop / grid_gain) is unchanged.
    //
    // `k == 2` keeps the fully unrolled hot path bit-identical to the
    // established 2-D golden trajectory: same explicit-inverse
    // scalars, same pass structure, same operation order. The `k > 2`
    // path is the same three passes over the column-major data with a
    // general (division-free) quadratic form.
    //
    // `gradfun` keeps the Cholesky-based kernel for the remaining
    // (non-hot) callers (`gradfunvec` at `get_ans`, the `min_gradient`
    // certificate); the two are numerically identical to ~1e-13
    // relative on the VALUE.
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
        if (std::getenv("NPFIXEDCOMPY_PROFILE") != nullptr)
            ++objevals_;
        if (dim_ == 2) {
            const double m0 = x[0], m1 = x[1];
            // COLUMN-major (n x 2): the x coordinate is the first column
            // and the y coordinate the second — NOT interleaved.
            // (The 2-D data-read lesson: the row-major `data_[2*i]` /
            // `[2*i+1]` read paired consecutive x's as (x, y)
            // coordinates and minimized the objective of a GARBAGE
            // dataset — measured certificate gain -518 vs the correct
            // ~+7e-3.) The R port reads via `x(i, 0)` / `x(i, 1)`; the
            // two column pointers do the same.
            const double* xd0 = xcol_[0];
            const double* xd1 = xcol_[1];
            const double* dd = dens_.data();
            const double* fd = fullden_.data();
            double* qb = buf_q_.data();
            // Pass 1: log single-component densities. The quadratic form
            // uses the explicit inverse (0 divisions):
            // d^T beta^-1 d = inv00*d0^2 + (inv01+inv10)*d0*d1 + inv11*d1^2.
            for (Eigen::Index i = 0; i < len_; ++i) {
                const double d0 = xd0[i] - m0;
                const double d1 = xd1[i] - m1;
                const double quad =
                    inv00_ * d0 * d0 + (inv01_ + inv10_) * d0 * d1 +
                    inv11_ * d1 * d1;
                qb[i] = -0.5 * quad - base_;
            }
            // Pass 2: the SAME Eigen SIMD exp as the `dnormNDarray`
            // fast path.
            Eigen::Map<Eigen::VectorXd> tm(buf_t_.data(), len_);
            tm = Eigen::Map<const Eigen::VectorXd>(qb, len_).array().exp() *
                 scale_;
            // Pass 3: the three scalar reductions, one fused loop, no
            // heap. The gradient accumulates the WEIGHTED sums
            // q_i = t_i * fd[i] (true directional derivative, see
            // above); ansd0 itself keeps the original unweighted
            // product (dens - t) * fd, bit-identical.
            const double* tb = buf_t_.data();
            double s0 = 0.0, s1 = 0.0, ansd0 = 0.0;
            for (Eigen::Index i = 0; i < len_; ++i) {
                const double t = tb[i];
                const double q = t * fd[i];
                s0 += (m0 - xd0[i]) * q;
                s1 += (m1 - xd1[i]) * q;
                ansd0 += (dd[i] - t) * fd[i];
            }
            // ansd1 = [s0 s1] * beta^-1 (explicit inverse; see the ctor).
            grad[0] = s0 * inv00_ + s1 * inv01_;
            grad[1] = s0 * inv10_ + s1 * inv11_;
            return ansd0;
        }
        // General k >= 3: the same three passes, general division-free
        // quadratic form over the explicit inverse.
        const double* dd = dens_.data();
        const double* fd = fullden_.data();
        double* qb = buf_q_.data();
        for (Eigen::Index i = 0; i < len_; ++i) {
            for (Eigen::Index a = 0; a < dim_; ++a)
                buf_d_[a] = xcol_[a][i] - x[a];
            double quad = 0.0;
            for (Eigen::Index a = 0; a < dim_; ++a)
                for (Eigen::Index b = 0; b < dim_; ++b)
                    quad += bi_[static_cast<std::size_t>(a) * dim_ + b] *
                            buf_d_[a] * buf_d_[b];
            qb[i] = -0.5 * quad - base_;
        }
        Eigen::Map<Eigen::VectorXd> tm(buf_t_.data(), len_);
        tm = Eigen::Map<const Eigen::VectorXd>(qb, len_).array().exp() *
             scale_;
        const double* tb = buf_t_.data();
        for (Eigen::Index a = 0; a < dim_; ++a)
            buf_s_[a] = 0.0;
        double ansd0 = 0.0;
        for (Eigen::Index i = 0; i < len_; ++i) {
            const double t = tb[i];
            const double q = t * fd[i];
            for (Eigen::Index a = 0; a < dim_; ++a)
                buf_s_[a] += (x[a] - xcol_[a][i]) * q;
            ansd0 += (dd[i] - t) * fd[i];
        }
        // grad[a] = sum_b s[b] * (beta^-1)[a, b].
        for (Eigen::Index a = 0; a < dim_; ++a) {
            double g = 0.0;
            for (Eigen::Index b = 0; b < dim_; ++b)
                g += buf_s_[b] * bi_[static_cast<std::size_t>(a) * dim_ + b];
            grad[a] = g;
        }
        return ansd0;
    }

    // New-support-point search (port of the 2-D relaxed search,
    // docs/PERF.md §4a.3, generalized to the tensor-product grid). For
    // every grid cell the L-BFGS-B run starts from the PREVIOUS call's
    // refined point for that cell (the CNM working-set hot start) when
    // it is still inside the cell, else from the cell midpoint.
    // `minimize` evaluates its start point FIRST, and the objective-value
    // early stop `f_stop = 0` (the negative-gain early stop) makes it
    // return IMMEDIATELY when that gain — or any later evaluated gain —
    // is negative: a negative grid point costs exactly ONE evaluation
    // and is accepted as-is (the weights are re-solved by
    // `computeweights` anyway), and a cell whose refined point
    // re-verifies negative skips the search entirely. No cell ever
    // costs more evaluations than the original loop, because the start
    // point is either the original midpoint or a hot start.
    // WARM-ROOT POLICY: a run terminated by the early stop returns the
    // first NEGATIVE point, not a minimum — storing it as the next
    // call's start lets the per-cell local search converge to a
    // positive local minimum and MISS a negative region. So
    // early-stopped runs DROP their entry (the next call restarts from
    // the midpoint), and only a CONVERGED non-negative minimum is
    // stored.
    Eigen::MatrixXd solvegrad(const Eigen::VectorXd& dens, double tol) const {
        setdens(dens);
        Eigen::MatrixXd ans(0, dim_);
        double fval;
        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = tol;
        param.max_linesearch = 100;
        param.max_iterations = 100;
        param.f_stop = 0.0;
        LBFGSpp::LBFGSBSolver<double> solver(param);
        const std::size_t Gm = static_cast<std::size_t>(G_ - 1);
        // Per-axis strides so that cell index `idx` has axis-0 the
        // SLOWEST varying digit and axis k-1 the fastest: for k == 2
        // this is exactly the original `i * (G-1) + j` (outer axis 0,
        // inner axis 1) loop order and key.
        std::vector<std::size_t> stride(dim_);
        std::size_t acc = 1;
        for (Eigen::Index a = dim_ - 1; a >= 0; --a) {
            stride[a] = acc;
            acc *= Gm;
        }
        Eigen::VectorXd lb(dim_), ub(dim_), xval(dim_);
        // `warm_` PERSISTS across calls (no clear): every cell is
        // visited on every sweep, so each entry is overwritten in place.
        for (Eigen::Index idx = 0; idx < ncells_; ++idx) {
            for (Eigen::Index a = 0; a < dim_; ++a) {
                const std::size_t c =
                    static_cast<std::size_t>(idx) / stride[a] % Gm;
                lb[a] = gridpoints_(static_cast<Eigen::Index>(c), a);
                ub[a] = gridpoints_(static_cast<Eigen::Index>(c) + 1, a);
                xval[a] = 0.5 * (lb[a] + ub[a]);
            }
            // Start point: the previous call's refined point for this
            // cell when it is still inside it, else the cell midpoint.
            const auto it = warm_.find(idx);
            if (it != warm_.end()) {
                const Eigen::VectorXd& r = it->second;
                bool inside = true;
                for (Eigen::Index a = 0; a < dim_; ++a)
                    inside = inside && r[a] > lb[a] && r[a] < ub[a];
                if (inside)
                    xval = r;
            }
            solver.minimize(*this, xval, fval, lb, ub);
            if (fval < 0.0) {
                // Early-stopped (negative) run: drop the warm entry;
                // next call restarts from the midpoint.
                warm_.erase(idx);
                ans.conservativeResize(ans.rows() + 1, dim_);
                ans.bottomRows(1) = xval.transpose();
            } else {
                // Converged non-negative minimum: a safe hot start.
                warm_[idx] = xval;
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
            mu0.conservativeResize(mu0.rows() + newpoints.rows(), dim_);
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
                         "[npnormND] iters=%ld objevals=%ld "
                         "n=%ld dim=%ld grid=%ld support=%ld\n",
                         iter_, objevals_, (long)len_, (long)dim_,
                         (long)gridpoints_.rows(), (long)resultpt_.rows());
    }

    // Final answer (port of the R `get_ans`): the non-fixed points
    // stacked over the fixed ones, plus the minimum d0 gradient and the
    // likelihood.
    struct Ans {
        std::vector<std::vector<double>> pt;  // k x dim (row-major)
        std::vector<double> pr;
        double ll;
        long iter;
        int convergence;
        double min_gradient;
        // Grid-level certificate: the minimum gain over all grid-cell
        // midpoints at the final mixture (computed in `get_ans`; the
        // N-d analog of the 1-D `grid_gain`, docs/PERF.md §4a.3).
        double grid_gain = std::numeric_limits<double>::quiet_NaN();
        std::vector<std::vector<double>> beta;
        std::string family;
        std::string flag;
    };
    Ans get_ans() const {
        const Eigen::Index nres = resultpt_.rows();
        const Eigen::Index nfix = mu0fixed_.rows();
        Eigen::MatrixXd mu0new(nres + nfix, dim_);
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
        Eigen::MatrixXd _gk;
        gradfunvec(resultpt_, fdens, maxgrad, _gk, true, false);
        double ming = 0.0;
        if (maxgrad.size() > 0)
            ming = maxgrad.minCoeff();
        else
            ming = 0.0;

        Ans a;
        const Eigen::Index k = mu0new.rows();
        a.pt.resize(k);
        for (Eigen::Index i = 0; i < k; ++i)
            for (Eigen::Index c = 0; c < dim_; ++c)
                a.pt[i].push_back(mu0new(i, c));
        a.pr.assign(pi0new.data(), pi0new.data() + pi0new.size());
        // Grid-level certificate (N-d): the gains over ALL grid-cell
        // midpoints at the FINAL mixture — exactly the zero-cost
        // candidate set the search itself starts from (each cell's
        // L-BFGS-B run evaluates its midpoint first), so a negative
        // value here means a direction outside both the support and
        // the search's own hot starts. Informational only (grid
        // resolution dependent), like the 1-D `grid_gain`; it does not
        // gate anything.
        {
            Eigen::VectorXd mid(dim_);
            std::vector<std::size_t> stride(dim_);
            std::size_t acc = 1;
            for (Eigen::Index c = dim_ - 1; c >= 0; --c) {
                stride[c] = acc;
                acc *= static_cast<std::size_t>(G_ - 1);
            }
            double gmin = std::numeric_limits<double>::infinity();
            for (Eigen::Index idx = 0; idx < ncells_; ++idx) {
                for (Eigen::Index c = 0; c < dim_; ++c) {
                    const std::size_t g =
                        static_cast<std::size_t>(idx) / stride[c] %
                        static_cast<std::size_t>(G_ - 1);
                    mid[c] = 0.5 *
                             (gridpoints_(static_cast<Eigen::Index>(g), c) +
                              gridpoints_(static_cast<Eigen::Index>(g) + 1, c));
                }
                Eigen::VectorXd gg(dim_);
                const double gm = (*this)(mid, gg);
                if (gm < gmin)
                    gmin = gm;
            }
            a.grid_gain = (gmin == std::numeric_limits<double>::infinity())
                              ? std::numeric_limits<double>::quiet_NaN()
                              : gmin;
        }
        a.ll = lossfunction(fdens);
        a.iter = iter_;
        a.convergence = convergence_;
        a.min_gradient = ming;
        a.beta.resize(dim_);
        for (Eigen::Index i = 0; i < dim_; ++i)
            for (Eigen::Index j = 0; j < dim_; ++j)
                a.beta[i].push_back(beta_(i, j));
        a.family = (dim_ == 2) ? "npnorm2D" : "npnormND";
        a.flag = "d1";
        return a;
    }

    void setdens(const Eigen::VectorXd& d) const {
        dens_ = d;
        // Per-call invariants of `gradfun`, cached once per `solvegrad`:
        // `fullden = 1/(dens + precompute)` and the single-component
        // scaling `scale = 1 - pi0fixed.sum()`. Both are constant for
        // all objective/gradient evaluations inside one per-cell search
        // pass (dens does not change until the next `setdens`), so
        // computing them once instead of ~10^5 times saves ~10^6
        // divisions and one `pi0fixed.sum()` per evaluation. The
        // arithmetic applied to the cached values is bit-identical to
        // the old per-call form.
        fullden_ = (d + precompute_).cwiseInverse();
        scale_ = 1.0 - pi0fixed_.sum();
    }

private:
    void fprintf_pts(const Eigen::MatrixXd& mu0, const Eigen::VectorXd& pi0) {
        std::fprintf(stderr, "support points:\n");
        for (Eigen::Index i = 0; i < mu0.rows(); ++i) {
            std::fprintf(stderr, "  ");
            for (Eigen::Index c = 0; c < mu0.cols(); ++c)
                std::fprintf(stderr, "%s%g", c ? " " : "", mu0(i, c));
            std::fprintf(stderr, "\n");
        }
        std::fprintf(stderr, "probabilities:\n");
        for (Eigen::Index i = 0; i < pi0.size(); ++i)
            std::fprintf(stderr, "  %g\n", pi0[i]);
    }

    Eigen::MatrixXd data_;
    Eigen::MatrixXd mu0fixed_;
    Eigen::VectorXd pi0fixed_;
    Eigen::MatrixXd beta_;
    // Inverse of `beta_`, precomputed once: it is a fixed argument of
    // the family, and `gradfun` re-computed it on every objective
    // evaluation.
    Eigen::MatrixXd beta_inv_;
    Eigen::MatrixXd initpt_;
    Eigen::VectorXd initpr_;
    Eigen::MatrixXd gridpoints_;  // (G x dim): axis c is column c
    const Eigen::Index len_;
    const Eigen::Index dim_;
    mutable Eigen::VectorXd precompute_;
    mutable Eigen::VectorXd dens_;
    // Cached invariants of the per-cell objective (see `setdens`).
    mutable Eigen::VectorXd fullden_;
    mutable double scale_ = 1.0;
    // Explicit inverse of `beta_` row-major (the general hot path reads
    // it division-free); the four scalars the unrolled 2-D path reads
    // directly (same values).
    std::vector<double> bi_;
    double inv00_ = 0.0, inv01_ = 0.0, inv10_ = 0.0, inv11_ = 0.0;
    double base_ = 0.0;
    // Per-evaluation scratch (sized once in the ctor): `buf_q_` holds
    // the log densities, `buf_t_` the exponentiated ones, `buf_d_` the
    // per-point displacement (general path), `buf_s_` the weighted
    // sums.
    mutable Eigen::VectorXd buf_q_, buf_t_, buf_d_, buf_s_;
    // COLUMN-major (n x dim) data: each axis is a contiguous column;
    // the hot path reads these pointers, not an interleaved layout.
    std::vector<const double*> xcol_;
    // Tensor-product grid: `G_` points per axis, `ncells_` cells.
    Eigen::Index G_ = 0;
    Eigen::Index ncells_ = 0;
    mutable long iter_;
    // NPFIXEDCOMPY_PROFILE=1 diagnostic: L-BFGS-B objective evaluations.
    mutable long objevals_ = 0;
    // Per-cell refined roots from PREVIOUS `solvegrad` calls (the CNM
    // working-set hot start), keyed by the cell index (axis 0 slowest,
    // axis k-1 fastest; for k == 2 the original `i * (G-1) + j`).
    // Persists across calls; a root that re-verifies negative is
    // dropped, a stale one just serves as a cheaper L-BFGS-B start.
    mutable std::map<Eigen::Index, Eigen::VectorXd> warm_;
    mutable int convergence_;
    const int verbose_;
    Eigen::MatrixXd resultpt_;
    Eigen::VectorXd resultpr_;
};

}  // namespace fam
}  // namespace npfc

#endif  // NPFIC_FAMND_H
