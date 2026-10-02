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
          beta_inv_(beta_.inverse()) {
        precompute_ = mapping(mu0fixed_, pi0fixed_);
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
        const Eigen::VectorXd fullden = (dens + precompute_).cwiseInverse();
        const double scale = 1.0 - pi0fixed_.sum();
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
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
        if (std::getenv("NPFIXEDCOMPY_PROFILE") != nullptr)
            ++objevals_;
        double ansd0;
        Eigen::RowVectorXd grad1(2);
        gradfun(x, dens_, ansd0, grad1, true, true);
        grad = grad1.transpose();
        return ansd0;
    }

    // New-support-point search: box-constrained L-BFGS-B from the midpoint of
    // every grid cell; keep the cells whose objective is negative (port of
    // the R `solvegrad`).
    Eigen::MatrixXd solvegrad(const Eigen::VectorXd& dens, double tol) const {
        Eigen::MatrixXd ans(0, 2);
        double fval;
        Eigen::VectorXd xval(2), lb(2), ub(2);
        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = tol;
        param.max_linesearch = 100;
        param.max_iterations = 100;
        LBFGSpp::LBFGSBSolver<double> solver(param);
        setdens(dens);
        const Eigen::Index G = gridpoints_.rows();
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

        Eigen::VectorXd maxgrad;
        Eigen::MatrixXd _g2;
        gradfunvec(resultpt_, mapping(resultpt_, resultpr_), maxgrad, _g2, true,
                   false);
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
        a.ll = lossfunction(mapping(resultpt_, resultpr_));
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

    void setdens(const Eigen::VectorXd& d) const { dens_ = d; }

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
    mutable long iter_;
    // NPFIXEDCOMPY_PROFILE=1 diagnostic: L-BFGS-B objective evaluations.
    mutable long objevals_ = 0;
    mutable int convergence_;
    const int verbose_;
    Eigen::MatrixXd resultpt_;
    Eigen::VectorXd resultpr_;
};

}  // namespace fam
}  // namespace npfc

#endif  // NPFIC_FAM2D_H
