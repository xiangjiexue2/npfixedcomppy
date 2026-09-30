// npfc_corrmatrix.h — projection of a symmetric matrix onto the cone of
// correlation matrices (PSD with unit diagonal).
//
// Faithful port of npfixedcomp2's `correlationmatrixcpp` (src/correlationmatrix.cpp),
// the Qi & Sun (2006) projected-gradient method. Every helper (eigendecomposition
// convention, the dual-gradient `Corrsub_gradient`, the `PCA` reconstruction, the
// `omega_mat` / `Jacobian_matrix` / `precond_matrix` CG pieces and the outer
// backtracking loop) is reproduced verbatim so the result agrees with R to
// working precision. The `MyEigen` row-reversal quirk is kept: the probe
// (tests/probe_corrmat.R) shows the R output equals the plain eigen-clipping
// projection to ~1e-11, so the quirk is neutralised by the algorithm — keeping
// it guarantees parity with R by construction.
//
// This is the only genuinely heavy primitive needed by `covestEB` /
// `covestEB.cor`; the rest of those entry points is thin NumPy orchestration.
#ifndef NPFIC_CORRMATRIX_H
#define NPFIC_CORRMATRIX_H

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace npfc {
namespace corr {

// Eigen decomposition with the R package's exact conventions: eigenvalues in
// DESCENDING order, eigenvector matrix row-reversed. (See file comment.)
inline void my_eigen(const Eigen::MatrixXd& X, Eigen::MatrixXd& eigvec,
                     Eigen::VectorXd& eigval) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
    eig.compute(X);
    eigvec.resize(X.rows(), X.cols());
    eigvec = eig.eigenvectors().rowwise().reverse();
    eigval.resize(X.rows());
    eigval = eig.eigenvalues().reverse();
}

inline void corrsub_gradient(const Eigen::VectorXd& y, const Eigen::VectorXd& lambda,
                             const Eigen::MatrixXd& P, const Eigen::VectorXd& b,
                             const int& n, double& f, Eigen::VectorXd& Fy) {
    int r = static_cast<int>((lambda.array() > 0).count());
    if (r > 0) {
        Fy = P.leftCols(r).cwiseAbs2() * lambda.head(r);
        f = 0.5 * lambda.head(r).squaredNorm() - b.dot(y);
    } else {
        Fy.resize(n);
        Fy.setZero();
        f = 0.0;
    }
}

inline void pca(Eigen::MatrixXd& X, const Eigen::VectorXd& lambda,
                const Eigen::MatrixXd& P, const Eigen::VectorXd& b, const int& n) {
    if (n == 1) {
        X.resize(b.size(), 1);
        X.leftCols<1>() = b;
    } else {
        int r = static_cast<int>((lambda.array() > 0).count());
        if (r > 1 && r < n) {
            if (r <= 2) {
                X = P.leftCols(r) * lambda.head(r).asDiagonal() * P.leftCols(r).transpose();
            } else {
                X += P.rightCols(n - r) * lambda.tail(n - r).cwiseAbs().asDiagonal() *
                     P.rightCols(n - r).transpose();
            }
        } else {
            if (r == 0) {
                X.resize(n, n);
                X.setZero();
            } else {
                if (r == 1)
                    X = lambda[0] * P.leftCols<1>() * P.leftCols<1>().transpose();
            }
        }
        Eigen::VectorXd d = X.diagonal().cwiseMax(b);
        X.diagonal() = d;
        d = b.cwiseQuotient(d).cwiseSqrt();
        X = X.cwiseProduct(d * d.transpose());
    }
}

inline void omega_mat(const Eigen::VectorXd& lambda, const int& n,
                      Eigen::MatrixXd& Omega12) {
    int r = static_cast<int>((lambda.array() > 0).count());
    if (r > 0) {
        if (r < n) {
            Omega12.resize(r, n - r);
            Omega12 = lambda.head(r).replicate(1, n - r);
            Omega12 = Omega12.cwiseQuotient(Omega12.rowwise() - lambda.tail(n - r).transpose());
        } else {
            Omega12.resize(n, n);
            Omega12.setOnes();
        }
    } else {
        Omega12.resize(0, 0);
    }
}

inline void jacobian_matrix(const Eigen::VectorXd& d, const Eigen::MatrixXd& Omega12,
                            const Eigen::MatrixXd& P, const int& n, Eigen::VectorXd& Vd) {
    int r = static_cast<int>(Omega12.rows());
    Vd.resize(n);
    Vd.setZero();
    if (r > 0) {
        if (r < n) {
            Eigen::VectorXd hh = (P.leftCols(r) *
                                  Omega12.cwiseProduct(P.leftCols(r).transpose() * d.asDiagonal() *
                                                       P.rightCols(n - r)))
                                     .cwiseProduct(P.rightCols(n - r))
                                     .rowwise()
                                     .sum() *
                                  2.;
            if (static_cast<double>(r) < static_cast<double>(n) / 2.) {
                Vd = (P.leftCols(r) * P.leftCols(r).transpose()).cwiseAbs2() * d + hh +
                     1e-10 * d;
            } else {
                Eigen::MatrixXd PP2 = P.rightCols(n - r) * P.rightCols(n - r).transpose();
                Vd = d + PP2.cwiseAbs2() * d + hh -
                     2. * d.cwiseProduct(PP2.diagonal()) + 1e-10 * d;
            }
        } else {
            Vd = (1. + 1e-10) * d;
        }
    }
}

inline void precond_matrix(const Eigen::MatrixXd& Omega12, const Eigen::MatrixXd& P,
                           const int& n, Eigen::VectorXd& c) {
    int r = static_cast<int>(Omega12.rows());
    c.resize(n);
    c.setOnes();
    if (r > 1) {
        Eigen::MatrixXd H = P.cwiseAbs2().transpose();
        if (static_cast<double>(r) < static_cast<double>(n) / 2.) {
            c = H.topRows(r).colwise().sum().cwiseAbs2().transpose() +
                2. * (H.topRows(r).transpose() * Omega12)
                          .cwiseProduct(H.bottomRows(n - r).transpose())
                          .rowwise()
                          .sum();
        } else {
            if (r < n) {
                c = (H.colwise().sum().cwiseAbs2() -
                     H.bottomRows(n - r).colwise().sum().cwiseAbs2() -
                     2. * H.topRows(r)
                              .cwiseProduct((Eigen::MatrixXd::Ones(Omega12.rows(), Omega12.cols()) -
                                            Omega12) *
                                           H.bottomRows(n - r))
                              .colwise()
                              .sum())
                        .transpose();
            }
        }
    }
    c = c.cwiseMax(1e-8);
}

inline void pre_cg(const Eigen::VectorXd& b, const double& tol, const int& iter_CG,
                   const Eigen::VectorXd& c, const Eigen::MatrixXd& Omega12,
                   const Eigen::MatrixXd& P, const int& n, Eigen::VectorXd& p) {
    Eigen::VectorXd r(b);
    double n2b = b.norm(), tolb = tol * n2b;
    p.resize(n);
    p.setZero();
    Eigen::VectorXd z = r.cwiseQuotient(c);
    double rz1 = r.dot(z), rz2 = 1.;
    Eigen::VectorXd d(z);
    Eigen::VectorXd w;
    double denom;
    for (int k = 1; k <= iter_CG; ++k) {
        if (k > 1) {
            d = z + (rz1 / rz2) * d;
        }
        jacobian_matrix(d, Omega12, P, n, w);
        denom = d.dot(w);
        if (denom <= 0) {
            p = d / d.norm();
            break;
        } else {
            p += (rz1 / denom) * d;
            r -= (rz1 / denom) * w;
        }
        z = r.cwiseQuotient(c);
        if (r.norm() <= tolb) {
            break;
        }
        rz2 = rz1;
        rz1 = r.dot(z);
    }
}

// Project a symmetric n×n matrix `G` onto the correlation-matrix cone.
// `tau` (default 0) shifts the target diagonal; `tol` is the duality-gap
// tolerance (covestEB passes 1e-3). Returns the projected n×n matrix.
inline Eigen::MatrixXd correlationmatrixcpp(const Eigen::MatrixXd& G1,
                                            const double tau = 0.0,
                                            const double tol = 1e-6) {
    int n = static_cast<int>(G1.rows());
    Eigen::MatrixXd G(G1);
    Eigen::VectorXd b(n);
    b.setOnes();
    if (tau > 0) {
        b.array() -= tau;
        G.diagonal().array() -= tau;
    }

    Eigen::VectorXd b0(b);
    double error_tol = std::max(tol, 1e-12);

    Eigen::VectorXd y(n), Fy(n);
    y.setZero();
    Fy.setZero();

    int Iter_whole = 200, Iter_inner = 20, Iter_CG = 200, iter_k = 0;
    double tol_CG = 1e-2, G_1 = 1e-4;

    Eigen::VectorXd x0(y), c(n), d(n);
    c.setOnes();
    d.setZero();
    double val_G = G.squaredNorm() * 0.5;

    Eigen::MatrixXd X(G);
    X.diagonal() += y;
    X = (X + X.transpose()) / 2.;

    Eigen::MatrixXd P;
    Eigen::VectorXd lambda;
    my_eigen(X, P, lambda);

    double f0;
    corrsub_gradient(y, lambda, P, b0, n, f0, Fy);
    double val_dual = val_G - f0;
    pca(X, lambda, P, b0, n);
    double val_obj = (X - G).squaredNorm() * 0.5,
           gap = (val_obj - val_dual) / (1. + std::fabs(val_dual) + std::fabs(val_obj));
    double f = f0;
    b = b0 - Fy;

    double norm_b = b.norm(), norm_b0 = b0.norm() + 1., norm_b_rel = norm_b / norm_b0;
    Eigen::MatrixXd Omega12;
    omega_mat(lambda, n, Omega12);
    x0 = y;

    double slope;
    int k_inner;
    while (gap > error_tol && norm_b_rel > error_tol && iter_k < Iter_whole) {
        precond_matrix(Omega12, P, n, c);
        pre_cg(b, tol_CG, Iter_CG, c, Omega12, P, n, d);
        slope = d.dot(Fy - b0);
        y = x0 + d;
        X = G;
        X.diagonal() += y;
        X = (X + X.transpose()) / 2.;
        my_eigen(X, P, lambda);
        corrsub_gradient(y, lambda, P, b0, n, f, Fy);
        k_inner = 0;

        while (k_inner <= Iter_inner &&
               f > f0 + G_1 * std::pow(.5, static_cast<double>(k_inner)) * slope + 1e-6) {
            k_inner++;
            y = x0 + std::pow(0.5, static_cast<double>(k_inner)) * d;
            X = G;
            X.diagonal() += y;
            X = (X + X.transpose()) * 0.5;
            my_eigen(X, P, lambda);
            corrsub_gradient(y, lambda, P, b0, n, f, Fy);
        }

        x0 = y;
        f0 = f;
        val_dual = val_G - f0;
        pca(X, lambda, P, b0, n);
        val_obj = (X - G).squaredNorm() * 0.5;
        gap = (val_obj - val_dual) / (1. + std::fabs(val_dual) + std::fabs(val_obj));

        iter_k++;
        b = b0 - Fy;
        norm_b = b.norm();
        norm_b_rel = norm_b / norm_b0;

        omega_mat(lambda, n, Omega12);
    }

    X.diagonal().array() += tau;
    return X;
}

}  // namespace corr
}  // namespace npfc

#endif  // NPFIC_CORRMATRIX_H
