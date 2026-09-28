// npfc_nnls.cpp — non-negative least-squares (Lawson & Hanson, `pnnls.f`
// `k = 0`) and the `pnnlssum_` / `pnnqp_` weight-subproblem wrappers. A
// line-for-line port of npfixedcomppy/src/nnls.rs.
//
// The main `nnls` works on a raw column-major buffer (each column is a
// contiguous slice, which is exactly how Householder reflections and the
// back-substitution index the matrix). Counters (`nsetp`, `npp1`, `iz1`, ...)
// are kept 1-based to track the Fortran line-for-line; only the matrix/vector
// indexing converts.
#include "npfc_nnls.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>

namespace npfc {
namespace nnls {

namespace {

// G1: orthogonal rotation mapping `(a, b) -> (sig, 0)`.
inline void g1(double a, double b, double& c, double& s, double& sig) {
    if (std::abs(a) > std::abs(b)) {
        const double xr = b / a;
        const double yr = std::sqrt(1.0 + xr * xr);
        c = std::copysign(1.0, a) / yr;  // a.signum()
        s = c * xr;
        sig = std::abs(a) * yr;
    } else if (b != 0.0) {
        const double xr = a / b;
        const double yr = std::sqrt(1.0 + xr * xr);
        s = std::copysign(1.0, b) / yr;  // b.signum()
        c = s * xr;
        sig = std::abs(b) * yr;
    } else {
        // Fortran G1 zero case: C = 1, S = 0 (identity), SIG = 0.
        c = 1.0;
        s = 0.0;
        sig = 0.0;
    }
}

// H12 (construct, NCV = 0): build a Householder reflection from the pivot
// vector `col[lpivot0..m)`, setting `up` and `col[lpivot0] = cl`. Leaves
// `up` unchanged (the caller initialises it to 0) when the pivot is zero.
inline void hh_construct(double* col, std::size_t lpivot0, std::size_t m,
                         double& up) {
    const std::size_t l1 = lpivot0 + 1;  // first row zeroed (0-based)
    double cl = std::abs(col[lpivot0]);
    for (std::size_t j = l1; j < m; ++j)
        cl = std::max(cl, std::abs(col[j]));
    if (cl <= 0.0)
        return;
    const double clinv = 1.0 / cl;
    double sm = (col[lpivot0] * clinv) * (col[lpivot0] * clinv);
    for (std::size_t j = l1; j < m; ++j) {
        const double v = col[j] * clinv;
        sm += v * v;
    }
    cl = cl * std::sqrt(sm);
    // H12: `IF (U(1,LPIVOT) <= 0) GOTO 50; CL=-CL` — negate only when the
    // pivot element is strictly positive (numerically safe Householder).
    if (col[lpivot0] > 0.0)
        cl = -cl;
    up = col[lpivot0] - cl;
    col[lpivot0] = cl;
}

// H12 (apply): apply the reflection `(up, sub)` starting at row `lpivot0`,
// where `cl` is the post-construct pivot value `col[lpivot0]`.
inline void hh_apply(double* target, const double* sub, std::size_t lpivot0,
                     std::size_t m, double up, double cl) {
    const std::size_t l1 = lpivot0 + 1;
    double b = up * cl;
    if (b >= 0.0)
        return;
    b = 1.0 / b;
    double sm = target[lpivot0] * up;
    for (std::size_t i = l1; i < m; ++i)
        sm += target[i] * sub[i - l1];
    if (sm == 0.0)
        return;
    sm *= b;
    target[lpivot0] += sm * up;
    for (std::size_t i = l1; i < m; ++i)
        target[i] += sm * sub[i - l1];
}

// Back-substitution on the implicit upper-triangular factor (label 400).
// 1-based `nsetp`; `index` maps a set position -> 1-based column number.
inline void solve_tri(const double* A, const int* index, std::size_t nsetp,
                      double* zz, std::size_t m) {
    long jj = 0;
    for (std::size_t l = 1; l <= nsetp; ++l) {
        const std::size_t ip = nsetp + 1 - l;
        if (l != 1) {
            // Fortran label 400: DO 410 II=1,IP (INCLUSIVE: II runs 1..ip);
            // ZZ(IP+1) in 1-based is zz[ip] in 0-based.
            for (std::size_t ii = 1; ii <= ip; ++ii)
                zz[ii - 1] -=
                    A[(ii - 1) + (std::size_t)(jj - 1) * m] * zz[ip];
        }
        jj = index[ip - 1];
        zz[ip - 1] /= A[(ip - 1) + (std::size_t)(jj - 1) * m];
    }
}

}  // namespace

std::vector<double> nnls(std::vector<double> A, std::vector<double> B,
                         std::size_t m, std::size_t n) {
    double* a = A.data();
    double* b = B.data();
    std::vector<double> x(n, 0.0);
    std::vector<int> index(n);
    for (std::size_t i = 0; i < n; ++i)
        index[i] = static_cast<int>(i + 1);
    std::vector<double> w(n, 0.0);
    std::vector<double> zz(m);
    std::vector<double> sub(m);
    std::vector<double> zzl(m);
    int iter = 0;
    const int itmax = static_cast<int>(3 * n);
    int iz1 = 1;
    const int iz2 = static_cast<int>(n);
    int nsetp = 0;
    int npp1 = 1;
    // Fortran `GO TO 60` after a rejection skips the dual recomputation;
    // mirror that with a flag so a rejected column is not re-selected.
    bool skip_dual = false;

    while (true) {
        if (iz1 > iz2 || nsetp >= static_cast<int>(m))
            break;
        // dual vector for set Z (skipped right after a rejection)
        if (!skip_dual) {
            for (int iz = iz1; iz <= iz2; ++iz) {
                const int j = index[iz - 1];
                const std::size_t j0 = static_cast<std::size_t>(j - 1);
                double sm = 0.0;
                for (int l = npp1; l <= static_cast<int>(m); ++l)
                    sm += a[(l - 1) + j0 * m] * b[l - 1];
                w[j0] = sm;
            }
        }
        skip_dual = false;
        // largest positive dual
        double wmax = 0.0;
        int izmax = 0;
        for (int iz = iz1; iz <= iz2; ++iz) {
            const int j = index[iz - 1];
            if (w[j - 1] > wmax) {
                wmax = w[j - 1];
                izmax = iz;
            }
        }
        if (wmax <= 0.0)
            break;
        const int iz = izmax;
        const int j = index[iz - 1];
        const std::size_t j0 = static_cast<std::size_t>(j - 1);
        const double asave = a[(npp1 - 1) + j0 * m];
        // linear-independence test
        double unorm = 0.0;
        for (int l = 1; l <= nsetp; ++l)
            unorm += a[(l - 1) + j0 * m] * a[(l - 1) + j0 * m];
        unorm = std::sqrt(unorm);
        // construct the Householder reflection on column j
        double up = 0.0;
        hh_construct(a + j0 * m, static_cast<std::size_t>(npp1 - 1), m, up);
        const int nsub = static_cast<int>(m) - npp1;
        for (int k = 0; k < nsub; ++k)
            sub[k] = a[j0 * m + npp1 + k];
        const double cl = a[(npp1 - 1) + j0 * m];
        bool accept = false;
        if (unorm + std::abs(cl) * 0.01 - unorm > 0.0) {
            for (int i = 0; i < static_cast<int>(m); ++i)
                zzl[i] = b[i];
            hh_apply(zzl.data(), sub.data(), static_cast<std::size_t>(npp1 - 1),
                     m, up, cl);
            const double ztest = zzl[npp1 - 1] / cl;
            accept = ztest > 0.0;
            if (accept)
                for (int i = 0; i < static_cast<int>(m); ++i) b[i] = zzl[i];
        }
        if (!accept) {
            // REJECT: restore the column, zero its dual, and re-scan the
            // existing W without recomputing it (Fortran `GO TO 60`).
            // Recomputing the dual here would restore the rejected column's
            // positive value, re-select it, and loop forever on
            // near-dependent columns (the CLL hang).
            a[(npp1 - 1) + j0 * m] = asave;
            w[j0] = 0.0;
            skip_dual = true;
            continue;
        }
        // commit: move j from Z to P
        index[iz - 1] = index[iz1 - 1];
        index[iz1 - 1] = j;
        iz1 += 1;
        nsetp = npp1;
        npp1 += 1;
        for (int jz = iz1; jz <= iz2; ++jz) {
            const int jj = index[jz - 1];
            const std::size_t jj0 = static_cast<std::size_t>(jj - 1);
            hh_apply(a + jj0 * m, sub.data(), static_cast<std::size_t>(npp1 - 2),
                     m, up, cl);
        }
        for (int l = npp1; l <= static_cast<int>(m); ++l)
            a[(l - 1) + j0 * m] = 0.0;
        w[j0] = 0.0;
        // Fortran label 400 reads ZZ() as the work area holding the
        // transformed B (Q*B) at this point.
        for (int i = 0; i < static_cast<int>(m); ++i)
            zz[i] = b[i];
        solve_tri(a, index.data(), static_cast<std::size_t>(nsetp), zz.data(), m);
        // secondary loop
        while (true) {
            iter += 1;
            if (iter > itmax)
                break;
            double alpha = 2.0;
            int jj = 0;
            for (int ip = 1; ip <= nsetp; ++ip) {
                const int l = index[ip - 1];
                const std::size_t l0 = static_cast<std::size_t>(l - 1);
                if (zz[ip - 1] <= 0.0) {
                    const double t = -x[l0] / (zz[ip - 1] - x[l0]);
                    if (alpha > t) {
                        alpha = t;
                        jj = ip;
                    }
                }
            }
            if (alpha == 2.0) {
                for (int ip = 1; ip <= nsetp; ++ip) {
                    const int i = index[ip - 1];
                    x[i - 1] = zz[ip - 1];
                }
                break;
            }
            for (int ip = 1; ip <= nsetp; ++ip) {
                const int l = index[ip - 1];
                const std::size_t l0 = static_cast<std::size_t>(l - 1);
                x[l0] += alpha * (zz[ip - 1] - x[l0]);
            }
            // label 260: move an infeasible coefficient from P back to Z
            while (true) {
                const int i = index[jj - 1];
                const std::size_t i0 = static_cast<std::size_t>(i - 1);
                x[i0] = 0.0;
                if (jj != nsetp) {
                    jj += 1;
                    for (int j2 = jj; j2 <= nsetp; ++j2) {
                        const int ii = index[j2 - 1];
                        const std::size_t ii0 = static_cast<std::size_t>(ii - 1);
                        index[j2 - 2] = index[j2 - 1];
                        // Fortran: CALL G1 (A(J-1,II), A(J,II), CC, SS,
                        // A(J-1,II)) — the fifth argument stores SIG back into
                        // A(J-1,II), i.e. the rotated pivot element; A(J,II)
                        // is then zeroed explicitly.
                        double cc, ss, sig;
                        g1(a[(j2 - 2) + ii0 * m], a[(j2 - 1) + ii0 * m], cc, ss,
                           sig);
                        a[(j2 - 2) + ii0 * m] = sig;
                        a[(j2 - 1) + ii0 * m] = 0.0;
                        for (int l = 0; l < static_cast<int>(n); ++l) {
                            if (l != static_cast<int>(ii0)) {
                                const double temp =
                                    a[(j2 - 2) + static_cast<std::size_t>(l) * m];
                                a[(j2 - 2) + static_cast<std::size_t>(l) * m] =
                                    cc * temp + ss * a[(j2 - 1) + static_cast<std::size_t>(l) * m];
                                a[(j2 - 1) + static_cast<std::size_t>(l) * m] =
                                    -ss * temp + cc * a[(j2 - 1) + static_cast<std::size_t>(l) * m];
                            }
                        }
                        const double temp = b[j2 - 2];
                        b[j2 - 2] = cc * temp + ss * b[j2 - 1];
                        b[j2 - 1] = -ss * temp + cc * b[j2 - 1];
                    }
                }
                npp1 = nsetp;
                nsetp -= 1;
                // Fortran: IZ1=IZ1-1; INDEX(IZ1)=I  (1-based -> 0-based iz1-1)
                iz1 -= 1;
                index[iz1 - 1] = i;
                bool ok = true;
                for (int j2 = 1; j2 <= nsetp; ++j2) {
                    const int i2 = index[j2 - 1];
                    if (x[i2 - 1] <= 0.0) {
                        jj = j2;
                        ok = false;
                        break;
                    }
                }
                if (ok)
                    break;
            }
            for (int i = 0; i < static_cast<int>(m); ++i)
                zz[i] = b[i];
            solve_tri(a, index.data(), static_cast<std::size_t>(nsetp), zz.data(),
                      m);
        }
    }
    return x;
}

std::vector<double> pnnlssum(const double* a, std::size_t m, std::size_t n,
                             const double* b, double sum) {
    // AA = [ (A*sum).colwise() - b ; 1 1 ... 1 ]  with shape (m+1) x n
    std::vector<double> aa((m + 1) * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i)
            aa[i + j * (m + 1)] = a[i + j * m] * sum - b[i];
        aa[m + j * (m + 1)] = 1.0;
    }
    std::vector<double> bb(m + 1, 0.0);
    bb[m] = 1.0;
    std::vector<double> x = nnls(std::move(aa), std::move(bb), m + 1, n);
    double s = 0.0;
    for (double v : x)
        s += v;
    if (s != 0.0 && std::isfinite(s)) {
        for (double& v : x)
            v = v / s * sum;
    } else {
        std::fill(x.begin(), x.end(), sum / static_cast<double>(n));
    }
    return x;
}

std::vector<double> pnnqp(const double* q, std::size_t m, const double* p,
                          double sum) {
    const auto& qe = Eigen::Map<const Eigen::MatrixXd>(q, m, m);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(qe);
    // Eigen returns eigenvalues in ascending order, but the Rust port sorted
    // the (eigenvalue, eigenvector-column) pairs ascending explicitly to be
    // robust to a solver that does not guarantee ordering; match that.
    std::vector<std::pair<double, int>> pairs;
    pairs.reserve(m);
    for (int i = 0; i < static_cast<int>(m); ++i)
        pairs.emplace_back(eig.eigenvalues()(i), i);
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    // `index = count(eigval > eigval.last()*1e-15)`, then the LAST `index`
    // (largest) eigen-directions are kept.
    const double threshold = pairs[m - 1].first * 1e-15;
    int index = 0;
    for (const auto& pr : pairs)
        if (pr.first > threshold)
            ++index;
    const int start = static_cast<int>(m) - index;
    if (index == 0) {
        return std::vector<double>(m, sum / static_cast<double>(m));
    }
    // A_R (index x m) in the column-major layout `pnnlssum` reads: element
    // (jj, i) at a[jj + i*index]. R builds it as
    // (eigmat.rightCols(index) * sqrt(lam).asDiagonal()).transpose().
    std::vector<double> a(static_cast<std::size_t>(index) * m, 0.0);
    for (int jj = 0; jj < index; ++jj) {
        const double ev = pairs[start + jj].first;
        const int col = pairs[start + jj].second;
        const double r = std::sqrt(ev);
        for (int i = 0; i < static_cast<int>(m); ++i)
            a[static_cast<std::size_t>(jj) + static_cast<std::size_t>(i) * index] =
                eig.eigenvectors()(i, col) * r;
    }
    std::vector<double> b(index, 0.0);
    for (int jj = 0; jj < index; ++jj) {
        const double ev = pairs[start + jj].first;
        const int col = pairs[start + jj].second;
        double s = 0.0;
        for (int i = 0; i < static_cast<int>(m); ++i)
            s += -eig.eigenvectors()(i, col) * p[i];
        b[jj] = s / std::sqrt(ev);
    }
    return pnnlssum(a.data(), static_cast<std::size_t>(index), m, b.data(), sum);
}

}  // namespace nnls
}  // namespace npfc
