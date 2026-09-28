// npfc_nnls.h — constrained non-negative least-squares subproblems used by
// the family `computeweights` (a faithful port of the R package's `pnnls.f`
// netlib NNLS, `k = 0` case, and the `pnnlssum_` / `pnnqp_` wrappers from
// miscfuns.h).
//
// Matrix convention throughout: column-major (Fortran `A(MDA, N)`). A matrix
// is passed as a flattened buffer of length `m * n` where element `(i, j)`
// (0-based) is at `buf[i + j * m]`.
#ifndef NPFIC_NNLS_H
#define NPFIC_NNLS_H

#include <cstddef>
#include <vector>

namespace npfc {
namespace nnls {

// Lawson & Hanson non-negative least squares.
// Solves `min ||A x - b||_2  s.t.  x >= 0`. `a` is `m x n` (column-major),
// `b` has length `m`; both are CONSUMED (moved in) and may be overwritten
// internally. Returns `x` of length `n`.
std::vector<double> nnls(std::vector<double> a, std::vector<double> b,
                         std::size_t m, std::size_t n);

// `pnnlssum_`: NNLS with an appended homogeneous sum-to-one row and `sum`
// scaling, then the solution is normalised so it sums to exactly `sum`.
// `a` is `m x n` (column-major), `b` length `m`. Returns length `n`.
std::vector<double> pnnlssum(const double* a, std::size_t m, std::size_t n,
                             const double* b, double sum);

// `pnnqp_`: the large-data (`len > 1000`) weight subproblem, which
// eigen-decomposes the positive-semidefinite Gram matrix `q` (`m x m`,
// column-major, symmetric) and reduces to a `pnnlssum` on the significant
// eigen-directions. Returns length `m`.
std::vector<double> pnnqp(const double* q, std::size_t m, const double* p,
                          double sum);

}  // namespace nnls
}  // namespace npfc

#endif  // NPFIC_NNLS_H
