/* npfc_core.h — C ABI of the C++/Eigen heavy-compute core.
 *
 * All heavy kernels (normal pdf matrix fills, mixture-density mappings) run
 * in C++ with Eigen: SIMD vectorization is decided by the compiler at build
 * time (the build script adds the best /arch: the machine supports), and
 * multi-threading uses OpenMP *if the OpenMP runtime was detected at build
 * time*, otherwise the same code compiles down to clean serial loops.
 *
 * Memory policy: every buffer is either caller-owned (passed with an exact
 * capacity) or owned by an Eigen object that frees itself. The library never
 * returns a bare pointer; the only exception is the scratch pool returned by
 * npfc_pool_create, which the caller MUST release with npfc_pool_free.
 * All element counts are checked before any write; a bad call returns 0
 * (or -1) and writes nothing.
 */
#ifndef NPF_FIXEDCOMPY_CORE_H
#define NPF_FIXEDCOMPY_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Capability bitmask (npfc_caps). Bit 0 = OpenMP available at runtime,
 * bit 1 = built with AVX2, bit 2 = built with AVX512F, bit 3 = FMA. */
#define NPFIC_HAS_OPENMP   (1u << 0)
#define NPFIC_HAS_AVX2     (1u << 1)
#define NPFIC_HAS_AVX512F  (1u << 2)
#define NPFIC_HAS_FMA      (1u << 3)

/* Build-time / run-time info for diagnostics. All out-params optional. */
void npfc_info(unsigned *caps, int *omp_threads, unsigned *simd_level);

/* ------------------------------------------------------------------ */
/* Scratch pool: one aligned buffer reused across calls.              */
/* Return NULL only on allocation failure. Free with npfc_pool_free.  */
/* ------------------------------------------------------------------ */
void *npfc_pool_create(size_t n_elems);
void  npfc_pool_free(void *p);

/* ------------------------------------------------------------------ */
/* npfc_kmat_norm: fill the (n x g) normal-pdf kernel, COLUMN-MAJOR:
 *   out[j * n + i] = exp(-0.5 ((x[i]-mu[j])/beta)^2 - ln(beta) - LN_SQRT_2PI)
 * `out` must hold at least n*g doubles. Returns 1 on success, 0 on
 * invalid args (n or g == 0, NULL pointers, or capacity too small). */
int npfc_kmat_norm(const double *x, long n,
                   const double *mu, long g,
                   double beta,
                   double *out);

/* ------------------------------------------------------------------ */
/* npfc_mapping_norm: mixture density out[i] = sum_j pi[j] * N(x[i];mu[j],beta).
 * `out` must hold n doubles. Returns 1 on success, 0 on invalid args. */
int npfc_mapping_norm(const double *x, long n,
                      const double *mu, long m,
                      const double *pi, double beta,
                      double *out);

/* ------------------------------------------------------------------ */
/* npfc_gemv: y = A^T * x, A (n x m) column-major, x length n,
 * y length m (caller-owned). Returns 1 on success, 0 on invalid args. */
int npfc_gemv(const double *A, long n, long m,
              const double *x, double *y);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NPF_FIXEDCOMPY_CORE_H */
