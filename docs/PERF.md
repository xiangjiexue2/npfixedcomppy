# Performance

This document records the performance architecture of `npfixedcomppy` and
the evidence behind it (measured on the development machine: i7-8700K,
MSVC 14.44, Eigen 5.0.0 vendored, Python 3.11.9).

## 1. Where the time goes

Phase profile of a full `computemixdist` call
(`NPFIXEDCOMPY_PROFILE=1` prints one line to stderr, median of 3 runs;
all values in ms unless marked). The profiled `total` covers the C++
engine phases only; the wall times in section 4 additionally include
Python-side binning and grid generation:

| case | iters | total | solvegrad | mapping | weights | collapse | loss |
|------|-------|-------|-----------|---------|---------|----------|------|
| `nptll` β=5, n=5000 | 31 | **2.47 s** | 2200 | 0.3 | 9.1 | 261 | 0.8 |
| `npnormcll`, n=1000 | 19 | **0.99 s** | 459 | 2.8 | 498 | 31 | 0.1 |
| `npnormadw`, n=5000 | 36 | **413 ms** | 182 | 40 | 77 | 112 | 1.7 |
| `nptllw` β=∞, n=5000 | 15 | **144 ms** | 123 | 0.1 | 6.8 | 14 | 0.3 |
| `npnormllw`, n=5000 | 15 | **129 ms** | 108 | 2.2 | 10.0 | 8.9 | 0.3 |
| `nptll` β=∞, n=5000 | 15 | **47 ms** | 38 | 0.1 | 4.3 | 5.1 | 0.4 |
| `npnormll`, n=5000 | 15 | **19.2 ms** | 13.2 | 0.1 | 3.4 | 2.3 | 0.2 |
| `npnormad`, n=1000 | 11 | **19.5 ms** | 15.7 | 0.0 | 1.3 | 2.4 | 0.1 |

Reading the profile:

* **`nptll` β=5** — dominated by `solvegrad`: the derivative-free
  support-point search evaluates many *off-grid* points per iteration, and
  each off-grid point is one full data-column of the non-central-t kernel
  (`dnt` = AS-243 series, the most expensive kernel in the package). The
  kernel column cache eliminates the *on-grid* re-evaluations (≈1.4 s of
  the pre-cache 4.4 s); what remains is the irreducible off-grid work plus
  the 260 ms of `collapse` remapping.
* **`npnormcll`** — dominated by `weights` (≈50 %): the per-iteration
  constrained NNLS subproblem (`pnnlssum` for n ≤ 1000, `pnnqp` beyond),
  plus the grid sweep in `solvegrad`. These are the algorithm's intrinsic
  small-matrix solves, not re-evaluation waste.
* **binned normal families (`npnormllw` / `npnormcvmw` / `npnormadw`)** —
  `solvegrad` again dominates: the grid sweep rebuilds an
  (nbins × ngrid) trapezoid fill at every candidate support point. The
  fill dimension depends on the current support spread, so no column
  caching is possible (section 3); the fill itself is a column-major
  `ddiscnorm_m` / `pnorm_disc_m` written to mirror R's `dnormarray`
  accumulation order bit-for-bit, and it is within a few percent of R's
  fill time on this build.
* **`npnormadw`** additionally spends ~112 ms per fit in `collapse`
  (AD's re-weighting changes the support set more aggressively than
  LL/CVM).

## 2. Compute core: C++/Eigen (Eigen-internal parallel GEMM when probed)

The whole compute stack — engine, support-point solvers, constrained NNLS,
gradient and weight sweeps, kernel-matrix fills, the mixture density —
lives in `cpp/` (pybind11 module `npfixedcomppy._core`, built by
`setup.py` with MSVC `/std:c++17 /O2 /arch:AVX2` by default, GCC/Clang
`-O3 -mavx2` elsewhere).

**The SIMD level is decided at BUILD time, not runtime**, by the
`NPFIC_ARCH` environment variable: `avx2` (default), `avx`, `avx512`,
`native` (GCC/Clang `-march=native` only), or empty for the plain x86-64
SSE2 baseline. The build artifact only runs on hardware with that
instruction set or a superset; there is no runtime ISA dispatch and the
source is unchanged. Note the MSVC and GCC/Clang flags do **not** behave
identically: MSVC's `/arch:AVX2` also lets the optimizer contract
multiply-add pairs into FMA instructions (documented MSVC behaviour under
the default `/fp:precise`); GCC/Clang keep FMA off unless `-mfma` is
added (it is not). The parity gates validate the shipping build;
cross-architecture bit-identity is not a requirement.

Two measured consequences of moving from the SSE2 baseline to AVX2 on
this machine:

* the wide binned fills (`npnormllw`, `nptllw`) got ~2× faster
  (354 → 130 ms at n=5000), closing the historical gap with R to within
  a few percent (section 4);
* the small un-binned normal cases got ~4 ms slower (`npnormll` n=1000:
  7.8 → 11.6 ms) — short vectors with AVX2 tails; `NPFIC_ARCH=` restores
  the baseline. The t-family and all larger cases are unaffected.

**There is no hand-written OpenMP — the only threading in the package is
Eigen's own, enabled by the build's OpenMP probe.**
All hand-written `#pragma omp` was stripped (it interacted badly with
Eigen's internal thread handling and made results machine-dependent).
Instead, `setup.py` probe-compiles the C++ compiler with its OpenMP flag
(`cl /openmp` on MSVC, `g++`/`clang++ -fopenmp` on Unix) and, when the
probe passes, adds the flag to the extension build — which is exactly the
flag that defines `_OPENMP`, the whole of Eigen's compile-time gate
(`EIGEN_HAS_OPENMP` in `eigen/Eigen/Core`). Eigen's internal parallel
GEMM/GEMV then runs in the mapping/computeweights hot paths while every
other loop in the package stays serial (the bit-identical reference order);
when the compiler lacks the flag, the build falls back to serial Eigen with
no other change.
The determinism contract:

* for a **fixed build and a fixed `OMP_NUM_THREADS`**, identical inputs
  give bit-identical outputs — Eigen's parallel GEMM splits its reduction
  into a fixed number of blocks and combines them in a fixed order, which
  is what the GOLD parity gates rely on;
* a different thread count may only reorder floating-point reductions
  inside Eigen's parallel GEMM — round-off-scale differences, well within
  the parity gates (ll 1e-9, pt 1e-6).

Environment knobs: `NPFIXEDCOMPY_PROFILE=1` (per-phase timing) and, for
OpenMP builds, `OMP_NUM_THREADS` (sizes Eigen's pool; see above for the
effect on results).

## 3. The kernel column cache

The dominant waste before the cache: the engine re-evaluates the *same*
kernel columns on every pass of every outer iteration — the `solvegrad`
grid sweep, `mapping`, the `computeweights` fill, the `collapse`
remapping, the end-of-iteration mapping. For the expensive kernels
(non-central-t, normal CDF, one-parameter normal) that is 3–8 redundant
evaluations of each `(n × m)` matrix per iteration.

A kernel column `K[:, mu]` depends ONLY on `(data, beta, mu)` — never on
the current weights, density, or iteration — so each distinct `mu` is
evaluated **exactly once per fit** (in `prepare` for grid points, lazily
on first use for off-grid points) and every consumer reads the shared
column.

Correctness is preserved, not approximated:

* a cached column is produced by the exact same per-row evaluation as a
  fresh fill (ascending `i`, identical arithmetic), so it is
  **bit-identical** to an uncached computation — only redundant
  re-evaluation is removed, never the arithmetic or its order;
* consumers that build an `(n × m)` matrix from the columns keep the
  reference column-by-column accumulation;
* all parity suites pass `TOTAL BAD: 0` unchanged with the cache in
  place.

Measured effect (this build):

| case | before cache | after cache |
|------|-------------|-------------|
| `nptll` β=5, n=5000 | 4.43 s | **2.47 s** (−44 %) |
| `npnormcll`, n=1000 | 1.21 s | **0.99 s** (−19 %) |
| `npnormad`, n=1000 | 60 ms | **19.5 ms** |
| `npnormcvm`, n=1000 | — | **36 ms** |
| `nptll` β=∞, n=1000 | 43 ms | **36 ms** |
| `npnormll`, n=1000 / n=5000 | 8 ms / 18 ms | **11.6 ms / 19.2 ms** |

**The cache does not apply to the three binned normal families
(`npnormllw` / `npnormcvmw` / `npnormadw`).** Their kernel matrix
`K(bins × grid)` is a trapezoid fill whose *width* (the number of
component columns `N`) depends on the current support spread, so the
same `(bin, grid)` point is a *different* kernel at different sweep
points — there is no `(x, mu)` column to reuse. (`nptllw` is the
exception: its binned-t kernel is the CDF difference
`Φ(x; μ−h) − Φ(x; μ)`, a pure function of `(data, df, h, mu)`, so it
does use the column cache for off-grid points.) The three binned normal
families instead pay for the fill with a column-major rewrite of R's
`dnormarray` layout (no row-major temp, no transpose, hoisted
constants) that is bit-identical to the reference accumulation.

## 4. Comparison with R `npfixedcomp2` (same machine)

Py: median of 5 (`tests/perf_baseline.py` / `tests/bench_binned.py`);
R: best of 3 (`tests/bench_r.R` / `tests/bench_binned_r.R`), R 4.6.1 +
RcppEigen 3.4.0 compiled by R's default MSVC flags (SSE2 baseline).

Un-binned:

| case | `npfixedcomppy` | R | speedup |
|------|-----------------|---|---------|
| `npnormll`, n=1000 | 11.6 ms | 19 ms | 1.6× |
| `npnormcvm`, n=1000 | 36.0 ms | — | — |
| `npnormad`, n=1000 | 38.0 ms | 59 ms | 1.5× |
| `npnormcll`, n=1000 | 1027 ms | 1924 ms | 1.9× |
| `nppoisll`, n=1000 | 0.56 ms | 5 ms | 8.9× |
| `nptll` β=∞, n=1000 | 36.5 ms | 211 ms | 5.8× |
| `npnormll`, n=5000 | 33.6 ms | — | — |
| `nptll` β=5, n=5000 | 3.10 s | 49.7 s | **16.0×** |
| `npnormll`, n=50000 | 370 ms | — | — |
| `estpi0` (norm), n=1000 | 29.1 ms | 97 ms | 3.3× |

Binned (`order = -3`):

| case | `npfixedcomppy` | R | ratio |
|------|-----------------|---|-------|
| `npnormllw`, n=5000 | 130.5 ms | 120 ms | 0.9× |
| `npnormcvmw`, n=5000 | 133.1 ms | 110 ms | 0.8× |
| `npnormadw`, n=5000 | 420.7 ms | 170 ms | 0.4× * |
| `nptllw` β=∞, n=5000 | 221.8 ms | 1970 ms | **8.9×** |
| `nptllw` β=5, n=5000 | 2.06 s | 29.3 s | **14.2×** |
| `estpi0` (`npnormllw`), n=5000 | 563 ms | 1330 ms | 2.4× |
| `npnormllw`, n=20000 | 485.7 ms | 540 ms | 1.1× |
| `nptllw` β=∞, n=20000 | 875 ms | 7250 ms | **8.3×** |
| `nptllw` β=5, n=20000 | 2488 ms | (minutes; not recorded) | — |

\* `npnormadw` fits are run-to-run non-deterministic in both
implementations (R wanders 6 basins, `ll` 0.17697…0.18252, over 13
runs), so this ratio is indicative, not structural — a slower basin on
the Py side inflates it.

Reading: the big wins remain the t-family (un-binned β=5, 16×; binned,
8–14×), where the AS-243 kernel cache and the cheap CDF-difference binned
fill dominate R's per-point re-evaluation. The binned *normal* families
are at parity with R (within ~25 %) after the column-major fill rewrite
and the AVX2 build: both sides fill the same trapezoid with the same
bit-exact accumulation, and both call the scalar libm `exp` — neither
Eigen 5.0.0 nor R's RcppEigen 3.4.0 ships a double-precision SIMD
`pexp` for x86 (grep-verified in both packet headers), so the fill is
`exp`-bound on both sides and no wider instruction set can touch it.

## 5. What was *not* done, and why

* **Hand-written OpenMP** — stripped entirely (see section 2); it was a
  crash/determinism hazard and the gains are re-obtained bit-safely by
  the cache.
* **SIMD GEMM library (`faer`)** — earlier measurements (tall-and-thin
  `n × m` with `m ≤ ~20` production columns) showed GEMM libraries lose to
  a streaming dot-loop at these shapes; not adopted.
* **Threading the remaining work** — the two remaining hotspots (the
  off-grid `dnt` evaluations in `solvegrad`, the NNLS in `weights`) are
  embarrassingly parallel over `n`, but per the design constraint they
  stay serial (no hand-written OpenMP; only Eigen's compile-time-gated
  parallel GEMM/GEMV is in play). The measured speedups versus R already
  come from SIMD + the cache.
* **Geometric-series fill shortcut (rejected)** — filling
  `D(x; μ−kδ)` as `D(x; μ)·rᵏ` was ~2.3× faster on the binned trapezoid,
  but algebraically wrong: `log D` is quadratic in `k`, so the ratio
  `D_{k+1}/D_k` is not constant. The parity gate caught the systematic
  1.2e-4 `ll` offset it produced; the branch was deleted, not demoted,
  because a bit-exactness hazard with a plausible API is worse than a
  missed optimization.

## 6. Reproducing the evidence

```bat
cd npfixedcomppy
.venv\Scripts\python.exe tests\perf_baseline.py        :: wall-time table (un-binned)
.venv\Scripts\python.exe tests\bench_binned.py         :: wall-time table (binned)
set NPFIXEDCOMPY_PROFILE=1
.venv\Scripts\python.exe -c "import numpy as np, npfixedcomppy as n; x=np.random.default_rng(3).normal(0,1,5000); n.computemixdist(x, method='nptll', beta=5)"
:: per-phase timing line (n=5000, nptll beta=5) to stderr
cd tests
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" bench_r.R                :: R reference times
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" bench_binned_r.R         :: R binned times
```

Parity gates (all must report `TOTAL BAD: 0`):

```bat
cd npfixedcomppy
for %A in (npnormll nptll cvmadcll pois density binned coveb posteriormean) do .venv\Scripts\python.exe tests\verify_%A.py
```
