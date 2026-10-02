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

### 4a. The `npnorm2Dll` (2-D) exception

The 2-D experimental family is still the one documented case where this
port is slower than R. A hand-unrolled 2×2 density-kernel fast path
(see below) closed most of the old ~28× gap on the reference run:

| case (same machine, same data n=300, same initial mix & grid) | `npfixedcomppy` | R | ratio |
|---|---|---|---|
| `computemixdist(X, method="npnorm2Dll")`, 2-D n=300 | ≈ 4.3 s (median of 5) | ≈ 0.75 s (best of 3) | ~5.7× slower |

Verified decomposition (this build):

* **The kernels are bit-exact and ≈ 7× cheaper per call.** The 2-D
  family only ever calls `dnormNDarray` with `dim == 2`. The original
  port mirrored the R source line-for-line — a per-point dynamic
  `dec.solve(d)` — which heap-allocates two `VectorXd` temporaries per
  data point (≈ 600 mallocs per call at n = 300) that MSVC `/O2` cannot
  escape-analyze away (the R build's GCC `-O2` sink-eliminates the
  same code). That made the density kernel ≈ 89 % of the fit's wall
  time (`dnormNDarray` n=300, m=1: ≈ 44 µs/call; 21.1 s fit). The fix
  is a hand-unrolled 2×2 fast path in `npfc_kernels.h`: the Cholesky
  factors come from the same `unblocked` algorithm Eigen's `LLT` runs
  for size < 32 (l11 = √S00, l21 = S10/l11, l22 = √(S11−l21²)) and each
  point is the two-stage triangular solve of `dec.solve(d)` (forward
  with L, backward with Lᵀ) dotted with `d`, written out as scalars —
  same factors, same operation order, no temporaries.
  `dnormNDarray` n=300, m=1 is now ≈ 6.5 µs/call and `dnpnormND`
  ≈ 7.4 µs/call (the `dim ≠ 2` path is unchanged).
* **The cross-evaluation parity is unchanged by the fast path.** PY's
  kernel evaluated at R's final points gives ll = 848.8877528768896
  (R's own: 848.88775287688986), and R's kernel at PY's final points
  gives 848.7472820349071 (PY's own: 848.747282034907) — both
  cross-evaluations match an independent numpy recomputation to
  < 1e-12, and the fit's reported `ll` / iteration trajectory are
  bit-identical to the pre-optimization build.
* **The remaining cost is the L-BFGS-B support-point search.** The
  family runs one box-constrained L-BFGS-B problem per grid cell
  (103 × 103 = 10,609 sub-problems per outer iteration × 6 iterations)
  plus a weight subproblem and a collapse per iteration.
  `NPFIXEDCOMPY_PROFILE=1` reports `objevals=421810` for the n=300
  reference run. At ≈ 7.4 µs per `dnpnormND` call, the density kernel
  is ≈ 3.1 s of the 4.3 s; the rest is the per-evaluation gradient
  temporaries in `gradfun` (`fullden`, `temp`, the 300×2 replicated
  `murep` — again heap temporaries MSVC keeps while the R build's GCC
  eliminates them) and the per-cell L-BFGSpp machinery (problem
  construction, Cauchy-point / subspace-minimization bookkeeping,
  backtracking). The per-cell early-exit (`projgnorm ≤ ε` on the
  initial midpoint) fires for the empty cells in both implementations.
* **The result is a valid — in the reference run, slightly better —
  local optimum.** The search accepts a grid cell only on a *strictly*
  negative objective, which is exquisitely sensitive to the
  floating-point trajectory; PY and R each converge deterministically
  (6 iterations both sides) to different optima, PY's NLL 848.747282034907
  vs R's 848.88775287688986 (lower is better). `tests/verify_2d_same.py`
  pins all of this as the parity gate for the family.

This is inherited algorithmic structure, not a port defect: the R
package documents `npnorm2Dll` as *experimental and possibly very slow*,
and the per-cell L-BFGS-B loop is the documented design. Folding the
per-evaluation gradient temporaries into scalars (or a cell-pruning
heuristic) would be the obvious next speed-up; neither has been added
because both change the floating-point trajectory of the support-point
search and therefore which (valid) optimum is found, which the
match-to-R contract treats as out of scope for this family.

#### 4a.1 Solver replacement evaluated: CppNumericalSolvers, NLopt, and a wider sweep

The per-cell solver was evaluated against **CppNumericalSolvers**
(`PatWie/CppNumericalSolvers`, GPL/MIT dual-licensed) and **NLopt**
(not present in this environment), and a wider literature sweep of
bound-constrained and mixture-model algorithms was performed. No
rejected candidate was vendored into the package — the tree adds no
external dependency, and the default path is byte-identical to the
pre-evaluation build (`verify_2d_same.py` green: ll
848.7472820349071 on the reference run, two-run bit-determinism).

* **cppoptlib's `LbfgsbSolver` is a different implementation**, not a
  build of the same code: its convergence test is a hard-coded
  projected-gradient norm ≤ 1e-4 (LBFGSpp tests the *unprojected*
  gradient against the caller's `tol`); its More–Thuente line search
  is run on the **unconstrained** function (bounds are clamped only
  after `findMinimize()` returns, and the post-hoc clamp does not
  re-evaluate the objective), so its accepted steps and its `fval`
  reporting differ from LBFGSpp's; and its history update
  reconstructs `M` with a dense `M.inverse()` every iteration instead
  of LBFGSpp's rank-1/2 recursion. At n = 2 these costs are tiny, so
  the wall-time difference is almost entirely the extra iterations.
* **Measured live (n=300 2-D run, same data / initial mix / grid /
  tol), via a temporary adapter that was removed afterwards:**

  | | default LBFGSpp | cppoptlib `LbfgsbSolver` |
  |---|---|---|
  | wall | ≈ 8.2 s | ≈ 20.1 s |
  | objective evaluations | 769,533 | 1,959,715 (2.5×) |
  | result ll / iter / support | 982.7562335069146 / 9 / 6 | 982.75623387… / 9 / 6 |

  The cppoptlib optimum differs from the R-matching one by ~3.7e-10 in
  ll — a different (valid) local optimum, which the match-to-R contract
  treats as a trajectory break. Its per-evaluation cost is in fact
  comparable (≈ 10.2 µs vs ≈ 10.7 µs), so the 2.4× wall ratio is the
  2.5× evaluation ratio, i.e. convergence policy, not kernel cost.
* **NLopt**: `LD_LBFGS` is a C port of the same LBFGS-B paper
  implementation with a per-evaluation C-callback hop — the same
  algorithmic behavior plus a cross-language boundary, with no
  convergence advantage.
* **Wider sweep — nothing both faster *and* trajectory-compatible:**
  * **LMBOPT** (Neumaier & Azmi, arXiv:1410.7714) — the best-documented
    bound-constrained solver (active-set + Wolfe line search, super-
    linear convergence, "finitely terminating" for quadratic-like
    problems). Its advantage is aimed at general/n-to-large problems;
    at n = 2 the extra active-set machinery buys nothing against a
    2-D LBFGS-B, and its published implementation is MATLAB
    (`Matlab_LMBOPT`), not a vendorable C++ core.
  * **Box-constrained L-MQN / projected-gradient family** (Pytlak,
    ASACG, box L-BFGS variants): same asymptotic class as LBFGS-B
    (quasi-Newton, super-linear); the differences are in
    active-set/box handling, not in convergence speed at this
    dimension, and each is a different algorithm — any swap changes
    the per-cell trajectory and hence the final optimum.
  * **Nelder–Mead / derivative-free**: O(ε⁻²)-style complexity and no
    tolerance semantics comparable to the solver's `tol`; the
    objective is smooth (we have an analytic gradient), so dropping
    gradient information is strictly worse, and a different
    algorithm again.
  * **Mixture-model support-search alternatives** (the statistical
    literature on component initialization for EM): directional-
    derivative screening on the grid (evaluating only the best
    coordinate direction per point, as in several fast component-search
    EM variants), Metropolis–Hastings random candidate selection, and
    support-reduction / vertex-direction active-set methods. These
    change the *selection algorithm* itself (which cells are accepted,
    in what order, under what test), i.e. the R trajectory by
    construction — out of scope for a faithful port, and none of them
    is faster in the "fewer L-BFGS-B evaluations per accepted cell"
    sense anyway, since the per-cell solve only runs on candidates.
  * **Cell-level early-exit tuning** (e.g. testing the midpoint's
    gradient before the line search): the vendored LBFGS-B already
    exits before its first line search whenever the projected gradient
    norm at the midpoint is ≤ tol, which is exactly why most empty
    cells are cheap; the measured 8.2 s is dominated by the
    non-exit cells, where a genuinely cheaper solve would be needed —
    and per the paragraphs above, none exists that keeps the
    trajectory.

**Unifying the `d0`/`d1` gradient flags** (one 3-vector gradient
instead of the two booleans) was evaluated and rejected: every 1-D
family always requests both components, and the `d0`-only branch
serves only the R `npnorm2Dll` gradient convention (d0 = the
probability-direction scalar, d1 = the 2 support-point directions).
Collapsing the flags would be a cosmetic refactor — it cannot change
any floating-point trajectory, and the per-evaluation cost is set by
`dnpnormND`, not by which subset of the gradient is requested.

## 5. What was *not* done, and why

* **Hand-written OpenMP** — stripped entirely (see section 2); it was a
  crash/determinism hazard and the gains are re-obtained bit-safely by
  the cache.
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
for %A in (npnormll nptll cvmadcll pois density binned coveb posteriormean kernels 2d_same) do .venv\Scripts\python.exe tests\verify_%A.py
```

`tests/verify_kernels.py` carries the 30-case 1-D/2-D kernel gold set
(`ref_new.txt` + the R-dumped inputs, recorded against live R 4.6.1);
`tests/verify_2d_same.py` carries the `npnorm2Dll` gold set
(`parity_2d.csv`, `grid_2d.csv`, the initial mix, and R's final points
`r3_pts_2d.csv` / `r3_pr_2d.txt`, recorded by `C:\...\rebuild\verify2d.R`).

2-D wall times: `.venv\Scripts\python.exe bench_2d.py` (Py) and
`Rscript bench_2d_r.R` (R), both reading the shared `parity_2d.csv`.
