# Performance

This document records the performance architecture of `npfixedcomppy` and
the evidence behind it (measured on the development machine: i7-8700K,
MSVC 14.44, Eigen 5.0.0 vendored, Python 3.11.9).

## 1. Where the time goes

Phase profile of a full `computemixdist` call
(`NPFIXEDCOMPY_PROFILE=1` prints one line to stderr, median of 3 runs;
all values in ms unless marked; re-measured on the current build with
`tests/reprofile_phases.py`). The profiled `total` covers the C++
engine phases only; the wall times in section 4 additionally include
Python-side binning and grid generation. `evals` is the count of
`(mu, dens)` gradient evaluations inside the `solvegrad` loop:

| case | iters | total | solvegrad | mapping | weights | collapse | loss | evals |
|------|-------|-------|-----------|---------|---------|----------|------|-------|
| `nptll` β=5, n=5000 | 31 | **2.36 s** | 2102 | 0.2 | 8.6 | 249 | 0.8 | 3733 |
| `npnormcll`, n=1000 | 19 | **0.98 s** | 467 | 2.4 | 482 | 29 | 0.1 | 21043 |
| `npnormadw`, n=5000 | 36 | **332 ms** | 168 | 39 | 74 | 49 | 1.5 | 4314 |
| `nptllw` β=∞, n=5000 | 15 | **130 ms** | 109 | 0.1 | 6.6 | 14 | 0.3 | 1777 |
| `npnormllw`, n=5000 | 15 | **31 ms** | 15 | 2.3 | 8.6 | 5.0 | 0.3 | 1755 |
| `nptll` β=∞, n=5000 | 15 | **45 ms** | 36 | 0.1 | 4.4 | 4.7 | 0.4 | 1758 |
| `npnormll`, n=5000 | 15 | **17.7 ms** | 11.2 | 0.1 | 3.7 | 2.4 | 0.2 | 1799 |
| `npnormad`, n=1000 | 11 | **18.5 ms** | 14.6 | 0.0 | 1.4 | 2.4 | 0.1 | 1334 |

Reading the profile:

* **`nptll` β=5** — dominated by `solvegrad`: the derivative-free
  support-point search evaluates many *off-grid* points per iteration, and
  each off-grid point is one full data-column of the non-central-t kernel
  (`dnt` = AS-243 series, the most expensive kernel in the package). The
  kernel column cache eliminates the *on-grid* re-evaluations (≈1.4 s of
  the pre-cache 4.4 s); what remains is the irreducible off-grid work plus
  the ≈ 249 ms of `collapse` remapping. The pdf's `df`-level constants
  (`log df`, `√((df+2)/df)`, the `gammln` pair) are precomputed once per
  run into `stats::DntConst` (used via `dnt_c`, bit-identical to `dnt`),
  so each off-grid evaluation skips that work.
* **`npnormcll`** — dominated by `weights` (≈50 %): the per-iteration
  constrained NNLS subproblem (`pnnlssum` for n ≤ 1000, `pnnqp` beyond),
  plus the grid sweep in `solvegrad`. These are the algorithm's intrinsic
  small-matrix solves, not re-evaluation waste.
* **binned normal families (`npnormllw` / `npnormcvmw` / `npnormadw`)** —
  before the grid-fill memo (section 3) `solvegrad` dominated: the grid
  sweep rebuilt an (nbins × ngrid) trapezoid fill at every candidate
  support point (`npnormllw` n=5000: ~108 ms of a 129 ms total). With
  the memo, the grid sweep is served from the pinned grid fill and
  `npnormllw` totals **31 ms**; what remains in its `solvegrad` is the
  off-grid single-point candidate fills (each is a new `mu` vector, so a
  fresh fill) plus the cheap d1 pdf part. `npnormadw` (332 ms total)
  still churns: its AD re-weighting keeps producing new support sets, so
  most fills miss the memo, and ~49 ms goes to `collapse`.

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
  a few percent — the grid-fill memo (section 3) has since cut
  `npnormllw` further to ≈ 37 ms wall, now ~3× ahead of R (section 4);
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

Environment knobs: `NPFIXEDCOMPY_PROFILE=1` (per-phase timing, plus the
`solvegrad` evaluation count), `NPFIC_REFINE_STEPS` (experimental: caps
the per-candidate refinement steps inside `brmin`/`dfmin`; default `-1`
= unlimited = shipped behaviour — see section 5), and, for OpenMP
builds, `OMP_NUM_THREADS` (sizes the Eigen pool; see above for the
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

**The column cache does not apply to the three binned normal families
(`npnormllw` / `npnormcvmw` / `npnormadw`).** Their kernel matrix
`K(bins × grid)` is a trapezoid fill whose *width* (the number of
component columns `N`) depends on the current support spread, so the
same `(bin, grid)` point is a *different* kernel at different sweep
points — there is no `(x, mu)` column to reuse. (`nptllw` is the
exception: its binned-t kernel is the CDF difference
`Φ(x; μ−h) − Φ(x; μ)`, a pure function of `(data, df, h, mu)`, so it
does use the column cache for off-grid points.) The fill itself is a
column-major rewrite of R's `dnormarray` layout (no row-major temp, no
transpose, hoisted constants) that is bit-identical to the reference
accumulation.

**Grid-fill memo (the binned normal families).** What the column cache
cannot reuse, the memo *can*: within one fit, the binned fills are
requested for only a small set of repeated `mu` vectors — the grid, the
current support set, a few solver-interior points — so
`detail::KernelMemo` caches the last fill keyed by the *exact* `mu`
vector, and the grid fill is pinned in `prepare` (served from
`grid_m`). A served matrix is bit-identical to a fresh
`ddiscnorm_m` / `pnorm_disc_m` with the same arguments; a different
`mu` vector simply fills fresh. This removed the per-candidate refill
that used to dominate the binned `solvegrad`: `npnormllw`, n=5000, went
from ≈ 131 ms wall (grid sweep ~108 ms of a 129 ms engine total) to
≈ 31 ms engine / ≈ 37 ms wall — now ~3× **faster** than R's ≈ 120 ms
(what remains is the off-grid single-point candidate fills, each a new
`mu` vector, plus the cheap d1 pdf part). `npnormadw` benefits less —
its AD re-weighting keeps producing new support sets, so most fills
miss the memo — and still trails R (332 ms vs ≈ 170 ms).

## 4. Comparison with R `npfixedcomp2` (same machine)

Py: median of 5 (`tests/perf_baseline.py` / `tests/bench_binned.py`);
R: best of 3 (`tests/bench_r.R` / `tests/bench_binned_r.R`), R 4.6.1 +
RcppEigen 3.4.0 compiled by R's default MSVC flags (SSE2 baseline).

Un-binned:

| case | `npfixedcomppy` | R | speedup |
|------|-----------------|---|---------|
| `npnormll`, n=1000 | 10.7 ms | 19 ms | 1.8× |
| `npnormcvm`, n=1000 | 34.6 ms | — | — |
| `npnormad`, n=1000 | 35.5 ms | 59 ms | 1.7× |
| `npnormcll`, n=1000 | 1009 ms | 1924 ms | 1.9× |
| `nppoisll`, n=1000 | 0.57 ms | 5 ms | 8.8× |
| `nptll` β=∞, n=1000 | 36.2 ms | 211 ms | 5.8× |
| `npnormll`, n=5000 | 27.2 ms | — | — |
| `nptll` β=5, n=5000 | 2.95 s | 49.7 s | **16.8×** |
| `npnormll`, n=50000 | 326 ms | — | — |
| `estpi0` (norm), n=1000 | 24.9 ms | 97 ms | 3.9× |

Binned (`order = -3`):

| case | `npfixedcomppy` | R | ratio |
|------|-----------------|---|-------|
| `npnormllw`, n=5000 | 37.1 ms | 120 ms | **3.2×** |
| `npnormcvmw`, n=5000 | 118.7 ms | 110 ms | 0.9× |
| `npnormadw`, n=5000 | 386.7 ms | 170 ms | 0.4× * |
| `nptllw` β=∞, n=5000 | 206.1 ms | 1970 ms | **9.6×** |
| `nptllw` β=5, n=5000 | 2.01 s | 29.3 s | **14.6×** |
| `estpi0` (`npnormllw`), n=5000 | 83.8 ms | 1330 ms | **15.9×** |
| `npnormllw`, n=20000 | 157.5 ms | 540 ms | **3.4×** |
| `nptllw` β=∞, n=20000 | 779.4 ms | 7250 ms | **9.3×** |
| `nptllw` β=5, n=20000 | 2293.6 ms | (minutes; not recorded) | — |

\* `npnormadw` fits are run-to-run non-deterministic in both
implementations (R wanders 6 basins, `ll` 0.17697…0.18252, over 13
runs), so this ratio is indicative, not structural — a slower basin on
the Py side inflates it.

Reading: the big wins remain the t-family (un-binned β=5, 16.8×;
binned, 9–15×), where the AS-243 kernel cache and the cheap
CDF-difference binned fill dominate R's per-point re-evaluation. The
binned normal families now *split*: `npnormllw` is ~3× faster than R —
the grid-fill memo (section 3) serves the per-iteration grid sweep from
the precomputed grid fill, while R refills the trapezoid at every
candidate point; `npnormcvmw` is at parity (≈ 119 ms vs 110 ms, its
support sets churn enough that most fills miss the memo); `npnormadw`
still trails R (387 ms vs 170 ms) for the same reason plus its
aggressive `collapse` re-weighting. Where a binned fit *does* pay for a
fresh trapezoid fill, both sides fill the same matrix with the same
bit-exact accumulation and both call the scalar libm `exp` — neither
Eigen 5.0.0 nor R's RcppEigen 3.4.0 ships a double-precision SIMD
`pexp` for x86 (grep-verified in both packet headers) — so a fresh fill
is `exp`-bound on both sides and no wider instruction set can touch it;
the remaining binned gap is *how often* each side refills, and the memo
closes it for the LL-style sweep.

### 4a. The `npnorm2Dll` (2-D) exception

The 2-D experimental family is still the one documented case where this
port is slower than R. Two fast paths — a hand-unrolled 2×2 density
kernel and an explicit-inverse objective (§4a.2, the default) — closed
most of the old ~28× gap on the reference run:

| case (same machine, same data n=300, same initial mix & grid) | `npfixedcomppy` | R | ratio |
|---|---|---|---|
| 2-D n=300, fast path (default) | ≈ 2.7 s (median of 5) | ≈ 0.75 s (best of 3) | ~3.5× slower |
| 2-D n=300, R-identical path (`NPFIC_2D_EXACT=1`) | ≈ 4.3 s (median of 5) | — | — |

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
  (R's own: 848.88775287688986), and R's kernel at the R-identical
  path's final points gives 848.7472820349071 (the exact path's own:
  848.747282034907) — both cross-evaluations match an independent
  numpy recomputation to < 1e-12. The kernel fast path itself is
  bit-exact: with `NPFIC_2D_EXACT=1` the fit reproduces the
  pre-objective-fast-path build's ll / iteration trajectory bit-for-bit.
  The **default** fast objective path (§4a.2) differs from that
  trajectory by ~1e-13 in the objective values (ll 848.7611702303334
  on the reference run) and accepts a slightly different cell set on
  some data.
* **The remaining cost is the L-BFGS-B support-point search.** The
  family runs one box-constrained L-BFGS-B problem per grid cell
  (103 × 103 = 10,609 sub-problems per outer iteration × 6 iterations)
  plus a weight subproblem and a collapse per iteration. The per-cell
  early-exit (`projgnorm ≤ ε` on the initial midpoint, executed on the
  first evaluation) is what makes most cells cheap; the cost is the
  non-exit cells. On the n=300 reference run `NPFIXEDCOMPY_PROFILE=1`
  reports `objevals=421810` for **both** objective paths — the
  trajectories there accept the same cells, and the whole 4.3 s →
  2.7 s gain is the per-evaluation kernel cost (≈ 10.1 µs exact →
  ≈ 6.3 µs fast, §4a.2). On the second n=300 benchmark the ~1e-13
  kernel difference *does* flip the near-zero per-cell objective signs
  of a few cells, the accepted set changes, and the evaluation count
  itself drops 769,533 → 251,704 (wall 7.0 s → 2.5 s).
* **The result is a valid — in the reference run, slightly better —
  local optimum, on both objective paths.** The search accepts a grid
  cell only on a *strictly* negative objective, which is exquisitely
  sensitive to the floating-point trajectory; both paths converge
  deterministically in 6 iterations: exact path NLL 848.747282034907,
  fast path 848.7611702303334, R's 848.88775287688986 (lower is
  better). The main components of the two PY mixtures agree within
  ≈ 0.06 in location. `tests/verify_2d_same.py` pins all of this as the
  family's gate (per-path determinism, ll bands, kernel parity vs R,
  and fast-vs-exact mixture similarity).

This is inherited algorithmic structure, not a port defect: the R
package documents `npnorm2Dll` as *experimental and possibly very slow*,
and the per-cell L-BFGS-B loop is the documented design. The
per-evaluation gradient temporaries *have* now been folded into scalars
and the invariant re-computations removed (§4a.2); that changed the
floating-point trajectory of the support-point search — which the old
match-to-R contract forbade — and is only the default now because the
contract was relaxed to *similar result, faster wall time* (the
R-identical path remains available via `NPFIC_2D_EXACT=1`). The
remaining ~3.5× gap is the per-cell L-BFGS-B evaluation cost itself
(421,810 objective evaluations at ≈ 6.3 µs on the reference run —
251,704 at ≈ 9.9 µs on the second benchmark), which §4a.1 concludes
no faster *and* similar-result solver can reduce.

#### 4a.1 Solver replacement evaluated: CppNumericalSolvers, NLopt, and a wider sweep

The per-cell solver was evaluated against **CppNumericalSolvers**
(`PatWie/CppNumericalSolvers`, GPL/MIT dual-licensed) and **NLopt**
(not present in this environment), and a wider literature sweep of
bound-constrained and mixture-model algorithms was performed. No
rejected candidate was vendored into the package — the tree adds no
external dependency. The default path is *no longer* byte-identical to
the pre-evaluation build: the contract relaxed to *similar result,
faster wall time* (§4a.2 added the fast objective path as the default);
the R-identical behavior is still exactly reproducible via
`NPFIC_2D_EXACT=1`, and `verify_2d_same.py` is green on both paths.

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
  tol), via a temporary adapter that was removed afterwards — the
  baseline column is the pre-fast-path (Cholesky-kernel) build:**

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

#### 4a.2 Objective fast path (new default) and the relaxed contract

With the contract relaxed from *bit-exact match to R* to *faster, with
ll/mixture similar to R's result*, the per-cell objective was re-derived
for n = 2 (it is the only caller of the full gradient pair):

* **Explicit-inverse quadratic form.** `dᵀΣ⁻¹d` is computed as
  `inv00·d0² + (inv01+inv10)·d0·d1 + inv11·d1²` from the explicit 2×2
  inverse of `beta` (precomputed once in the constructor) — 0 divisions
  per point where the Cholesky-solve kernel needs 4. The normalizing
  constant keeps R's exact grouping
  (`LN_SQRT_2PI·2 + 0.5·log det`).
* **Invariant caching.** `fullden = 1/(dens+precompute)` (300
  divisions) and the component scale are recomputed on *every*
  objective evaluation in the old code; they are constant for a whole
  `solvegrad` pass, so they are cached in `setdens` — ≈ 10⁸ divisions
  saved per fit with bit-identical values.
* **Preallocated buffers, same SIMD exp.** The two `n`-length
  temporaries are constructor-sized buffers; the exponentiation is
  still Eigen's SIMD `array().exp()` (the identical instruction
  sequence the R-matching kernel uses — a scalar `std::exp` drifts by
  1 ulp and was rejected).
* **Measured (same build; `NPFIXEDCOMPY_PROFILE=1`), reference run**
  (`tests/parity_2d.csv`):

  | | fast path (default) | exact path (`NPFIC_2D_EXACT=1`) |
  |---|---|---|
  | wall (median of 5) | 2654 ms | 4252 ms |
  | objective evaluations | 421,810 | 421,810 |
  | per-evaluation | ≈ 6.3 µs | ≈ 10.1 µs |
  | ll / iters / components | 848.7611702303334 / 6 / 5 | 848.7472820349071 / 6 / 5 |

  The two trajectories accept the **same cells** there, so the entire
  gain is the per-evaluation kernel cost (0 divisions, no heap
  temporaries, cached invariants — ≈ 1.6×).
* **Second n=300 benchmark** (`itercmp/iterdata_2d.csv`):

  | | fast path | exact path |
  |---|---|---|
  | wall | ≈ 2.5 s | ≈ 7.0 s |
  | objective evaluations | 251,704 | 769,533 (3.1×) |
  | per-evaluation | ≈ 9.9 µs | ≈ 9.1 µs |
  | ll / iters / components | 983.1383384870023 / 8 / 6 | 982.7562335069146 / 9 / 7 |

  Here the ~1e-13 kernel difference flips the near-zero per-cell
  objective signs of a few cells, the accepted set changes, and the
  wall-time gain is the **evaluation count** (2.8× = 3.1× fewer
  evaluations; per-evaluation cost ≈ unchanged). Per-evaluation
  figures include the per-cell L-BFGSpp machinery and vary ≈ ±10 %
  between runs.
* **Similarity, measured (both datasets, default grid/init/tol):**
  reference run — both paths 6 iters on the same cell set, main
  components (weight ≥ 0.02) within 0.057 in location, |Δll| =
  0.0139; iterdata benchmark — fast ll 983.1383384870023, 8 iters,
  6 components vs exact ll 982.7562335069146, 9 iters, 7 components
  (fast drops one w≈0.02 minor component; both are valid and beat
  R 1.1.0003's 986.236 on that data). The parity gate now checks
  similarity (per-path determinism, iter count, ll bands, kernel
  parity vs R, main-component proximity) rather than bit-identity.
* **Escape hatch.** `NPFIC_2D_EXACT=1` restores the original
  Cholesky-based kernel — the bit-exact 0.2.1 trajectory — at the old
  cost.

## 5. What was *not* done, and why

* **Capping the per-candidate refinement (`NPFIC_REFINE_STEPS`) —
  measured A/B, kept as an experimental knob, default unchanged.** The
  hypothesis: capping the refinement steps each candidate runs inside
  `brmin`/`dfmin` ("less work per step") saves more wall than the extra
  outer iterations cost. A/B at n=30000 with identical data / initial
  points / grid / tol (`tests/refine_ab.py`; the evaluation count via
  `tests/refine_ab_evals.py`, fresh subprocess per arm, 3 timed runs,
  median):

  | method | arm | wall | iters | evals | ll |
  |--------|-----|------|-------|-------|----|
  | `nptll` β=3 | unlimited (shipped) | 3669 ms | 2 | 219 | 44729.103420 |
  | `nptll` β=3 | cap=3 | 3528 ms | 2 | 214 | 44729.103420 |
  | `nptll` β=3 | cap=2 | 3435 ms | 2 | 212 | 44729.103420 |
  | `nptll` β=3 | cap=1 | 3374 ms | 2 | 210 | 44729.103420 |
  | `npnormll` β=1 | unlimited (shipped) | 271 ms | 18 | 2358 | 42693.066732 |
  | `npnormll` β=1 | cap=3 | 231 ms | 17 | 2184 | 42693.066931 |
  | `npnormll` β=1 | cap=2 | 599 ms | 60 | 7537 | 42693.066723 |
  | `npnormll` β=1 | cap=1 | 129 ms | 12 | 1412 | 42693.070212 |

  No arm beats the shipped behaviour uniformly: the 2-iteration `nptll`
  run gains only ~1–7 % at bit-identical ll (the refinement is cheap
  relative to the 30k-point `dnt` columns), while the `npnormll` case
  shows the other side of the trade — cap=2 needs 60 iterations (2.2×
  the wall) and cap=1 drifts off the trajectory (different k and ll).
  The default stays `-1` (unlimited); the knob is kept for further
  experiments, and the `NPFIXEDCOMPY_PROFILE` line now reports the
  `solvegrad` evaluation count (`evals=`) so such experiments are
  measurable.
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
.venv\Scripts\python.exe tests\reprofile_phases.py     :: re-measure the section-1 phase table
.venv\Scripts\python.exe tests\refine_ab.py            :: NPFIC_REFINE_STEPS A/B (wall/iters/ll)
.venv\Scripts\python.exe tests\refine_ab_evals.py      :: same A/B, evals from the PROFILE line
set NPFIXEDCOMPY_PROFILE=1
.venv\Scripts\python.exe -c "import numpy as np, npfixedcomppy as n; x=np.random.default_rng(3).normal(0,1,5000); n.computemixdist(x, method='nptll', beta=5)"
:: per-phase timing line (n=5000, nptll beta=5, with evals=) to stderr
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
The per-objective-path medians behind the §4a tables come from
`tests/measure_2d.py` — once as-is (fast path, the default) and once
with `NPFIC_2D_EXACT=1` (the R-identical path); `NPFIXEDCOMPY_PROFILE=1`
on either run prints the `objevals` line.
