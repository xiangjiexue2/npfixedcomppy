# Performance

This document records the performance architecture of `npfixedcomppy` and
the evidence behind it (measured on the development machine).

## 1. Where the time goes

Phase profile of a full `computemixdist` call
(`NPFIXEDCOMPY_PROFILE=1` prints one line to stderr, median of 3 runs):

| case | iters | total | solvegrad | mapping | weights | collapse | loss |
|------|-------|-------|-----------|---------|---------|----------|------|
| `nptll` β=5, n=5000 | 31 | **2.99 s** | ≈2.5 s | 0.3 | 8.8 | 260 | 0.8 |
| `npnormcll`, n=1000 | 19 | **0.99 s** | 462 | 2.2 | 501 | 29 | 0.1 |
| `nptll` β=∞, n=1000 | 15 | **36 ms** | ≈20 | 1 | 8 | 6 | 0.5 |
| `npnormll`, n=5000 | 15 | **18 ms** | ≈10 | <1 | 2 | 3 | 0.3 |

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
* Everything else (mapping after the cache, loss, collapse) is single-digit
  milliseconds.

## 2. Compute core: serial C++/Eigen

The whole compute stack — engine, support-point solvers, constrained NNLS,
gradient and weight sweeps, kernel-matrix fills, the mixture density —
lives in `cpp/` (pybind11 module `npfixedcomppy._core`, built by
`setup.py` with MSVC `/std:c++17 /O2 /arch:AVX2`, GCC/Clang
`-O3 -mavx2` elsewhere).

**SIMD is decided by the compiler at BUILD time, not at runtime.**
`/arch:AVX2` lets the compiler emit the widest instructions the build host
supports — `exp` auto-vectorizes to packed math sequences and the scalar
loops get full autovec. The same source rebuilds cleanly on a machine
without AVX2 (override with `NPFIC_ARCH=`, everything falls back to
scalar, same code path). No runtime ISA dispatch.

**There is no OpenMP and no thread pool — the solver is fully serial.**
All hand-written `#pragma omp` was stripped (it interacted badly with
Eigen's internal thread handling and made results machine-dependent);
Eigen's own compile-time parallel GEMM is disabled, so the process owns
exactly zero worker threads. The consequence:

* **bit-identical determinism** — identical inputs give bit-identical
  outputs on every machine, which the GOLD parity gates rely on;
* no thread-pool oversubscription to tune, no `OMP_*`/`RAYON_*` knobs.

The one environment knob is `NPFIXEDCOMPY_PROFILE=1` (per-phase timing).

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
* all five parity suites (npnormll / nptll / cvmadcll / pois / density)
  pass `TOTAL BAD: 0` unchanged with the cache in place.

Measured effect (this build):

| case | before cache | after cache |
|------|-------------|-------------|
| `nptll` β=5, n=5000 | 4.43 s | **3.0 s** (−32 %) |
| `npnormcll`, n=1000 | 1.21 s | **0.99 s** (−19 %) |
| `npnormad`, n=1000 | 60 ms | **37 ms** (−38 %) |
| `npnormcvm`, n=1000 | — | **36 ms** |
| `nptll` β=∞, n=1000 | 43 ms | **36 ms** |
| `npnormll`, n=1000 / n=5000 | 8 ms / 18 ms | **8 ms / 18 ms** (already near the floor) |

## 4. Comparison with R `npfixedcomp2` (same machine)

Median of 3, `computemixdist`, recorded via `bench_r.R` /
`tests/perf_baseline.py`:

| case | `npfixedcomppy` | R | speedup |
|------|-----------------|---|---------|
| `npnormll`, n=1000 | 7.8 ms | 15 ms | 1.9× |
| `nptll` β=∞, n=1000 | 36 ms | 212 ms | 5.8× |
| `nptll` β=5, n=5000 | 3.0 s | 50.0 s | 16.6× |
| `npnormcll`, n=1000 | 988 ms | 1.95 s | 2.0× |
| `npnormad`, n=1000 | 37 ms | 38 ms | 1.0× |
| `nppoisll`, n=1000 | 0.62 ms | 5 ms | 8.1× |
| `estpi0` (norm), n=1000 | 25 ms | — | — |

The big wins are the expensive-kernel cases (β=5 t, 16.6×). The
`npnormad` row is at parity: its AD loss weights are per-iteration and the
fit converges in few iterations, so the cache removes little absolute
time.

## 5. What was *not* done, and why

* **Hand-written OpenMP** — stripped entirely (see section 2); it was a
  crash/determinism hazard and the gains are re-obtained bit-safely by the
  cache.
* **SIMD GEMM library (`faer`)** — earlier measurements (tall-and-thin
  `n × m` with `m ≤ ~20` production columns) showed GEMM libraries lose to
  a streaming dot-loop at these shapes; not adopted.
* **Threading the remaining work** — the two remaining hotspots (the
  off-grid `dnt` evaluations in `solvegrad`, the NNLS in `weights`) are
  embarrassingly parallel over `n`, but any thread pool reintroduces the
  determinism/oversubscription trade-offs above. Kept serial per the
  design constraint; the measured speedups versus R already come from the
  single-threaded core + the cache.

## 6. Reproducing the evidence

```bat
cd npfixedcomppy
.venv\Scripts\python.exe tests\perf_baseline.py          :: wall-time table
set NPFIXEDCOMPY_PROFILE=1
.venv\Scripts\python.exe tests\profile_n5000.py          :: per-phase line (n=5000)
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" bench_r.R   :: R reference times
```

Parity gates (all must report `TOTAL BAD: 0`):

```bat
.venv\Scripts\python.exe tests\verify_npnormll.py
.venv\Scripts\python.exe tests\verify_nptll.py
.venv\Scripts\python.exe tests\verify_cvmadcll.py
.venv\Scripts\python.exe tests\verify_pois.py
.venv\Scripts\python.exe tests\verify_density.py
```
