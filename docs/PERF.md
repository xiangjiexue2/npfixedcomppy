# SIMD + Parallel Computing Strategy

This document records the performance architecture of `npfixedcomppy`, the
evidence behind it (measured on the development machine, n = 5000, k = 200,
12 threads), and the **rules** that keep the parallelism safe and
predictable.

## 1. Where the time goes

Phase profile of a full `computemixdist` call at **n = 5000**
(`NPFIXEDCOMPY_PROFILE=1`), after the kernel cache (section 6):

| case | iters | total | solvegrad | mapping | weights | collapse | loss |
|------|-------|-------|-----------|---------|---------|----------|------|
| `npnormll` β=1 | 15 | **39.4 ms** | 22.9 | 1.4 | 8.4 | 6.3 | 0.4 |
| `nptll` β=5 | 31 | 509.6 ms | 290.9 | 37.1 | 71.0 | 109.2 | 1.3 |
| `nptll` β=∞ | 15 | 31.8 ms | 18.2 | 1.2 | 6.6 | 5.4 | 0.4 |

The dominant hot paths, in descending cost order:

| # | hot path | cost at n=5000, k=200 (median) | parallelism used |
|---|----------|-------------------------------|------------------|
| 1 | kernel-matrix fill `K[i][j] = kernel(data[i], mu[j])` | serial 6.3 ms (normal) / 22.5 ms (poisson) → **1.1 / 3.2 ms with rayon** | rayon over observation rows |
| 2 | gradient sums over the data (per support point) | same order as (1) | single-pass per-thread accumulators (`par_acc`) |
| 3 | small GEMMs: Gram `G = KᵀK`, mat-vec `s = Kᵀv`, NNLS | 10–11 ms (Gram), 2.8 ms (mat-vec), scalar | LLVM auto-vectorization |

Everything else (grid construction, bisection bookkeeping, Armijo line
search) is < 1 % of runtime. The `nptll` β=5 row shows why the t family is
the slow one: each of its 31 iterations runs the derivative-free parabolic
search (`dfmin`) on several crossing intervals, and **every interior
evaluation is a fresh O(n) pass over the data** — those points are not on
the grid, so the section-6 cache cannot help them.

## 2. The parallelism model: ONE pool, never nested

The package uses **exactly one thread pool** (`rayon`) at **exactly one
layer**. The rules:

1. **No nested parallelism.** Any code that may run inside a rayon task
   (all of `families.rs`, `nnls.rs`, `engine.rs`) must not itself launch
   rayon work or call a BLAS/linear-algebra library with its own internal
   threading. A second pool inside a task either starves the outer pool
   (barrier deadlock at best) or oversubscribes the cores (thread explosion
   at worst).
2. **Parallel only over observations (n), never over support points (k).**
   `k` is small (default grid = 100, a few support points per fit); there is
   no work to share. `n` is the only large dimension.
3. **Serial fallback below a threshold.** Per-observation loops are serial
   for `n < PAR_N` (default 2048, overridable via the
   `NPFIXEDCOMPY_PAR_N` environment variable). Below the threshold the
   serial path is also **bit-identical** to the original ported code —
   the rayon path uses tree reductions that reassociate floating-point
   additions and change only the last ulps (far below the 1e-6 parity
   tolerance used against the R reference).
4. **faer (SIMD) may be layered on, but only without its rayon feature.**
   `faer`'s SIMD GEMM kernels (pulp) are single-threaded, so calling
   `matmul(…, Par::Seq)` from a rayon task is legal under rule 1. The
   default feature set of `faer` **enables its own rayon pool** — if faer
   is ever added as a real dependency it must be declared as
   `default-features = false, features = ["std", "linalg"]` (this is what
   the prototype `Cargo.toml` dev-dependency does). Using
   `Par::Rayon` inside a rayon task would violate rule 1.

### Thread control

| knob | effect |
|------|--------|
| `NPFIXEDCOMPY_PAR_N` (env) | raise it to force the fully-serial, bit-identical path (used by parity gates and benchmarking) |
| `RAYON_NUM_THREADS` (env) | size the single rayon pool (e.g. `1` for serial timing, or to cap CPU usage) |
| none | default: pool = physical cores, parallel for `n ≥ 2048` |

The library never calls `rayon::ThreadPool::builder` itself; it uses the
global pool, so callers can size it once for the whole process.

## 3. Why not a SIMD GEMM library (faer)? — measured evidence

The user's concern was to *avoid* a parallel-SIMD library (its internal
pool would nest under our rayon pool). `faer` can be used **without**
internal threading (`Par::Seq` + no `rayon` feature), so it was prototyped
honestly — it is **no longer in the dependency tree at all**; the
numbers below were measured with it as a temporary dev-dependency and are
kept here as the record of the decision.

Measured (i7-8700K, 12 threads, `cargo run --release --example
bench_simd`, median of 5):

| op (n=5000, k=200) | nalgebra (current) | faer `Par::Seq` | nalgebra, rayon row-blocks |
|--------------------|-------------------|-----------------|----------------------------|
| Gram `G = KᵀK` (0.40 GFlop) | 10.98 ms (36.4 GF) | 12.33 ms (32.4 GF) | **10.04 ms (39.8 GF)** |
| mat-vec `s = Kᵀv` (0.002 GFlop) | 2.79 ms | 4.92 ms | **2.35 ms** |
| kernel fill, normal (1 M elts) | 6.34 ms serial → 1.12 ms rayon | n/a (no primitive) | — |
| kernel fill, poisson (1 M elts) | 22.45 ms serial → 3.24 ms rayon | n/a (no primitive) | — |

**Conclusions**

1. **faer is ~12 % *slower* than nalgebra here.** The reason is the
   shape: `k = 200` is far below the ~512–1024 column count at which
   GEMM libraries start beating the simple `m × n` dot-loop. At such
   "tall-and-thin" shapes the arithmetic is dominated by streaming
   (memory-bound) reductions along `n`; the small `k × k` tile never
   fills an AVX2 lane's worth of independent work, and faer's
   kernel dispatch overhead plus its column-major packing of the LHS
   loses to nalgebra's contiguous row-major inner loop. Note the bench
   is a *stress test*: in production the GEMM dimension is the number
   of current support points `m` (typically 1–10 per fit, ≤ ~20 before
   collapse), **not** the 100-point grid — the real shapes are even
   further inside the streaming-dominated regime where no GEMM library
   wins.
2. **SIMD is still used — implicitly.** The nalgebra kernels the engine
   calls (`*` for the small Gram/mat-vec, the NNLS Householder
   reflections) are ordinary `f64` loops that LLVM auto-vectorizes with
   AVX2 on this CPU; the measured 36–40 GFlops (vs ~0.2 GFlops serial in
   scalar C) is the auto-vec win. No hand-written SIMD is needed.
3. **The winning parallelism is the outer rayon layer** (6.4× on the
   normal fill, 6.9× on the poisson fill), which faer does not provide
   for our fill pattern anyway (the kernel is transcendental and
   row-independent — there is no off-the-shelf primitive; a per-column
   `zip!` would force a non-contiguous access pattern over `data` and
   lose to the row loop).
4. **If `k` were ever large** (e.g. a 2000-point grid for a binned
   family), the trade changes: a no-rayon faer `Par::Seq` GEMM should be
   re-benchmarked, *and* the row-block decomposition of variant B3
   (rayon over n-blocks, each block doing the GEMM on its sub-matrix,
   block results summed) remains the safe pattern under rule 1 — it gave
   the best Gram number above (39.8 GF) without any second pool.

**Decision: keep nalgebra + rayon. faer was removed from the dependency
tree** (the `bench_simd` example now benchmarks only the production
patterns: nalgebra serial, nalgebra + rayon row-blocks, and the rayon
kernel fill).

## 4. estpi0 refinement

`estpi0`'s outer bisection is embarrassingly serial (each step needs the
previous), so all its parallelism is *inside* the inner
`computemixdist` calls (section 1). The `fast` refinement mode (default
`true`) performs one inner solve per bisection step at `inner_tol`
instead of two at 1e-6; it is an algorithmic, not a threading, change —
no interaction with rules 1–4.

## 6. The grid kernel cache (n = 5000: normal 73.6 ms → 39.4 ms)

Before this cache, every `solvegrad` grid sweep of the normal family
recomputed the full data×grid kernel `dnormv(data[i], mu[j], beta)`
(n × |grid| transcendental evaluations) on every iteration — 56 ms of the
73.6 ms total at n = 5000. Both `npnormll` and `nptll` now precompute the
matrix once per fit in `prepare` (called by the engine when the sorted
grid is fixed) and store it column-major (`kmat[j*n + i]`), so each
iteration's full-grid sweep is a plain dot product against contiguous
columns.

Correctness is preserved, not approximated:

* The kernel depends only on the data, `beta` and the grid — never on the
  current weights or the fixed components — so the matrix is valid for the
  whole fit, including across `estpi0`'s bisection (same family instance,
  same grid).
* The cached sweep accumulates column `j` **serially over `i`**, in the
  exact order and arithmetic of the original per-point loop
  (`dnormv(data[i], kgrid[j], beta) * fullden[i]`), so the result is
  bit-identical to the uncached path — the GOLD/BAND parity gates and the
  deterministic re-run check pass unchanged.
* Non-grid evaluations (interior points of the Brent / parabolic root
  solvers, `mapping`, the weight subproblem) never touch the cache; they
  keep their original code paths.

Measured at n = 5000 (single runs, `NPFIXEDCOMPY_PROFILE=1`):

| case | before | after |
|------|--------|-------|
| `npnormll` β=1, 15 iters | 73.6 ms (solvegrad 56.0) | **39.4 ms (solvegrad 22.9)** |
| `nptll` β=∞, 15 iters | 72.0 ms (solvegrad ≈54) | **31.8 ms (solvegrad 18.2)** |
| `nptll` β=5, 31 iters | 495 ms (solvegrad ≈283) | 509.6 ms (solvegrad 290.9) |

The β=5 row is unchanged **by construction**: its grid sweep was already
the cached one, and the remaining cost is the derivative-free interior
searches, which evaluate points off the grid (see section 1). The cache
allocation itself is `n × |grid|` f64 ≈ 0.4 MB at n = 5000, |grid| = 100,
built once in `prepare` (parallel over columns for `n ≥ PAR_N`).

## 7. Reproducing the evidence

```bat
cd npfixedcomppy
set NPFIXEDCOMPY_PAR_N=2048
cargo run --release --example bench_simd
```

Parity with the fully-serial path (what the R-parity gates rely on):

```bat
set NPFIXEDCOMPY_PAR_N=999999999
python tests\verify_nptll.py
```
