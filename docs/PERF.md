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
`(mu, dens)` gradient evaluations inside the `solvegrad` loop;
`freshcols` / `freshms` count the off-grid kernel columns that had to
be evaluated for the first time inside that loop (the ones the
section-3 column cache cannot serve) and time that work;
`certskips` (0.3.0) counts the d1 sign-change intervals the section-4d
crossing certificate proved gain-positive and skipped (0 when the
gate is off or never fires):

| case | iters | total | solvegrad | mapping | weights | collapse | loss | evals | freshcols | freshms |
|------|-------|-------|-----------|---------|---------|----------|------|-------|-----------|---------|
| `nptll` β=5, n=5000 | 11 | **266 ms** | 183 | 0.1 | 2.9 | 80 | 0.3 | 1191 | 61 | 255 |
| `npnormcll`, n=1000 | 31 | **1233 ms** | 374 | 5.5 | 802 | 53 | 0.2 | 24425 | 11363 | 346 |
| `npnormadw`, n=5000 | 17 | **184 ms** | 93 | 23 | 48 | 20 | 0.7 | 1859 | 0 | 0 |
| `nptllw` β=∞, n=5000 | 18 | **35 ms** | 10 | 0.1 | 6.4 | 18 | 0.3 | 1852 | 53 | 22 |
| `npnormllw`, n=5000 | 18 | **21 ms** | 2.2 | 2.9 | 11 | 4.6 | 0.3 | 1852 | 0 | 0 |
| `nptll` β=∞, n=5000 | 18 | **21 ms** | 11 | 0.1 | 4.2 | 5.9 | 0.5 | 1852 | 53 | 5.5 |
| `npnormll`, n=5000 | 17 | **9.9 ms** | 3.8 | 0.1 | 3.1 | 2.8 | 0.3 | 1828 | 157 | 6.8 |
| `npnormad`, n=1000 | 13 | **19.7 ms** | 14.5 | 0.0 | 1.6 | 3.3 | 0.2 | 1419 | 138 | 20.5 |

(0.2.3 build; 0.2.2's row for `nptll` β=5 was 2334 ms / 3733 evals /
455 fresh columns — see section 4c for the delta.)

Reading the profile:

* **`nptll` β=5** — still dominated by `solvegrad`, but the 0.2.3
  support-search relaxation (section 4c) cut the gradient evaluations
  3733 → 1191 (3.1×) and the fresh off-grid columns 455 → 61 (7.5×):
  183 ms of solvegrad wall (the 61 fresh columns account for 255 ms of
  new-column wall time — each fresh column is one full data-column of
  the non-central-t kernel, `dnt` = AS-243 series, the most expensive
  kernel in the package) plus ≈ 80 ms of `collapse` remapping. The
  kernel column cache eliminates the *on-grid* re-evaluations (≈1.4 s of
  the pre-cache 4.4 s); the pdf's `df`-level constants (`log df`,
  `√((df+2)/df)`, the `gammln` pair) are precomputed once per run into
  `stats::DntConst` (used via `dnt_c`, bit-identical to `dnt`).
* **The `d0` families' cost is the fresh off-grid columns.** The
  `freshms`/`total` split on the current build: `nptll` β=5, n=5000 —
  255 ms of 266 ms (61 fresh columns); `nptllw` — 22 ms of 35 ms
  (53 fresh columns). Before the 0.2.3 relaxation the same cases ran
  2304 / 1053 ms of fresh-column work (455 / 300 columns) — the
  negative-gain early stop (a candidate refinement returns as soon as
  ANY evaluated point has negative gain; the caller's sign filter accepts
  any negative point, the exact minimum is not needed) and the CNM
  re-verification (section 4c) mean most candidates never reach the
  parabolic refinement that used to mint fresh columns.
* **`npnormcll`** — dominated by `weights` (≈65 %): the per-iteration
  constrained NNLS subproblem (`pnnlssum` for n ≤ 1000, `pnnqp` beyond),
  plus the grid sweep in `solvegrad`. These are the algorithm's
  intrinsic small-matrix solves, not re-evaluation waste. Note 0.2.3
  also changed WHAT it computes: the free grid-point acceptance (section
  4c) admits a richer support set (92 → 107 points) and a slightly
  better ll (4.8e-6 relative vs the 0.2.2 fit), at 19 → 31 iterations —
  the one family where the relaxation costs wall time (section 4c).
* **binned normal families (`npnormllw` / `npnormcvmw` / `npnormadw`)** —
  the grid sweep is served from the pinned grid fill (section 3), so
  `npnormllw` totals **21 ms** engine / **28 ms** wall; what remains in
  its `solvegrad` is the off-grid single-point candidate fills (each is
  a new `mu` vector, so a fresh fill) plus the cheap d1 pdf part.
  `npnormadw` (184 ms total, 36 → 17 iterations under the relaxation)
  still churns: its AD re-weighting keeps producing new support sets, so
  most fills miss the memo, and ~20 ms goes to `collapse`.

### §1 中文注释

完整 `computemixdist` 调用的分阶段 profile（`NPFIXEDCOMPY_PROFILE=1`
打印一行到 stderr，3 次取中位数；单位 ms；profiled `total` 只覆盖
C++ 引擎各阶段，§4 的 wall 另含 Python 侧分箱与网格生成；`evals` 是
`solvegrad` 循环内的 `(mu, dens)` 梯度求值数；`freshcols`/`freshms`
是循环内首次求值的离网格核列数与耗时——§3 列缓存服务不了的）。要点：

* **`nptll` β=5** — 仍由 `solvegrad` 主导，但 0.2.3 支撑点搜索放宽
  （§4c）把梯度求值 3733 → 1191（3.1×）、fresh 列 455 → 61（7.5×）；
  61 列 fresh `dnt`（AS-243 级数，包内最贵核）占 255 ms，加 ≈80 ms
  `collapse` 重映射。核列缓存消掉的是*在网格*重求值（缓存前 4.4 s
  中的 ≈1.4 s）；pdf 的 `df` 级常数（`log df`、`√((df+2)/df)`、
  `gammln` 对）每 run 预算进 `stats::DntConst`（经 `dnt_c`，与
  `dnt` 位级一致）。
* **`d0` 族的成本就是 fresh 离网格列**：`nptll` β=5 为 255/266 ms
  （61 列）、`nptllw` 为 22/35 ms（53 列）；0.2.3 放宽前同用例
  fresh 工作 2304/1053 ms（455/300 列）——负增益早停（任一求值点
  增益为负即返回；调用方符号过滤接受任一负点，不要求精确最小）+
  CNM 再校验（§4c）使多数候选到不了曾产生 fresh 列的抛物精化。
* **`npnormcll`** — 由 `weights` 主导（≈65%）：每轮约束 NNLS 子问题
  （n ≤ 1000 用 `pnnlssum`，更大用 `pnnqp`）+ 网格扫描——算法固有
  小矩阵求解，不是重复求值浪费。0.2.3 放宽还改变了它算的内容：
  接受更丰富支撑集（92 → 107 点）、ll 略好（相对 0.2.2 拟合
  4.8e-6），代价 19 → 31 轮——唯一放宽使 wall 变慢的族（§4c）。
* **分箱正态族** — 网格扫描由钉住的网格填充服务（§3），
  `npnormllw` 引擎 21 ms / wall 28 ms；其 `solvegrad` 剩的是离网格
  单点候选填充（每个是新 `mu` 向量，必然 fresh）加廉价的 d1 pdf
  部分。`npnormadw`（总 184 ms，36 → 17 轮）仍翻搅——AD 重加权
  不断产生新支撑集，填充大多 miss memo，≈20 ms 进 `collapse`。

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
`solvegrad` evaluation count and the fresh-column count/time),
`NPFIC_REFINE_STEPS` (experimental: caps the per-candidate refinement
steps inside `brmin`/`dfmin`; default `-1` = unlimited = shipped
behaviour — see section 5), and, for OpenMP builds,
`OMP_NUM_THREADS` (sizes the Eigen pool; see above for the effect on
results). (The `NPFIC_WARM` knob documented in section 4b was REMOVED in
0.2.3 — its `=2` CNM re-verification arm is now the always-on engine
behaviour, section 4c.)

### §2 中文注释

整个计算栈——引擎、支撑点求解器、约束 NNLS、梯度/权重扫描、核矩阵
填充、混合密度——都在 `cpp/`（pybind11 模块 `npfixedcomppy._core`，
`setup.py` 默认 MSVC `/std:c++17 /O2 /arch:AVX2`，GCC/Clang
`-O3 -mavx2`）。

**SIMD 级别在构建期而非运行期决定**，由 `NPFIC_ARCH` 环境变量选择：
`avx2`（默认）/ `avx` / `avx512` / `native`（仅 GCC/Clang
`-march=native`）/ 空 = 纯 x86-64 SSE2 基线。构建产物只能跑在该指令
集（或超集）硬件上；无运行期 ISA 分发，源码不变。注意 MSVC 与
GCC/Clang flag 行为**不一致**：MSVC `/arch:AVX2` 会让优化器把乘加对
收缩为 FMA（`/fp:precise` 下的文档化 MSVC 行为）；GCC/Clang 不加
`-mfma` 则 FMA 不开（实际未加）。parity 门验证的是出货构建；跨架构
位级一致不是要求。

SSE2 基线 → AVX2 的两个实测后果：

* 宽分箱填充（`npnormllw`、`nptllw`）约 2×（n=5000 354 → 130 ms），
  与 R 的历史差距缩到几个百分点内——网格填充 memo（§3）之后
  `npnormllw` 进一步到 ≈37 ms wall，比 R 快 ≈3×（§4）；
* 小样本未分箱正态变慢 ≈4 ms（`npnormll` n=1000：7.8 → 11.6 ms）
  ——短向量的 AVX2 尾部效应；`NPFIC_ARCH=` 可恢复基线。t 族与所有
  大样本用例不受影响。

**包内没有手写 OpenMP——唯一的线程来自 Eigen 自身，由构建期 OpenMP
探测打开。** 所有手写 `#pragma omp` 已移除（与 Eigen 内部线程处理
相互作用，且结果变得机器相关）。改为：`setup.py` 用编译器的 OpenMP
flag（MSVC `cl /openmp`、Unix `g++`/`clang++ -fopenmp`）探测编译，
通过则把 flag 加进扩展构建——正是定义 `_OPENMP` 的 flag，即 Eigen
编译期门的整体（`eigen/Eigen/Core` 的 `EIGEN_HAS_OPENMP`）。于是
mapping/computeweights 热路径上 Eigen 内部并行 GEMM/GEMV 生效，其余
循环保持串行（位级一致的参考顺序）；编译器无该 flag 时构建回退
串行 Eigen，别无他变。确定性契约：

* **固定构建 + 固定 `OMP_NUM_THREADS`** 下，相同输入 → 位级相同输出
  ——Eigen 并行 GEMM 把归约分成固定数量的块、按固定顺序合并，GOLD
  parity 门依赖的正是这一点；
* 换线程数至多重排 Eigen 并行 GEMM 内的浮点归约——舍入尺度差异，
  远在 parity 门（ll 1e-9、pt 1e-6）之内。

环境旋钮：`NPFIXEDCOMPY_PROFILE=1`（分阶段计时 + `solvegrad` 求值
数与 fresh 列数/耗时）、`NPFIC_REFINE_STEPS`（实验性：限制
`brmin`/`dfmin` 内每候选精化步数；默认 `-1` = 不限制 = 出货行为，
§5）、OpenMP 构建下的 `OMP_NUM_THREADS`（调 Eigen 线程池，对结果的
影响见上）。§4b 记载的 `NPFIC_WARM` 旋钮已于 0.2.3 **移除**——其
`=2` CNM 再校验臂现为引擎常开行为（§4c）。

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

### §3 中文注释

缓存前的主要浪费：引擎在**每一轮外层迭代的每个环节**重复求值*相同*
的核列——`solvegrad` 网格扫描、`mapping`、`computeweights` 填充、
`collapse` 重映射、轮末 mapping。对昂贵核（非中心 t、正态 CDF、
单参数正态）即每轮 3–8 次冗余的 `(n × m)` 矩阵求值。

核列 `K[:, mu]` **只**依赖 `(data, beta, mu)`——与当前权重、密度、
迭代轮次无关——所以每个不同 `mu` **每次拟合只求值一次**（网格点
在 `prepare` 里、离网格点首次使用时惰性求值），所有消费方读共享
列。

正确性被保持而非近似：

* 缓存列由与 fresh 填充完全相同的逐行求值产生（升序 `i`、相同算术），
  与无缓存计算**位级一致**——只去掉冗余重求值，从不改变算术或其
  顺序；
* 从列组装 `(n × m)` 矩阵的消费方保留参考的逐列累加；
* 全部 parity 套件在缓存就位后仍 `TOTAL BAD: 0` 不变。

实测效果（本构建）：`nptll` β=5 n=5000：4.43 s → **2.47 s**（−44%）；
`npnormcll` n=1000：1.21 s → **0.99 s**（−19%）；`npnormad` n=1000：
60 ms → **19.5 ms**；`npnormcvm` n=1000：**36 ms**；`nptll` β=∞
n=1000：43 ms → **36 ms**；`npnormll` n=1000/5000：8/18 ms →
**11.6/19.2 ms**（小样本 AVX2 尾部，见 §2）。

**列缓存不适用于三个分箱正态族（`npnormllw` / `npnormcvmw` /
`npnormadw`）。** 其核矩阵 `K(bins × grid)` 是梯形填充，*宽度*
（分量列数 `N`）依赖当前支撑分布的展宽——同一个 `(bin, grid)` 点在
不同扫描点是*不同*的核，没有可复用的 `(x, mu)` 列。（`nptllw` 是
例外：其分箱 t 核是 CDF 差值 `Φ(x; μ−h) − Φ(x; μ)`，纯
`(data, df, h, mu)` 的函数，因此离网格点用列缓存。）填充本身是 R
`dnormarray` 布局的列主序重写（无行主序临时量、无转置、常量外提），
与参考累加位级一致。

**网格填充 memo（分箱正态族）。** 列缓存复不了的，memo 能复：一次
拟合内，分箱填充只为很少一组重复的 `mu` 向量被请求——网格、当前
支撑集、若干求解器内部点——所以 `detail::KernelMemo` 按*精确* `mu`
向量键控缓存最近一次填充，网格填充在 `prepare` 钉住（由 `grid_m`
服务）。被服务的矩阵与同参数 fresh `ddiscnorm_m` / `pnorm_disc_m`
位级一致；不同 `mu` 向量则 fresh 填充。这消掉了曾主导分箱
`solvegrad` 的逐候选重填：`npnormllw` n=5000 从 ≈131 ms wall
（129 ms 引擎总计中网格扫描 ≈108 ms）降到 ≈31 ms 引擎 / ≈37 ms
wall——现在比 R 的 ≈120 ms **快 ≈3×**（剩下的是离网格单点候选
填充，每个是新 `mu` 向量，加廉价的 d1 pdf 部分）。`npnormadw` 受益
较少——AD 重加权不断产生新支撑集，填充大多 miss memo——仍落后于 R
（332 ms vs ≈170 ms）。

## 4. Comparison with R `npfixedcomp2` (same machine)

Py: median of 5 (`tests/perf_baseline.py` / `tests/bench_binned.py`);
R: best of 3 (`tests/bench_r.R` / `tests/bench_binned_r.R`), R 4.6.1 +
RcppEigen 3.4.0 compiled by R's default MSVC flags (SSE2 baseline).

(Py numbers are the 0.2.3 build, re-measured with the same scripts; the
0.2.2 column is shown for the cases the 0.2.3 relaxation (§4c) moved
most. R numbers are unchanged.) The 0.3.0 build leaves the 1-D engine
and kernels untouched and the k = 2 normal family bit-identical
(§4a), so these numbers carry over.

Un-binned:

| case | `npfixedcomppy` (0.2.3) | 0.2.2 | R | speedup |
|------|-------------------------|-------|---|---------|
| `npnormll`, n=1000 | 5.5 ms | 10.7 | 19 ms | 3.5× |
| `npnormcvm`, n=1000 | 32.9 ms | 34.6 | — | — |
| `npnormad`, n=1000 | 36.8 ms | 35.5 | 59 ms | 1.6× |
| `npnormcll`, n=1000 | 1269 ms | 1009 | 1924 ms | 1.5× |
| `nppoisll`, n=1000 | 0.45 ms | 0.57 | 5 ms | 11× |
| `nptll` β=∞, n=1000 | 17.5 ms | 36.2 | 211 ms | 12× |
| `npnormll`, n=5000 | 19.4 ms | 27.2 | — | — |
| `nptll` β=5, n=5000 | **857 ms** | 2.95 s | 49.7 s | **58×** |
| `npnormll`, n=50000 | 215 ms | 326 | — | — |
| `estpi0` (norm), n=1000 | 14.3 ms | 24.9 | 97 ms | 6.8× |

Binned (`order = -3`):

| case | `npfixedcomppy` (0.2.3) | 0.2.2 | R | ratio |
|------|-------------------------|-------|---|-------|
| `npnormllw`, n=5000 | 27.9 ms | 37.1 | 120 ms | **4.3×** |
| `npnormcvmw`, n=5000 | 152.8 ms | 118.7 | 110 ms | 0.8× |
| `npnormadw`, n=5000 | 240.4 ms | 386.7 | 170 ms | 0.7× * |
| `nptllw` β=∞, n=5000 | 112.1 ms | 206.1 | 1970 ms | **17.6×** |
| `nptllw` β=5, n=5000 | 559.6 ms | 2.01 s | 29.3 s | **52×** |
| `estpi0` (`npnormllw`), n=5000 | 53.4 ms | 83.8 | 1330 ms | **25×** |
| `npnormllw`, n=20000 | 65.4 ms | 157.5 | 540 ms | **8.3×** |
| `nptllw` β=∞, n=20000 | 213.2 ms | 779.4 | 7250 ms | **34×** |
| `nptllw` β=5, n=20000 | 985.2 ms | 2293.6 | (minutes; not recorded) | — |

\* `npnormadw` fits are run-to-run non-deterministic in both
implementations (R wanders 6 basins, `ll` 0.17697…0.18252, over 13
runs), so this ratio is indicative, not structural — a slower basin on
the Py side inflates it.

Reading: the 0.2.3 support-search relaxation (§4c) widens the t-family
lead to 58× (un-binned β=5) and 17–52× (binned) — the relaxed interval
refinement cuts the fresh off-grid `dnt` columns 455 → 61 at n=5000.
The un-binned normal families gain 1.3–2× (the `fl` cache and the
zero-cost grid acceptance). The binned normal families still *split*:
`npnormllw` is now 4–8× faster than R (28 ms vs 120 ms at n=5000;
65 ms vs 540 ms at n=20000) — the grid-fill memo (section 3) serves
the per-iteration grid sweep from the precomputed grid fill, and the
relaxation shrinks the off-grid candidate work further; `npnormadw`
narrows to ~0.7× (240 ms vs 170 ms, 36 → 17 outer iterations) but
still trails, for the same memo-churn reason plus its aggressive
`collapse` re-weighting; `npnormcvmw` is ~0.8× (153 ms vs 110 ms) —
the relaxation visits a slightly slower basin (its `ll` stays within
R's band, §4c). Where a binned fit *does* pay for a fresh trapezoid
fill, both sides fill the same matrix with the same bit-exact
accumulation and both call the scalar libm `exp` — neither Eigen 5.0.0
nor R's RcppEigen 3.4.0 ships a double-precision SIMD `pexp` for x86
(grep-verified in both packet headers) — so a fresh fill is `exp`-bound
on both sides and no wider instruction set can touch it; the remaining
binned gap is *how often* each side refills, and the memo closes it for
the LL-style sweep.

### §4 中文注释

测量口径：Py 为 5 次中位数（`tests/perf_baseline.py` /
`tests/bench_binned.py`）；R 为 3 次最佳（`tests/bench_r.R` /
`tests/bench_binned_r.R`），R 4.6.1 + RcppEigen 3.4.0、R 默认 MSVC
flag（SSE2 基线）。Py 列是 0.2.3 构建（同一脚本重测；0.2.3 放宽
§4c 移动最多的用例给出 0.2.2 对照列；R 数字不变）。0.3.0 构建未
动一维引擎与核、k = 2 正态族位级一致（§4a），故这些数字继续有效。

解读：0.2.3 支撑点搜索放宽（§4c）把 t 族领先扩到 58×（未分箱 β=5）
与 17–52×（分箱）——放宽的区间精化把 n=5000 的 fresh 离网格 `dnt`
列 455 → 61。未分箱正态族得 1.3–2×（`fl` 缓存 + 零成本网格接受）。
分箱正态族仍然*分化*：`npnormllw` 比 R 快 4–8×（n=5000 28 ms vs
120 ms；n=20000 65 ms vs 540 ms）——网格填充 memo（§3）直接服务每轮
网格扫描，放宽又缩小离网格候选工作；`npnormadw` 缩到 ~0.7×
（240 ms vs 170 ms，36 → 17 轮）但仍落后——同一 memo 翻搅原因 +
激进的 `collapse` 重加权；`npnormcvmw` ~0.8×（153 ms vs 110 ms）
——放宽访问了一个略慢的 basin（其 `ll` 仍在 R 的 band 内，§4c）。
分箱拟合*确实*付一次 fresh 梯形填充时，两侧用同一位级累加填同一
矩阵、都调标量 libm `exp`——Eigen 5.0.0 与 R 的 RcppEigen 3.4.0
都不带 x86 双精度 SIMD `pexp`（两侧 packet 头 grep 证实）——所以
fresh 填充两侧都受 `exp` 束缚、更宽指令集也救不了；剩下的分箱差距
是*重填频率*，memo 已为 LL 型扫描消掉它。

\* `npnormadw` 两侧实现都逐次运行不确定（R 13 次运行漂过 6 个
basin，`ll` 0.17697…0.18252），该比值仅供参考不是结构性的——Py 侧
较慢的 basin 会把它抬高。

### 4a. The `npnormND` (N-D normal) exception — k = 2 is the R `npnorm2Dll`

The multivariate-normal family — the R `npnorm2Dll` generalized to N
dimensions in 0.3.0 (`npnormND`; the R name is kept as the k = 2
alias) — is still the one documented case where this port is slower
than R (k = 2). Two fast paths — a hand-unrolled 2×2 density kernel
and an explicit-inverse objective (§4a.2) — plus the per-cell hot
start (§4a.3) and the true directional derivative (§4a.4) closed the
old ~28× gap to ~1.2× on the reference run:

| case (same machine, same data n=300, same initial mix & grid) | `npfixedcomppy` | R | ratio |
|---|---|---|---|
| 2-D n=300 (`npnorm2Dll` alias) | ≈ 0.91 s (median of 5) | ≈ 0.75 s (best of 3) | ~1.2× slower |

The R-identical objective path (`NPFIC_2D_EXACT`) — ≈ 4.3 s (median of
5) on the same data in 0.2.x builds — was **removed in 0.3.0**: the
relaxed contract (similar density, ll no worse — never a bit-exact
trajectory) made it unnecessary, and the N-D kernel supersedes its
Cholesky path. The fast/exact A/B tables below are that build's
historical record.

Verified decomposition (measured on the 0.2.3 build; the k = 2
trajectory carries over bit-identically to 0.3.0):

* **The kernels are bit-exact and ≈ 7× cheaper per call.** The k = 2
  case only ever calls `dnormNDarray` with `dim == 2` (k > 2 uses the
  general `LLT` path). The original
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
  (R's own: 848.88775287688986); an independent numpy recomputation
  matches to < 1e-12. The kernel fast path is bit-exact: the k = 2
  fit on the reference data still lands at ll
  848.7472072261477 / 10 iters / 5 support points in 0.3.0 — bit-
  identical to the 0.2.3 build (the unrolled 2×2 kernel and the
  column-major `(n, 2)` read are unchanged by the N-D generalization).
  The **fast objective** path (§4a.2) differs from the pre-objective-
  fast-path trajectory by ~1e-13 in the objective values and, through
  its hot-start + negative-gain early-stop search (§4a.3) and its
  true directional derivative (§4a.4), accepts a ≈ 1.9× smaller cell
  set (216,955 vs 421,810 objective evaluations on the reference
  run — the 421,810 figure is the removed R-identical path's).
* **The remaining cost is the L-BFGS-B support-point search.** The
  family runs one box-constrained L-BFGS-B problem per grid cell
  (103 × 103 = 10,609 sub-problems per outer iteration × 6 iterations)
  plus a weight subproblem and a collapse per iteration. The per-cell
  early-exit (`projgnorm ≤ ε` on the initial midpoint, executed on the
  first evaluation) is what makes most cells cheap; the cost is the
  non-exit cells. `NPFIXEDCOMPY_PROFILE=1` on the n=300 reference run
  reports `objevals=216955` for the fast path (the removed
  R-identical path ran 421,810 — the hot-start + negative-gain
  early-stop search (§4a.3) and the true directional derivative (§4a.4)
  together accept a ≈ 1.9× smaller cell set). The 4.3 s → 0.95 s
  end-to-end gain recorded in 0.2.x is the product of the per-
  evaluation kernel cost (≈ 10.2 µs exact → ≈ 4.4 µs fast) and the
  ~1.9× fewer cell evaluations. On the second n=300 benchmark the
  same split holds (769,533 → 316,253 evaluations, wall 8.0 s → 1.7
  s; fast path still 1.7 s / 316,253 in 0.3.0 — the k = 2 trajectory
  is bit-identical).
* **The result is a valid — in the reference run, slightly better —
  local optimum.** The search accepts a grid cell only on a *strictly*
  negative objective, which is exquisitely sensitive to the floating-
  point trajectory; the fit converges deterministically (10 iters):
  NLL 848.7472072261477 (R's: 848.88775287688986, lower is better).
  Against R's fit, the density similarity measured in 0.3.0 is
  rel-L1 0.0158 / max |log-density| difference 0.169, and the main
  components (w ≥ 0.02) sit in the same two modes — max location
  distance 0.675 (the FAST fit keeps one extra small shoulder
  component).
  `tests/verify_2d_same.py` pins all of this as the family's gate
  (two-run determinism, the ll, kernel parity vs R, density
  similarity vs R, and the grid-level certificate below).
* **The grid-level certificate (`grid_gain`) caught a real defect and
  now guards the path.** `get_ans` reports `grid_gain`, the minimum of
  the gain over ALL (G−1)² cell midpoints at the final mixture — the
  exact zero-cost candidate set the search itself starts each cell
  from. At the reference fit the corrected paths give `grid_gain`
  ≈ +4.7e-3 (fast) and +2.7e-7 (exact), both non-negative with 0
  negative cells, and an independent numpy recomputation from the
  public kernels matches the C++ value to < 1e-13. A *large* negative
  `grid_gain` means the objective was being evaluated on a corrupted
  dataset — the signature of the column-major/row-major data-read bug
  (§4a.3), which the certificate exposed (`grid_gain` ≈ −518, 3586
  negative cells) before the fix. The parity gate now fails on
  `grid_gain < −1`; small negatives (≈ 1e-3) are the documented
  early-stop hot-start artifact and are acceptable.

This is inherited algorithmic structure, not a port defect: the R
package documents `npnorm2Dll` as *experimental and possibly very slow*,
and the per-cell L-BFGS-B loop is the documented design. The
per-evaluation gradient temporaries *have* now been folded into scalars
and the invariant re-computations removed (§4a.2); that changed the
floating-point trajectory of the support-point search — which the old
match-to-R contract forbade — and is only acceptable because the
contract was relaxed to *similar result, faster wall time* (the
R-identical path was removed with the 0.3.0 N-D generalization). The
remaining ~1.2× gap (vs R, k = 2) is the per-cell L-BFGS-B evaluation
cost itself (216,955 objective evaluations at ≈ 4.4 µs on the reference
run — 316,253 at ≈ 5.5 µs on the second benchmark), which §4a.1
concludes no faster *and* similar-result solver can reduce.

#### 4a.1 Solver replacement evaluated: CppNumericalSolvers, NLopt, and a wider sweep

The per-cell solver was evaluated against **CppNumericalSolvers**
(`PatWie/CppNumericalSolvers`, GPL/MIT dual-licensed) and **NLopt**
(not present in this environment), and a wider literature sweep of
bound-constrained and mixture-model algorithms was performed. No
rejected candidate was vendored into the package — the tree adds no
external dependency. The default path is *no longer* byte-identical to
the pre-evaluation build: the contract relaxed to *similar result,
faster wall time* (§4a.2 added the fast objective path as the default);
the R-identical trajectory was reproducible via `NPFIC_2D_EXACT=1`
until 0.3.0, when the path was removed (see §4a).

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

#### 4a.2 Objective fast path (the family's objective) and the relaxed contract

With the contract relaxed from *bit-exact match to R* to *faster, with
ll/mixture similar to R's result*, the per-cell objective was re-derived
for k = 2 (its only caller is the full gradient pair; k > 2 reuses the
same code with the general `LLT` kernel):

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
  (`tests/parity_2d.csv`). The fast/exact columns are the 0.2.x A/B
  record (the exact path was removed in 0.3.0); the 0.3.0 build
  reproduces the fast column bit-for-bit:

  | | fast path (0.3.0) | exact path (`NPFIC_2D_EXACT=1`, 0.2.x) |
  |---|---|---|
  | wall (median of 5) | 913 ms | 4295 ms |
  | objective evaluations | 216,955 | 421,810 |
  | per-evaluation | ≈ 4.2 µs | ≈ 10.2 µs |
  | ll / iters / components | 848.7472072261477 / 10 / 5 | 848.7472820349071 / 6 / 5 |

  On the reference run the fast path's gain (≈ 4.5× wall) is the
  product of the per-evaluation kernel cost (0 divisions, no heap
  temporaries, cached invariants — ≈ 2.3×) and the ≈ 1.9× smaller
  per-cell evaluation work (hot start + negative-gain early stop,
  §4a.3, plus the true directional derivative, §4a.4).
* **Second n=300 benchmark** (`itercmp/iterdata_2d.csv`; same
  historical split, fast column reproduced in 0.3.0):

  | | fast path | exact path (0.2.x) |
  |---|---|---|
  | wall | ≈ 1.7 s | ≈ 8.0 s |
  | objective evaluations | 316,253 | 769,533 (2.4×) |
  | per-evaluation | ≈ 5.5 µs | ≈ 10.4 µs |
  | ll / iters / components | 982.7545058419914 / 14 / 7 | 982.7562335069146 / 9 / 7 |

  Here the wall-time gain (≈ 4.6×) is the product of the per-evaluation
  fast kernel (≈ 10.4 → ≈ 5.5 µs, ≈ 1.9×) and the ≈ 2.4× fewer
  evaluations from the hot start and the true directional derivative
  (§4a.3, §4a.4). Per-evaluation figures include the per-cell L-BFGSpp
  machinery and vary ≈ ±10 % between runs.
* **Similarity, measured (both datasets, default grid/init/tol,
  post-fix, true directional derivative):** reference run — 10 iters,
  5 components (0.2.x: exact 6 iters); main components (weight ≥
  0.02) against R's fit within 0.675 in location (0.2.x fast-vs-exact:
  0.0066), |Δll| = 0.14 vs R (0.2.x fast-vs-exact: 7.5e-5);
  iterdata benchmark — 14 iters, 7 components (0.2.x: exact 9);
  0.2.x fast-vs-exact: main components within 0.027, |Δll| = 1.7e-3
  (both paths beat R 1.1.0003's 986.236 on that data). The parity
  gate now checks similarity (two-run determinism, iter count, the ll,
  kernel parity vs R, density similarity vs R — rel-L1 < 0.02,
  max |logdiff| < 0.25 — and main-component proximity < 0.70) plus
  the grid-level certificate `grid_gain > −1` (§4a.3) rather than
  bit-identity.
* **No escape hatch remains.** `NPFIC_2D_EXACT=1` — the 0.2.x
  selector for the original Cholesky-based kernel (the bit-exact
  0.2.1 trajectory, at the old cost) — was removed with the 0.3.0 N-D
  generalization: the relaxed contract no longer requires bit-exact
  trajectories, and the general `LLT` kernel supersedes its Cholesky
  path. The k = 2 fast trajectory itself is bit-identical to 0.2.3, so
  nothing in the package's recorded goldens changed.

#### 4a.3 The per-cell hot start and the `grid_gain` certificate (new since 0.2.3)

The family launches one L-BFGS-B problem **per grid cell** (103 ×
103 = 10,609 cells per sweep for the default 104 × 104 2-D grid;
(G−1)^k per axis for k dimensions), originally started from the cell
midpoint every sweep. This build adds two mechanisms:

* **Warm roots + the `f_stop` early exit.** The previous `solvegrad`
  call's converged point for a cell (keyed by grid position, persistent
  across calls — every cell is visited on every sweep) is reused as
  the next call's start when it is still inside the cell. The objective
  hook `f_stop = 0` makes the *first* evaluation of that point also the
  acceptance test: a warm root that re-verifies still negative is
  accepted at one evaluation (the grid evaluation and the acceptance
  that used to surround it collapse into it), and its warm entry is
  dropped so the next call restarts from the midpoint — the zero-cost
  negative-point acceptance of §4c, generalized to per-cell minima.
  A converged non-negative warm root is a safe hot start (re-verified
  at one evaluation). Measured effect (jointly with the true
  directional derivative, §4a.4): the per-sweep accepted-cell work
  drops 421,810 → 216,955 objective evaluations on the reference run
  (1.9×) and 769,533 → 316,253 on the second benchmark (2.4×).
* **The `grid_gain` certificate caught a real defect.** The 0.2.3
  FAST objective had read the `(n, 2)` data **row-major**
  (`data_[2*i]`, `data_[2*i+1]`) while Eigen stores it **column-major**
  — it optimized the objective on a *corrupted dataset* (consecutive
  x-values paired as `(x, y)` coordinates). The `grid_gain` certificate
  reported by `get_ans` exposed it: ≈ −518 with 3,586 negative cells on
  the reference fit, versus ≈ +4.7e-3 with 0 negative cells once the
  two columns are read correctly (`data_.col(0)` / `data_.col(1)`;
  the pre-true-gradient build gave +7.2e-7).
  The family's similarity band (`dmu < 0.08` in 0.2.x; `dmu < 0.70`
  vs R in 0.3.0) would have *missed* the defect — the corrupted fit's
  main components still landed within
  ≈ 0.06 of the uncorrupted fit's — so `tests/verify_2d_same.py` now
  also fails on `grid_gain < −1`. Post-fix, the independent numpy
  recomputation of the certificate (from the public kernels, all
  (G−1)² cell midpoints) matches the C++ value to < 1e-13 (FAST
  +4.7e-3 on the reference fit; the 0.2.x exact path gave +2.7e-7).
* **A 2-D kernel-column cache was evaluated and rejected.** The 1-D
  engine's column cache (§3) works because its support roots lie on a
  precomputable grid+refinement set; the 2-D cells are 2-D regions and
  L-BFGS-B's line search evaluates arbitrary *interior* points, so a
  per-cell column cache would cover only a small fraction of the
  evaluations. The per-evaluation fast kernel (§4a.2) is the 2-D
  answer instead.

#### 4a.4 The true directional derivative (new in the current build)

The family's per-cell objective is the *gain* of the cell,
`g(μ) = (dens(μ) − K(μ)) / F` with `K` the unnormalised KDE
(`s·Σ_i N(x_i; μ, β)`) and `F = dens + precompute` (constant in μ).
L-BFGS-B needs the directional derivative of `g` along its search
direction. The R package — and the original port of it — instead
passes `B·Σ_i t_i(μ − x_i) = −∇_μ K(μ)`: the gradient of the
unnormalised KDE, not of the gain. That is proportional to `∇_μ g`
only when `1/F` is ~constant over the data; in general L-BFGS-B was
following *KDE directions* of the gain objective. The 1-D `d1`
families do it right — their `a1` IS `dg/dμ` (each point weighted by
`1/F_i`), which is what makes their `brmin`/`dfmin` true derivative
methods.

The `operator()` now accumulates the weighted sum `Σ_i
(t_i/F_i)(μ − x_i)` instead (≈ 3 extra FLOPs per point in pass 3 —
`t_i` and `1/F_i` are already hot from the objective pass; for k > 2
the same formula holds with the general `LLT` kernel, `B` the k×k
weight matrix); the OBJECTIVE arithmetic (`ansd0`) is left
bit-identical, so every value-based decision (`fval < 0`, `f_stop`,
`grid_gain`) is unchanged.

Two effects:

* **The trajectory changes (for the better).** The search now descends
  the true objective: the reference fit moves from ll
  848.7479272517137 (6 iters) to ll 848.7472072261477 (10 iters) —
  *closer* to the exact path (|Δll| vs the exact
  848.7472820349071: 6.4e-4 → 7.5e-5) — and the per-cell evaluation
  work drops with it (216,955 vs the pre-change 352,940 on the
  reference run; 316,253 vs 642,681 on the second benchmark).
* **A stale-curvature guard in `LBFGSB.h`.** With a gradient that no
  longer equals −∇K, the BFGS curvature history can go stale and the
  Cauchy/subspace direction stops descending, which made the line
  search throw ("the moving direction does not decrease the objective
  function value"). The standard L-BFGS-B remedy is applied: when
  `grad·d >= 0` the BFGS history is reset and the steepest-descent
  direction is taken (which descends whenever `grad != 0`, guaranteed
  here because the convergence test returns only for
  `projgnorm <= epsilon <= ||grad||`). The guard is inactive on
  trajectories that never stall — it fires only when the curvature
  history actually goes stale.

#### 4a.5 N-D (k > 2) performance and the tensor-product grid cost (0.3.0)

Profiled on this build (`NPFIXEDCOMPY_PROFILE=1`, `n=360`, 3-Gaussian
generator data `tests/parity_3d.csv`, medians of 3–5):

| k | grid (pts/axis) | cells (L−1)^k | wall | objevals | per-eval |
|---|---|---|---|---|---|
| 2 | 104 (default) | 103² = 10,609 | ≈ 913 ms | 216,955 | ≈ 4.2 µs |
| 3 | 18 (explicit) | 17³ = 4,913 | ≈ 2.05 s | 301,910 | ≈ 6.8 µs |
| 3 | 12 (explicit) | 11³ = 1,331 | ≈ 617 ms | 54,895 | ≈ 11.2 µs |

Findings:

* **The wall time is the per-cell L-BFGS-B evaluation count, as at
  k = 2** — the count is ≈ (G−1)^k × (accepted fraction) and tracks
  the cell count (301,910 / 4,913 ≈ 61.4 per cell vs 216,955 /
  10,609 ≈ 20.5 at k = 2; the k = 3 data is harder — wider, flatter
  gain basins keep more cells non-exit).
* **The per-evaluation cost RISES on coarser grids** (11.2 vs 6.8 µs,
  12 vs 18 pts/axis): a larger cell is a harder box-constrained
  sub-problem — the line search takes more steps and the midpoint
  early-exit fires less. So the wall time is U-shaped in the grid
  resolution: too fine → too many cells; too coarse → too much
  per-cell work. 15–20 points/axis is the sweet spot for the
  n ≈ 360, 3-component cases measured here.
* **The default grid is too large for k ≥ 3**: it pads each axis to
  the longest marginal (≈ 104 points at k = 3), i.e. 103³ ≈ 1.1 M
  cells — the same 2-D cell count per AXIS, cubed. Pass an explicit
  `gridpoints` (the 18 pts/axis run above: 4,913 cells, 2.05 s).
* **The tensor-product overhead is otherwise small**: the grid itself
  is materialized once (L × k doubles), the warm-start map is one
  entry per cell (k doubles, ≤ 4,913 entries at 18³), and the k-D
  `LLT` kernel adds ≈ 2.5× per-point cost over the unrolled k = 2
  fast path (12 flops + 4 divisions → the general Cholesky solve) —
  visible in the 6.8 µs vs 4.2 µs per-evaluation at similar grid
  density. No further micro-optimization of the k > 2 kernel was
  attempted; the §4a.1 conclusion (the L-BFGS-B machinery, not the
  kernel, is the floor) applies, and §4a.2's unrolled kernel could be
  generalized to 3-D later if ever needed.

#### §4a 中文注释

多元正态族（0.3.0 由 R `npnorm2Dll` 泛化为 N 维的 `npnormND`；R 名
保留为 k = 2 别名）是本移植中**文档化的唯一比 R 慢**的族（k = 2）。
四条改动把旧 ~28× 差距收窄到参考运行的 ~1.2×：手写展开 2×2 密度
内核（§4a.2 前段）、显式逆目标函数快路径（§4a.2）、逐格热启动
（§4a.3）、真方向导数（§4a.4）。R 同款目标函数路径
（`NPFIC_2D_EXACT`，0.2.x 下同数据 ≈4.3 s）已**在 0.3.0 移除**：
放宽契约（相似 density、ll 不差于历史版本——从不要求位级一致轨迹）
使其不再必要，N-D 内核取代了其 Cholesky 路径；下文 fast/exact A/B
表是该构建的历史记录。

**§4a.1 求解器替换评估（CppNumericalSolvers、NLopt、更广扫描）。**
被否决的候选均未 vendor 进包（树不加外部依赖）。默认路径自评估后
*不再*与评估前构建字节一致：契约已放宽为"结果相似、wall 更快"
（§4a.2 的快路径成为默认）；R 同款轨迹在 0.3.0 移除该路径前可用
`NPFIC_2D_EXACT=1` 复现。

* **cppoptlib 的 `LbfgsbSolver` 是另一份实现**，不是同一代码的
  构建：收敛判据是硬编码投影梯度范数 ≤ 1e-4（LBFGSpp 用*未投影*
  梯度对调用方 `tol` 判）；More–Thuente 线搜索跑在**无约束**函数上
  （bounds 只在 `findMinimize()` 返回后钳位、事后钳位不重求目标），
  接受的步与 `fval` 报告都不同；历史更新每轮用稠密 `M.inverse()`
  重建而非 rank-1/2 递推。n = 2 时这些成本微小，wall 差几乎全来自
  额外迭代。
* **实测（n=300 2-D 运行，临时 adapter，测后已移除；基线列是
  快路径前 Cholesky 内核构建）**：LBFGSpp ≈ 8.2 s / 769,533 次目标
  求值 vs cppoptlib ≈ 20.1 s / 1,959,715（2.5×）。cppoptlib 最优与
  R 匹配最优 ll 差 ~3.7e-10——另一个（有效的）局部最优，匹配 R
  契约视其为轨迹断裂；单次求值成本其实相当（≈10.2 vs ≈10.7 µs），
  2.4× wall 比就是 2.5× 求值比——收敛策略，不是内核成本。
* **NLopt**：`LD_LBFGS` 是同一篇 LBFGS-B 论文的 C 移植，每求值多
  一次 C 回调跳——同一算法行为加跨语言边界，无收敛优势。
* **更广扫描——没有任何候选既更快又轨迹兼容**：LMBOPT（n = 2 时
  额外 active-set 机构买不到东西，且发表实现是 MATLAB 非可 vendor
  的 C++ 核心）；Box L-MQN / 投影梯度族（与 LBFGS-B 同一渐近类，
  差异在 active-set/box 处理——任何替换都改逐格轨迹与最终最优）；
  Nelder–Mead / 无导数（O(ε⁻²)、无可比 `tol` 语义，目标光滑且有
  解析梯度，丢梯度信息严格更差）；混合模型支撑点搜索替代（方向
  导数筛选、MH 随机候选、支撑缩减 active-set——改的是*选择算法*
  本身，即按构造改 R 轨迹；且在"每接受格更少的 L-BFGS-B 求值"
  意义下也没有更快）；cell 级早退调优（vendored LBFGS-B 本就在
  首步线搜索前、中点投影梯度范数 ≤ tol 时退出——这正是大多数空
  格便宜的原因；8.2 s 由不退出的格主导，需要真正更便宜的求解——
  而按上述段落，保持轨迹的不存在）。
* **`d0`/`d1` 梯度 flag 统一被评估并否决**：每个一维族总请求两个
  分量，`d0`-only 分支只服务 R `npnorm2Dll` 梯度约定（d0 = 概率
  方向标量，d1 = 2 个支撑点方向）；合并 flag 是化妆式重构，改不了
  任何浮点轨迹，单次求值成本由 `dnpnormND` 而非请求子集决定。

**§4a.2 目标函数快路径与放宽契约。** 契约从*与 R 位级一致*放宽到
*更快、ll/混合与 R 结果相似*后，per-cell 目标对 k = 2 重推（k > 2
复用同代码 + 通用 `LLT` 内核）：**显式逆二次型**（`dᵀΣ⁻¹d` 用
`beta` 显式 2×2 逆的系数——构造时预算一次——点积展开，每点 0 次
除法 vs Cholesky 解的 4 次；归一常数保持 R 的精确分组）；**不变量
缓存**（`fullden = 1/(dens+precompute)` 与分量 scale 对整轮
`solvegrad` 是常数，旧代码每次目标求值重算，现缓存在 `setdens`——
每次拟合省 ≈10⁸ 次除法，值位级一致）；**预分配缓冲 + 同一 SIMD
exp**（Eigen `array().exp()` 与匹配 R 的核同一指令序列——标量
`std::exp` 偏 1 ulp 被否决）。实测（参考运行，fast 列即 0.3.0
构建、与 exact 路径位级复现）：wall 913 ms vs 4295 ms、求值
216,955 vs 421,810（≈4.2 vs ≈10.2 µs/次）、ll 848.7472072261477 /
10 轮 / 5 分量 vs 848.7472820349071 / 6 / 5；第二个 n=300 基准同
比例（≈1.7 s vs ≈8.0 s、316,253 vs 769,533）。~4.5× wall 收益 =
单次求值快内核（~2.3×）× 每格求值工作 ≈1.9–2.4× 少（§4a.3 热启动
+ 负增益早停、§4a.4 真方向导数）。**相似性实测**：参考运行 10 轮、
5 分量（0.2.x exact 6 轮），主分量（权重 ≥ 0.02）对 R 位置差 < 0.675
（0.2.x fast-vs-exact 0.0066）、|Δll| = 0.14 vs R；iterdata 基准
14 轮 7 分量、0.2.x fast-vs-exact 主分量差 < 0.027、|Δll| = 1.7e-3
（两路径都优于 R 1.1.0003 的 986.236）。parity 门现查相似性（两次
运行确定性、迭代数、ll、与 R 的核 parity、与 R 的 density 相似
rel-L1 < 0.02 与 max |logdiff| < 0.25、主分量邻近 < 0.70）加网格级
证书 `grid_gain > −1`（§4a.3），而不再是位级一致。**已无逃生舱**：
`NPFIC_2D_EXACT=1` 随 0.3.0 N-D 泛化移除；k = 2 快轨迹本身与 0.2.3
位级一致，包内记录的金标无变化。

**§4a.3 逐格热启动与 `grid_gain` 证书（0.2.3 起新增）。** 该族对
**每个网格单元**启动一个 L-BFGS-B（默认 104×104 2-D 网格即每 sweep
103×103 = 10,609 格；k 维 (G−1)^k），最初每 sweep 都从格中点起步。
本构建加两个机制：**温热根 + `f_stop` 早退**——上一轮 `solvegrad`
该格收敛点（按网格位置键控、跨轮持久——每格每 sweep 都访问）若仍
在格内则作本轮起点；目标 hook `f_stop = 0` 使该点的*首次*求值兼任
接受测试：再验证仍为负的温热根一次求值即接受（网格求值与环绕它的
接受测试坍缩进它），其 warm 条目随后丢弃、下轮从中点重启——§4c
零成本负点接受推广到逐格极小；收敛非负的温热根是安全热启动（一次
求值再验证）。联合 §4a.4 实测：每 sweep 接受格工作从 421,810 降到
216,955 次目标求值（参考运行 1.9×）、769,533 → 316,253（第二基准
2.4×）。**`grid_gain` 证书抓到一个真实缺陷**：0.2.3 FAST 目标把
`(n, 2)` 数据按*行主序*读（`data_[2i]`、`data_[2i+1]`）而 Eigen 存
*列主序*——等于在*被损坏的数据集*上优化（相邻 x 值被配成 (x, y)
坐标）；`get_ans` 报告的 `grid_gain` 暴露它：参考拟合 ≈ −518、
3,586 个负格，而两列读对（`data_.col(0)`/`data_.col(1)`）后 ≈ +4.7e-3、
0 负格（真梯度前构建为 +7.2e-7）。该族相似性 band（0.2.x
`dmu < 0.08`；0.3.0 对 R `dmu < 0.70`）本会*漏掉*该缺陷——损坏拟合
的主分量仍落在未损坏拟合 ≈0.06 内——故 `tests/verify_2d_same.py`
现在也在 `grid_gain < −1` 时失败。修复后独立 numpy 重算证书（公开
内核、全部 (G−1)² 格中点）与 C++ 值差 < 1e-13。**2-D 核列缓存被
评估并否决**：一维列缓存（§3）成立是因为支撑根在可预算的网格+精化
集上；2-D 格是二维区域、L-BFGS-B 线搜索求值任意*内部*点，逐格列
缓存只能覆盖很小比例——答案是单次求值快内核（§4a.2）。

**§4a.4 真方向导数（本构建新增）。** per-cell 目标是格的*gain*
`g(μ) = (dens(μ) − K(μ)) / F`，`K` 为未归一化 KDE、`F = dens +
precompute`（对 μ 常数）。L-BFGS-B 需要 `g` 沿搜索方向的方向导数；
R 包及原移植传的是 `B·Σ t_i(μ − x_i) = −∇_μ K(μ)`——未归一化 KDE
的梯度，不是 gain 的，仅当 `1/F` 近似为常数时与 `∇_μ g` 成比例；
一般情形 L-BFGS-B 追的是 gain 目标的 *KDE 方向*。一维 `d1` 族做对
了——其 `a1` 就是 `dg/dμ`（每点按 `1/F_i` 加权）。现在 `operator()`
改为累加加权和 `Σ (t_i/F_i)(μ − x_i)`（pass 3 每点 ≈3 个额外
FLOP——`t_i` 与 `1/F_i` 在目标 pass 已热；k > 2 同式、`B` 为 k×k
权重矩阵）；目标算术（`ansd0`）保持位级一致，所有值判定
（`fval < 0`、`f_stop`、`grid_gain`）不变。两个效果：**轨迹变好**
——参考拟合 ll 848.7479272517137（6 轮）→ 848.7472072261477
（10 轮），距 exact 路径更近（|Δll| 6.4e-4 → 7.5e-5），逐格求值
工作随之降（参考 352,940 → 216,955；第二基准 642,681 → 316,253）；
**`LBFGSB.h` 的过期曲率守卫**——梯度不再等于 −∇K 后 BFGS 曲率历史
可过期、Cauchy/子空间方向停止下降，使线搜索抛
"the moving direction does not decrease the objective function
value"；应用 L-BFGS-B 标准补救：`grad·d >= 0` 时重置 BFGS 历史并
取最速下降方向（`grad ≠ 0` 时必下降，此处有保证——收敛判据只在
`projgnorm <= epsilon <= ||grad||` 时返回）。守卫对从不卡壳的轨迹
不激活，只在曲率历史真的过期时触发。

**§4a.5 N-D（k > 2）性能与张量积网格成本（0.3.0）。** 本构建 profile
（`n=360`、3 高斯生成器数据 `tests/parity_3d.csv`，3–5 次中位数）：
k=2 默认 104 点/轴（103² = 10,609 格）≈913 ms / 216,955 次求值 /
≈4.2 µs；k=3 显式 18 点/轴（17³ = 4,913 格）≈2.05 s / 301,910 /
≈6.8 µs；k=3 显式 12 点/轴（11³ = 1,331 格）≈617 ms / 54,895 /
≈11.2 µs。结论：**wall 就是逐格 L-BFGS-B 求值数**（计数 ≈
(G−1)^k × 接受比例；k=3 每格 ≈61.4 次 vs k=2 ≈20.5——k=3 数据更
难，更宽更平的 gain basin 让更多格不早退）；**单次求值成本随网格
变粗而升高**（11.2 vs 6.8 µs）——大格是更难 box 约束子问题（线搜索
步多、中点早退少触发），故 wall 对网格分辨率呈 U 形：太细 → 格太
多、太粗 → 每格工作太多，n ≈ 360 / 3 分量情形 15–20 点/轴最甜；
**默认网格对 k ≥ 3 太大**——每轴 pad 到最长 marginal（k=3 ≈104
点）即 103³ ≈ 1.1 M 格，应显式传 `gridpoints`（18 点/轴：4,913 格、
2.05 s）；**张量积开销本身很小**——网格一次物化（L × k 个 double）、
warm-start 映射每格一项、k-D `LLT` 核比展开 k=2 快路径每点贵 ≈2.5×
（12 flops + 4 除法 → 通用 Cholesky 解，见 6.8 vs 4.2 µs）；k > 2
内核未再微优化（§4a.1 结论——L-BFGS-B 机构而非内核是 floor——
适用），§4a.2 的展开内核日后如有需要可推广到 3-D。

### 4b. Support-point hot start (the `NPFIC_WARM` experiment; the knob
was removed in 0.2.3 — see §4c)

The grid is fixed for the solver's life, so every outer iteration
searches the same sign-change intervals (d1) / triples (d0) with a
*slightly* different gradient. `NPFIC_WARM` exploits that: the
PREVIOUS call's refined root for an interval (indexed stably by grid
position) is reused in the NEXT call. Two arms were measured
(`tests/bench_warm_ab.py`, n=5000 per case, `order=-3`, identical
data/initial mix/grid/tol, median of 5):

* **`NPFIC_WARM=1` — seed (quality-preserving).** The previous root is
  passed as the *first interior point* of `brmin`/`dfmin`; the search
  still converges to the true root of the NEW gradient, so the support
  set found is unchanged — only the kernel-column evaluation count
  drops (a seed near the root shrinks the bracket for Brent and
  replaces the first parabolic step of `dfmin`). Measured:

  | case | shipped (0) | seed (1) | speedup | Δll | max rel. |Δdens| |
  |---|---|---|---|---|---|
  | `nptll` β=5, n=5000 | 2957 ms (18 it, 2290 evals, 455 fresh) | 2736 ms (2274, 416) | 1.08× | −1.0e-10 | 4.9e-09 |
  | `nptllw` β=5, n=5000 | 1505 ms (12 it, 1524, 300) | 1407 ms (1515, 274) | 1.07× | +4.6e-09 | 8.4e-08 |
  | `npnormcll` β=1000, n=5000 | 3221 ms (61 it, 20678, 15326) | 2622 ms (59, 18468, 12407) | 1.23× | +5.2e-06 | 9.2e-06 |
  | `npnormll` β=1, n=5000 | 23.3 ms (11 it, 1428, 297) | 20.7 ms (1396, 265) | 1.12× | +7.3e-12 | 1.1e-10 |
  | `npnormad` β=1, n=5000 | 276 ms (18 it, 2395, 201) | 263 ms (2341, 198) | 1.05× | −6.4e-12 | 2.7e-09 |
  | `nppoisll`, n=5000 | 2.0 ms (28 it, 3101, 0) | 1.8 ms (3085, 0) | 1.09× | −1.5e-11 | 1.7e-10 |

  (`evals, freshcols` in parentheses.) The ll agreement is at the
  outer-loop `tol` scale (1e-6) by construction — both arms solve the
  same problem to the same tolerance; on five of the six cases it is
  in fact bit-identical to ~1e-9. Gains track exactly the fresh-column
  work the seed saves: largest where the `d0`/NNLS-heavy cases refine
  many candidates (`npnormcll` 1.23×), negligible where the search is
  already cheap (`npnormad` 1.05×, `nppoisll` — no kernel columns at
  all — 1.09× on iteration count only).
* **`NPFIC_WARM=2` — aggressive (CNM working-set re-verification; A/B
  reference only).** The previous root is re-verified with ONE
  gradient-value evaluation and accepted when still negative,
  skipping the search entirely — the column-generation working-set
  hot-start of the CNM scheme (Wang 2007, *J. R. Statist. Soc. B* —
  full references in section 7). Measured up to
  2.39× (`npnormcll` 3221 → 1346 ms, 61 → 39 iterations) but the
  quality is not preserved where the support roots drift between
  outer iterations: `npnormll` lands on a worse optimum
  (ll +4.16e-2, mixture density off by rel. 4.9e-3) and `npnormad`
  by +1.7e-3, while on `nptllw` it even *lost* time (0.83× — the
  accepted stale roots changed the weight trajectory and added
  iterations). It violates the package's "the estimate's ll must not
  be worse than the historical (shipped) result" acceptance rule, so
  it is kept as an experimental arm, not a default.
* **0.2.3: the knob was removed — the `=2` arm's behaviour is always
  on.** Under the relaxed acceptance contract (density parity within
  tolerance; the estimated `ll` not worse than the historical version;
  a similar — not R-trajectory-matching — optimum), the quality
  objection that kept `=2` experimental (a slightly different local
  optimum where the support roots drift) no longer blocks it. The 0.2.3
  engine makes the whole support-search relaxation (§4c — CNM
  re-verification plus zero-cost negative-grid acceptance plus
  negative-gain early stop) the default, and all ten parity suites
  report `TOTAL BAD: 0` on the re-recorded trajectory. The A/B tables
  above are historical — recorded on the 0.2.2 build where the knob
  existed (`tests/bench_warm_ab.py` still runs, but every arm now
  behaves identically, the environment variable is no longer read).

#### §4b 中文注释

网格在求解器生命周期内固定，所以每轮外层迭代在**同一批**变号区间
（d1）/三元组（d0）上、以*略不同*的梯度搜索。`NPFIC_WARM` 利用
这一点：上一轮调用对某区间的精化根（按网格位置稳定索引）在下轮
复用。两支实测（`tests/bench_warm_ab.py`，每用例 n=5000、
`order=-3`、同数据/初始混合/网格/tol、5 次中位数）：

* **`NPFIC_WARM=1`——seed（保质量）**：上轮根作为 `brmin`/`dfmin`
  的*第一个内部点*；搜索仍收敛到新梯度的真根，找到的支撑集不变
  ——只有核列求值数下降（近根的 seed 缩小 Brent 括号、替换
  `dfmin` 首个抛物步）。实测 1.05–1.23×（`npnormcll` 最大 1.23×
  ——`d0`/NNLS 重的用例精化候选最多；`npnormad` 1.05%、`nppoisll`
  1.09% 最小——搜索本就便宜或根本没有核列）。ll 一致在外层 `tol`
  尺度（1e-6）以内——按构造两臂解到同一容差的同一问题；六个用例
  中五个实际上 ~1e-9 位级一致。
* **`NPFIC_WARM=2`——激进（CNM 工作集再校验；仅作 A/B 参考）**：
  上轮根用**一次**梯度值求值再验证、仍为负即接受、跳过整个搜索
  ——CNM 方案（Wang 2007 *J. R. Statist. Soc. B*——完整引用见 §7）
  的列生成工作集热启动。
  实测最高 2.39×
  （`npnormcll` 3221 → 1346 ms、61 → 39 轮），但支撑根在外轮之间
  漂移时质量不保：`npnormll` 落到较差最优（ll +4.16e-2、density
  相对差 4.9e-3）、`npnormad` +1.7e-3，`nptllw` 甚至*变慢*
  （0.83×——接受的过期根改变权重轨迹、增加迭代）。违反包
  "估计的 ll 不得劣于历史（出货）结果"的验收规则，故保留为实验
  臂、不作默认。
* **0.2.3：旋钮移除——`=2` 臂行为常开。** 放宽验收契约（density
  容忍内一致、ll 不差于历史版本、相似而非匹配 R 轨迹的最优）下，
  当初让 `=2` 留在实验区的质量异议（支撑根漂移处的略不同局部最优）
  不再构成阻碍。0.2.3 引擎把整个支撑点搜索放宽（§4c——CNM 再
  校验 + 零成本负网格接受 + 负增益早停）设为默认，10 个 parity
  套件在重录轨迹上全部 `TOTAL BAD: 0`。上表为历史记录（记录于
  旋钮尚存的 0.2.2 构建；`tests/bench_warm_ab.py` 仍跑，但所有臂
  行为已相同，环境变量不再被读取）。

### 4c. 0.2.3 — the relaxed support search (always on; measured)

The 0.2.2 engine searched for the *exact* minimum of the gain inside
every sign-change interval (d1) / triple (d0). 0.2.3 makes three
relaxations that exploit the fact the search only ever needs to *find a
negative point* — the caller's sign filter accepts any of them:

1. **Zero-cost negative-grid acceptance.** The grid sweep already
   evaluates the gain at every grid point (free, from the cached grid
   kernel). If an interval endpoint — or a triple's middle grid point —
   already has a negative gain, that grid point is a valid new support
   point at **zero additional evaluations**; the interval's warm root is
   kept for a later call. Before 0.2.3 every such interval still ran the
   full refinement.
2. **CNM working-set re-verification (the old `NPFIC_WARM=2` arm).**
   For the remaining intervals, the previous call's refined root is
   re-verified with **one** gradient evaluation and accepted when still
   negative — skipping the refinement entirely.
3. **Negative-gain early stop inside `brmin`/`dfmin`.** The refinements
   now return as soon as *any* evaluated point has negative gain (the
   `d1` check piggybacks on the same kernel column the refinement
   evaluates; the `d0` check is the parabolic point itself).

Each accepted point is still a *valid* new support point (negative
gain ⇒ the outer iteration decreases the loss), and the `Npmix` result
gains a **`grid_gain`** field — the minimum gain over *all* grid points
at the final estimate — so the certificate is exposed, not hidden:
`min_gradient` (support directions) plus `grid_gain` (grid directions)
together certify no negative direction inside the grid; `grid_gain` is
grid-resolution-dependent and can legitimately be slightly negative
(measured −8.2e-4 on `npnormcvm`, −2.8e-3 on `npnormadw`), so the
acceptance gate is `min_gradient ≥ -1e-4` with `grid_gain` reported for
information.

**Wall-time effect (same data / init / grid / tol, median of 5):**

| case | 0.2.2 | 0.2.3 | speedup | iters |
|------|-------|-------|---------|-------|
| `nptll` β=5, n=5000 | 2957 ms (engine 2334) | **857 ms** (engine 266) | **3.5×** | 31 → 11 |
| `nptll` β=∞, n=5000 | 130 ms | 81 ms (engine 21) | 1.6× | 15 → 18 |
| `nptllw` β=∞, n=5000 | 206 ms | 112 ms | 1.8× | 15 → 18 |
| `nptllw` β=5, n=5000 | 2.01 s | **559 ms** | **3.6×** | 12 → 11 |
| `nppoisll`, n=5000 | 2.0 ms | 0.54 ms | 3.7× | — |
| `npnormll`, n=1000 | 10.7 ms | 5.5 ms | 1.9× | — |
| `npnormllw`, n=5000 | 37.1 ms | 27.9 ms | 1.3× | 15 → 18 |
| `npnormadw`, n=5000 | 387 ms | 240 ms | 1.6× | 36 → 17 |
| `npnormcll`, n=1000 | 1009 ms | **1269 ms** | **0.8×** | 19 → 31 |

The gradient-evaluation drop at n=5000, n=1000: `nptll` β=5
3733 → 1191 evals, 455 → 61 fresh columns (§1); `npnormcll`
21043 → 24425 evals (the opposite direction — see below).

**Quality effect (documented, verified in the parity gates):**

* Most families land at a **better or equal** ll: the t-family and
  un-binned normal `ll`s agree with 0.2.2 to ~1e-6 (the outer `tol`
  scale) or better; the binned families agree with R within their
  existing BAND (R is non-deterministic on
  `npnormcvmw`/`npnormadw`). The one documented exception is
  `npnormcll` (its flat correlation landscape has 6e-5-scale basins, so
  basin choice is not identified by the tolerance): 0.2.3's fit is
  4.8e-6 (relative) better than the 0.2.2 fit but 5.5e-5 *worse* than
  R's from-scratch optimum (ll −60.20387 vs −60.20717, 107 vs 92
  support points) — a different valid optimum, gated on
  `min_gradient ≥ -1e-4` (no negative direction) plus the
  recorded-trajectory GOLD (`tests/verify_cvmadcll.py`).
* The trade: the relaxation accepts *any* negative point, so on
  landscapes with flat, nearly-equal basins it can settle on a
  **different valid optimum** than the exact-minimum search —
  `npnormcvm` lands 8.4e-6 above R's observed band (R itself wanders
  1.5e-5 across runs), `npnormcll` visits a 107-point support set (vs
  92) with 31 outer iterations, which is why it is the one family
  where 0.2.3 is *slower* (0.8×). All ten parity suites report
  `TOTAL BAD: 0` on the re-recorded deterministic trajectory.
* `estpi0` (target-statistic bisection) is unaffected in spirit: its
  final mixture is not a free optimum (it pins the point-mass weight to
  hit the threshold), so the KKT certificate does not apply to it; the
  parity gates check its statistic hit, not `min_gradient`.

#### §4c 中文注释

0.2.2 引擎在每个变号区间（d1）/三元组（d0）内搜索 gain 的*精确*
最小值。0.2.3 做三个放宽，利用"搜索只需要*找一个负点*"这一事实
——调用方的符号过滤接受其中任意一个：

1. **零成本负网格接受**：网格扫描本就已（由缓存网格核免费地）求出
   每个网格点的 gain；若区间端点——或三元组的中网格点——增益已经
   为负，该网格点就是零额外求值的有效新支撑点；该区间的温热根
   保留给后续调用。0.2.3 之前每个这样的区间仍会跑完整精化。
2. **CNM 工作集再校验（旧 `NPFIC_WARM=2` 臂）**：对剩余区间，上一
   轮的精化根用**一次**梯度求值再验证、仍为负即接受——完全跳过
   精化。
3. **`brmin`/`dfmin` 内负增益早停**：精化在*任一*求值点增益为负时
   立即返回（`d1` 检查搭同一核列的便车；`d0` 检查就是抛物点本身）。

每个被接受点仍是*有效*新支撑点（负增益 ⟹ 外层迭代降损失）；`Npmix`
结果新增 **`grid_gain`** 字段——最终估计上*全部*网格点的最小增益，
证书被暴露而非隐藏：`min_gradient`（支撑方向）+ `grid_gain`（网格
方向）共同证明网格内无负方向；`grid_gain` 依赖网格分辨率、可合法
略负（实测 `npnormcvm` −8.2e-4、`npnormadw` −2.8e-3），故验收门是
`min_gradient ≥ -1e-4`，`grid_gain` 供参考。

**wall 效果**（同数据/初始化/网格/tol，5 次中位数）：`nptll` β=5
n=5000 2957 → **857 ms**（3.5×，31 → 11 轮）；`nptllw` β=5
2.01 s → **559 ms**（3.6×）；`nppoisll` 3.7×；`npnormll` n=1000
1.9×；`npnormllw` 1.3×；`npnormadw` 1.6×（36 → 17 轮）；
**`npnormcll` 0.8×（1009 → 1269 ms，19 → 31 轮）——唯一变慢的
族**。n=5000/1000 的梯度求值变化：`nptll` β=5 3733 → 1191 次、
455 → 61 fresh 列（§1）；`npnormcll` 21043 → 24425（反向——见下）。

**质量效果（已记录、parity 门已验证）**：

* 多数族落在**更好或相等**的 ll：t 族与未分箱正态的 ll 与 0.2.2
  差 ~1e-6（外层 `tol` 尺度）或更好；分箱族与 R 在既有 BAND 内一致
  （R 在 `npnormcvmw`/`npnormadw` 上本身不确定）。唯一记录例外是
  `npnormcll`（平坦相关景观有 6e-5 尺度的 basin，basin 选择不被
  容差标识）：0.2.3 拟合比 0.2.2 拟合好 4.8e-6（相对），但比 R 的
  从零最优差 5.5e-5（ll −60.20387 vs −60.20717、107 vs 92 支撑
  点）——另一个有效最优，由 `min_gradient ≥ -1e-4`（无负方向）+
  记录轨迹 GOLD（`tests/verify_cvmadcll.py`）把关。
* 代价：放宽接受*任意*负点，所以在平坦、近相等的 basin 景观上可能
  停在比精确最小搜索**不同的有效最优**——`npnormcvm` 落在 R 观察
  band 上方 8.4e-6（R 自身跨运行漂 1.5e-5）、`npnormcll` 访问
  107 点支撑集（vs 92）用 31 轮，因此它是 0.2.3 *变慢*的那一族。
  10 个 parity 套件在重录的确定性轨迹上 `TOTAL BAD: 0`。
* `estpi0`（目标统计量二分）精神上不受影响：其最终混合不是自由
  最优（把点质量权重钉住以命中阈值），KKT 证书不适用；parity 门
  查它的统计量命中而非 `min_gradient`。

### 4d. 1-D second-order crossing certificate (`NPFIC_1D_CERT`, 0.3.0)

The d1 sign-change loop (section 4c, step 2) still paid a CNM
re-verification evaluation — and, when that failed, a `brmin`
refinement — for intervals whose *endpoint gains are both
non-negative*: there, a valid root can exist only if the gain dips
back below zero inside the interval. Such an interval can be
excluded at zero cost when the family provides a curvature bound.

The grid sweep already gives, at both endpoints, the gain `pv` and
the gain derivative `pg` (= the d1 `a1`), and `NpNormLL` supplies
`M ≥ −inf a0″` (all `mu`): with `a0″ = −(1−pi0fixed) β⁻³
Σᵢ flᵢ (zᵢ²−1) φ(zᵢ)`, `zᵢ = (xᵢ−μ)/β`, `flᵢ = 1/(densᵢ+preᵢ)`, and
the pointwise bound `(z²−1) φ(z) ≤ 2e^{−3/2}/√(2π)` (maximum at
`z = √3`),

    M = (1−pi0fixed) · 2e^{−3/2}/√(2π) · Σᵢ flᵢ / β³.

`Σ fl` is one extra accumulator in the per-solve `fl` pass (existing
O(n); the `fl` values and the `dens_dot_fl` accumulation order are
unchanged, so cached results stay bit-identical). The tangent
parabolas at the two endpoints bound `a0` from below over the whole
interval, and that bound being concave puts its interval minimum at
an endpoint, so

    min a0 ≥ min( pv(a), pv(b),
                  pv(a) + pg(a)·h − ½M·h²,
                  pv(b) − pg(b)·h − ½M·h² ),   h = b − a.

Positive ⇒ `a0 > 0` on the whole interval ⇒ neither the CNM check
nor `brmin` can find a valid root ⇒ both are skipped (one
gradient evaluation plus the refinement, per firing interval). The
warm root is KEPT — the certificate is state-dependent (it uses the
current `dens`), so a later iteration may make the old root
verifiable again. The gate is on by default; `NPFIC_1D_CERT=0`
restores the pre-0.3.0 loop exactly; only `NpNormLL` implements
`gain_curv_bound` (default −1 = never fires), so the t/Poisson/CLL/
AD/binned/N-D families are untouched. Soundness: a standalone
numerical harness re-derives the gain and its derivatives for two
`npnormll` states (a near self-consistent one-component fit and a
three-component weight-transfer scan over 20 states), high-
resolution re-sweeps every interval, and checks the four-tangent
bound against the true interval minimum — **zero violations on all
99 scanned intervals, zero unsound skips** (the single candidate
interval that reaches the CNM stage is certified positive and
indeed safe). Engine-level: gate on vs off returns
**bit-identical** ll/pt/pr/iter/min_gradient/grid_gain on the
`npnormll` suite (computemixdist / n=5000 / fixed component /
estpi0) — the gate only skips provably-positive intervals — and
the full `run_all_verify.py` (11 suites) passes `TOTAL BAD: 0`.

Measured: on the current benchmark data the certificate **never
fires** (`certskips=0` on all four npnormll benchmark runs — the
`½M h²` term ≈ 16 dominates the endpoint gains on those
datasets), so A/B wall is neutral (n=5000 median 22.1 ms off vs
22.4 ms on over 30 reps, noise level) while the accepted-point set
and all trajectories are bit-identical. The bound is deliberately
state-independent and uses the pointwise maximum of
`(z²−1)φ(z)`, so it is loose exactly where the kernel is flat —
it fires on data where endpoint gains are large relative to the
grid spacing (e.g. wide-support fits, coarse grids). Kept on by
default: per-candidate cost is a few nanoseconds when idle, and it
is a zero-cost certificate for future families to override
`gain_curv_bound` with (the d0 path has no `a1` analogue and
deliberately does not use it).

#### §4d 中文注释

d1 变号区间循环（§4c 第 2 步）在**两端增益均非负**的区间上仍会花
一次 CNM 再校验求值——若仍为负还要跑 `brmin` 精化。这类区间存在
有效根的充要条件是 gain 在区间内部下穿 0；若族提供曲率界
`M ≥ −inf a0″`（对所有 `mu`），整个区间可以在**零成本**下排除。

网格扫描已免费给出两端点的 gain `pv` 与 gain 导数 `pg`（= d1 的
`a1`）；正态极大似然族的界来自
`a0″ = −(1−pi0fixed) β⁻³ Σᵢ flᵢ (zᵢ²−1) φ(zᵢ)`（`zᵢ = (xᵢ−μ)/β`，
`flᵢ = 1/(densᵢ+preᵢ)`）加上逐点界 `(z²−1)φ(z) ≤
2e^{−3/2}/√(2π)`（最大值在 `z = √3` 取到）：

    M = (1−pi0fixed) · 2e^{−3/2}/√(2π) · Σᵢ flᵢ / β³。

`Σ fl` 是每次求解的 `fl` 通道里多出的一个累加器（既有 O(n)；`fl`
取值与 `dens_dot_fl` 累加顺序不变，缓存结果保持位级一致）。两端点
的切线抛物线把整个区间的 gain 从下方钉住，而该下界是凹函数、区间
最小必在端点：

    min a0 ≥ min( pv(a), pv(b),
                  pv(a) + pg(a)·h − ½M·h²,
                  pv(b) − pg(b)·h − ½M·h² )，h = b − a。

下界 > 0 ⟹ 整区间 `a0 > 0` ⟹ CNM 校验与 `brmin` 都找不到有效根
⟹ 两者整体跳过（每次触发省一次梯度求值加一次精化）。温热根**保留**
——证书是状态相关的（用的是当前 `dens`），后续迭代可能让旧根重新
可验证。默认常开；`NPFIC_1D_CERT=0` 精确回到 0.3.0 之前的循环；
只有 `NpNormLL` 覆写 `gain_curv_bound`（默认 −1 = 永不触发），
t/Poisson/CLL/AD/分箱/N-D 族不受影响。soundness：独立数值 harness
对两个 npnormll 状态（近自洽的单分量拟合、三分量权重转移扫描
20 个状态）重推导 gain 及其导数、对每个区间高分辨率重扫、把
四端点下界与区间真最小值比对——**99 个扫描区间零违例、零误跳**
（唯一到达 CNM 阶段的候选区间被证书判正且确实安全）。引擎级：
开关证书跑 npnormll 套件（computemixdist / n=5000 / 固定分量 /
estpi0），ll/pt/pr/iter/min_gradient/grid_gain **位级一致**（门
只跳过可证明为正的区间）；`run_all_verify.py` 全套 11 个套件
`TOTAL BAD: 0`。

实测：在当前基准数据上证书**从不触发**（4 个 npnormll 基准运行
`certskips=0`——`½M h² ≈ 16` 项压过端点增益），故 A/B wall 中性
（n=5000 中位数 关 22.1 ms vs 开 22.4 ms，30 次重复，噪声级），
接受点集与全部轨迹位级一致。界是刻意状态无关的、用了 `(z²−1)φ(z)` 的逐点
最大值，所以在核平坦处偏松——它在"端点增益相对网格间距大"的数据
上会触发（宽支撑拟合、粗网格）。保持默认常开：空闲时每候选点仅
几纳秒；对后续族是零成本钩子（覆写 `gain_curv_bound` 即可）；d0
路径无 `a1` 对应量，刻意不使用。

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
  measurable. (The table was recorded on the 0.2.2 exact-minimum
  search; under the 0.2.3 relaxed search (§4c) the refinement is
  usually skipped or cut short, so the cap rarely binds — the knob
  remains, but its effect is now secondary to the relaxation.)
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

#### §5 中文注释

* **限制每候选精化步数（`NPFIC_REFINE_STEPS`）——已 A/B 实测，保留为
  实验旋钮，默认不变。** 假设：给 `brmin`/`dfmin` 内每候选的精化
  步数设上限（"每步更少工作"）省下的 wall 多于额外外轮的成本。
  n=30000、同数据/初始点/网格/tol 的 A/B（`tests/refine_ab.py`；
  求值数经 `tests/refine_ab_evals.py` 的 PROFILE 行，每臂独立子
  进程、3 次计时取中位数）：2 轮的 `nptll` β=3 在位级一致 ll 下只
  得 ~1–7%（精化相对 3 万点 `dnt` 列本就便宜）；`npnormll` β=1 展
  示另一面——cap=2 要 60 轮（2.2× wall）、cap=1 漂出轨迹（不同 k
  与 ll）。没有一臂统一胜过出货行为；默认保持 `-1`（不限制）；
  旋钮保留供后续实验，且 `NPFIXEDCOMPY_PROFILE` 行现报告
  `solvegrad` 求值数（`evals=`）使此类实验可测。表记录于 0.2.2
  精确最小搜索；在 0.2.3 放宽搜索（§4c）下精化通常被跳过或截
  断，cap 很少绑定——旋钮仍在，但效果已次于放宽本身。
* **手写 OpenMP**——全部移除（见 §2）；它是崩溃/确定性隐患，收益
  已由缓存位级安全地重新获得。
* **把剩余工作线程化**——剩余两个热点（`solvegrad` 中离网格 `dnt`
  求值、`weights` 中 NNLS）对 `n` 是平凡可并行的，但按设计约束保持
  串行（无手写 OpenMP；只有 Eigen 编译期门控的并行 GEMM/GEMV
  参与）。对 R 的实测加速本就来自 SIMD + 缓存。
* **几何级数填充捷径（否决）**——把 `D(x; μ−kδ)` 填成
  `D(x; μ)·rᵏ` 在分箱梯形上快 ~2.3×，但代数上是错的：`log D` 对
  `k` 是二次的，比值 `D_{k+1}/D_k` 不是常数。parity 门抓住了它
  产生的系统性 1.2e-4 `ll` 偏移；该分支被*删除*而非降级——一个
  有合理 API 的位级一致隐患比一个错失的优化更糟。

## 6. Reproducing the evidence

```bat
cd npfixedcomppy
.venv\Scripts\python.exe tests\perf_baseline.py        :: wall-time table (un-binned)
.venv\Scripts\python.exe tests\bench_binned.py         :: wall-time table (binned)
.venv\Scripts\python.exe tests\reprofile_phases.py     :: re-measure the section-1 phase table
.venv\Scripts\python.exe tests\refine_ab.py            :: NPFIC_REFINE_STEPS A/B (wall/iters/ll)
.venv\Scripts\python.exe tests\refine_ab_evals.py      :: same A/B, evals from the PROFILE line
.venv\Scripts\python.exe tests\bench_warm_ab.py        :: historical NPFIC_WARM A/B (§4b; 0.2.3: all arms identical)
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
`r3_pts_2d.csv` / `r3_pr_2d.txt`, recorded by `C:\...\rebuild\verify2d.R`);
`tests/verify_3d.py` carries the 3-D `npnormND` gold set
(`parity_3d.csv`, the explicit 18³ grid) — structural +
self-consistency, since there is no R reference for k = 3.

2-D wall times: `.venv\Scripts\python.exe bench_2d.py` (Py) and
`Rscript bench_2d_r.R` (R), both reading the shared `parity_2d.csv`.
The §4a.2 fast-path median comes from `tests/measure_2d.py`;
`NPFIXEDCOMPY_PROFILE=1` on the run prints the `objevals` line.

#### §6 中文注释

复现入口（脚本均见上代码块）：`tests/perf_baseline.py`（未分箱
wall 表）、`tests/bench_binned.py`（分箱 wall 表）、
`tests/reprofile_phases.py`（重测 §1 分阶段表）、
`tests/refine_ab.py` / `tests/refine_ab_evals.py`（
`NPFIC_REFINE_STEPS` A/B：wall/迭代/ll 与 PROFILE 行的 evals）、
`tests/bench_warm_ab.py`（历史 `NPFIC_WARM` A/B，§4b；0.2.3 起各臂
相同）；`NPFIXEDCOMPY_PROFILE=1` 运行向 stderr 打分阶段计时行
（含 `evals=`）。R 参考时间：`tests/bench_r.R` /
`bench_binned_r.R`（Rscript）。

parity 门（全部须 `TOTAL BAD: 0`，代码块见上）：
`tests/verify_kernels.py` 携带 30 例 1-D/2-D 核金标集
（`ref_new.txt` + R 导出输入，对照活体 R 4.6.1 记录）；
`tests/verify_2d_same.py` 携带 `npnorm2Dll` 金标集（`parity_2d.csv`、
`grid_2d.csv`、初始混合、R 最终点 `r3_pts_2d.csv` / `r3_pr_2d.txt`，
由 `C:\...\rebuild\verify2d.R` 记录）；`tests/verify_3d.py` 携带 3-D
`npnormND` 金标集（`parity_3d.csv`、显式 18³ 网格）——k = 3 无 R
参考，故走结构 + 自洽门。2-D wall 时间：
`.venv\Scripts\python.exe bench_2d.py`（Py）与
`Rscript bench_2d_r.R`（R），共读 `parity_2d.csv`；§4a.2 快路径
中位数来自 `tests/measure_2d.py`，运行加 `NPFIXEDCOMPY_PROFILE=1`
打印 `objevals` 行。

## 7. References

The algorithm references for this package:

1. **Wang (2007)** — the support-point (column-generation) search
   for the non-parametric MLE of a mixing distribution that the
   engine implements (the CNM scheme; §4b/§4c re-verify its
   working set with one gradient evaluation per candidate):
   Y. Wang, "On Fast Computation of the Non-Parametric Maximum
   Likelihood Estimate of a Mixing Distribution", *Journal of the
   Royal Statistical Society, Series B (Statistical Methodology)*
   69(2), 185–198, 2007. DOI: 10.1111/j.1467-9868.2007.00583.x.
2. **Xue & Wang (2023)** — the null-proportion / density
   simultaneous estimation that `estpi0` implements:
   X. Xue, Y. Wang, "A Nonparametric Mixture Approach to Density
   and Null Proportion Estimation in Large-Scale Multiple
   Comparison Problems", *Australasian Journal of Statistics*
   65(1), 49–75, 2023. DOI: 10.1111/anzs.12383.

The fixed-component (`mu0`/`pi0`) mechanism is the same extension
the R package `npfixedcomp2` builds on top of Wang (2007); an
earlier draft of this list cited a Wang & Taylor (2013)
fixed-component paper for it, but that item could not be verified
against Crossref and has been withdrawn.

#### §7 中文注释

本包的算法文献（完整出处，代码内短引用指向本节）：

1. **Wang (2007)** —— 本引擎实现的非参数混合分布极大似然估计的
   支撑点（列生成）搜索（CNM 方案；§4b/§4c 以每候选点一次梯度
   求值再校验其工作集）：
   Y. Wang, "On Fast Computation of the Non-Parametric Maximum
   Likelihood Estimate of a Mixing Distribution", *Journal of the
   Royal Statistical Society, Series B (Statistical Methodology)*
   69(2), 185–198, 2007. DOI: 10.1111/j.1467-9868.2007.00583.x。
2. **Xue & Wang (2023)** —— 空假设比例与密度同时估计，
   `estpi0` 实现的机制：X. Xue, Y. Wang, "A Nonparametric Mixture
   Approach to Density and Null Proportion Estimation in
   Large-Scale Multiple Comparison Problems", *Australasian Journal
   of Statistics* 65(1), 49–75, 2023. DOI: 10.1111/anzs.12383。

固定分量（`mu0`/`pi0`）机制即 R 包 `npfixedcomp2` 在 Wang (2007)
之上的同一扩展；早期稿曾引一篇 Wang & Taylor (2013) 的固定分量
论文作其出处，经 Crossref 多路核实无法确认该文存在，已撤除。
