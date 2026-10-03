# npfixedcomppy

A Python package (C++/Eigen core) for **non-parametric estimation of
mixing distributions** in several parametric families — a reimplementation
of the R package [`npfixedcomp2`](https://CRAN). Algorithms follow
Wang (2007) and the extensions added in `npfixedcomp2`.

## 1. What this package does

Two entry points (R-compatible names and semantics):

* **`computemixdist(v, method, ...)`** — estimate a discrete mixing
  distribution `G = sum_j pi_j Delta_{mu_j}` for the kernel indexed by
  `method`, optionally with **fixed components** (support points/weights
  that are included in the mixture but never updated).
* **`estpi0(v, method, ...)`** — estimate the mixing distribution
  **together with the proportion of point mass at zero**.

Supported families (`method` argument):

| `method`     | family       | loss function                | structural parameter `beta`            |
|--------------|--------------|------------------------------|----------------------------------------|
| `npnormll`   | normal       | maximum likelihood           | scale (default `1`)                    |
| `npnormcvm`  | normal       | Cramér–von Mises distance    | scale (default `1`)                    |
| `npnormad`   | normal       | Anderson–Darling distance    | scale (default `1`)                    |
| `nptll`      | t            | maximum likelihood           | degrees of freedom (default `inf`)     |
| `npnormcll`  | 1-param normal | maximum likelihood         | number of observations (user supplied) |
| `nppoisll`   | poisson      | maximum likelihood           | none (unused)                          |
| `npnormllw`  | normal (binned)  | maximum likelihood         | scale (default `1`)                    |
| `npnormcvmw` | normal (binned)  | Cramér–von Mises distance  | scale (default `1`)                    |
| `npnormadw`  | normal (binned)  | Anderson–Darling distance  | scale (default `1`)                    |
| `nptllw`     | t (binned)       | maximum likelihood         | degrees of freedom (default `inf`)     |
| `npnorm2Dll` | bivariate normal | maximum likelihood (experimental) | covariance matrix (default `I₂`) |

The binned ("`...w`") variants pre-bin the observations onto the grid
`h = 10^order` (default `order = -3`, i.e. `h = 10^-3`; round-down, as in
R's `bin`) and fit the binned kernel; they are intended for large samples
and take `order` as an extra argument (ignored by the un-binned families).

`npnorm2Dll` takes an `(n, 2)` data matrix and fits a bivariate normal
mixture (the structural parameter `beta` is the `2 x 2` covariance matrix,
default the identity). It runs its own L-BFGS-B support-point search (the
L-BFGS-B of yixuan/LBFGSpp, as in the R package) instead of the 1-D
engine; the R package documents it as *experimental and possibly very
slow*. On this build it is still the slowest family relative to R
(n=300: ≈ 2.7 s vs ≈ 0.8 s, same data — the residual gap is the per-cell
L-BFGS-B search machinery, after a hand-unrolled 2×2 density kernel *and*
a 0-division explicit-inverse objective fast path cut the old ~28× gap to
~3.5×) — see [`docs/PERF.md`](docs/PERF.md). The per-cell objective has
two paths: a fast path (default, ≈ 1.6× faster per evaluation on the
reference run and ≈ 2.8× end-to-end on the second n=300 benchmark,
results *similar* to R — main components within ≈ 0.06, ll within
≈ 0.01) and the R-identical bit-exact path, selectable with
`NPFIC_2D_EXACT=1`; the kernels themselves remain bit-exact with R, and
both paths land on a valid (in the reference run, actually slightly
better) local optimum.

**Compute architecture:** `Python → C++/Eigen` — a single pybind11
extension, `npfixedcomppy._core`. The whole compute stack (engine,
support-point solvers, constrained NNLS, gradient and weight sweeps,
kernel-matrix fills, mixture density) runs in C++/Eigen, with
**Eigen 5.0.0** vendored header-only under `eigen/`. The **SIMD level is
decided at build time** (`/arch:AVX2` by default; override with the
`NPFIC_ARCH` env var — see §3). There are no hand-written OpenMP loops:
`setup.py` probe-compiles the compiler with its OpenMP flag at build time
and, when supported, adds the flag to the extension build so Eigen's own
compile-time parallel GEMM/GEMV is enabled (serial Eigen otherwise); for a
fixed build and thread count, identical inputs therefore give
bit-identical outputs (unlike the R package on parallel platforms).
Results are designed to match the R package to working precision (ll to
relative error ~1e-9, support points ~1e-6); the parity gates in §5
enforce this. Details and measured evidence:
[`docs/PERF.md`](docs/PERF.md).

### 本包的作用

`npfixedcomppy` 是一个 Python 包（C++/Eigen 内核），用于若干参数族的
**非参数混合分布估计**——R 包 [`npfixedcomp2`](https://CRAN) 的重写版本。
算法遵循 Wang (2007) 及 `npfixedcomp2` 的扩展。

两个入口（与 R 包同名、同语义）：

- **`computemixdist(v, method, ...)`** — 估计离散混合分布
  `G = sum_j pi_j Delta_{mu_j}`，可选带**固定分量**（参与混合但从不更新
  的支撑点/权重）。
- **`estpi0(v, method, ...)`** — 同时估计混合分布与**零处点质量比例**。

支持的族（`method` 参数）：

| `method`     | 族             | 损失函数                     | 结构参数 `beta`                |
|--------------|----------------|------------------------------|--------------------------------|
| `npnormll`   | 正态           | 极大似然                     | 尺度（默认 `1`）               |
| `npnormcvm`  | 正态           | Cramér–von Mises 距离        | 尺度（默认 `1`）               |
| `npnormad`   | 正态           | Anderson–Darling 距离        | 尺度（默认 `1`）               |
| `nptll`      | t              | 极大似然                     | 自由度（默认 `inf`）           |
| `npnormcll`  | 一参数正态     | 极大似然                     | 样本量（须用户提供）           |
| `nppoisll`   | 泊松           | 极大似然（计数数据）         | 无（不使用）                   |
| `npnormllw`  | 正态（分箱）   | 极大似然                     | 尺度（默认 `1`）               |
| `npnormcvmw` | 正态（分箱）   | Cramér–von Mises 距离        | 尺度（默认 `1`）               |
| `npnormadw`  | 正态（分箱）   | Anderson–Darling 距离        | 尺度（默认 `1`）               |
| `nptllw`     | t（分箱）      | 极大似然                     | 自由度（默认 `inf`）           |
| `npnorm2Dll` | 双变量正态     | 极大似然（实验性）           | 协方差矩阵（默认 `I₂`）        |

分箱（``"...w"``）变体先将观测值合并到网格 `h = 10^order`（默认
`order = -3`，即 `h = 10^-3`；向下取整，与 R 的 `bin` 一致），再拟合分
箱核；面向大样本场景，`order` 为额外参数（非分箱族忽略）。

`npnorm2Dll` 接收 `(n, 2)` 数据矩阵，拟合双变量正态混合分布（结构参数
`beta` 为 `2 x 2` 协方差矩阵，默认单位阵）。它走自带的 L-BFGS-B 支撑点
搜索（与 R 包相同，即 yixuan/LBFGSpp 的 L-BFGS-B），不复用一维引擎；
R 包将其标注为*实验性、可能很慢*。本构建下它仍是相对 R 最慢的族
（n=300：≈ 2.7 s vs ≈ 0.8 s，同一数据——在引入手写展开的 2×2 密度内核
与 0 除法显式逆目标函数快路径后，旧 ~28× 差距已收窄到 ~3.5×，残余差距
在逐格 L-BFGS-B 求解器机制），见
[`docs/PERF.md`](docs/PERF.md)。逐格目标函数有两条路径：快路径（默认，
参考运行上单次求值约 1.6× 更快、第二个 n=300 基准上端到端约 2.8× 更快，
结果与 R **类似**——主成分位置差 ≈ 0.06 内、ll 差 ≈ 0.01 内）与 R 同款
逐位一致路径（`NPFIC_2D_EXACT=1` 选用）；密度内核本身与 R 逐位一致，
两条路径都落在有效（参考运行下甚至略优）的局部最优。

**计算架构：** `Python → C++/Eigen`——单一 pybind11 扩展
`npfixedcomppy._core`。整个计算栈（引擎、支撑点求解、约束非负最小二乘、
梯度与权重扫描、核矩阵填充、混合密度）全部在 C++/Eigen 中完成，其中
**Eigen 5.0.0** 以 header-only 形式 vendored 在 `eigen/` 下。**SIMD 级别
在构建期决定**（默认 `/arch:AVX2`，可用环境变量 `NPFIC_ARCH` 覆盖——见
§3）。无手写 OpenMP：`setup.py` 在构建期用编译器的 OpenMP 参数做一次
探测编译，支持时把该参数加入扩展构建，从而启用 Eigen 自身的编译期并行
GEMM/GEMV（不支持则为串行 Eigen）；对固定构建与线程数，相同输入恒产生
逐位相同的输出（并行平台上的 R 包做不到）。结果设计为与 R 包达到工作
精度一致（ll 相对误差 ~1e-9、支撑点 ~1e-6），由 §5 的 parity 门保证。
细节与实测证据见 [`docs/PERF.md`](docs/PERF.md)。

## 2. What's new in 0.2.1 (vs 0.2.0)

**New: the 2-D family `npnorm2Dll`** — bivariate normal mixture via
`computemixdist(v, method="npnorm2Dll", beta=...)`, porting R's
`computemixdist.npnorm2Dll` (including its experimental L-BFGS-B
support-point search, the `(n, 2)` data convention, the covariance-matrix
`beta`, and the fixed-component arguments `mu0`/`pi0`):

* `v` is an `(n, 2)` array (a flat length-`2n` vector is reshaped);
* `beta` is the `2 x 2` covariance matrix (default the identity);
* `mix` / `gridpoints` accept the 2-D forms `{"pt": (k, 2), "pr": (k,)}`;
  when omitted, the defaults are built exactly as R's wrapper builds them
  (per-marginal `initial.npnorm` / `gridpoints.npnorm`, tensor-product
  combination);
* the result is an `Npmix` whose `pt` is a list of `(x, y)` pairs and whose
  `beta` holds the covariance matrix.

**New: the 1-D mixture density/cdf kernels (R's `dnp*` / `pnp*` layer)** —
exported at the top level, all backed by the C++/Eigen core:

* `dnpnorm`, `pnpnorm`, `dnpnormc`, `dnpt`, `pnpt`, `dnpdiscnorm`,
  `pnpdiscnorm`, `dnppois`, `pnppois`, `dnpdisct` (1-D families, both
  `lg` log-space forms);
* `dnpnormND` (2-D mixture density) and `dnormNDarray` (the `(n, k)`
  multivariate-normal pdf kernel, log form available).

Every one of these reproduces the R package's **exact** code paths
(including R's quirks, e.g. the `dnppois` log-path normal-term
copy-paste and the `dnpnormc` NaNs at point-mass support points) to
relative error ≤ 1.2e-15 on the 30-case parity set; the log-space `pgamma`
helper was rebuilt on top of the shared regularized-gamma core to match
R's log branch bit-for-bit (a log-space Lentz CF diverged by up to ~1e-4
in the far tail, which the `pnppois` `lg=TRUE` gate caught).

**Eigen version pinned in the docs** — the vendored Eigen under `eigen/`
is **5.0.0** (`EIGEN_WORLD_VERSION 3`, the semver start point); this is
now stated explicitly in §1, §2, and `docs/PERF.md`. Build, flags, and
parity gates are unchanged.

**Kernel bug fixes verified against live R 4.6.1** (the 0.2.x parity files
predate these): `pnorm` far-tail `lg=TRUE` routing, `ppois` `±Inf`
semantics, `pgamma` pole/sign handling, `gammln` pole/sign, the
`logspacesub` branch, Poisson `x < 0`, `pnpt` non-log quirk, and the
`dnppois` log-space behavior. All ten test suites in §5 report
`TOTAL BAD: 0` on the rebuilt extension.

**`npnorm2Dll` objective fast path (new default) + solver evaluation** —
with the contract relaxed from *bit-exact match to R* to *faster, with
ll/mixture similar to R's result*, the per-cell L-BFGS-B objective was
re-derived for n = 2:

* **Fast path (default)**: the quadratic form uses the explicit 2×2
  inverse of `beta` (0 divisions per point vs 4 for the Cholesky-solve
  kernel), the per-evaluation invariants (`1/(dens+precompute)`, the
  component scale) are cached once per `solvegrad` instead of on every
  objective evaluation (≈ 10⁸ divisions per fit), and the two
  `n`-length temporaries are preallocated buffers — while the
  exponentiation stays the same Eigen SIMD `exp` as the R-matching
  kernel. Reference run (n=300): ≈ 10.1 → ≈ 6.3 µs per objective
  evaluation (same 421,810 evaluations, same accepted cell set),
  4.3 s → 2.7 s (≈ 5.7× → ≈ 3.5× vs R). On the second n=300 benchmark
  the ~1e-13 numerical difference also changes the accepted-cell
  trajectory, and the evaluation count itself drops 769,533 →
  251,704 (wall 7.0 s → 2.5 s).
* **Similarity, not identity**: on the reference data both paths
  converge in 6 iters to a 5-component mixture whose main components
  agree within ≈ 0.06 in location (ll 848.761 fast vs 848.747 exact
  vs 848.888 R); on the second benchmark the fast path drops one
  w≈0.02 minor component (k 7 → 6, ll 982.756 → 983.138, both valid,
  no worse than R's 1.1.0003 result of 986.236). The parity gate
  (`tests/verify_2d_same.py`) now checks *similarity*: determinism per
  path, iter count, ll bands, kernel parity vs R (still bit-exact), and
  main-component proximity between the two paths.
* **R-identical escape hatch**: `NPFIC_2D_EXACT=1` selects the
  original Cholesky-based kernel — the bit-exact 0.2.1 trajectory (ll
  848.7472820349071 on the reference run).
* **Alternative solvers were also evaluated** (rejected ones are **not**
  vendored; nothing was added to the tree): CppNumericalSolvers'
  `LbfgsbSolver` (GPL/MIT dual) measured live via a temporary adapter
  and removed — a different implementation (hard-coded 1e-4
  projected-gradient test, unbounded More–Thuente step clamped to the
  box afterwards, dense `M.inverse()` history update) that converged
  **later** (20.1 s vs 8.2 s, 1.96 M vs 0.77 M objective evaluations)
  to a different (valid) optimum; NLopt's `LD_LBFGS` is a C port of the
  same LBFGS-B with a per-evaluation C-callback hop (same behavior plus
  overhead); a wider literature sweep (LMBOPT, box-constrained L-MQN /
  projected-gradient families, Nelder–Mead, mixture-model
  support-search alternatives) found nothing that both converges faster
  **and** lands on a similar mixture — the best bound-constrained
  solvers are high-dimension algorithms (LMBOPT is Matlab-only in its
  published form; Nelder–Mead is gradient-free, O(ε⁻²)); and the
  statistical alternatives change the *selection algorithm* itself.
  Details in [`docs/PERF.md`](docs/PERF.md) §4a.1–§4a.2.
* **Unifying the `d0`/`d1` gradient flags** into one 3-vector remains
  rejected: every 1-D family always requests the full pair, and the
  `d0`-only branch exists solely for R's `npnorm2Dll` gradient
  convention — a cosmetic change with no measurable benefit.

### 0.2.1 更新内容（相对 0.2.0）

**新增：二维族 `npnorm2Dll`** —— 双变量正态混合分布，经
`computemixdist(v, method="npnorm2Dll", beta=...)` 调用，移植 R 的
`computemixdist.npnorm2Dll`（含其实验性 L-BFGS-B 支撑点搜索、`(n, 2)`
数据约定、协方差矩阵 `beta` 与固定分量参数 `mu0`/`pi0`）：

* `v` 为 `(n, 2)` 数组（长度 `2n` 的扁向量会被自动重塑）；
* `beta` 为 `2 x 2` 协方差矩阵（默认单位阵）；
* `mix` / `gridpoints` 接受二维形式 `{"pt": (k, 2), "pr": (k,)}`；缺省时
  与 R 的包装器完全一致地构造默认值（逐边缘 `initial.npnorm` /
  `gridpoints.npnorm`，张量积组合）；
* 返回 `Npmix`，其 `pt` 为 `(x, y)` 点对列表，`beta` 存放协方差矩阵。

**新增：一维混合密度/分布函数核（R 的 `dnp*` / `pnp*` 层）** ——
顶层导出，全部由 C++/Eigen 核支撑：

* `dnpnorm`、`pnpnorm`、`dnpnormc`、`dnpt`、`pnpt`、`dnpdiscnorm`、
  `pnpdiscnorm`、`dnppois`、`pnppois`、`dnpdisct`（一维族，均含 `lg`
  对数空间形式）；
* `dnpnormND`（二维混合密度）与 `dnormNDarray`（`(n, k)` 多元正态 pdf
  核，可选对数形式）。

上述每个函数都精确复现 R 包的**原始代码路径**（包括 R 的怪癖，如
`dnppois` 对数路径的正态项复制粘贴错误、`dnpnormc` 在点质量支撑点处
的 NaN），在 30 例对拍集上相对误差 ≤ 1.2e-15；对数空间 `pgamma` 辅助
函数改为基于共享正则化伽马核重建，与 R 的对数分支逐位一致（此前对数
空间 Lentz CF 在远尾偏差高达 ~1e-4，被 `pnppois` 的 `lg=TRUE` 门捕获）。

**文档中明确标注 Eigen 版本** —— `eigen/` 下 vendored 的 Eigen 为
**5.0.0**（`EIGEN_WORLD_VERSION 3`，semver 起点），现已在 §1、§2 与
`docs/PERF.md` 中显式标注；构建参数与 parity 门不变。

**针对 R 4.6.1 实测的内核修正**（0.2.x 的对拍文件早于这些修正）：
`pnorm` 远尾 `lg=TRUE` 路由、`ppois` 的 `±Inf` 语义、`pgamma` 极点/符号
处理、`gammln` 极点/符号、`logspacesub` 分支、Poisson `x < 0`、`pnpt`
非对数怪癖、`dnppois` 对数空间行为。重建后 §5 全部 10 个测试套件报告
`TOTAL BAD: 0`。

**`npnorm2Dll` 目标函数快路径（新默认）+ 求解器评估** —— 契约从
*与 R 逐位一致*放宽为*更快、且 ll/混合与 R 结果类似*后，逐格
L-BFGS-B 目标函数为 n = 2 重新推导：

* **快路径（默认）**：二次型改用 `beta` 的显式 2×2 逆（每点 0 次除法，
  Cholesky 求解内核为 4 次）；每次求值不变量（`1/(dens+precompute)`、
  分量缩放）在每次 `solvegrad` 缓存一次而非逐次求值重算（每次拟合省
  ≈ 10⁸ 次除法）；两个长度为 n 的临时向量改为预分配缓冲——指数化仍与
  R 同款内核使用同一条 Eigen SIMD `exp` 指令序列。参考运行（n=300）：
  单次求值 ≈ 10.1 → ≈ 6.3 µs（求值次数同为 421,810、接受的格集相同），
  4.3 s → 2.7 s（相对 R ≈ 5.7× → ≈ 3.5×）。第二个 n=300 基准上
  ~1e-13 的数值差异还改变了被接受的格轨迹，求值数本身
  769,533 → 251,704（wall 7.0 s → 2.5 s）。
* **类似而非相同**：参考数据上两条路径均 6 次迭代收敛到 5 成分
  混合，主成分位置差 ≈ 0.06 内（ll 848.761 fast vs 848.747 exact
  vs 848.888 R）；第二个基准上 fast 少保留一个 w≈0.02 的小成分
  （k 7 → 6，ll 982.756 → 983.138，均为有效最优、不劣于 R 1.1.0003
  的 986.236）。parity 门（`tests/verify_2d_same.py`）现在检查的是
  *类似性*：每路径确定性、迭代数、ll 带宽、与 R 的**核**对拍（仍
  逐位一致）、两路径主成分接近度。
* **R 同款逃生门**：`NPFIC_2D_EXACT=1` 选用原 Cholesky 内核——即
  bit-exact 的 0.2.1 轨迹（参考运行 ll 848.7472820349071）。
* **替代求解器也一并评估**（被否决者**未** vendored，代码树零新增
  依赖）：CppNumericalSolvers 的 `LbfgsbSolver`（GPL/MIT 双许可）曾
  经临时适配器实测后撤出——另一实现（投影梯度容差硬编码 1e-4、
  无界 More–Thuente 步长事后 clamp 回盒、稠密 `M.inverse()` 历史
  更新），实测**收敛更晚**（20.1 s vs 8.2 s、1.96 M vs 0.77 M 次
  求值）且最优解不同（仍有效）；NLopt 的 `LD_LBFGS` 是同一 LBFGS-B
  的 C 移植 + 逐次求值 C 回调开销；更广的文献调研（LMBOPT、盒约束
  L-MQN / 投影梯度族、Nelder–Mead、混合模型支撑点搜索的算法级替代）
  未找到既**收敛更快**又落在类似混合的求解器——盒约束最优求解器是
  面向高维的算法（LMBOPT 发表形态为 Matlab；Nelder–Mead 无梯度、
  O(ε⁻²)）；统计侧替代改变的是*选择算法*本身。详见
  [`docs/PERF.md`](docs/PERF.md) §4a.1–§4a.2。
* **unify(d0/d1)** 维持**否决**：一维各族永远取全量梯度对，
  `d0`-only 分支只为 R `npnorm2Dll` 的梯度约定（d0 = 概率方向，
  d1 = 支撑点方向）而存在，合并纯属表面改动、无可测收益。

### What's new in 0.2.0 (vs 0.1.0)

**New: the four binned ("`...w`") families** — `npnormllw`,
`npnormcvmw`, `npnormadw`, `nptllw` (for `computemixdist` **and**
`estpi0`):

* observations are pre-binned onto the grid `h = 10^order`
  (default `order = -3`, round-down as in R's `bin`);
* new binned kernel machinery in C++/Eigen: binned-grid generation
  (`gridpoints_*_w`), the binned-t kernel (CDF-difference form), and the
  binned normal fill — a column-major rewrite of R's `dnormarray`
  accumulation that is bit-identical to the reference (no row-major
  temp, no transpose);
* the `order` argument is now accepted (and required in principle) by
  `computemixdist` / `estpi0` for these families.

**New: `NPFIC_ARCH` build-time SIMD control** — the SIMD level is now
explicitly chosen at build time by the `NPFIC_ARCH` environment variable
(`avx2` default / `avx` / `avx512` / `native` / empty = SSE2 baseline)
instead of implicit compiler defaults. On this build, AVX2 closes the
historical gap with R on the binned normal families
(`npnormllw`, n=5000: 354 ms → 130 ms, R 120 ms) and keeps the binned
t family 8–14× ahead of R; all parity gates stay green (see
[`docs/PERF.md`](docs/PERF.md)).

**Cleanup & consistency**

* all stale "binned variants not ported yet" notes removed from
  `npfc.py` / `__init__.py` docstrings and the README;
* the package tree is now build-relevant only: ad-hoc debug probes,
  crash repros, experiment scratch (root-level `.obj`/`.pdb`/`.log`/
  `.mdmp`/`.bin`, `examples/` Rust experiment, `target/`), and test
  scratch (parity logs, one-off R probes, intermediate bench outputs)
  removed; `.gitignore` tightened accordingly;
* `pyproject.toml` is the single version source (`0.2.0`), mirrored in
  `setup.py`; `npfc.__version__` / `npfixedcomppy.__version__` report
  `0.2.0-cpp` from the compiled extension;
* parity gates for the binned families (`tests/verify_binned.py`) and the
  binned bench pair (`tests/bench_binned.py` / `tests/bench_binned_r.R`)
  are part of the standard test set (10 suites, §5).

#### 0.2.0 更新内容（相对 0.1.0）

**新增：四个分箱（``"...w"``）族** —— `npnormllw`、`npnormcvmw`、
`npnormadw`、`nptllw`（`computemixdist` **和** `estpi0` 均支持）：

- 观测值先合并到网格 `h = 10^order`（默认 `order = -3`，向下取整，与
  R 的 `bin` 一致）；
- C++/Eigen 内新增分箱机制：分箱网格生成（`gridpoints_*_w`）、分箱 t
  核（CDF 差值形式）、分箱正态填充——列主序重写 R 的 `dnormarray`
  累加，与参考实现逐位一致（无行主序临时量、无转置）；
- `computemixdist` / `estpi0` 接受新的 `order` 参数（仅分箱族使用）。

**新增：`NPFIC_ARCH` 构建期 SIMD 控制** —— SIMD 级别改由环境变量
`NPFIC_ARCH` 在构建期显式选择（`avx2` 默认 / `avx` / `avx512` /
`native` / 空值 = SSE2 基线），取代原先隐式的编译器默认。本构建下
AVX2 补上了分箱正态族相对 R 的历史差距（`npnormllw`，n=5000：
354 ms → 130 ms，R 120 ms），分箱 t 族保持 8–14× 领先；全部 parity 门
保持绿色（见 [`docs/PERF.md`](docs/PERF.md)）。

**清理与一致性**

- 移除 `npfc.py` / `__init__.py` docstring 与 README 中所有"分箱变体尚
  未移植"的过期说明；
- 包树现在只保留与构建相关的内容：删除一次性调试探针、崩溃复现、实验
  草稿（根目录 `.obj`/`.pdb`/`.log`/`.mdmp`/`.bin`、`examples/` Rust
  实验、`target/`）以及测试草稿（parity 日志、一次性 R 探针、中间 bench
  输出），`.gitignore` 相应收紧；
- `pyproject.toml` 为唯一版本源（`0.2.0`），`setup.py` 同步；
  `npfc.__version__` / `npfixedcomppy.__version__` 由编译扩展报告
  `0.2.0-cpp`；
- 分箱族的 parity 门（`tests/verify_binned.py`）与分箱 bench 对
  （`tests/bench_binned.py` / `tests/bench_binned_r.R`）纳入标准测试集
  （10 个套件，见 §5）。

## 3. Installation

Prerequisites: Python ≥ 3.9 with `pip`, a C++17 compiler (MSVC 2019+
on Windows, GCC/Clang on Linux/macOS), and `pybind11` (installed
automatically as a build dependency).

### From a GitHub clone (compile from source)

```bat
git clone git@github.com:xiangjiexue2/npfixedcomppy.git
cd npfixedcomppy
pip install .                 # or: pip install -e .  (editable, for development)
```

`pip install .` invokes `setup.py`, which:

1. compiles the C++/Eigen extension `npfixedcomppy._core`
   (sources: `cpp/npfc_py.cpp`, `cpp/npfc_nnls.cpp`; headers from
   `cpp/` and the vendored `eigen/`) with the platform compiler
   (`/std:c++17 /O2` on MSVC, `-std=c++17 -O3` on GCC/Clang) plus the
   SIMD flags below;
2. packages `python/npfixedcomppy/` (the Python front-end) together with
   the built extension into an installable wheel.

**SIMD level (`NPFIC_ARCH`)** — set it before the build; the artifact
only runs on hardware with that instruction set or a superset:

| `NPFIC_ARCH` | effect (MSVC / GCC-Clang)                              |
|--------------|--------------------------------------------------------|
| *(unset)*    | `avx2` — `/arch:AVX2` / `-mavx2` (default; what ships) |
| `avx`        | `/arch:AVX` / `-mavx`                                   |
| `avx512`     | `/arch:AVX512` / `-mavx512*`                            |
| `native`     | no effect on MSVC (falls back to `avx2`) / `-march=native` |
| *(empty)*    | plain x86-64 baseline (SSE2, no `/arch` flag)           |

```bat
:: MSVC (cmd.exe): choose a different SIMD level
set NPFIC_ARCH=avx512
pip install .

:: bash
NPFIC_ARCH=avx512 pip install .
```

Note: MSVC's `/arch:AVX2` also enables FMA contraction (documented MSVC
behaviour under the default `/fp:precise`); GCC/Clang do not. The
parity gates (§5) are the arbiter of whether a given level is safe for
the bit-exactness requirements.

### Packaging / building wheels

```bash
# sdist + wheel for the current interpreter
pip wheel .                    # or: python setup.py bdist_wheel

# cross-platform wheels (as CI does, .github/workflows/main.yml)
pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
```

### 安装

前置条件：Python ≥ 3.9（含 `pip`）、C++17 编译器（Windows 用 MSVC
2019+，Linux/macOS 用 GCC/Clang）；`pybind11` 会作为构建依赖自动安装。

**从 GitHub 拉取后编译：**

```bat
git clone git@github.com:xiangjiexue2/npfixedcomppy.git
cd npfixedcomppy
pip install .                 # 开发期可用：pip install -e .（editable）
```

`pip install .` 调用 `setup.py`：

1. 用平台编译器编译 C++/Eigen 扩展 `npfixedcomppy._core`（源文件：
   `cpp/npfc_py.cpp`、`cpp/npfc_nnls.cpp`；头文件来自 `cpp/` 与 vendored
   的 `eigen/`），MSVC 下为 `/std:c++17 /O2`，GCC/Clang 下为
   `-std=c++17 -O3`，外加下面的 SIMD 参数；
2. 把 `python/npfixedcomppy/`（Python 前端）与编译出的扩展打包成 wheel。

**SIMD 级别（`NPFIC_ARCH`）** —— 构建前设置；产物只能在该指令集（或
超集）的硬件上运行：

| `NPFIC_ARCH` | 效果（MSVC / GCC-Clang）                                   |
|--------------|------------------------------------------------------------|
| *（未设置）* | `avx2` — `/arch:AVX2` / `-mavx2`（默认，即发布构建）        |
| `avx`        | `/arch:AVX` / `-mavx`                                       |
| `avx512`     | `/arch:AVX512` / `-mavx512*`                                |
| `native`     | MSVC 下无效（回退 `avx2`） / `-march=native`                |
| *（空值）*   | 纯 x86-64 基线（SSE2，无 `/arch` 参数）                     |

```bat
:: MSVC（cmd.exe）：选择其他 SIMD 级别
set NPFIC_ARCH=avx512
pip install .
```

注意：MSVC 的 `/arch:AVX2` 同时启用 FMA 收缩（MSVC 在默认
`/fp:precise` 下的既定行为）；GCC/Clang 不会。对逐位精确性要求是否
安全，以 §5 的 parity 门为准。

**打包 / 构建 wheel：**

```bash
# 当前解释器的 sdist + wheel
pip wheel .                    # 或：python setup.py bdist_wheel

# 跨平台 wheel（CI 的方式，见 .github/workflows/main.yml）
pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
```

## 4. Usage

```python
import numpy as np
from npfixedcomppy import computemixdist, estpi0

rng = np.random.default_rng(123)
x = rng.normal(loc=[0, 2], scale=1, size=1000)

# non-parametric maximum-likelihood mixing distribution (normal family)
res = computemixdist(x, method="npnormll")
print(res)                 # Npmix(family='npnorm', beta=1.0, n_support=..., ...)
print(res.pt)              # support points
print(res.pr)              # weights
print(res.mix["pt"])       # R-style accessor, same as res.pt

# estimate the proportion at 0 simultaneously
res0 = estpi0(x, method="npnormll", val=2)

# binned family (large samples): grid h = 10**order
resb = computemixdist(x, method="npnormllw", order=-3)

# fixed components: (mu0, pi0) included in the mixture, never updated
resf = computemixdist(x, method="npnormll", mu0=-0.5, pi0=0.3)

# bivariate normal mixture (experimental; slow — see §6)
from numpy import eye
res2d = computemixdist(np.column_stack([x, rng.normal(size=1000)]),
                       method="npnorm2Dll", beta=eye(2))
```

### 快速上手

```python
import numpy as np
from npfixedcomppy import computemixdist, estpi0

rng = np.random.default_rng(123)
x = rng.normal(loc=[0, 2], scale=1, size=1000)

# 正态族的非参数极大似然混合分布
res = computemixdist(x, method="npnormll")
print(res)                 # Npmix(family='npnorm', beta=1.0, ...)
print(res.pt)              # 支撑点
print(res.pr)              # 权重
print(res.mix["pt"])       # R 风格访问器，与 res.pt 相同

# 同时估计零处点质量比例
res0 = estpi0(x, method="npnormll", val=2)

# 分箱族（大样本）：网格 h = 10**order
resb = computemixdist(x, method="npnormllw", order=-3)

# 固定分量：(mu0, pi0) 参与混合、从不更新
resf = computemixdist(x, method="npnormll", mu0=-0.5, pi0=0.3)

# 双变量正态混合（实验性，较慢——见 §6）
res2d = computemixdist(np.column_stack([x, rng.normal(size=1000)]),
                       method="npnorm2Dll", beta=np.eye(2))
```

### API

**`computemixdist(v, method="npnormll", mu0=None, pi0=None, beta=None,
order=-3, mix=None, gridpoints=None, tol=1e-6, maxit=100, verbose=0)
-> Npmix`**

Estimates a discrete mixing distribution `G = sum_j pi_j Delta_{mu_j}` for
the kernel indexed by `method`, optionally with fixed components
(support points/weights included in the mixture but never updated),
matching the R `computemixdist` dispatcher.

- `v`: array_like of observations (counts for `nppoisll`,
  correlations in (-1, 1) for `npnormcll`).
- `method`: one of the ten families in §1.
- `mu0` / `pi0`: fixed components. Default `[0.0]` for both (a
  zero-weight degenerate fixed component at the origin, as in R);
  `sum(pi0) < 1` is required, estimated components are scaled to the
  remaining mass.
- `beta`: structural parameter; required for `npnormcll` (sample size).
- `order`: binning level, binned ("`...w`") families only (default `-3`,
  grid `h = 10^order`, round-down as in R); ignored by the un-binned
  families.
- `mix`: optional initial mixing `{"pt": [...], "pr": [...]}`; derived
  from a histogram of the data when omitted.
- `gridpoints`: optional candidate support points for the new-point
  search; derived from a histogram when omitted.
- `tol`, `maxit`: outer-loop convergence tolerance / iteration cap.
- Returns an `Npmix` (below).

**`estpi0(v, method="npnormll", beta=None, val=None, order=-3, mix=None,
gridpoints=None, tol=1e-6, verbose=0, fast=True, relax=False,
inner_tol=1e-4) -> Npmix`**

Fits `computemixdist` with the point mass at the origin *estimated rather
than fixed*: it searches for the fixed weight `pi0` of a point mass at 0
such that the family's hypothesis statistic (loss difference between the
fit with the point mass and the unconstrained fit) reaches the threshold
`val` — a likelihood-ratio-style test for the ML families, a distance
statistic for the distance families. Recommended `val`: `2` for the
likelihood families, `0.1` for `npnormcvm`, `1` for `npnormad`
(`None` uses these defaults). `fast=True` (default) is several times
faster while preserving the `tol` guarantee on the statistic;
`fast=False` reproduces the R/C++ refinement bit-for-bit.

**`Npmix`** — plain dataclass, attribute access or the R-compatible
`mix` accessor (`r.mix["pt"]` / `r.mix["pr"]`):

| field          | meaning                                                        |
|----------------|----------------------------------------------------------------|
| `pt`, `pr`     | support points (incl. fixed components, ascending) and weights |
| `beta`         | structural parameter actually used                             |
| `family`       | `"npnorm"` / `"npt"` / `"npnormc"` / `"nppois"`                |
| `min_gradient` | min gradient w.r.t. a new support point (≤ 0 at convergence)   |
| `ll`           | loss at the estimate (−log-likelihood or the distance)         |
| `flag`         | `"d0"` (derivative-free search) / `"d1"` (improved Brent)      |
| `iter`         | outer iterations performed                                     |
| `convergence`  | 0 = tolerance met, 1 = hit `maxit`                             |

Also exported for the R-package correspondence layer: `posteriormean`,
`covestEB`, `covestEB_cor` (`CovEBResult`), the `FAMILIES` registry, and
`__version__`. See the module docstrings in `python/npfixedcomppy/` for
the full parameter reference.

**Mixture density / cdf kernels (the R `dnp*` / `pnp*` layer)** — all run
in the C++/Eigen core, take array-like `x`/`mu0`/`pi0`, and return an
`ndarray`:

| function | meaning (R name in `x`) |
|----------|--------------------------|
| `dnpnorm(x, mu0, pi0, stdev=1.0, lg=False)` | normal mixture pdf (R `dnpnorm`) |
| `pnpnorm(x, mu0, pi0, stdev=1.0, lt=True, lg=False)` | normal mixture cdf (R `pnpnorm`) |
| `dnpnormc(x, mu0, pi0, n, lg=False)` | one-parameter normal (correlation) pdf — `npnormcll` kernel (R `dnpnormc`) |
| `dnpt(x, mu0, pi0, df, lg=False)` | non-central-t mixture pdf — `nptll` kernel (R `dnpt`) |
| `pnpt(x, mu0, pi0, df, lt=True, lg=False)` | non-central-t mixture cdf (R `pnpt`) |
| `dnpdiscnorm(x, mu0, pi0, stdev, h, lg=False)` | binned normal mixture pdf — `...w` normal families (R `dnpdiscnorm`) |
| `pnpdiscnorm(x, mu0, pi0, stdev, h, lt=True, lg=False)` | binned normal mixture cdf (R `pnpdiscnorm`) |
| `dnppois(x, mu0, pi0, stdev=1.0, lg=False)` | Poisson mixture pdf — `nppoisll` kernel (R `dnppois`; `stdev` only reaches the `lg` normal-term quirk) |
| `pnppois(x, mu0, pi0, lt=True, lg=False)` | Poisson mixture cdf (R `pnppois`) |
| `dnpdisct(x, mu0, pi0, df, h, lg=False)` | binned t mixture pdf — `nptllw` kernel (R `dnpdisct`) |
| `dnpnormND(x, mu0, pi0, sigma, lg=False)` | 2-D (bivariate) normal mixture pdf — `npnorm2Dll` kernel (R `dnpnormND`) |
| `dnormNDarray(x, mu0, sigma, lg=False)` | the `(n, k)` multivariate-normal pdf kernel (R `dnormNDarray_`) |

These reproduce R's exact code paths — including its quirks (the
`dnppois` `lg` path adds a normal term for the extra support points;
`dnpnormc` is NaN at point-mass support points) — to relative error
≤ 1.2e-15 against live R 4.6.1 (30-case gold set,
`tests/verify_kernels.py`).

### API

**`computemixdist(v, method="npnormll", mu0=None, pi0=None, beta=None,
order=-3, mix=None, gridpoints=None, tol=1e-6, maxit=100, verbose=0)
-> Npmix`**

对 `method` 指定的核估计离散混合分布 `G = sum_j pi_j Delta_{mu_j}`，
可选带固定分量（参与混合但从不更新的支撑点/权重），与 R 的
`computemixdist` 分发器一致。

- `v`：观测值（`nppoisll` 为计数，`npnormcll` 为 (-1, 1) 内的相关系数）。
- `method`：§1 中十个族之一。
- `mu0` / `pi0`：固定分量。默认均为 `[0.0]`（原点处零权重的退化固定
  分量，与 R 一致）；要求 `sum(pi0) < 1`，估计分量按剩余质量缩放。
- `beta`：结构参数；`npnormcll` 必须显式给出（样本量）。
- `order`：分箱级别，仅分箱（``"...w"``）族使用（默认 `-3`，网格
  `h = 10^order`，向下取整，与 R 一致）；非分箱族忽略。
- `mix`：可选初始混合分布 `{"pt": [...], "pr": [...]}`；缺省时由数据
  直方图导出。
- `gridpoints`：可选的候选支撑点；缺省时由数据直方图导出。
- `tol`、`maxit`：外层循环收敛容差 / 迭代上限。
- 返回 `Npmix`（见下）。

**`estpi0(v, method="npnormll", beta=None, val=None, order=-3, mix=None,
gridpoints=None, tol=1e-6, verbose=0, fast=True, relax=False,
inner_tol=1e-4) -> Npmix`**

以**估计而非固定**的原点处点质量拟合 `computemixdist`：搜索零处点质量
的固定权重 `pi0`，使该族的假设检验统计量（带点质量的拟合与无约束拟合
的损失差）达到阈值 `val` —— 似然族为似然比型检验、距离族为距离统计量。
推荐 `val`：似然族 `2`，`npnormcvm` 用 `0.1`，`npnormad` 用 `1`
（`None` 时采用这些默认值）。`fast=True`（默认）快数倍且保持统计量
`tol` 精度保证；`fast=False` 与 R/C++ 的细化过程逐位一致。

**`Npmix`** —— 纯 dataclass，属性访问或 R 兼容的 `mix` 访问器
（`r.mix["pt"]` / `r.mix["pr"]`）：

| 字段           | 含义                                                         |
|----------------|--------------------------------------------------------------|
| `pt`、`pr`     | 支撑点（含固定分量，升序）与权重                             |
| `beta`         | 实际使用的结构参数                                           |
| `family`       | `"npnorm"` / `"npt"` / `"npnormc"` / `"nppois"`              |
| `min_gradient` | 对新支撑点的最小梯度（收敛解处 ≤ 0）                         |
| `ll`           | 估计处的损失（负对数似然或距离）                             |
| `flag`         | `"d0"`（无导数搜索） / `"d1"`（改进 Brent）                  |
| `iter`         | 执行的外层迭代数                                             |
| `convergence`  | 0 = 达到容差，1 = 达到 `maxit`                               |

另导出 R 包对应层用的 `posteriormean`、`covestEB`、`covestEB_cor`
（`CovEBResult`）、`FAMILIES` 注册表与 `__version__`。完整参数说明见
`python/npfixedcomppy/` 中的模块 docstring。

**混合密度 / 分布函数核（R 的 `dnp*` / `pnp*` 层）** —— 全部运行于
C++/Eigen 核，接受 array-like 的 `x`/`mu0`/`pi0`，返回 `ndarray`：

| 函数 | 含义（括号内为 R 函数名） |
|------|---------------------------|
| `dnpnorm(x, mu0, pi0, stdev=1.0, lg=False)` | 正态混合 pdf（R `dnpnorm`） |
| `pnpnorm(x, mu0, pi0, stdev=1.0, lt=True, lg=False)` | 正态混合 cdf（R `pnpnorm`） |
| `dnpnormc(x, mu0, pi0, n, lg=False)` | 一参数正态（相关）pdf —— `npnormcll` 核（R `dnpnormc`） |
| `dnpt(x, mu0, pi0, df, lg=False)` | 非中心 t 混合 pdf —— `nptll` 核（R `dnpt`） |
| `pnpt(x, mu0, pi0, df, lt=True, lg=False)` | 非中心 t 混合 cdf（R `pnpt`） |
| `dnpdiscnorm(x, mu0, pi0, stdev, h, lg=False)` | 分箱正态混合 pdf —— `...w` 正态族（R `dnpdiscnorm`） |
| `pnpdiscnorm(x, mu0, pi0, stdev, h, lt=True, lg=False)` | 分箱正态混合 cdf（R `pnpdiscnorm`） |
| `dnppois(x, mu0, pi0, stdev=1.0, lg=False)` | 泊松混合 pdf —— `nppoisll` 核（R `dnppois`；`stdev` 仅出现在 `lg` 的正态项怪癖中） |
| `pnppois(x, mu0, pi0, lt=True, lg=False)` | 泊松混合 cdf（R `pnppois`） |
| `dnpdisct(x, mu0, pi0, df, h, lg=False)` | 分箱 t 混合 pdf —— `nptllw` 核（R `dnpdisct`） |
| `dnpnormND(x, mu0, pi0, sigma, lg=False)` | 双变量正态混合 pdf —— `npnorm2Dll` 核（R `dnpnormND`） |
| `dnormNDarray(x, mu0, sigma, lg=False)` | `(n, k)` 多元正态 pdf 核（R `dnormNDarray_`） |

它们精确复现 R 的原始代码路径——包括 R 的怪癖（`dnppois` 的 `lg`
路径对额外支撑点加入正态项；`dnpnormc` 在点质量支撑点处为 NaN）——
在 30 例金标集上（`tests/verify_kernels.py`）相对误差 ≤ 1.2e-15
（对照 R 4.6.1 实测值）。

## 5. Testing & parity with R

Run from the package root with the venv's Python (script-style, each
exits non-zero on failure):

- `python tests\verify_npnormll.py` — normal MLE: bit-level **GOLD**
  checks (ll/pt/pr vs recorded R references, deterministic re-run) plus
  tolerance **BAND** checks where R itself is run-to-run
  non-deterministic.
- `python tests\verify_nptll.py` — t-family MLE (finite and infinite
  degrees of freedom, fixed components, estpi0), bit-level.
- `python tests\verify_cvmadcll.py` — Cramér–von Mises, Anderson–Darling
  and correlation families incl. estpi0: GOLD for the deterministic R
  cases (CLL), BAND where R is non-deterministic (CVM/AD).
- `python tests\verify_pois.py` — Poisson MLE (CM + estpi0) on a shared
  R-generated data file, bit-level.
- `python tests\verify_density.py` — recomputed-density invariants
  (`ll == -sum log d(x_i | returned mixture)`) and estpi0 threshold
  invariants, all families.
- `python tests\verify_binned.py` — the four binned ("`...w`") families:
  GOLD (ll to 1e-9, pt/pr at 1e-6/1e-4, deterministic re-run) where R is
  deterministic (LLW / TLLW / LLW_FIX), BAND where R is non-deterministic
  (CVMW / ADW).
- `python tests\verify_coveb.py` — `covestEB` / `covestEB_cor` against
  R-recorded intermediates (projection primitive strict, pipeline at
  1e-2).
- `python tests\verify_posteriormean.py` — posterior means of `fun(pt)`
  against R-recorded values (strict 1e-9 on the pure kernel path).
- `python tests\verify_kernels.py` — the 30-case 1-D/2-D **kernel**
  parity gate (`dnp*` / `pnp*` / `dnormNDarray` against live R 4.6.1
  gold values; finite values ≤ 1.2e-15 relative, NaNs at identical
  positions).
- `python tests\verify_2d_same.py` — the `npnorm2Dll` gate (a
  *similarity* gate since the objective fast path, §4a.2 of
  [`docs/PERF.md`](docs/PERF.md)): per-path two-run determinism and
  convergence for both the fast (default) and the R-identical
  (`NPFIC_2D_EXACT=1`) objective path, ll bands, per-`solvegrad`
  `ll` self-consistency, and **kernel** parity (PY's `dnpnormND` at
  R's final points reproduces R's ll to 1e-9), plus a
  no-worse-than-R local-optimum check and a fast-vs-exact mixture
  proximity check (main components within 0.08 in location,
  |Δll| < 2e-2).
- `python tests\perf_baseline.py` — wall-time baseline of the public API.
- `python tests\bench_binned.py` — wall-time bench for the binned
  families (the R-side counterpart is `tests\bench_binned_r.R`).

All ten parity suites must report `TOTAL BAD: 0`. The package is
**deterministic per build**: for a fixed build and a fixed
`OMP_NUM_THREADS`, identical inputs produce bit-identical outputs (unlike
the R package on parallel platforms).

### 测试与 R 包一致性

在包根目录下用 venv 的 Python 运行（脚本式，失败时退出码非零）：

- `python tests\verify_npnormll.py` — 正态极大似然：逐位 **GOLD** 检查
  （ll/pt/pr 对比录制的 R 参考、确定性重跑）+ R 本身逐次非确定处的
  容差 **BAND** 检查。
- `python tests\verify_nptll.py` — t 族极大似然（有限/无限自由度、固定
  分量、estpi0），逐位。
- `python tests\verify_cvmadcll.py` — Cramér–von Mises、Anderson–Darling
  与相关族（含 estpi0）：R 确定（CLL）用 GOLD，R 非确定（CVM/AD）用
  BAND。
- `python tests\verify_pois.py` — 泊松极大似然（CM + estpi0），共读
  R 生成的数据文件，逐位。
- `python tests\verify_density.py` — 重算密度不变量
  （`ll == -sum log d(x_i | 返回的混合分布)`）与 estpi0 阈值不变量，
  全部族。
- `python tests\verify_binned.py` — 四个分箱（``"...w"``）族：R 确定的
  用例（LLW / TLLW / LLW_FIX）用 GOLD（ll 1e-9，pt/pr 1e-6/1e-4，
  确定性重跑），R 非确定的用例（CVMW / ADW）用 BAND。
- `python tests\verify_coveb.py` — `covestEB` / `covestEB_cor` 对比
  R 录制的中间量（投影基元严格门，管线 1e-2）。
- `python tests\verify_posteriormean.py` — `fun(pt)` 的后验均值对比
  R 录制的值（纯核路径严格 1e-9）。
- `python tests\verify_kernels.py` — 30 例一维/二维**核**对拍门
  （`dnp*` / `pnp*` / `dnormNDarray` 对比 R 4.6.1 实测金标值；有限值
  相对误差 ≤ 1.2e-15，NaN 位置一致）。
- `python tests\verify_2d_same.py` — `npnorm2Dll` 门（目标函数快路径
  之后改为*类似性*门，见
  [`docs/PERF.md`](docs/PERF.md) §4a.2）：快路径（默认）与 R 同款
  （`NPFIC_2D_EXACT=1`）两条路径各自的双次运行确定性与收敛、ll 带宽、
  每次 `solvegrad` 的 `ll` 自洽，以及**核**对拍（PY 的 `dnpnormND` 在
  R 最终点上复现 R 的 ll 至 1e-9），并校验两条路径的局部最优均不劣于
  R、以及两路径混合接近度（主成分位置差 < 0.08、|Δll| < 2e-2）。
- `python tests\perf_baseline.py` — 公开 API 的耗时基线。
- `python tests\bench_binned.py` — 分箱族耗时 bench（R 侧对应脚本为
  `tests\bench_binned_r.R`）。

十个 parity 套件必须全部报告 `TOTAL BAD: 0`。本包对**给定构建与线程数**
是确定性的：相同输入恒产生逐位相同的输出（并行平台上的 R 包做不到这一
点）。

## 6. Performance

The whole solver is C++/Eigen (SIMD level decided at build time,
`/arch:AVX2` by default; Eigen's own parallel GEMM/GEMV is enabled when
the build's OpenMP probe passes); the kernel column cache — each kernel column
`K[:, mu]` computed once per fit and reused by the mapping, gradient,
weight, and collapse passes — is the main speed-up. Typical wall time on
this build (median of 5, warmup excluded):

| case | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormll")`, n=1000 | ≈ 12 ms | ≈ 19 ms |
| `computemixdist(x, method="nptll", beta=inf)`, n=1000 | ≈ 37 ms | ≈ 211 ms |
| `computemixdist(x, method="nptll", beta=5)`, n=5000 | ≈ 3.1 s | ≈ 50 s |
| `computemixdist(x, method="npnormcll", beta=1000)`, n=1000 | ≈ 1.0 s | ≈ 1.9 s |
| `computemixdist(x, method="npnormad")`, n=1000 | ≈ 38 ms | ≈ 59 ms |
| `computemixdist(x, method="nppoisll")`, n=1000 | ≈ 0.6 ms | ≈ 5 ms |
| `estpi0(x, method="npnormll")`, n=1000 | ≈ 29 ms | ≈ 97 ms |
| `computemixdist(X, method="npnorm2Dll")`, n=300 (2-D) | ≈ 2.7 s | ≈ 0.8 s |

Binned (`order = -3`, i.e. `h = 10^-3`):

| case | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormllw")`, n=5000 | ≈ 131 ms | ≈ 120 ms |
| `computemixdist(x, method="npnormadw")`, n=5000 | ≈ 421 ms * | ≈ 170 ms * |
| `computemixdist(x, method="nptllw")`, n=5000 | ≈ 222 ms | ≈ 1.97 s |
| `computemixdist(x, method="nptllw", beta=5)`, n=5000 | ≈ 2.1 s | ≈ 29.3 s |
| `estpi0(x, method="npnormllw")`, n=5000 | ≈ 563 ms | ≈ 1.33 s |

\* `npnormadw` fits wander between basins run-to-run in both
implementations, so this ratio is indicative.

The 2-D row above is the documented *exception*: `npnorm2Dll` is an
experimental family whose per-outer-iteration support-point search
launches one L-BFGS-B problem **per grid cell** (103 × 103 for the
default grid) and, unlike the 1-D engine, it does not use the kernel
column cache. Two fast paths now cut the old ~28× gap to ~3.5×:
(i) a hand-unrolled 2×2 density-kernel fast path (cutting `dnpnormND`
from ≈ 0.4 ms/call to ≈ 7.4 µs/call at n=300 — the old gap's main
source, since MSVC keeps the per-point heap temporaries that the R
build's GCC eliminates), and (ii) an objective fast path with an
explicit-inverse quadratic form (0 divisions/point), per-`solvegrad`
invariant caching, and preallocated buffers — the default, which cuts
the per-evaluation cost ≈ 10.1 → ≈ 6.3 µs on the reference run (same
421,810 evaluations, same accepted cell set; on the second n=300
benchmark the ~1e-13 difference from the R-identical kernel also
changes the accepted cells, dropping the evaluation count 769,533 →
251,704). The density kernels remain bit-exact with R, and both paths
land on a similar valid optimum
(main components within ≈ 0.06, ll within ≈ 0.01); `NPFIC_2D_EXACT=1`
selects the bit-exact R-identical path. See
[`docs/PERF.md`](docs/PERF.md) for the breakdown.

For the `t` family with small degrees of freedom the cost grows with `n`
(super-linearly — the per-point `dnt` kernel has no cheap identity), so
very large `n` is slow; the normal families scale linearly. The binned
normal families run within ~25 % of R's binned time (bit-exact
column-major fill, scalar `exp` on both sides), while the binned `t`
family — whose kernel is a cheap CDF difference — is ~8–14× faster. See
[`docs/PERF.md`](docs/PERF.md) for the phase profiles, the kernel column
cache, and the full comparison with R.

Environment knobs: `NPFIXEDCOMPY_PROFILE=1` prints a per-phase timing
line (solvegrad / mapping / loss / weights / collapse) to stderr;
`NPFIC_2D_EXACT=1` selects the bit-exact R-identical `npnorm2Dll`
objective path (default: the fast path, §4a.2 of
[`docs/PERF.md`](docs/PERF.md)); `OMP_NUM_THREADS` sizes Eigen's pool
(when the build has OpenMP) without changing the results.

### 性能

整个求解器为 C++/Eigen（SIMD 级别在构建期决定，默认 `/arch:AVX2`；构建期
OpenMP 探测通过时启用 Eigen 自身的并行 GEMM/GEMV）；内核列缓存——每个内核
列 `K[:, mu]` 每次拟合只计算一次，供 mapping、梯度、
权重、collapse 各环节复用——是主要的加速手段。本构建典型耗时（5 次取中
位数，排除预热）：

| 用例 | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormll")`，n=1000 | ≈ 12 ms | ≈ 19 ms |
| `computemixdist(x, method="nptll", beta=inf)`，n=1000 | ≈ 37 ms | ≈ 211 ms |
| `computemixdist(x, method="nptll", beta=5)`，n=5000 | ≈ 3.1 s | ≈ 50 s |
| `computemixdist(x, method="npnormcll", beta=1000)`，n=1000 | ≈ 1.0 s | ≈ 1.9 s |
| `computemixdist(x, method="npnormad")`，n=1000 | ≈ 38 ms | ≈ 59 ms |
| `computemixdist(x, method="nppoisll")`，n=1000 | ≈ 0.6 ms | ≈ 5 ms |
| `estpi0(x, method="npnormll")`，n=1000 | ≈ 29 ms | ≈ 97 ms |
| `computemixdist(X, method="npnorm2Dll")`，n=300（二维） | ≈ 2.7 s | ≈ 0.8 s |

分箱（`order = -3`，即 `h = 10^-3`）：

| 用例 | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormllw")`，n=5000 | ≈ 131 ms | ≈ 120 ms |
| `computemixdist(x, method="npnormadw")`，n=5000 | ≈ 421 ms * | ≈ 170 ms * |
| `computemixdist(x, method="nptllw")`，n=5000 | ≈ 222 ms | ≈ 1.97 s |
| `computemixdist(x, method="nptllw", beta=5)`，n=5000 | ≈ 2.1 s | ≈ 29.3 s |
| `estpi0(x, method="npnormllw")`，n=5000 | ≈ 563 ms | ≈ 1.33 s |

\* `npnormadw` 在两侧实现中都存在逐次运行的 basin 漂移，该比值仅作参考。

上表 2-D 一行是**有记录的例外**：`npnorm2Dll` 是实验性族，其每轮外层
迭代的支撑点搜索对**每个网格单元**各启动一个 L-BFGS-B 问题（默认网格
103 × 103），且与一维引擎不同，它不使用内核列缓存。两条快路径把旧
~28× 差距收窄到 ~3.5×：(i) 手写展开的 2×2 密度内核快路径（
`dnpnormND` 从 ≈ 0.4 ms/次降到 ≈ 7.4 µs/次，n=300——旧差距的主要来源：
MSVC 保留了逐点堆临时对象，而 R 构建的 GCC 会消除它们）；(ii) 目标函数
快路径（默认）——显式逆矩阵二次型（每点 0 次除法）、每次 `solvegrad`
的不变量缓存、预分配缓冲——参考运行上把单次求值从 ≈ 10.1 µs 降到
≈ 6.3 µs（求值次数同为 421,810、接受的格集相同；第二个 n=300 基准上
与 R 同款内核 ~1e-13 的数值差异还改变了被接受的格，求值数
769,533 → 251,704）。密度核本身仍与 R 逐位一致，两条路径都落在
类似的**有效**最优（主成分位置差 ≈ 0.06 内、ll 差 ≈ 0.01 内）；
`NPFIC_2D_EXACT=1` 选用 bit-exact 的 R 同款路径。分解见
[`docs/PERF.md`](docs/PERF.md)。

t 族在小自由度下 `dnt` 核没有廉价恒等式，成本随 `n` 超线性增长，`n` 很
大时较慢；正态族为线性。分箱正态族与 R 的分箱耗时相差约 25% 以内（列
主序、逐位精确的填充，两侧同为标量 `exp`）；分箱 t 族的核是廉价的 CDF
差值，快约 8–14 倍。分阶段 profile、内核列缓存及与 R 的完整对比见
[`docs/PERF.md`](docs/PERF.md)。

环境旋钮：`NPFIXEDCOMPY_PROFILE=1` 向 stderr 输出分阶段计时
（solvegrad / mapping / loss / weights / collapse）；`NPFIC_2D_EXACT=1`
选用 `npnorm2Dll` 的 bit-exact R 同款目标函数路径（默认：快路径，
见 [`docs/PERF.md`](docs/PERF.md) §4a.2）；`OMP_NUM_THREADS` 调整
Eigen 线程池大小（OpenMP 构建时），不改变结果。
