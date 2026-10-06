# npfixedcomppy

A Python package (C++/Eigen core) for **non-parametric estimation of
mixing distributions** in several parametric families — a reimplementation
of the R package [`npfixedcomp2`](https://CRAN). The support-point
algorithm follows Wang (2007, "On Fast Computation of the Non-Parametric
Maximum Likelihood Estimate of a Mixing Distribution", *J. R. Statist.
Soc. B* 69(2), 185–198) and the extensions added in `npfixedcomp2`; the
null-proportion (`estpi0`) machinery follows Xue & Wang (2023, "A
Nonparametric Mixture Approach to Density and Null Proportion Estimation
in Large-Scale Multiple Comparison Problems", *Aust. N.Z. J. Statist.*
65(1), 49–75). Full reference list: §7.

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
| `npnormND`   | multivariate normal (k ≥ 2) | maximum likelihood | covariance matrix (default `Iₖ`) |
| `npnorm2Dll` | k = 2 alias of `npnormND` (R name) | maximum likelihood (experimental) | covariance matrix (default `I₂`) |

The binned ("`...w`") variants pre-bin the observations onto the grid
`h = 10^order` (default `order = -3`, i.e. `h = 10^-3`; round-down, as in
R's `bin`) and fit the binned kernel; they are intended for large samples
and take `order` as an extra argument (ignored by the un-binned families).

`npnormND` takes an `(n, k)` data matrix (k ≥ 2) and fits a
k-variate normal mixture (the structural parameter `beta` is the
`k x k` covariance matrix, default the identity). It runs its own
L-BFGS-B support-point search (the L-BFGS-B of yixuan/LBFGSpp, as in
the R package) instead of the 1-D engine; the R package documents
its 2-D version as *experimental and possibly very slow*. At
k = 2 — invokable as `method="npnorm2Dll"`, the R name (API
compatibility) — it is still the slowest family relative to R on
this build (n=300: ≈ 0.91 s vs ≈ 0.8 s, same data — the residual gap
is the per-cell L-BFGS-B search machinery, after a hand-unrolled 2×2
density kernel, a 0-division explicit-inverse objective fast path,
the per-cell hot start *and* the true directional derivative cut the
old ~28× gap to ~1.2×) — see
[`docs/PERF.md`](docs/PERF.md). The k = 2 trajectory is bit-
identical to 0.2.3, the density kernels remain bit-exact with R, and
the fit converges deterministically to a valid local optimum whose
density agrees with the R reference within tolerance (k = 2 gate:
rel-L1 0.0158, max |log diff| 0.169 —
`tests/verify_2d_same.py`). Higher k is slower: the search grid is
the tensor product of the per-axis marginal grids (k = 3: 18 points
per axis ⇒ 4,913 cells, ≈ 2.2 s at n = 360) — see §6.

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
The acceptance contract (relaxed from 0.2.3, all families): the fit
density must agree with the R reference within tolerance, and the
estimated `ll` must not be worse than the historical version — the
solver no longer matches R's floating-point trajectory (the 0.2.3
relaxed support search, "What's new in 0.2.3" above, settles on a
similar — frequently slightly better — local optimum); the parity
gates in §5 enforce the contract. Determinism within a build is
unchanged: identical inputs give bit-identical outputs. Details and
measured evidence: [`docs/PERF.md`](docs/PERF.md).

### 本包的作用

`npfixedcomppy` 是一个 Python 包（C++/Eigen 内核），用于若干参数族的
**非参数混合分布估计**——R 包 [`npfixedcomp2`](https://CRAN) 的重写版本。
支撑点算法遵循 Wang (2007, "On Fast Computation of the Non-Parametric
Maximum Likelihood Estimate of a Mixing Distribution", *J. R. Statist.
Soc. B* 69(2), 185–198) 及 `npfixedcomp2` 的扩展；零处点质量比例
（`estpi0`）机制遵循 Xue & Wang (2023, "A Nonparametric Mixture Approach
to Density and Null Proportion Estimation in Large-Scale Multiple
Comparison Problems", *Aust. N.Z. J. Statist.* 65(1), 49–75)。
完整引用见 §7。

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
| `npnormND`   | k 元正态（k ≥ 2） | 极大似然                   | 协方差矩阵（默认 `Iₖ`）        |
| `npnorm2Dll` | `npnormND` 的 k = 2 别名（R 名） | 极大似然（实验性） | 协方差矩阵（默认 `I₂`）        |

分箱（``"...w"``）变体先将观测值合并到网格 `h = 10^order`（默认
`order = -3`，即 `h = 10^-3`；向下取整，与 R 的 `bin` 一致），再拟合分
箱核；面向大样本场景，`order` 为额外参数（非分箱族忽略）。

`npnormND` 接收 `(n, k)` 数据矩阵（k ≥ 2），拟合 k 元正态混合分布
（结构参数 `beta` 为 `k x k` 协方差矩阵，默认单位阵）。它走自带的
L-BFGS-B 支撑点搜索（与 R 包相同，即 yixuan/LBFGSpp 的 L-BFGS-B），
不复用一维引擎；R 包将其 2-D 版本标注为*实验性、可能很慢*。k = 2
——以 `method="npnorm2Dll"`（R 名，API 兼容）调用——在本构建下仍是
相对 R 最慢的族（n=300：≈ 0.91 s vs ≈ 0.8 s，同一数据——手写展开的
2×2 密度内核、0 除法显式逆目标函数、逐格热启动与真方向导数把旧
~28× 差距收窄到 ~1.2×，残余差距在逐格 L-BFGS-B 求解器机制），见
[`docs/PERF.md`](docs/PERF.md)。k = 2 轨迹与 0.2.3 位级一致，密度
内核与 R 逐位一致，拟合确定性地收敛到有效局部最优，且密度与 R
参考在容忍范围内一致（k = 2 门：rel-L1 0.0158、max |log diff|
0.169，见 `tests/verify_2d_same.py`）。更高 k 更慢：搜索网格是各轴
marginal 网格的张量积（k = 3：每轴 18 点 ⇒ 4,913 格，n = 360 时
≈ 2.2 s）——见 §6。

**计算架构：** `Python → C++/Eigen`——单一 pybind11 扩展
`npfixedcomppy._core`。整个计算栈（引擎、支撑点求解、约束非负最小二乘、
梯度与权重扫描、核矩阵填充、混合密度）全部在 C++/Eigen 中完成，其中
**Eigen 5.0.0** 以 header-only 形式 vendored 在 `eigen/` 下。**SIMD 级别
在构建期决定**（默认 `/arch:AVX2`，可用环境变量 `NPFIC_ARCH` 覆盖——见
§3）。无手写 OpenMP：`setup.py` 在构建期用编译器的 OpenMP 参数做一次
探测编译，支持时把该参数加入扩展构建，从而启用 Eigen 自身的编译期并行
GEMM/GEMV（不支持则为串行 Eigen）；对固定构建与线程数，相同输入恒产生
逐位相同的输出（并行平台上的 R 包做不到）。验收契约（0.2.3 起放宽，
适用于所有族）：拟合密度须与 R 参考在容忍范围内一致，且估计的 `ll`
不差于历史版本——求解器不再逐位匹配 R 的浮点轨迹（0.2.3 的支撑点
搜索放宽，见上文 "What's new in 0.2.3"，落在类似的、往往略优的局部
最优上），由 §5 的 parity 门保证。构建内确定性不变：相同输入恒产生
逐位相同的输出。细节与实测证据见 [`docs/PERF.md`](docs/PERF.md)。

## 2. What's new in 0.3.0 (vs 0.2.3)

**The 2-D family `npnorm2Dll` is generalized to N dimensions —
`npnormND` — and the R-identical `NPFIC_2D_EXACT` objective path is
removed.**

* **New family `npnormND` (k ≥ 2)** — the k-variate normal mixture,
  `computemixdist(X, method="npnormND", beta=...)` with `X` of shape
  `(n, k)` and `beta` the `k x k` covariance matrix (default the
  identity):

  * `method="npnorm2Dll"` is kept as the **k = 2 alias** (the R name,
    API compatibility); the k = 2 result is **bit-identical** to
    0.2.3 (ll 848.7472072261477 / 10 iters / 5 components on the
    reference data; ≈ 0.91 s wall).
  * Kernel: the general `dnormNDarray` (Eigen `LLT`) for k > 2;
    k = 2 keeps the hand-unrolled 2×2 fast path, so the 2-D
    trajectory is untouched.
  * The default grid / initial mixing distribution generalize R's
    2-D construction: per-axis `gridpoints_npnorm` /
    `initial_npnorm` combined as a tensor product (each axis padded
    to the longest). Explicit `gridpoints` (an `m x k` matrix) and
    `mix` (`{"pt": (m, k), "pr": (m,)}`) are accepted for any k.
  * The per-cell L-BFGS-B search, the true directional derivative
    (`1/F_i`-weighted; `docs/PERF.md` §4a.4), the per-cell hot start,
    and the `grid_gain` certificate all carry over to k dimensions.

* **Removed: the `NPFIC_2D_EXACT=1` objective path** — the R-identical
  Cholesky kernel (0.2.1's trajectory, ≈ 4.3 s on the reference run).
  The relaxed contract (density similar within tolerance, `ll` no
  worse than the historical version) does not require bit-exact
  trajectories, and the general k-D kernel supersedes its Cholesky
  path; the k = 2 fast trajectory itself is bit-identical to 0.2.3, so
  no golden changed. (History and the A/B record: the 0.2.1 section
  below and `docs/PERF.md` §4a.)

* **New: the 1-D second-order crossing certificate**
  (`NPFIC_1D_CERT`, default on) — in the d1 support-point search, a
  sign-change interval whose *endpoint gains are both non-negative*
  is now skipped at zero cost when the family's curvature bound
  `M ≥ −inf a0″` proves the gain stays positive over the whole
  interval (the sweep's free endpoint values plus `M` bound the
  tangent-parabola lower bound; `NpNormLL` supplies
  `M = (1−pi0fixed)·2e^{−3/2}/√(2π)·Σfl/β³`, all other families
  never fire). Sound: a standalone numerical harness (two `npnormll`
  states, high-resolution re-sweep of every interval) reports zero
  bound violations and zero unsound skips; certificate on/off is
  bit-identical on the `npnormll` suite and `run_all_verify.py` is
  all green; measured `certskips=0` on the current benchmarks (no
  wall change there — the `½M h²` term dominates the endpoint gains
  on those datasets). Full record: `docs/PERF.md` §4d.

* **N-D performance** — the search grid is the tensor product of the
  per-axis marginals, (L−1)^k cells for L points per axis, so
  k ≥ 3 is combinatorially slower than k = 2: k = 3, n = 360,
  explicit 18-points-per-axis grid (17³ = 4,913 cells): ≈ 2.2 s
  (11 iters, 4 components, `ll` 1430.711238303976 —
  `tests/parity_3d.csv`). The default grid at k = 3 would be
  ≈ 103³ ≈ 1.1 M cells — pass a coarser explicit `gridpoints` for
  k ≥ 3 (see `docs/PERF.md` §4a).

* **Tests** — `tests/verify_2d_same.py` is now the density-similarity
  gate vs the R reference (rel-L1 < 0.02, max |log-density|
  difference < 0.25, main components within 0.70 in location, `ll`
  no worse than R, `grid_gain > −1`); new `tests/verify_3d.py` pins
  the 3-D trajectory (determinism, self-consistency recomputed with
  the public `dnpnormND` kernel, the `grid_gain` certificate, main
  components near the three known generator modes). All eleven parity
  suites report `TOTAL BAD: 0`.

Version sources bumped to `0.3.0` (`pyproject.toml` single source,
`setup.py` mirrored; the compiled extension reports `0.3.0`).

## 2. What's new in 0.2.3 (vs 0.2.2)

**Relaxed support search — the engine's default is now "find ANY valid
new support point", not "find the exact gain minimum"** (always on;
measured in [`docs/PERF.md`](docs/PERF.md) §4c). The 0.2.2 engine spent
most of its time refining the *exact* minimum of the gain inside every
sign-change interval (d1) / triple (d0) — but the caller's sign filter
accepts *any* negative point, so the exact minimum was never needed.
Three relaxations, each verified against the acceptance contract (fit
density within tolerance of the R reference; the estimated `ll` not
worse than the historical version):

1. **Zero-cost negative-grid acceptance** — the grid sweep already
   evaluates the gain at every grid point (free, from the cached grid
   kernel); a negative grid point is a valid new support point at zero
   additional evaluations.
2. **CNM working-set re-verification** — the previous call's refined
   root for an interval is re-verified with ONE gradient evaluation and
   accepted when still negative (the column-generation hot start of the
   CNM scheme; Wang 2007 — full references in §7).
   This was the experimental
   `NPFIC_WARM=2` arm in 0.2.2; under the relaxed acceptance contract it
   is now the default and **the `NPFIC_WARM` knob has been removed**.
3. **Negative-gain early stop in `brmin`/`dfmin`** — a candidate
   refinement returns as soon as any evaluated point has negative gain.

Measured effect (same machine, same data; median of 5):

| case | 0.2.2 | 0.2.3 | speedup |
|------|-------|-------|---------|
| `nptll` β=5, n=5000 | 2.95 s | **857 ms** | **3.5×** (vs R 49.7 s: **58×**) |
| `nptllw` β=5, n=5000 | 2.01 s | **560 ms** | **3.6×** (vs R: **52×**) |
| `npnormll`, n=1000 | 10.7 ms | 5.5 ms | 1.9× |
| `npnormllw`, n=5000 | 37.1 ms | 27.9 ms | 1.3× (vs R 120 ms: **4.3×**) |
| `npnormadw`, n=5000 | 387 ms | 240 ms | 1.6× |
| `npnormcll`, n=1000 | 1.01 s | 1.27 s | **0.8×** (richer support set: 92 → 107 points) |

The `npnormcll` row is the documented trade, both ways: on its flat
correlation landscape (6e-5-scale basins) the relaxation settles on a
slightly different (valid) support set — 4.8e-6 relative *better* than
0.2.2's fit but 5.5e-5 above R's from-scratch optimum (107 vs 92
points, 31 outer iterations) — which is why it is the one family where
0.2.3 is *slower* (0.8×). The KKT certificate (`min_gradient ≥ -1e-4`,
no negative direction) holds, and the parity gate checks it. All ten
parity suites report `TOTAL BAD: 0` on the re-recorded deterministic
trajectory (R's references remain the fit-quality band; the goldens
now pin *this build's* trajectory).

**New `Npmix` field: `grid_gain`** — the minimum gain over *all* grid
points at the final estimate (the grid-level certificate). Together
with `min_gradient` (the support directions) it certifies that no
negative direction exists inside the grid; `grid_gain` is
grid-resolution-dependent and can legitimately be slightly negative
(−8.2e-4 on `npnormcvm`, −2.8e-3 on `npnormadw` in the test data), so
it is reported for information, not gated — the exception being the
2-D family, where `tests/verify_2d_same.py` fails on `grid_gain < −1`
(a large negative means the objective ran on a corrupted dataset;
[`docs/PERF.md`](docs/PERF.md) §4a.3).

Version sources bumped to `0.2.3` (`pyproject.toml` single source,
`setup.py` mirrored; the compiled extension reports `0.2.3`). The
`NPFIXEDCOMPY_PROFILE` line is unchanged (`evals=` / `freshcols=` /
`freshms=`); on the current build the `nptll` β=5, n=5000 case runs
1191 gradient evaluations / 61 fresh off-grid columns (vs 3733 / 455
pre-relaxation) — see [`docs/PERF.md`](docs/PERF.md) §1.

*Historical note: this section describes the 0.2.3 build; the current
build is 0.3.0. The 0.2.3 1-D engine, goldens, and the numbers above
carry over unchanged to 0.3.0, which only generalized the 2-D family to
N-D (`npnormND`) and removed the `NPFIC_2D_EXACT` objective path (see
"What's new in 0.3.0" above). The "ten parity suites" count is the
0.2.3 set — `tests/verify_3d.py` joined it in 0.3.0, making eleven.*

### 0.3.0 更新内容（相对 0.2.3）

**2-D 族 `npnorm2Dll` 泛化为 N 维——新族 `npnormND`——并移除 R 同款的
`NPFIC_2D_EXACT` 目标函数路径。**

* **新族 `npnormND`（k ≥ 2）**——k 元正态混合：
  `computemixdist(X, method="npnormND", beta=...)`，`X` 为 `(n, k)`，
  `beta` 为 `k x k` 协方差矩阵（默认单位阵）：

  * `method="npnorm2Dll"` 保留为 **k = 2 别名**（R 名，API 兼容）；
    k = 2 的结果与 0.2.3 **位级一致**（参考数据上
    ll 848.7472072261477 / 10 次迭代 / 5 成分；≈ 0.91 s wall）。
  * 核：k > 2 走通用 `dnormNDarray`（Eigen `LLT`）；k = 2 保留手写
    展开的 2×2 快路径，2-D 轨迹不受影响。
  * 默认网格/初始混合分布是 R 2-D 构造的 N-D 推广：每轴
    `gridpoints_npnorm` / `initial_npnorm` 取张量积（各轴补齐到最长
    轴）。任意 k 都接受显式 `gridpoints`（`m x k` 矩阵）与 `mix`
    （`{"pt": (m, k), "pr": (m,)}`）。
  * 逐格 L-BFGS-B 搜索、真方向导数（`1/F_i` 加权；
    `docs/PERF.md` §4a.4）、逐格热启动与 `grid_gain` 证书全部推广到
    k 维。

* **移除：`NPFIC_2D_EXACT=1` 目标函数路径**——R 同款 Cholesky 内核
  （0.2.1 的轨迹，参考运行 ≈ 4.3 s）。放宽后的验收契约（密度在容忍
  范围内相似、`ll` 不差于历史版本）不要求逐位轨迹一致，且通用 k-D
  核已取代其 Cholesky 路径；k = 2 的 fast 轨迹本身与 0.2.3 位级一致，
  故金标无变化。（历史与 A/B 记录见下文 0.2.1 节与
  `docs/PERF.md` §4a。）

* **新增：1-D 二阶穿 0 证书**（`NPFIC_1D_CERT`，默认常开）——d1
  支撑点搜索中，*两端增益均非负*的变号区间，在族的曲率界
  `M ≥ −inf a0″` 证明整区间 gain 保持为正时零成本跳过（扫描免费
  带出的端点值 + `M` 钉住切线抛物线下界；`NpNormLL` 提供
  `M = (1−pi0fixed)·2e^{−3/2}/√(2π)·Σfl/β³`，其余族永不触发）。
  Sound：独立数值 harness（两个 npnormll 状态、每个区间高分辨率
  重扫）零违例、零误跳；证书开/关在 npnormll 套件上位级一致、
  `run_all_verify.py` 全绿；当前基准上实测 `certskips=0`（wall
  无变化——那些数据集上 `½M h²` 项压过端点增益）。完整记录：
  `docs/PERF.md` §4d。

* **N-D 性能**——搜索网格是各轴 marginal 的张量积，每轴 L 点即
  (L−1)^k 格，所以 k ≥ 3 相对 k = 2 组合式变慢：k = 3、n = 360、
  显式 18 点/轴网格（17³ = 4,913 格）≈ 2.2 s（11 次迭代、4 成分、
  `ll` 1430.711238303976——`tests/parity_3d.csv`）。k = 3 的默认
  网格约为 103³ ≈ 110 万格——k ≥ 3 请显式传较粗的 `gridpoints`
  （见 `docs/PERF.md` §4a）。

* **测试**——`tests/verify_2d_same.py` 改为对 R 参考的密度相似度门
  （rel-L1 < 0.02、max |log-density| 差 < 0.25、主成分位置差 0.70
  内、`ll` 不劣于 R、`grid_gain > −1`）；新增 `tests/verify_3d.py`
  固定 3-D 轨迹（确定性、用公开 `dnpnormND` 核重算的自洽、
  `grid_gain` 证书、主成分贴近三个已知生成器模态）。全部 11 个
  parity 套件报告 `TOTAL BAD: 0`。

版本源升到 `0.3.0`（`pyproject.toml` 为唯一源、`setup.py` 同步；
编译扩展报告 `0.3.0`）。

### 0.2.3 更新内容（相对 0.2.2）

**支撑点搜索放宽——引擎默认改为“找到**任意**有效新支撑点”，而不是
“找到增益的精确最小值”**（默认常开；实测见
[`docs/PERF.md`](docs/PERF.md) §4c）。0.2.2 引擎把大部分时间花在逐个
精修每个变号区间（d1）/三元组（d0）内增益的**精确最小值**上——但调用
方的符号过滤接受**任意**负增益点，精确最小值从来不是必需的。三项放宽
均已按验收契约验证（拟合密度与 R 参考在容忍范围内一致；估计的 `ll`
不差于历史版本）：

1. **零成本负增益网格点接受** —— 网格扫描本就会在缓存的网格核上求出
   每个网格点的增益（免费）；负增益的网格点即有效新支撑点，无需任何
   额外求值。
2. **CNM 工作集再校验** —— 对每个区间，用一次梯度求值再校验上一次
   调用精化出的根，仍为负则直接接受（CNM 列生成的热启动；Wang 2007
   ——完整引用见 §7）。这就是 0.2.2 里实验性的
   `NPFIC_WARM=2` 臂；
   在放宽后的验收契约下它成为默认，**`NPFIC_WARM` 旋钮已移除**。
3. **`brmin`/`dfmin` 内负增益早停** —— 候选点精修一旦遇到负增益点
   立即返回。

实测效果（同机、同数据；5 次取中位数）：

| 用例 | 0.2.2 | 0.2.3 | 加速 |
|------|-------|-------|------|
| `nptll` β=5，n=5000 | 2.95 s | **857 ms** | **3.5×**（相对 R 49.7 s：**58×**） |
| `nptllw` β=5，n=5000 | 2.01 s | **560 ms** | **3.6×**（相对 R：**52×**） |
| `npnormll`，n=1000 | 10.7 ms | 5.5 ms | 1.9× |
| `npnormllw`，n=5000 | 37.1 ms | 27.9 ms | 1.3×（相对 R 120 ms：**4.3×**） |
| `npnormadw`，n=5000 | 387 ms | 240 ms | 1.6× |
| `npnormcll`，n=1000 | 1.01 s | 1.27 s | **0.8×**（支撑集更丰富：92 → 107 点） |

`npnormcll` 一行是**双向记录的代价**：在它平坦的相关系数景观（6e-5
量级的 basin）上，放宽搜索落在一个略有不同（有效）的支撑集上——比
0.2.2 的拟合在相对误差 4.8e-6 意义上*更好*，但比 R 从零拟合的最优
差 5.5e-5（107 vs 92 点、31 次外层迭代）——这也是它成为 0.2.3 中
唯一*变慢*（0.8×）的族的原因。KKT 证书（`min_gradient ≥ -1e-4`，
无负方向）成立，parity 门对其设门。全部 10 个 parity 套件在重录的
确定性轨迹上报告 `TOTAL BAD: 0`（R 参考保留为拟合质量带宽；金标现固定
的是**本构建自己的**轨迹）。

**`Npmix` 新字段：`grid_gain`** —— 最终估计下**全部网格点**上的最小
增益（网格级证书）。它与 `min_gradient`（支撑方向）一起证明网格内不
存在负方向；`grid_gain` 依赖网格分辨率，可能合法地略为负值（测试数据
上 `npnormcvm` −8.2e-4、`npnormadw` −2.8e-3），因此只作信息性输出，
不设门。

版本源升到 `0.2.3`（`pyproject.toml` 为唯一源、`setup.py` 同步；
编译扩展报告 `0.2.3`）。`NPFIXEDCOMPY_PROFILE` 行不变（`evals=` /
`freshcols=` / `freshms=`）；当前构建下 `nptll` β=5、n=5000 为 1191
次梯度求值 / 61 个 fresh 离网格核列（放宽前为 3733 / 455）——见
[`docs/PERF.md`](docs/PERF.md) §1。

*历史注记：本节描述 0.2.3 构建；当前构建为 0.3.0。0.2.3 的一维引擎、
金标与上文数字在 0.3.0 中不变地沿用，0.3.0 仅把 2-D 族泛化为 N-D
（`npnormND`）并移除了 `NPFIC_2D_EXACT` 目标函数路径（见上文
"0.3.0 更新内容"）。"10 个 parity 套件"是 0.2.3 时的集合——
`tests/verify_3d.py` 在 0.3.0 加入后为 11 个。*

## 2. What's new in 0.2.2 (vs 0.2.1)

**New: `NPFIC_WARM` — the support-point hot start, A/B-measured (default off).**
The grid is fixed for the solver's life, and every outer iteration
refines the *same* sign-change intervals (d1) / triples (d0) against a
slightly different gradient — so the root refined by the previous call
is an excellent starting point for the next one. The knob (read per
call) has two experimental arms:

* `NPFIC_WARM=1` (**seed**, quality-preserving): the previous root is
  passed as the first interior point of `brmin`/`dfmin`; the search
  still converges on the NEW gradient, so the support set is
  unchanged and only the kernel-column evaluation count drops. A/B at
  n=5000 with identical data/init/grid/tol (`tests/bench_warm_ab.py`):
  1.05–1.23× across the six 1-D families (e.g. `npnormcll`
  3221 → 2622 ms, `nptll` β=5 2957 → 2736 ms); Δll ≤ 5e-6 (the
  outer `tol` scale) and mixture densities match to relative ≤ 9e-6.
* `NPFIC_WARM=2` (**aggressive**, CNM working-set re-verification —
  A/B reference only): the previous root is re-verified with ONE
  gradient evaluation and accepted when still negative (the
  column-generation working-set hot start of the CNM scheme; Wang
  2007). Faster (up to 2.39× on `npnormcll`) but
  it can land on a *worse* optimum where the support roots drift
  between iterations (`npnormll`: Δll +4.2e-2, density off by
  relative 4.9e-3), violating the "the estimate's ll must not be
  worse than the historical version" acceptance rule — kept as an
  experimental arm, not a default.

The default stays `0` — the shipped, bit-identical behaviour. Full A/B
tables and the quality argument: [`docs/PERF.md`](docs/PERF.md) §4b.

**New: `freshcols` / `freshms` on the `NPFIXEDCOMPY_PROFILE` line.**
Beyond the `solvegrad` evaluation count, the line now reports the
number of off-grid kernel columns that had to be evaluated for the
first time inside the loop and the wall time of that work (the part
the section-3 column cache cannot serve). This quantified the `d0`
bottleneck — e.g. on the current build, 2304 of the 2334 ms of
`nptll` β=5, n=5000 is fresh columns (455 of them) — and motivated
the hot start above.

Version sources bumped to `0.2.2` (`pyproject.toml` single source,
`setup.py` mirrored; the compiled extension reports `0.2.2`). All ten
parity suites report `TOTAL BAD: 0` on the current build (the default
path is bit-identical to 0.2.1).

*Historical note: this section describes the 0.2.2 build; the current
build is 0.3.0, which superseded `NPFIC_WARM` (see above) and
re-recorded the trajectory goldens.*

### 0.2.2 更新内容（相对 0.2.1）

**新增：`NPFIC_WARM`——经 A/B 实测的支撑点热启动（默认关）。** 网格在
整个求解器生命周期内固定，外层每轮对**同一批**变号区间（d1）/三元组
（d0）用略不同的梯度再精修——因此上一次调用精化出的根就是下一次调用
的优秀起点。该旋钮（每次调用读取）有两个实验臂：

- `NPFIC_WARM=1`（**seed**，质量保持）：把上次的根作为 `brmin`/`dfmin`
  的首个内点传入；搜索仍对新梯度收敛，支撑集不变，只减少核列求值
  次数。n=5000、同数据/初值/网格/tol 的 A/B（`tests/bench_warm_ab.py`）：
  六个一维族 1.05–1.23×（如 `npnormcll` 3221 → 2622 ms、`nptll` β=5
  2957 → 2736 ms）；Δll ≤ 5e-6（外层 `tol` 量级），混合密度相对差
  ≤ 9e-6。
- `NPFIC_WARM=2`（**aggressive**，CNM 工作集再校验——仅 A/B 参考）：
  用一次梯度求值再校验上次的根，仍为负则直接接受（CNM 列生成的工作集
  热启动；Wang 2007）。更快（`npnormcll` 最高
  2.39×），但在支撑根漂移的族（如 `npnormll`：Δll +4.2e-2、密度相对
  差 4.9e-3）可能落在**更差**的最优，违反"ll 不差于历史版本"的验收
  规则——仅作实验臂保留，不作默认。

默认保持 `0`——shipped 逐位一致行为。完整 A/B 表与质量论证见
[`docs/PERF.md`](docs/PERF.md) §4b。

**新增：`NPFIXEDCOMPY_PROFILE` 行的 `freshcols` / `freshms`。** 除
`solvegrad` 求值次数外，现在报告循环内必须新求值的离网格核列数与该
工作耗时（列缓存覆盖不到的部分）。这把 `d0` 瓶颈量化了——当前构建下
`nptll` β=5、n=5000 的 2334 ms 中 2304 ms 是 fresh 列（455 列）——并
构成了上面热启动的动机。

版本源升到 `0.2.2`（`pyproject.toml` 为唯一源、`setup.py` 同步；
编译扩展报告 `0.2.2`）。当前构建上全部 10 个 parity 套件报告
`TOTAL BAD: 0`（默认路径与 0.2.1 逐位一致）。

*历史注记：本节描述 0.2.2 构建；当前构建为 0.3.0，已取代
`NPFIC_WARM`（见上）并重录了轨迹金标。*

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
* `dnpnormND` (the k-D normal mixture density, k ≥ 2) and
  `dnormNDarray` (the `(n, k)` multivariate-normal pdf kernel, log
  form available).

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

**`npnorm2Dll` objective fast path (new default), per-cell hot start,
the `grid_gain` certificate, and the true directional derivative**
(fast path: 0.2.3; the hot start, the certificate, the column-major
data-read fix, and the true directional derivative below: since the
0.2.3 tag, in the current build) — with the contract relaxed from
*bit-exact match to R* to *faster, with ll/mixture similar to R's
result*, the per-cell L-BFGS-B objective was re-derived for n = 2:

* **Fast path (default)**: the quadratic form uses the explicit 2×2
  inverse of `beta` (0 divisions per point vs 4 for the Cholesky-solve
  kernel), the per-evaluation invariants (`1/(dens+precompute)`, the
  component scale) are cached once per `solvegrad` instead of on every
  objective evaluation (≈ 10⁸ divisions per fit), and the two
  `n`-length temporaries are preallocated buffers — while the
  exponentiation stays the same Eigen SIMD `exp` as the R-matching
  kernel. Reference run (n=300): ≈ 10.2 → ≈ 4.4 µs per objective
  evaluation (≈ 2.3×), wall 4.3 s → 0.95 s (≈ 5.7× → ≈ 1.3× vs R);
  second n=300 benchmark: ≈ 10.4 → ≈ 5.5 µs, wall 8.0 s → 1.7 s
  (≈ 4.6×).
* **Per-cell hot start + negative-gain early stop.** The previous
  `solvegrad`'s converged point per grid cell is reused as the next
  call's L-BFGS-B start when it is still inside the cell, and the
  `f_stop = 0` early stop makes a still-negative warm root — or a
  negative grid point — cost exactly one evaluation. The accepted-cell
  work drops 421,810 → 216,955 objective evaluations on the reference
  run (1.9×) and 769,533 → 316,253 on the second benchmark (2.4×),
  jointly with the true directional derivative (details in
  [`docs/PERF.md`](docs/PERF.md) §4a.3).
* **True directional derivative (current build).** The FAST path's
  gradient used to be the gradient of the *unnormalised* KDE —
  proportional to the true gain gradient only when
  `1/(dens+precompute)` is ~constant over the data. It now accumulates
  the weighted sum `Σ_i (t_i/F_i)(μ − x_i)` (≈ 3 extra FLOPs/point,
  values already hot), so L-BFGS-B descends the actual objective; the
  objective *value* arithmetic stays bit-identical and the
  `NPFIC_2D_EXACT=1` trajectory (then the only other path) was
  bit-identical to the pre-change build. The reference fit moves to ll
  848.7472072261477 (10 iters), *closer* to the exact path (|Δll| vs
  it: 6.4e-4 → 7.5e-5); `LBFGSB.h` gains a stale-curvature restart
  guard (reset the BFGS history, take steepest descent when
  `grad·d >= 0`) that never fires on the exact path
  ([`docs/PERF.md`](docs/PERF.md) §4a.4).
* **Similarity, not identity**: on the reference data the exact path
  converges in 6 iters and the fast in 10, both to a 5-component
  mixture whose main components agree within ≈ 0.007 in location
  (ll 848.7472 fast vs 848.7473 exact vs 848.888 R, |Δll| ≈ 7.5e-5);
  on the second benchmark exact 9 / fast 14 iters, both 7-component
  (main components within ≈ 0.027 in location, |Δll| ≈ 1.7e-3, both
  valid and better than R's 1.1.0003 result of 986.236). The parity
  gate (`tests/verify_2d_same.py`) now checks *similarity*:
  determinism per path, iter count, ll bands, kernel parity vs R
  (still bit-exact), main-component proximity between the two paths,
  and the grid-level certificate below.
* **The `grid_gain` certificate caught a real bug.** The FAST
  objective as of the 0.2.3 build read the `(n, 2)` data **row-major**
  while Eigen stores it **column-major** — it optimized the objective
  on a corrupted dataset (consecutive x-values paired as `(x, y)`). The
  certificate reported by `get_ans` — the minimum gain over all
  (G−1)² cell midpoints at the final mixture — exposed it: ≈ −518 with
  3,586 negative cells on the reference fit (the correct value is
  ≈ +7e-7 with 0 negative cells), while the family's similarity band
  (`dmu < 0.08`) still *passed* because the corrupted mixture's main
  components landed within ≈ 0.06 of the exact path's. The fix reads
  the two columns directly (`data_.col(0)` / `data_.col(1)`), and the
  parity gate now fails on `grid_gain < −1` (post-fix: +4.7e-3 fast,
  +2.7e-7 exact on the reference fit; an independent numpy
  recomputation matches the C++ value to < 1e-13).
* **R-identical escape hatch**: `NPFIC_2D_EXACT=1` selected the
  original Cholesky-based kernel — the bit-exact 0.2.1 trajectory (ll
  848.7472820349071 on the reference run). **Removed in 0.3.0** with
  the N-D generalization ("What's new in 0.3.0").
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

**1-D performance: grid-fill memoization + per-solve invariant caching**
(the numbers in §6/§4 of this README are re-measured on the current
build; `tests/reprofile_phases.py` re-runs the `docs/PERF.md` §1 phase
table):

* **Binned normal families — grid-fill memoization.** The binned
  trapezoid/cdf fills are now memoized on the exact `mu` vector
  (`detail::KernelMemo`): the per-iteration grid sweep (the dominant
  binned cost) and the small support-set fills are served from the
  precomputed grid fill instead of being refilled at every candidate
  point, and a served matrix is bit-identical to a fresh fill. This is
  what takes `npnormllw`, n=5000 from ≈ 131 ms to ≈ 37 ms wall (now
  ~3× **faster** than R's ≈ 120 ms) and `estpi0(npnormllw)` from
  ≈ 563 ms to ≈ 84 ms. `npnormadw` improves less — its `collapse`
  re-weighting churns the support sets — and still trails R; see the
  corrected note in [`docs/PERF.md`](docs/PERF.md) §3.
* **All 1-D families — per-solve invariant caching.** The `dens` vector
  is fixed for the whole support-point search of each `solvegrad`, so
  the per-family invariant arrays every `gradfun`/`gradfunvec` call
  used to recompute (`1/(dens+precompute)` and its dot product, the AD
  `s1` / `Σ w2/(1-fl)` terms, the CVM difference array, the
  count-weighted LL variants) are now built ONCE per `solvegrad` by a
  per-family `prepare_solve` and reused by every candidate evaluation
  — same loop, same accumulation order, so a hit is bit-identical; a
  `dens` mismatch rebuilds on the spot (only the end-of-fit `finish()`
  path). Measured solvegrad deltas: `nptll` β=5, n=5000 2200 →
  2102 ms; `npnormcll`, n=1000 459 → 467 ms (unchanged within noise —
  its cost is the NNLS, not the sweep); the small cases are within ~5 %.
* **`dnt` constants hoisted.** The non-central-t pdf's `df`-level
  constants (`log df`, `√((df+2)/df)`, the `gammln` pair) are computed
  once per run into `stats::DntConst`; the cached `dnt_c` substitutes
  them and is bit-identical to `dnt`.
* **`NPFIC_REFINE_STEPS` — experimental A/B knob, default unchanged.**
  Caps the refinement steps each candidate runs inside `brmin`/`dfmin`
  ("less work per step, the outer loop then needs more iterations").
  Measured A/B at n=30000, identical data/init/grid/tol
  (`tests/refine_ab.py`, `tests/refine_ab_evals.py`): no arm beats the
  shipped behaviour uniformly — the 2-iteration-converging `nptll` case
  gains only ~1–7 % wall at bit-identical ll, while `npnormll` at
  cap=2 needs 60 iterations (599 ms vs 271 ms shipped) and cap=1
  drifts off the trajectory. The default stays `-1` (unlimited, the
  shipped behaviour); the knob is kept for further experiments. The
  `NPFIXEDCOMPY_PROFILE` line now also reports the `solvegrad`
  evaluation count (`evals=`).

All ten parity suites report `TOTAL BAD: 0` on the current build.

*Historical note: this section describes the 0.2.1 build; the current
build is 0.3.0. The 2-D family introduced here was named
`npnorm2Dll`; 0.3.0 generalized it to N dimensions as `npnormND`
(keeping `npnorm2Dll` as the k = 2 alias) and removed the
`NPFIC_2D_EXACT` objective path. The "ten parity suites" count is the
0.2.1 set — `tests/verify_3d.py` joined it in 0.3.0, making eleven.*

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
* `dnpnormND`（k 元正态混合密度，k ≥ 2）与 `dnormNDarray`（`(n, k)`
  多元正态 pdf 核，可选对数形式）。

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

**`npnorm2Dll` 目标函数快路径（新默认）+ 逐格热启动 + `grid_gain`
证书 + 真方向导数**（快路径：0.2.3；热启动、证书、下述列主序数据读取
修复与真方向导数：0.2.3 tag 之后、当前构建）—— 契约从*与 R 逐位一致*
放宽为*更快、且 ll/混合与 R 结果类似*后，逐格 L-BFGS-B 目标函数为
n = 2 重新推导：

* **快路径（默认）**：二次型改用 `beta` 的显式 2×2 逆（每点 0 次除法，
  Cholesky 求解内核为 4 次）；每次求值不变量（`1/(dens+precompute)`、
  分量缩放）在每次 `solvegrad` 缓存一次而非逐次求值重算（每次拟合省
  ≈ 10⁸ 次除法）；两个长度为 n 的临时向量改为预分配缓冲——指数化仍与
  R 同款内核使用同一条 Eigen SIMD `exp` 指令序列。参考运行（n=300）：
  单次求值 ≈ 10.2 → ≈ 4.4 µs（≈ 2.3×），wall 4.3 s → 0.95 s（相对 R
  ≈ 5.7× → ≈ 1.3×）；第二个 n=300 基准上 ≈ 10.4 → ≈ 5.5 µs，
  wall 8.0 s → 1.7 s（≈ 4.6×）。
* **逐格热启动 + 负增益早停**：上一轮 `solvegrad` 每格收敛点若仍在
  格内，则作为本轮 L-BFGS-B 起点；`f_stop = 0` 早停使仍为负的温热根
  ——或负网格点——只花一次求值。被接受格的求值数（与真方向导数合计）
  参考运行 421,810 → 216,955（1.9×）、第二个基准 769,533 →
  316,253（2.4×）（详见 [`docs/PERF.md`](docs/PERF.md) §4a.3）。
* **真方向导数（当前构建）**：FAST 路径的梯度过去是*未归一化* KDE
  的梯度——只有当 `1/(dens+precompute)` 在数据上近似为常数时，它才与
  真 gain 梯度成比例。现改为累加带权求和 `Σ_i (t_i/F_i)(μ − x_i)`
  （每点约 3 个额外 FLOP，值本已热），L-BFGS-B 真正沿目标函数下降；
  目标函数*值*的算术保持位级一致，`NPFIC_2D_EXACT=1` 轨迹（当时唯一
  的另一条路径）与改动前构建逐位一致。参考拟合移到
  ll 848.7472072261477（10 次迭代），与 exact 路径更近（|Δll|：
  6.4e-4 → 7.5e-5）；`LBFGSB.h` 新增过期曲率重启守卫（`grad·d >= 0`
  时重置 BFGS 历史、改走最速下降），在 exact 路径上从不触发
  （[`docs/PERF.md`](docs/PERF.md) §4a.4）。
* **类似而非相同**：参考数据上 exact 路径 6 次迭代、fast 路径 10 次
  迭代，均收敛到 5 成分混合，主成分位置差 ≈ 0.007 内（ll 848.7472
  fast vs 848.7473 exact vs 848.888 R，|Δll| ≈ 7.5e-5）；第二个基准
  上 exact 9 / fast 14 次迭代，均 7 成分（主成分位置差 ≈ 0.027 内、
  |Δll| ≈ 1.7e-3，均为有效最优、优于 R 1.1.0003 的 986.236）。
  parity 门（`tests/verify_2d_same.py`）现在检查的是*类似性*：每路径
  确定性、迭代数、ll 带宽、与 R 的**核**对拍（仍逐位一致）、两路径
  主成分接近度，以及下面的格级证书。
* **`grid_gain` 证书抓到一个真 bug**：0.2.3 构建的 FAST 目标函数按
  **行主序**读 `(n, 2)` 数据（Eigen 实际是**列主序**）——等于在*损坏的
  数据集*上优化目标（相邻 x 值被配成 `(x, y)` 坐标）。`get_ans` 报告的
  证书（最终混合下全部 (G−1)² 格中心的最小 gain）暴露了它：参考拟合
  ≈ −518、3,586 个负格（正确值 ≈ +7e-7、0 负格），而族的 similarity
  门（`dmu < 0.08`）却仍然*通过*——损坏混合的主成分落在 exact 路径
  ≈ 0.06 内。修复为直接读两列（`data_.col(0)` / `data_.col(1)`），
  parity 门现在在 `grid_gain < −1` 时失败（修复后参考拟合：fast
  +4.7e-3、exact +2.7e-7；独立 numpy 重算与 C++ 值 < 1e-13 吻合）。
* **R 同款逃生门**：`NPFIC_2D_EXACT=1` 曾选用原 Cholesky 内核——即
  bit-exact 的 0.2.1 轨迹（参考运行 ll 848.7472820349071）。
  **0.3.0 随 N-D 泛化移除**（见 "What's new in 0.3.0"）。
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

**一维性能：网格填充 memo + 每次求解的不变量缓存**（本 README §6/§4
的数字已在当前构建上重新实测；`tests/reprofile_phases.py` 可重跑
`docs/PERF.md` §1 的 phase 表）：

* **分箱正态族——网格填充 memo。** 分箱梯形/分布函数填充现按 `mu`
  向量精确键控做 memo（`detail::KernelMemo`）：每轮网格扫描（分箱族的
  主要开销）与小编撑集填充直接服用预备算好的网格填充，不再在每个候选
  点重填，服用的矩阵与新算逐位一致。`npnormllw`，n=5000 的 wall 从
  ≈ 131 ms 降到 ≈ 37 ms（现在**快于** R 的 ≈ 120 ms 约 3 倍），
  `estpi0(npnormllw)` 从 ≈ 563 ms 降到 ≈ 84 ms；`npnormadw` 改善较小
  （其 `collapse` 重加权使支撑集不断变化），仍落后于 R——修正后的
  说明见 [`docs/PERF.md`](docs/PERF.md) §3。
* **所有一维族——每次求解的不变量缓存。** `solvegrad` 的整个支撑点
  搜索期间 `dens` 固定，于是各族每次 `gradfun`/`gradfunvec` 调用原来
  都要重算的不变量数组（`1/(dens+precompute)` 及其点积、AD 的 `s1` /
  `Σ w2/(1-fl)` 项、CVM 的差值数组、带计数的 LL 变体）现在由各族的
  `prepare_solve` 在每次 `solvegrad` 只建一次，被每个候选点评估复用
  ——循环与累加顺序不变，命中即逐位一致；`dens` 不匹配时现场重算
  （只有拟合末尾的 `finish()` 路径）。实测 solvegrad 变化：
  `nptll` β=5，n=5000 为 2200 → 2102 ms；`npnormcll`，n=1000 为
  459 → 467 ms（噪声内不变——其开销在 NNLS 而非扫描）；小用例在
  ~5 % 以内。
* **`dnt` 常量上提。** 非中心 t 密度函数的 `df` 级常量（`log df`、
  `√((df+2)/df)`、`gammln` 差）每轮运行只算一次存入
  `stats::DntConst`；带缓存的 `dnt_c` 代入它们，与 `dnt` 逐位一致。
* **`NPFIC_REFINE_STEPS`——实验性 A/B 旋钮，默认不变。** 限制每个候选
  点在 `brmin`/`dfmin` 内走的精修步数（“每步少干点活，外层多迭代几
  轮”）。n=30000 上同数据/初值/网格/tol 的 A/B 实测
  （`tests/refine_ab.py`、`tests/refine_ab_evals.py`）：没有任何臂统一
  优于 shipped 行为——2 轮即收敛的 `nptll` 只省 ~1–7 % wall 且 ll
  逐位相同，而 `npnormll` 在 cap=2 需要 60 轮（599 ms vs shipped
  271 ms）、cap=1 轨迹漂移。默认保持 `-1`（不限制，即 shipped 行为）；
  旋钮保留供后续实验。`NPFIXEDCOMPY_PROFILE` 行现在还会打印
  `solvegrad` 的求值次数（`evals=`）。

当前构建上全部 10 个 parity 套件报告 `TOTAL BAD: 0`。

*历史注记：本节描述 0.2.1 构建；当前构建为 0.3.0。本节的 2-D 族当时
名为 `npnorm2Dll`；0.3.0 把它泛化为 N 维的 `npnormND`（保留
`npnorm2Dll` 为 k = 2 别名）并移除了 `NPFIC_2D_EXACT` 目标函数路径。
"10 个 parity 套件"是 0.2.1 时的集合——`tests/verify_3d.py` 在 0.3.0
加入后为 11 个。*

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
| `family`       | `"npnorm"` / `"npt"` / `"npnormc"` / `"nppois"` / `"npnorm2D"` (k = 2) / `"npnormND"` (k > 2) |
| `min_gradient` | min gradient w.r.t. a new support point (≤ 0 at convergence)   |
| `grid_gain`    | min gain over all grid points (1-D) / cell midpoints (N-D normal family) at the final estimate (grid-level certificate; informational — see "What's new in 0.2.3") |
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
| `dnpnormND(x, mu0, pi0, sigma, lg=False)` | k-D normal mixture pdf (k ≥ 2) — `npnormND` kernel; k = 2 is R `npnorm2Dll` (R `dnpnormND`) |
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
| `family`       | `"npnorm"` / `"npt"` / `"npnormc"` / `"nppois"` / `"npnorm2D"`（k = 2）/ `"npnormND"`（k > 2） |
| `min_gradient` | 对新支撑点的最小梯度（收敛解处 ≤ 0）                         |
| `grid_gain`    | 最终估计下全部网格点（一维）/格中心（N-D 正态族）的最小增益（网格级证书；信息性输出——见 "What's new in 0.2.3"） |
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
| `dnpnormND(x, mu0, pi0, sigma, lg=False)` | k 元正态混合 pdf（k ≥ 2）—— `npnormND` 核；k = 2 即 R `npnorm2Dll`（R `dnpnormND`） |
| `dnormNDarray(x, mu0, sigma, lg=False)` | `(n, k)` 多元正态 pdf 核（R `dnormNDarray_`） |

它们精确复现 R 的原始代码路径——包括 R 的怪癖（`dnppois` 的 `lg`
路径对额外支撑点加入正态项；`dnpnormc` 在点质量支撑点处为 NaN）——
在 30 例金标集上（`tests/verify_kernels.py`）相对误差 ≤ 1.2e-15
（对照 R 4.6.1 实测值）。

## 5. Testing & parity with R

Run from the package root with the venv's Python (script-style, each
exits non-zero on failure):

- `python tests\verify_npnormll.py` — normal MLE: **GOLD** checks
  (ll/pt/pr vs this build's recorded deterministic trajectory,
  deterministic re-run) plus tolerance **BAND** checks where R itself
  is run-to-run non-deterministic. Since 0.2.3's relaxed support
  search the goldens pin *this build's* trajectory; R's references
  remain the fit-quality band.
- `python tests\verify_nptll.py` — t-family MLE (finite and infinite
  degrees of freedom, fixed components, estpi0); goldens re-recorded
  for the 0.2.3 trajectory, R's values are the fit-quality band.
- `python tests\verify_cvmadcll.py` — Cramér–von Mises, Anderson–Darling
  and correlation families incl. estpi0: recorded-trajectory GOLD for
  CLL plus the KKT certificate (`min_gradient ≥ -1e-4`; `grid_gain`
  informational) and BAND where R is non-deterministic (CVM/AD).
- `python tests\verify_pois.py` — Poisson MLE (CM + estpi0) on a shared
  R-generated data file; goldens re-recorded for the 0.2.3 trajectory,
  R's value the fit-quality band.
- `python tests\verify_density.py` — recomputed-density invariants
  (`ll == -sum log d(x_i | returned mixture)`) and estpi0 threshold
  invariants, all families.
- `python tests\verify_binned.py` — the four binned ("`...w`") families:
  recorded-trajectory GOLD (ll to 1e-9, pt/pr at 1e-6/1e-4,
  deterministic re-run) where R is deterministic (LLW / TLLW /
  LLW_FIX), BAND where R is non-deterministic (CVMW / ADW).
- `python tests\verify_coveb.py` — `covestEB` / `covestEB_cor` against
  R-recorded intermediates (projection primitive strict; the
  end-to-end pipeline at 5e-1, because 0.2.3's relaxed search moves
  the inner npnormcll fit to a different valid support set and the
  posterior mean amplifies the support-point drift — the R-exact
  pipeline is strictly gated at 1e-10 in pipeline E).
- `python tests\verify_posteriormean.py` — posterior means of `fun(pt)`
  against R-recorded values (strict 1e-9 on the pure kernel path; the
  refit-based cases at 1e-2, the 0.2.3 relaxation moves the
  deterministic refit by ~2e-3 in a support point).
- `python tests\verify_kernels.py` — the 30-case 1-D/2-D **kernel**
  parity gate (`dnp*` / `pnp*` / `dnormNDarray` against live R 4.6.1
  gold values; finite values ≤ 1.2e-15 relative, NaNs at identical
  positions).
- `python tests\verify_2d_same.py` — the `npnorm2Dll` (k = 2) gate (a
  *similarity* gate, §4a of [`docs/PERF.md`](docs/PERF.md)): two-run
  determinism and convergence, the recorded ll, `ll` self-consistency,
  **kernel** parity (PY's `dnpnormND` at R's final points reproduces
  R's ll to 1e-9), a no-worse-than-R local-optimum check, and the
  density similarity vs R (rel-L1 < 0.02, max |log diff| < 0.25,
  main components within 0.70).
- `python tests\verify_3d.py` — the 3-D `npnormND` gate: two-run
  determinism, convergence, the recorded ll, `ll` self-consistency
  (recomputed with the public `dnpnormND` kernel), shape/family
  checks, the `grid_gain` certificate, and main components near the
  three known generator modes (no R reference exists for k = 3).
- `python tests\perf_baseline.py` — wall-time baseline of the public API.
- `python tests\bench_binned.py` — wall-time bench for the binned
  families (the R-side counterpart is `tests\bench_binned_r.R`).

All eleven parity suites must report `TOTAL BAD: 0`. The package is
**deterministic per build**: for a fixed build and a fixed
`OMP_NUM_THREADS`, identical inputs produce bit-identical outputs (unlike
the R package on parallel platforms).

### 测试与 R 包一致性

在包根目录下用 venv 的 Python 运行（脚本式，失败时退出码非零）：

- `python tests\verify_npnormll.py` — 正态极大似然：**GOLD** 检查
  （ll/pt/pr 对比本构建重录的确定性轨迹、确定性重跑）+ R 本身逐次
  非确定处的容差 **BAND** 检查。0.2.3 的支撑点搜索放宽之后，金标固定
  的是*本构建的*轨迹；R 参考保留为拟合质量带宽。
- `python tests\verify_nptll.py` — t 族极大似然（有限/无限自由度、固定
  分量、estpi0）；金标按 0.2.3 轨迹重录，R 值保留为拟合质量带宽。
- `python tests\verify_cvmadcll.py` — Cramér–von Mises、Anderson–Darling
  与相关族（含 estpi0）：CLL 用重录轨迹 GOLD + KKT 证书
  （`min_gradient ≥ -1e-4`；`grid_gain` 信息性），R 非确定（CVM/AD）
  用 BAND。
- `python tests\verify_pois.py` — 泊松极大似然（CM + estpi0），共读
  R 生成的数据文件；金标按 0.2.3 轨迹重录，R 值保留为拟合质量带宽。
- `python tests\verify_density.py` — 重算密度不变量
  （`ll == -sum log d(x_i | 返回的混合分布)`）与 estpi0 阈值不变量，
  全部族。
- `python tests\verify_binned.py` — 四个分箱（``"...w"``）族：R 确定的
  用例（LLW / TLLW / LLW_FIX）用重录轨迹 GOLD（ll 1e-9，pt/pr
  1e-6/1e-4，确定性重跑），R 非确定的用例（CVMW / ADW）用 BAND。
- `python tests\verify_coveb.py` — `covestEB` / `covestEB_cor` 对比
  R 录制的中间量（投影基元严格门；端到端管线 5e-1——0.2.3 的放宽
  搜索把内层 npnormcll 拟合挪到另一个有效支撑集，后验均值放大了
  支撑点漂移；R 精确管线在 pipeline E 以 1e-10 严格把关）。
- `python tests\verify_posteriormean.py` — `fun(pt)` 的后验均值对比
  R 录制的值（纯核路径严格 1e-9；基于 refit 的用例 1e-2——0.2.3
  的放宽使确定性 refit 的一个支撑点移动 ~2e-3）。
- `python tests\verify_kernels.py` — 30 例一维/二维**核**对拍门
  （`dnp*` / `pnp*` / `dnormNDarray` 对比 R 4.6.1 实测金标值；有限值
  相对误差 ≤ 1.2e-15，NaN 位置一致）。
- `python tests\verify_2d_same.py` — `npnorm2Dll`（k = 2）门（*类似性*
  门，见 [`docs/PERF.md`](docs/PERF.md) §4a）：双次运行确定性与收敛、
  录制的 ll、`ll` 自洽、**核**对拍（PY 的 `dnpnormND` 在 R 最终点上
  复现 R 的 ll 至 1e-9）、局部最优不劣于 R，以及密度相似度
  （rel-L1 < 0.02、max |log diff| < 0.25、主成分位置差 < 0.70）。
- `python tests\verify_3d.py` — 3-D `npnormND` 门：双次运行确定性、
  收敛、录制的 ll、`ll` 自洽（用公开的 `dnpnormND` 核重算）、shape/
  family 校验、`grid_gain` 证书、主成分贴近三个已知的生成器模态
  （k = 3 无 R 参考）。
- `python tests\perf_baseline.py` — 公开 API 的耗时基线。
- `python tests\bench_binned.py` — 分箱族耗时 bench（R 侧对应脚本为
  `tests\bench_binned_r.R`）。

十一个 parity 套件必须全部报告 `TOTAL BAD: 0`。本包对**给定构建与线程数**
是确定性的：相同输入恒产生逐位相同的输出（并行平台上的 R 包做不到这一
点）。

## 6. Performance

The whole solver is C++/Eigen (SIMD level decided at build time,
`/arch:AVX2` by default; Eigen's own parallel GEMM/GEMV is enabled when
the build's OpenMP probe passes); the kernel column cache — each kernel
column `K[:, mu]` computed once per fit and reused by the mapping,
gradient, weight, and collapse passes — plus the 0.2.3 relaxed
support search (see "What's new in 0.2.3" above) are the main
speed-ups. Typical wall time on this build (median of 5, warmup
excluded):

| case | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormll")`, n=1000 | ≈ 5.5 ms | ≈ 19 ms |
| `computemixdist(x, method="nptll", beta=inf)`, n=1000 | ≈ 17.5 ms | ≈ 211 ms |
| `computemixdist(x, method="nptll", beta=5)`, n=5000 | ≈ 857 ms | ≈ 50 s |
| `computemixdist(x, method="npnormcll", beta=1000)`, n=1000 | ≈ 1.27 s | ≈ 1.9 s |
| `computemixdist(x, method="npnormad")`, n=1000 | ≈ 37 ms | ≈ 59 ms |
| `computemixdist(x, method="nppoisll")`, n=1000 | ≈ 0.45 ms | ≈ 5 ms |
| `estpi0(x, method="npnormll")`, n=1000 | ≈ 14 ms | ≈ 97 ms |
| `computemixdist(X, method="npnorm2Dll")`, n=300 (2-D) | ≈ 0.91 s | ≈ 0.8 s |
| `computemixdist(X, method="npnormND")`, n=360 (3-D) | ≈ 2.2 s (18 pts/axis grid) | — (no R counterpart) |

Binned (`order = -3`, i.e. `h = 10^-3`):

| case | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormllw")`, n=5000 | ≈ 28 ms | ≈ 120 ms |
| `computemixdist(x, method="npnormadw")`, n=5000 | ≈ 240 ms * | ≈ 170 ms * |
| `computemixdist(x, method="nptllw")`, n=5000 | ≈ 112 ms | ≈ 1.97 s |
| `computemixdist(x, method="nptllw", beta=5)`, n=5000 | ≈ 560 ms | ≈ 29.3 s |
| `estpi0(x, method="npnormllw")`, n=5000 | ≈ 53 ms | ≈ 1.33 s |

\* `npnormadw` fits wander between basins run-to-run in both
implementations, so this ratio is indicative.

The k = 2 row above is the documented *exception*: `npnorm2Dll` (the R
name; in 0.3.0 the k = 2 alias of `npnormND`) is an experimental
family whose per-outer-iteration support-point search launches one
L-BFGS-B problem **per grid cell** (103 × 103 for the default 104 × 104
grid) and, unlike the 1-D engine, it does not use the kernel column
cache (evaluated and rejected — `docs/PERF.md` §4a.3: the line search
evaluates arbitrary interior points, so a per-cell column cache would
cover only a small fraction of the evaluations). Four changes cut the
old ~28× gap to ~1.2×: (i) a hand-unrolled 2×2 density-kernel fast
path (cutting the original `dec.solve(d)` port from ≈ 44 µs/call to
≈ 6.5 µs/call at n=300 for `dnormNDarray`, ≈ 7.4 µs for `dnpnormND`
— the old gap's main source, since MSVC keeps the per-point heap
temporaries that the R build's GCC eliminates); (ii) an objective fast
path with an explicit-inverse quadratic form (0 divisions/point),
per-`solvegrad` invariant caching, and preallocated buffers — the
0.2.x A/B record is ≈ 10.2 → ≈ 4.2 µs per evaluation on the reference
run (4.3 s → 0.95 s end-to-end; the 0.3.0 build measures ≈ 0.91 s);
(iii) a per-cell hot start + negative-gain early stop (the previous
`solvegrad`'s converged point per cell as the next call's L-BFGS-B
start; a still-negative warm root or a negative grid point costs one
evaluation); and (iv) the true directional derivative of the gain
objective (the R-ported unweighted form is the gradient of the
*unnormalised* KDE, aligned with the gain gradient only when 1/F is
~constant), which moves the trajectory to a closer, cheaper optimum
(|Δll| vs the removed R-identical path: 6.4e-4 → 7.5e-5 on the
reference run) behind a stale-curvature restart guard in `LBFGSB.h`.
(iii) and (iv) together drop the evaluation count 421,810 (the
R-identical path, 0.2.x) → 216,955 on the reference run and
769,533 → 316,253 on the second n=300 benchmark. The density kernels
remain bit-exact with R and the k = 2 trajectory is bit-identical to
0.2.3; the fit's density agrees with the R reference within
tolerance (rel-L1 0.0158, max |log diff| 0.169, main components
within 0.675 — `tests/verify_2d_same.py`). See
[`docs/PERF.md`](docs/PERF.md) for the breakdown.

N-D scaling: the search grid is the tensor product of the per-axis
marginals — (L−1)^k cells for L points per axis — so higher k is
combinatorially slower: k = 3 with an explicit 18-points-per-axis grid
(17³ = 4,913 cells) runs in ≈ 2.2 s at n = 360, while the default
grid (≈ 104 points per axis) would be 103³ ≈ 1.1 M cells — pass a
coarser explicit `gridpoints` for k ≥ 3 (`tests/verify_3d.py` uses
18 per axis).

For the `t` family with small degrees of freedom the cost grows with `n`
(super-linearly — the per-point `dnt` kernel has no cheap identity), so
very large `n` is slow; the normal families scale linearly. The binned
`npnormllw` family runs ~3× faster than R's binned normal (the grid
sweep reads the precomputed grid fill instead of refilling the
trapezoid at every candidate point; see the note in
[`docs/PERF.md`](docs/PERF.md) §3), while `npnormadw` still trails R
(its `collapse` re-weighting keeps the support sets churning, so the
fill memoization helps less); the binned `t` family — whose kernel is a
cheap CDF difference — is ~8–14× faster. See
[`docs/PERF.md`](docs/PERF.md) for the phase profiles, the kernel column
cache, and the full comparison with R.

Environment knobs: `NPFIXEDCOMPY_PROFILE=1` prints a per-phase timing
line (solvegrad / mapping / loss / weights / collapse, plus the
`solvegrad` evaluation count and the fresh-column count/time) to
stderr; `NPFIC_REFINE_STEPS` caps the
per-candidate refinement steps inside `brmin`/`dfmin` (experimental,
default `-1` = unlimited = shipped behaviour, see What's new);
`OMP_NUM_THREADS` sizes Eigen's pool (when the build has OpenMP)
without changing the results. `NPFIC_1D_CERT` (0.3.0, default on)
gates the 1-D second-order crossing certificate — d1 sign-change
intervals whose gain is proven positive by the family's curvature
bound skip the CNM check and `brmin` entirely (`NPFIC_1D_CERT=0`
restores the pre-0.3.0 loop; the PROFILE line adds a `certskips=`
counter — [`docs/PERF.md`](docs/PERF.md) §4d). The 0.2.2
`NPFIC_WARM` knob (support-point hot start) was **removed** in 0.2.3
— its `=2` CNM re-verification arm is now the always-on engine
behaviour ("What's new in 0.2.3",
[`docs/PERF.md`](docs/PERF.md) §4c).

### 性能

整个求解器为 C++/Eigen（SIMD 级别在构建期决定，默认 `/arch:AVX2`；构建期
OpenMP 探测通过时启用 Eigen 自身的并行 GEMM/GEMV）；内核列缓存——每个内核
列 `K[:, mu]` 每次拟合只计算一次，供 mapping、梯度、权重、collapse 各环节
复用——加上 0.2.3 的支撑点搜索放宽（见上文 "What's new in 0.2.3"），是
主要的加速手段。本构建典型耗时（5 次取中位数，排除预热）：

| 用例 | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormll")`，n=1000 | ≈ 5.5 ms | ≈ 19 ms |
| `computemixdist(x, method="nptll", beta=inf)`，n=1000 | ≈ 17.5 ms | ≈ 211 ms |
| `computemixdist(x, method="nptll", beta=5)`，n=5000 | ≈ 857 ms | ≈ 50 s |
| `computemixdist(x, method="npnormcll", beta=1000)`，n=1000 | ≈ 1.27 s | ≈ 1.9 s |
| `computemixdist(x, method="npnormad")`，n=1000 | ≈ 37 ms | ≈ 59 ms |
| `computemixdist(x, method="nppoisll")`，n=1000 | ≈ 0.45 ms | ≈ 5 ms |
| `estpi0(x, method="npnormll")`，n=1000 | ≈ 14 ms | ≈ 97 ms |
| `computemixdist(X, method="npnorm2Dll")`，n=300（二维） | ≈ 0.91 s | ≈ 0.8 s |
| `computemixdist(X, method="npnormND")`，n=360（三维） | ≈ 2.2 s（18 点/轴网格） | —（无 R 对应） |

分箱（`order = -3`，即 `h = 10^-3`）：

| 用例 | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormllw")`，n=5000 | ≈ 28 ms | ≈ 120 ms |
| `computemixdist(x, method="npnormadw")`，n=5000 | ≈ 240 ms * | ≈ 170 ms * |
| `computemixdist(x, method="nptllw")`，n=5000 | ≈ 112 ms | ≈ 1.97 s |
| `computemixdist(x, method="nptllw", beta=5)`，n=5000 | ≈ 560 ms | ≈ 29.3 s |
| `estpi0(x, method="npnormllw")`，n=5000 | ≈ 53 ms | ≈ 1.33 s |

\* `npnormadw` 在两侧实现中都存在逐次运行的 basin 漂移，该比值仅作参考。

上表 k = 2 一行是**有记录的例外**：`npnorm2Dll`（R 名；0.3.0 起为
`npnormND` 的 k = 2 别名）是实验性族，其每轮外层迭代的支撑点搜索对
**每个网格单元**各启动一个 L-BFGS-B 问题（默认 104 × 104 网格的
103 × 103），且与一维引擎不同，它不使用内核列缓存（已评估并否决——
`docs/PERF.md` §4a.3：线搜索会求值任意内部点，逐格列缓存只能覆盖很小
比例的求值）。四处改动把旧 ~28× 差距收窄到 ~1.2×：(i) 手写展开的 2×2
密度内核快路径（把最初逐行照搬的 `dec.solve(d)` 从 ≈ 44 µs/次降到
≈ 6.5 µs/次，n=300 的 `dnormNDarray`；`dnpnormND` ≈ 7.4 µs/次——旧
差距的主要来源：MSVC 保留了逐点堆临时对象，而 R 构建的 GCC 会消除
它们）；(ii) 目标函数快路径——显式逆矩阵二次型（每点 0 次除法）、每次
`solvegrad` 的不变量缓存、预分配缓冲——0.2.x A/B 记录：参考运行上单次
求值 ≈ 10.2 → ≈ 4.2 µs（端到端 4.3 s → 0.95 s；0.3.0 构建实测
≈ 0.91 s）；(iii) 逐格热启动 + 负增益早停（上一轮 `solvegrad` 每格
收敛点作为本轮 L-BFGS-B 起点；仍为负的温热根或负网格点只花一次求值）；
(iv) gain 目标函数的真方向导数（R 移植的无权重形式是*未归一化* KDE 的
梯度，只有 1/F 近似为常数时才与 gain 梯度对齐），把轨迹移到更近更便宜
的极小（|Δll| 相对已移除的 R 同款路径：6.4e-4 → 7.5e-5，参考运行），
其背后是 `LBFGSB.h` 中的过期曲率重启守卫。(iii) 与 (iv) 合计把求值数
从 R 同款路径（0.2.x）的 421,810 降到参考运行的 216,955、第二个 n=300
基准从 769,533 降到 316,253。密度核本身仍与 R 逐位一致，k = 2 轨迹与
0.2.3 位级一致；拟合密度与 R 参考在容忍范围内一致（rel-L1 0.0158、
max |log diff| 0.169、主成分位置差 0.675 内，`tests/verify_2d_same.py`）。
分解见 [`docs/PERF.md`](docs/PERF.md)。

N-D 扩展：搜索网格是各轴 marginal 的张量积——每轴 L 点即 (L−1)^k 格——
所以 k 更高时组合式变慢：k = 3 用显式 18 点/轴网格（17³ = 4,913 格）
在 n = 360 时 ≈ 2.2 s；而默认网格（每轴 ≈ 104 点）是 103³ ≈ 110 万
格——k ≥ 3 建议显式传较粗的 `gridpoints`（`tests/verify_3d.py` 用每
轴 18 点）。

t 族在小自由度下 `dnt` 核没有廉价恒等式，成本随 `n` 超线性增长，`n` 很
大时较慢；正态族为线性。分箱 `npnormllw` 族比 R 的分箱正态快约 3 倍
（网格扫描直接读取预备算好的网格填充，而不是在每个候选点重填梯形；见
[`docs/PERF.md`](docs/PERF.md) §3 的说明），`npnormadw` 仍落后于 R
（其 `collapse` 重加权使支撑集不断变化，填充 memo 帮助较小）；分箱
t 族的核是廉价的 CDF 差值，快约 8–14 倍。分阶段 profile、内核列缓存及
与 R 的完整对比见 [`docs/PERF.md`](docs/PERF.md)。

环境旋钮：`NPFIXEDCOMPY_PROFILE=1` 向 stderr 输出分阶段计时
（solvegrad / mapping / loss / weights / collapse，外加 `solvegrad`
的求值次数与 fresh 核列数/耗时）；`NPFIC_REFINE_STEPS` 限制
`brmin`/`dfmin` 内每个候选点的精修步数（实验性，默认 `-1` = 不限制
= shipped 行为，见 What's new）；`OMP_NUM_THREADS` 调整 Eigen 线程池
大小（OpenMP 构建时），不改变结果。0.2.2 的 `NPFIC_WARM` 旋钮
（支撑点热启动）已在 0.2.3 **移除**——其 `=2` CNM 再校验臂现为引擎
默认行为（"What's new in 0.2.3"、[`docs/PERF.md`](docs/PERF.md) §4c）。
`NPFIC_1D_CERT`（0.3.0，默认常开）门控 1-D 二阶穿 0 证书：把 d1
变号区间中可证明增益为正者整体跳过 CNM/`brmin` 工作
（`NPFIC_1D_CERT=0` 回到 0.3.0 之前的循环；PROFILE 行新增
`certskips=` 计数——[`docs/PERF.md`](docs/PERF.md) §4d）。

## 7. References

The algorithm references (full citations; the short forms elsewhere
in this file and in `cpp/` point here):

1. **Wang (2007)** — the support-point (column-generation) search
   for the non-parametric MLE of a mixing distribution, including the
   CNM working-set hot start: Y. Wang, "On Fast Computation of the
   Non-Parametric Maximum Likelihood Estimate of a Mixing
   Distribution", *Journal of the Royal Statistical Society, Series B*
   69(2), 185–198, 2007. DOI: 10.1111/j.1467-9868.2007.00583.x.
2. **Xue & Wang (2023)** — the null-proportion / density
   simultaneous estimation that `estpi0` implements: X. Xue, Y. Wang,
   "A Nonparametric Mixture Approach to Density and Null Proportion
   Estimation in Large-Scale Multiple Comparison Problems",
   *Australasian Journal of Statistics* 65(1), 49–75, 2023.
   DOI: 10.1111/anzs.12383.

The fixed-component (`mu0`/`pi0`) mechanism is the same extension the
R package `npfixedcomp2` builds on top of Wang (2007); an earlier
draft of this list cited a Wang & Taylor (2013) fixed-component paper
for it, but that item could not be verified against Crossref and has
been withdrawn.

### §7 中文注释

本包的算法文献（完整出处；本文件与 `cpp/` 源码中的短引用均指向
本节）：

1. **Wang (2007)** —— 非参数混合分布极大似然估计的支撑点（列生成）
   搜索，含 CNM 工作集热启动：Y. Wang, "On Fast Computation of the
   Non-Parametric Maximum Likelihood Estimate of a Mixing
   Distribution", *Journal of the Royal Statistical Society, Series B*
   69(2), 185–198, 2007. DOI: 10.1111/j.1467-9868.2007.00583.x。
2. **Xue & Wang (2023)** —— 空假设比例与密度同时估计（`estpi0`
   实现的机制）：X. Xue, Y. Wang, "A Nonparametric Mixture Approach
   to Density and Null Proportion Estimation in Large-Scale Multiple
   Comparison Problems", *Australasian Journal of Statistics*
   65(1), 49–75, 2023. DOI: 10.1111/anzs.12383。

固定分量（`mu0`/`pi0`）机制即 R 包 `npfixedcomp2` 在 Wang (2007)
之上的同一扩展；早期稿曾引一篇 Wang & Taylor (2013) 的固定分量
论文作其出处，经 Crossref 多路核实无法确认该文存在，已撤除。
