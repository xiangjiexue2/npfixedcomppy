# npfixedcomppy

A Rust + Python reimplementation of the R package
[`npfixedcomp2`](https://CRAN) for computing **non-parametric mixing
distribution** estimates for several parametric families, with support for

- estimating the mixing distribution with some components **fixed**;
- estimating the mixing distribution **together with the proportion of
  point mass at zero** (`estpi0`);
- multiple **loss functions** (maximum likelihood, Cramér–von Mises
  distance, Anderson–Darling distance).

All numerically heavy work (initial mixing distribution, grid points,
density / CDF / mapping / gradient evaluation, the constrained non-negative
least-squares subproblem, and the support-point solvers) is implemented in
Rust; the Python package is a thin, ergonomic front-end.

The algorithms follow Wang (2007) and the extensions in the `npfixedcomp2`
R package. The un-binned families are currently supported:

| `method`     | family    | loss function                | structural parameter `beta`            |
|--------------|-----------|------------------------------|----------------------------------------|
| `npnormll`   | normal    | maximum likelihood           | scale (default `1`)                    |
| `npnormcvm`  | normal    | Cramér–von Mises distance    | scale (default `1`)                    |
| `npnormad`   | normal    | Anderson–Darling distance    | scale (default `1`)                    |
| `nptll`      | t         | maximum likelihood           | degrees of freedom (default `inf`)     |
| `npnormcll`  | 1-param normal | maximum likelihood       | number of observations (user supplied) |
| `nppoisll`   | poisson   | maximum likelihood           | none (unused)                          |

The binned ("`...w`", `order = -k`) variants from the R package are **not
ported yet**.

## 中文说明

`npfixedcomppy` 是 R 包 [`npfixedcomp2`](https://CRAN) 的 **Rust + Python**
重写版本，用于计算若干参数族的**非参数混合分布**估计，支持：

- 带**固定分量**的混合分布估计；
- 同时估计**零处点质量比例**（`estpi0`）；
- 多种**损失函数**（极大似然、Cramér–von Mises 距离、Anderson–Darling
  距离）。

所有数值计算密集部分（初始混合分布、网格点、密度/CDF/mapping/梯度
求值、约束非负最小二乘子问题、支撑点求解器）均用 Rust 实现；Python 包
只是一个轻量、易用、与 R 接口兼容的前端。

算法遵循 Wang (2007) 及 `npfixedcomp2` 的扩展。目前已支持的
（非分箱）族：

| `method`     | 族       | 损失函数                     | 结构参数 `beta`                |
|--------------|----------|------------------------------|--------------------------------|
| `npnormll`   | 正态     | 极大似然                     | 尺度（默认 `1`）               |
| `npnormcvm`  | 正态     | Cramér–von Mises 距离        | 尺度（默认 `1`）               |
| `npnormad`   | 正态     | Anderson–Darling 距离        | 尺度（默认 `1`）               |
| `nptll`      | t        | 极大似然                     | 自由度（默认 `inf`）           |
| `npnormcll`  | 一参数正态 | 极大似然                   | 样本量（须用户提供）           |
| `nppoisll`   | 泊松     | 极大似然（计数数据）         | 无（不使用）                   |

R 包中的分箱（"`...w`"、`order = -k`）变体**尚未移植**。

## Installation

```bash
pip install -e .            # builds the Rust extension via maturin
```

A recent Rust toolchain (stable) and a C compiler for the platform are
required; `maturin` is pulled in by the build system.

## Quick start

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

# estimate the proportion at 0 simultaneously
res0 = estpi0(x, method="npnormll", val=2)
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

# 同时估计零处点质量比例
res0 = estpi0(x, method="npnormll", val=2)
```

## API

### `computemixdist(v, method="npnormll", mu0=None, pi0=None, beta=None,
mix=None, gridpoints=None, tol=1e-6, maxit=100, verbose=0)`

Estimates a discrete mixing distribution `G = sum_j pi_j Delta_{mu_j}` for
the kernel indexed by `method`, optionally with fixed components (support
points/weights that are included in the mixture but never updated), matching
the R `computemixdist` dispatcher.

- `mu0` / `pi0`: fixed components. Default `[0.0]` for both (a zero-weight
  degenerate fixed component at the origin, as in R). `sum(pi0) < 1` is
  required; the estimated components are scaled to the remaining mass.
- `beta`: structural parameter; required for `npnormcll` (sample size).
- `mix`: optional initial mixing `{"pt": [...], "pr": [...]}`; derived from
  a histogram of the data when omitted.
- `gridpoints`: optional candidate support points for the new-point search;
  derived from a histogram when omitted.
- Returns an `Npmix` with `pt`, `pr` (including the fixed components),
  `ll`, `min_gradient` (≤ 0 at a converged solution), `flag`, `iter` and
  `convergence` (0 = converged, 1 = hit `maxit`).

`estpi0(v, method, beta, val=2.0, mix=None, gridpoints=None, tol=1e-6,
verbose=0, fast=True, relax=False, inner_tol=1e-4)` additionally searches
for the point-mass weight at 0 such that the family's hypothesis statistic
reaches the threshold `val`. Recommended `val`: `2` for the likelihood
families, `0.1` for `npnormcvm`, `1` for `npnormad`. `fast=True` (default)
is several times faster while preserving the `tol` guarantee on the
statistic; `fast=False` reproduces the R/C++ refinement bit-for-bit. See the
module docstrings for the full parameter reference.

### `computemixdist` / `estpi0` 参数

- `mu0` / `pi0`：固定分量。默认均为 `[0.0]`（原点处零权重的退化固定分量，
  与 R 一致）。要求 `sum(pi0) < 1`；估计分量会按剩余质量
  `1 - sum(pi0)` 缩放。
- `beta`：结构参数；`npnormcll` 必须显式给出（样本量）。
- `mix`：可选的初始混合分布 `{"pt": [...], "pr": [...]}`；缺省时由数据
  直方图导出。
- `gridpoints`：可选的候选支撑点；缺省时由数据直方图导出。
- 返回值 `Npmix`：`pt`、`pr`（含固定分量）、`ll`、`min_gradient`
  （收敛解处 ≤ 0）、`flag`、`iter`、`convergence`（0 = 收敛，1 = 达到
  `maxit`）。

`estpi0` 额外搜索零处点质量权重，使该族的假设检验统计量达到阈值
`val`。推荐 `val`：似然族用 `2`，`npnormcvm` 用 `0.1`，`npnormad` 用
`1`。`fast=True`（默认）更快且保持统计量 `tol` 精度保证；`fast=False`
与 R/C++ 的细化过程逐位一致。完整参数说明见模块 docstring。

## Performance

All heavy work runs in Rust with a single `rayon` thread pool (parallel
over observations `n` only, serial and **bit-identical** below
`n = 2048`), and a per-fit **grid kernel cache** that turns each
iteration's full-grid gradient sweep into a dot product. At `n = 5000`
(single runs):

| case | wall time |
|------|-----------|
| `computemixdist(x, method="npnormll")` | ≈ 43 ms |
| `computemixdist(x, method="nptll", beta=inf)` | ≈ 82 ms |
| `computemixdist(x, method="nptll", beta=5)` | ≈ 630 ms |

For the `t` family with small degrees of freedom the cost grows with `n`
(super-linearly — the per-point `dnt` kernel has no cheap identity), so
very large `n` is slow; the normal families scale linearly. See
[`docs/PERF.md`](docs/PERF.md) for the full phase profile, the
parallelism rules, and the evidence for why a SIMD GEMM library (`faer`)
was *not* adopted.

Environment knobs: `NPFIXEDCOMPY_PAR_N` (serial threshold; set very high
to force the fully-serial, bit-identical path), `RAYON_NUM_THREADS`
(pool size).

### 性能

重计算全部在 Rust 中完成，只使用一个 `rayon` 线程池（仅对观测数 `n`
并行；`n < 2048` 时串行且**逐位一致**），并有按次拟合的**网格核缓存**：
每次迭代的全网格梯度扫描退化为点积。`n = 5000` 单次运行实测：

| 用例 | 耗时 |
|------|------|
| `computemixdist(x, method="npnormll")` | ≈ 43 ms |
| `computemixdist(x, method="nptll", beta=inf)` | ≈ 82 ms |
| `computemixdist(x, method="nptll", beta=5)` | ≈ 630 ms |

t 族在小自由度下 `dnt` 核没有廉价恒等式，成本随 `n` 超线性增长，`n` 很大
时较慢；正态族为线性。完整分阶段 profile、并行规则以及不采用 SIMD 矩阵
库（`faer`）的实测依据见 [`docs/PERF.md`](docs/PERF.md)。

环境旋钮：`NPFIXEDCOMPY_PAR_N`（串行阈值；设很大可强制完全串行、逐位
一致路径）、`RAYON_NUM_THREADS`（线程池大小）。

## Testing & parity with R

- `python -m pytest` — unit tests.
- `python tests\verify_npnormll.py` / `tests\verify_nptll.py` — parity
  gates against recorded R `npfixedcomp2` references: bit-level **GOLD**
  checks (ll/pt/pr, deterministic re-run) plus tolerance **BAND** checks
  where R itself is run-to-run non-deterministic (its Eigen/OpenMP
  reductions perturb the flat NPMLE surface).
- `python tests\verify_density.py` — recomputed-density invariants
  (`ll == -sum log d(x_i | returned mixture)`) and estpi0 threshold
  invariants.
- `python ..\test_fast_estpi0.py` — `fast`/`relax` vs legacy refinement
  bit-identical on all six families.

The package is **deterministic**: identical inputs always produce
bit-identical outputs (unlike the R package on parallel platforms).

### 测试与 R 包一致性

- `python -m pytest` — 单元测试。
- `python tests\verify_npnormll.py` / `tests\verify_nptll.py` — 针对
  录制的 R `npfixedcomp2` 参考值的一致性门：逐位 **GOLD** 检查（ll/pt/pr、
  确定性重跑）+ R 本身逐次非确定（Eigen/OpenMP 归约扰动平坦 NPMLE 面）
  处的容差 **BAND** 检查。
- `python tests\verify_density.py` — 重算密度不变量
  （`ll == -sum log d(x_i | 返回的混合分布)`）与 estpi0 阈值不变量。
- `python ..\test_fast_estpi0.py` — `fast`/`relax` 与 legacy 细化在全部
  六个族上逐位一致。

本包是**确定性**的：相同输入恒产生逐位相同的输出（并行平台上的 R 包
做不到这一点）。
