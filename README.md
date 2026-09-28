# npfixedcomppy

A Python reimplementation (C++/Eigen core) of the R package
[`npfixedcomp2`](https://CRAN) for computing **non-parametric mixing
distribution** estimates for several parametric families, with support for

- estimating the mixing distribution with some components **fixed**;
- estimating the mixing distribution **together with the proportion of
  point mass at zero** (`estpi0`);
- multiple **loss functions** (maximum likelihood, Cramér–von Mises
  distance, Anderson–Darling distance).

**Compute architecture:** `Python → C++/Eigen` (a single pybind11
extension, `npfixedcomppy._core`). The whole compute stack — engine,
support-point solvers, constrained NNLS, gradient and weight sweeps,
kernel-matrix fills, the mixture density — runs in C++/Eigen. The
**SIMD width is decided by the compiler at build time** (`/arch:AVX2` on
the build host, scalar fallback otherwise, same source) and the solver is
**fully serial: there is no OpenMP and no thread pool** — identical
inputs therefore always produce bit-identical outputs. A per-fit kernel
column cache (each kernel column is computed once and reused by every
consumer) is the main performance lever; details and measured evidence:
[`docs/PERF.md`](docs/PERF.md).

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

`npfixedcomppy` 是 R 包 [`npfixedcomp2`](https://CRAN) 的 **Python
（C++/Eigen 内核）**重写版本，用于计算若干参数族的**非参数混合分布**
估计，支持：

- 带**固定分量**的混合分布估计；
- 同时估计**零处点质量比例**（`estpi0`）；
- 多种**损失函数**（极大似然、Cramér–von Mises 距离、Anderson–Darling
  距离）。

**计算架构：** `Python → C++/Eigen`（单一 pybind11 扩展
`npfixedcomppy._core`）。整个计算栈——引擎、支撑点求解、约束非负最小
二乘、梯度与权重扫描、核矩阵填充、混合密度——全部在 C++/Eigen 中
完成。**SIMD 宽度由编译器在构建期决定**（构建主机上 `/arch:AVX2`，
不支持时退化为标量，同一份源码）；求解器为**完全串行：无 OpenMP、无线
程池**——因此相同输入恒产生逐位相同的输出。按拟合缓存内核列（每个内核
列只计算一次、所有消费者复用）是主要的性能手段；细节与实测证据见
[`docs/PERF.md`](docs/PERF.md)。

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
pip install -e .            # builds the C++/Eigen extension via setup.py + pybind11
```

The C++/Eigen core is compiled by `setup.py` with the platform C++
compiler (MSVC on Windows, GCC/Clang elsewhere) with `/O2 /arch:AVX2`
(override the SIMD level with `NPFIC_ARCH`, e.g. `NPFIC_ARCH=` for plain
x86-64), with Eigen vendored header-only under `eigen/`. `pybind11` is
required at build time. The build has no OpenMP and no Rust dependency.

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

The whole solver is serial C++/Eigen (SIMD decided at build time); the
kernel column cache — each kernel column `K[:, mu]` computed once per fit
and reused by the mapping, gradient, weight, and collapse passes — is the
main speed-up. Typical wall time on this build (median of 3):

| case | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormll")`, n=1000 | ≈ 8 ms | ≈ 15 ms |
| `computemixdist(x, method="nptll", beta=inf)`, n=1000 | ≈ 36 ms | ≈ 212 ms |
| `computemixdist(x, method="nptll", beta=5)`, n=5000 | ≈ 3.0 s | ≈ 50 s |
| `computemixdist(x, method="npnormcll", beta=1000)`, n=1000 | ≈ 1.0 s | ≈ 2.0 s |
| `computemixdist(x, method="npnormad")`, n=1000 | ≈ 37 ms | ≈ 38 ms |
| `computemixdist(x, method="nppoisll")`, n=1000 | ≈ 0.6 ms | ≈ 5 ms |

For the `t` family with small degrees of freedom the cost grows with `n`
(super-linearly — the per-point `dnt` kernel has no cheap identity), so
very large `n` is slow; the normal families scale linearly. See
[`docs/PERF.md`](docs/PERF.md) for the phase profiles, the kernel column
cache, and the comparison with R.

Environment knob: `NPFIXEDCOMPY_PROFILE=1` prints a per-phase timing
line (solvegrad / mapping / loss / weights / collapse) to stderr.

### 性能

整个求解器为串行 C++/Eigen（SIMD 在构建期决定）；内核列缓存——每个内核
列 `K[:, mu]` 每次拟合只计算一次，供 mapping、梯度、权重、collapse 各
环节复用——是主要的加速手段。本构建典型耗时（3 次取中位数）：

| 用例 | `npfixedcomppy` | R `npfixedcomp2` |
|------|-----------------|------------------|
| `computemixdist(x, method="npnormll")`，n=1000 | ≈ 8 ms | ≈ 15 ms |
| `computemixdist(x, method="nptll", beta=inf)`，n=1000 | ≈ 36 ms | ≈ 212 ms |
| `computemixdist(x, method="nptll", beta=5)`，n=5000 | ≈ 3.0 s | ≈ 50 s |
| `computemixdist(x, method="npnormcll", beta=1000)`，n=1000 | ≈ 1.0 s | ≈ 2.0 s |
| `computemixdist(x, method="npnormad")`，n=1000 | ≈ 37 ms | ≈ 38 ms |
| `computemixdist(x, method="nppoisll")`，n=1000 | ≈ 0.6 ms | ≈ 5 ms |

t 族在小自由度下 `dnt` 核没有廉价恒等式，成本随 `n` 超线性增长，`n` 很大
时较慢；正态族为线性。分阶段 profile、内核列缓存及与 R 的对比见
[`docs/PERF.md`](docs/PERF.md)。

环境旋钮：`NPFIXEDCOMPY_PROFILE=1` 向 stderr 输出分阶段计时
（solvegrad / mapping / loss / weights / collapse）。

## Testing & parity with R

Run from `tests/` with the venv's Python (script-style, each exits
non-zero on failure):

- `python tests\verify_npnormll.py` — normal MLE: bit-level **GOLD**
  checks (ll/pt/pr vs recorded R references, deterministic re-run) plus
  tolerance **BAND** checks where R itself is run-to-run non-deterministic.
- `python tests\verify_nptll.py` — t-family MLE (finite and infinite
  degrees of freedom, fixed components, estpi0), bit-level.
- `python tests\verify_cvmadcll.py` — Cramér–von Mises, Anderson–Darling
  and correlation families incl. estpi0: GOLD for the deterministic R
  cases (CLL), BAND where R is non-deterministic (CVM/AD).
- `python tests\verify_pois.py` — Poisson MLE (CM + estpi0) on a shared
  R-generated data file, bit-level.
- `python tests\verify_density.py` — recomputed-density invariants
  (`ll == -sum log d(x_i | returned mixture)`) and estpi0 threshold
  invariants, all six families.
- `python tests\perf_baseline.py` — wall-time baseline of the public API.

All five parity suites must report `TOTAL BAD: 0`. The package is
**deterministic**: identical inputs always produce bit-identical outputs
(unlike the R package on parallel platforms).

### 测试与 R 包一致性

用 venv 的 Python 在 `tests/` 下运行（脚本式，失败时退出码非零）：

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
  全部六个族。
- `python tests\perf_baseline.py` — 公开 API 的耗时基线。

五个 parity 套件必须全部报告 `TOTAL BAD: 0`。本包是**确定性**的：相同
输入恒产生逐位相同的输出（并行平台上的 R 包做不到这一点）。
