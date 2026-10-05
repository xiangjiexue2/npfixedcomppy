# solve 实现流程记录（npfixedcomppy C++/Eigen 引擎）

本文档记录 `computemixdist` / `estpi0` 的 C++ 求解实现的整体过程：
分层架构、外层主循环、支撑点搜索、权重子问题、collapse、`estpi0`
二分/插值、结果汇总，以及支撑这些过程的缓存层设计。代码位置以
`cpp/` 目录下的文件为准；本文是"大概过程"的速记，不是逐行注释。

## 1. 一句话概括

估计混合分布 = **外层列生成（column generation）循环**：

1. 当前有一组支撑点 `mu0` 与权重 `pi0`（加固定的 point-mass 分量）；
2. 在每个外层迭代里，先在**整个网格**上找"能继续降低损失"的新支撑点
   （一维梯度扫描 + 区间内精化），把找到的点**以权重 0 追加**进混合；
3. 在（扩大的）支撑点集上解一个**约束 NNLS 权重子问题**（非负、和为
   `1 - sum(pi0fixed)`），并用 Armijo 回溯线搜索保证损失下降；
4. **collapse**：合并过近的相邻支撑点（权重加权平均），删零权重点；
5. 损失改善 `< tol` → 收敛（`convergence = 0`）；超过 `maxit` →
   `convergence = 1`。

`estpi0` 在这个循环外面再包一层：对"零处 point-mass 权重 `sp`"做
二分 + 二次插值，使假设统计量 `hypofun(ll, minloss)` 命中目标值
`val`；每次二分都调用一次完整的 `computemixdist`（内层用
`tol = 1e-6, maxit = 100`）。

整个求解是确定性的：相同输入 → 逐位相同输出（与 R 包并行平台不同）。

## 2. 分层架构

| 层 | 文件 | 职责 |
|----|------|------|
| Python 前端 | `python/npfixedcomppy/npfc.py` | R 兼容的薄封装：参数校验、分箱族的预分箱（`h = 10**order`，向下取整，同 R `bin`）、默认网格/初始分布、`Npmix` 结果对象 |
| pybind11 绑定 | `cpp/npfc_py.cpp` | 每个族一个入口函数（`npnormll`、`nptll`、`npnormllw`、…），把 numpy 数组转 `std::vector`，构造 `MixSolver` + 族对象，调 `computemixdist` / `estpi0`，`finish()` 的结果转 dict |
| 引擎 | `cpp/npfc_engine.h` | `MixSolver`：外层主循环、`solvegrad`（支撑点搜索）、`brmin`/`dfmin`、`collapse`、`estpi0`/`estpi0_fast`、`finish`；混合适用工具 `sortmix`/`simplifymix`/`collapsemix`；`Family` 虚接口定义 |
| 一维族 | `cpp/npfc_families.h` | 11 个一维族的实现：`lossfunction`、`mapping`、`gradfun`/`gradfunvec`、`computeweights`、`prepare_solve`、`prepare`、per-solve 不变量缓存 |
| 多元正态族（N-D） | `cpp/npfc_famnd.h` | `npnormND`（R `npnorm2Dll` 的 N-D 泛化；k = 2 即 R 的双变量族，绑定名 `npnorm2Dll` 保留为别名）：自带一套外层循环（结构同上），权重子问题用 `LBFGSpp::LBFGSBSolver`（有界 LBFGS），核为 N-D 正态（k = 2 走手展开 2×2 Cholesky 快路径，k > 2 走 Eigen `LLT`；0.3.0 起无 R 同款 exact 路径） |
| 核函数 | `cpp/npfc_kernels.h` | 各核（正态、非中心 t 的 AS-243 级数、Poisson、相关系数、分箱梯形填充）+ `KernelColumnCache`（按 `mu` 缓存整列核值） |
| 网格/初始化 | `cpp/npfc_grid.h` | R `nspmix` 的移植：`whist` 直方图（diddle 规则）、`initial_npnorm`/`initial_nppois`（直方图式初始混合分布 + `disc` 排序归一）、`gridpoints_npnorm`（数据范围外扩 + 端点，默认 100 点）/`gridpoints_nppois`（sqrt 空间网格） |
| NNLS | `cpp/npfc_nnls.{h,cpp}` | Lawson–Hanson NNLS（`nnls`）；`pnnlssum`（追加"和为 1"行）；`pnnqp`（大数据 Gram 阵特征分解降维后回退 `pnnlssum`） |
| 有界 LBFGS | `cpp/LBFGSB.h` + `cpp/LBFGSpp/` | 仅多元正态族使用（Yixuan Qiu 的 LBFGSB 移植，MIT） |
| 统计/相关矩阵 | `cpp/npfc_stats.h`、`cpp/npfc_corrmatrix.h` | 特殊函数（`gammln`、`dnt` 常量 `DntConst` 等）、CLL 族相关矩阵工具 |

**线程模型**：包内没有手写 OpenMP 循环。`setup.py` 构建期探测编译器
是否支持 OpenMP flag；支持则加 flag，Eigen 自身的编译期门
（`EIGEN_HAS_OPENMP`）打开内部并行 GEMM/GEMV，不支持则为串行 Eigen。
`OMP_NUM_THREADS` 只调 Eigen 线程池大小，不改变结果。

## 3. 入口与预处理

以 `computemixdist(x, method="npnormll", beta, mu0fixed, pi0fixed,
initpt, initpr, gridpoints, tol, maxit)` 为例，一次调用的完整路径：

1. **Python 前端**（`npfc.py`）：
   - 校验/转换输入（1-D 有限值、`mu0fixed`/`pi0fixed` 合法性、
     `beta` 语义：`nptll` 的 `inf` = 退化为正态核、`npnormcll` 的
     `beta = n` 必须给定）；
   - 分箱族（`"...w"`）：先把观测预分箱到 `h = 10**order`（默认
     `order = -3`，向下取整），后续全部按分箱后的加权数据算；
   - 未给 `initpt`/`initpr` 时用 `whist` 直方图构造初始混合分布
     （`initial_npnorm`：直方图中点 + 密度权重，`disc` 排序归一）；
   - 未给 `gridpoints` 时用 `gridpoints_npnorm`（数据 min/max 外扩
     后取 100 点，端点即数据范围端点）。
2. **C++ 入口**（`npfc_py.cpp::npnormll`）：`vec_from` 拷贝为
   `std::vector<double>`，`std::make_unique<fam::NpNormLL>(data,
   mu0fixed, pi0fixed, beta)` 构造族对象（此时固定分量的密度
   `precompute` 尚未就绪），再构造 `MixSolver(fam, mu0fixed, pi0fixed,
   initpt, initpr, grid, verbose)`。
3. **`MixSolver` 构造函数**：
   - `std::sort` 网格点（R 在调 C++ 前先 `sort(gridpoints)`，这里
     忠实复刻）；
   - `warm_root_d1_` / `warm_root_d0_` 全置 NaN——上一轮调用精化出的
     根，按网格区间/三元组索引（CNM 工作集热启动的载体，0.2.3 起
     常开）；
   - `set_precompute()` → `fam_->set_fixed(mu0fixed, pi0fixed)`：族对象
     计算并缓存**固定分量的密度** `precompute`（整个 run 不变，
     只有 `estpi0` 二分改变 `pi0fixed` 时重算）；
   - `fam_->prepare(grid)`：一次性、按网格的预计算钩子。对贵核
     （`nptll` 的 AS-243 非中心 t、`npnormll` 的正态核）在这里把
     **整个 数据×网格 核矩阵**预算好——核只依赖数据、`beta`、网格，
     不依赖当前权重或固定分量，所以整个 run（包括 `estpi0` 的约
     15 次完整内层求解）都有效；此后每轮迭代的网格扫描退化为查表。

## 4. `Family` 虚接口（引擎与族的契约）

`MixSolver` 只认 `Family`（`npfc_engine.h`），每个族实现：

- `lossfunction(maps)`：给定混合密度（不含固定分量；族内部加回
  `precompute`）的损失值。似然族为负对数似然；CVM/AD 族为距离加
  `extrafun()` 里的常数项。
- `mapping(mu0, pi0)`：当前混合在数据上的密度（不含固定分量）。
- `gradfun(mu, dens, ctx, d0, d1, a0, a1)` / `gradfunvec(mu, …)`：损失
  对新支撑点 `mu` 的梯度。`a0` = 概率方向导数（"新增一个支撑点、
  给它单位概率"的收益 = **gain**；`a0 < 0` 即该点能降低损失），
  `a1` = 支撑点方向导数（仅 `d1` 族有，供 `brmin` 用）。
- `prepare_solve(dens)`：每次 `solvegrad` 开头调用一次，把搜索中每个
  候选点都会重算的不变量（如 `fl[i] = 1/(dens[i] + pre[i])`）缓存起来
  （0.2.2 的方案 A；`dens` 指针不匹配时原地重建，算术与顺序逐位一致）。
- `computeweights(mu0, pi0, dens)`：权重子问题（§7）。
- `precompute()`：固定分量密度；`set_fixed(…)`：更新之。
- `hypofun(ll, minloss)`：`estpi0` 的统计量（似然族 = `ll - minloss`；
  CVM/AD = 距离的单调变换）。
- `familydensity(x, mu0, pi0)`：族（未混合）密度在单点 `x` 的值
  （`estpi0` 的初始 `sp0` 用）。
- `prepare(grid)` / `flag()`（`"d0"` 或 `"d1"`）/ `kernel_fresh()`
  （PROFILE 行的 fresh 核列计数）等。

## 5. 外层主循环（`MixSolver::computemixdist`）

```
mu0 = initpt_;  pi0 = initpr_ * (1 - sum(pi0fixed_))   # 权重缩放到剩余概率
dens = mapping(mu0, pi0);  closs = loss(dens);  iter = 0
while (true):
    newpoints = solvegrad(dens, tol)          # ① 找新支撑点(§6)
    mu0 += newpoints;  pi0 += 0s;  sortmix(mu0, pi0)
    computeweights(mu0, pi0, dens)            # ② NNLS 权重 + Armijo(§7)
    collapse(mu0, pi0)                        # ③ 合并过近点(§8)
    iter += 1
    dens = mapping(mu0, pi0);  nloss = loss(dens)
    if closs - nloss < tol:  convergence = 0; break
    if iter > maxit:         convergence = 1; break
    closs = nloss
```

要点：

- **新点以权重 0 加入**，真正的权重由 ② 统一重算——所以 ① 只需要
  "点是否有效"（gain < 0），不需要点上的最优权重。这是 0.2.3
  放宽搜索（§6.3）成立的前提：调用方的符号过滤接受**任意**负点。
- `sortmix` 按 `mu0` 稳定排序；`simplifymix` 删 `|pi| <= 1e-14` 的点；
  两者都是 R `sortmix`/`simplifymix` 的逐位移植。
- 收敛判据是**相对损失改善** `closs - nloss < tol`（损失是"越小
  越好"的口径），与 R/C++ 参考实现一致。
- `verbose >= 1` 每轮打支撑点/权重；`verbose >= 2` 另打新点梯度与
  各阶段后的 loss；`NPFIXEDCOMPY_PROFILE=1` 结束时打一行分相耗时
  （`solvegrad/mapping/loss/weights/collapse` + `evals=` 梯度求值数 +
  `freshcols=`/`freshms=` fresh 核列数与耗时，见 docs/PERF.md §1）。

## 6. 支撑点搜索（`solvegrad`）

按族 flag 分两条路径。共同的第一步：**网格扫描**——对整条网格做
一次向量化的 `gradfunvec`（网格核列来自 `prepare` 的预算矩阵，扫描
本身是廉价 GEMV），得到每个网格点的 gain `pv` 与（d1）导数 `pg`。

### 6.1 d1 路径（`solvegradd1`，有导数：正态族、Poisson、CLL、AD）

1. 找所有**变号区间** `(gp[i], gp[i+1])`（`pg[i] < 0 < pg[i+1]`，
   即 gain 从负穿到正）；
2. 对每个变号区间，按"最便宜优先"取候选根（0.2.3 放宽，§6.3）：
   a. **零成本网格点接受**：若 `pv[i] < 0` 或 `pv[i+1] < 0`（扫描
      免费带出的值），直接取该网格点为根——它已是有效新支撑点，
      无需任何搜索；
   b. **CNM 工作集再校验**：否则取上一轮调用为同一区间精化出的根
      `warm_root_d1_[i]`（落在区间内时），用**一次** gain 求值
      再校验，仍为负则接受；
   c. 再否则跑 `brmin`（§6.2）做区间精化，结果写回
      `warm_root_d1_[i]` 供下轮热启动；
3. 汇总所有候选根，再做**一次向量的 gain 求值 + 符号过滤**
   （`vals[i] < 0` 才保留）——这是最终接受判据；
4. 端点检查：`gp[0]` 在 `pv[0] < 0 && pg[0] > 0` 时接受；
   `gp[last]` 在 `pv[last] < 0 && pg[last] < 0` 时接受（支撑空间
   端点外增益继续为负的情形）。

### 6.2 d0 路径（`solvegradd0`，无导数：CVM、分箱 LLW/TLLW 等）

无导数 → 在网格上做**三点差分**：对每个相邻三元组
`(gp[j], gp[j+1], gp[j+2])`，若 `pv` 先降后升
（`(pv[j+1]-pv[j]) < 0 && (pv[j+2]-pv[j+1]) > 0`），`gp[j+1]` 就是
三元组上的离散最小点，同样按"最便宜优先"处理：

a. **零成本接受**：`pv[j+1] < 0` → 直接取 `gp[j+1]`；
b. **CNM 再校验**：否则再校验上一轮的 `warm_root_d0_[j]`（一次求值）；
c. 再否则跑 `dfmin`（§6.2.1）精化三元组，写回 warm root；

端点：`gp[0]`（`pv[0] < 0` 且向内侧上升）、`gp[last]`
（`pv[last] < 0` 且向内侧下降）同 d1 逻辑。

d0 的 `dfmin` 比 d1 的 `brmin` 贵：它没有导数，精化每次要在全新的
内点求一次**整列** gain（一个 fresh 核列 = 一整个数据列的核求值，
`nptllw` 上是最贵的一步）；d1 的 `brmin` 复用同一列的导数部分，
相对便宜。

### 6.3 精化器：`brmin`（改进 Brent）与 `dfmin`（逐次抛物插值）

- **`brmin(lb, ub, dens, ctx, tol)`**：在变号区间上对 `a1`（导数）
  做改进 Brent——中点 + 割线/抛物式试探点 `s`，按符号关系收缩区间，
  收敛条件 `|fc|, |fs| < tol` 或区间宽 `< tol`（guard 上限 1000，
  可被 `NPFIC_REFINE_STEPS` 封顶，默认 `-1` 不限制）。
- **`dfmin(x1, fx1, …)`**：对三元组做逐次抛物插值——`newmin` 由三点
  拟合抛物线的顶点给新点（越界/NaN 时退回中点），按新点 gain 与
  三点极值的比较收缩 `[lb, ub]`；收敛时若 `fxx[2] < 0` 返回
  `xx[2]`，否则返回 NaN（该三元组不产生新点）。
- **负增益早停（0.2.3）**：两个精化器都在**任何**被求值点的
  `a0 < 0` 时立即返回该点——精确最小值从来不是目的，"找到任意
  有效负点"就够（调用方 §5 的符号过滤会再次把关）。d1 的早停检查
  搭在同一列求值上（`a0` 免费搭车），d0 的就是抛物新点本身。
  这是 0.2.3 相对 0.2.2 的主要提速来源（docs/PERF.md §4c：
  `nptll` β=5 n=5000 梯度求值 3733 → 1191、fresh 核列 455 → 61）。
- **证书**：接受的点都是 gain < 0 的合法新支撑点（外层损失必降）；
  最终解的"无负方向"由 `finish()` 的 `min_gradient`（支撑方向）+
  `grid_gain`（全网格）给出，见 §9。

## 7. 权重子问题（`computeweights` + Armijo）

给定（扩充后的）支撑点 `mu0`，重解权重 `pi0`：

1. 构造 `fp[i] = dens[i] + pre[i]`（当前混合密度 + 固定分量密度），
   `tp[i,j] = sp[i,j] / fp[i]`（`sp` = 核矩阵，列来自核列缓存），
   即把行缩放折进一次列并行填充，写出列主序 `n × m` 矩阵一次；
2. 解**约束 NNLS**（非负 + 权重和 = `1 - sum(pi0fixed)`）：
   - `n ≤ 1000`：`pnnlssum(tp, b, sum)`——Lawson–Hanson NNLS
     （`npfc_nnls.cpp`，C 风格移植），在数据行之外追加一行"和为 1"
     的同质约束，解完再归一到精确的和；
   - `n > 1000`：`pnnqp`——先对 Gram 阵 `q = tpᵀtp`（`m × m`，
     半正定）做特征分解，只保留显著特征方向，降维成一个小
     `pnnlssum`（避免 `n × m` 大 NNLS）；
3. **Armijo 回溯线搜索**（`checklossfun2`，基类默认实现）：
   方向 `eta = nw - pi0`，`sigma` 从 2 起每次减半、
   `alpha = 0.3333`，要求
   `loss(dens + sigma·diff) < loss(dens) + alpha·sigma·(p·eta)`，
   不满足且 `sigma < 1e-3` 才放弃（保留原 `pi0`）。
   这保证 ② 之后损失**单调不增**（配合 collapse 的 `ntol` 容差）。

分箱族的 `computeweights` 结构相同，只是核是"数据分箱计数 × 梯形
填充"：网格填充被 `prepare` 钉住（grid-fill memo，docs/PERF.md §3），
每次只有 off-grid 候选点触发一次 fresh 填充。

## 8. collapse（合并过近的支撑点）

每轮权重重算后调用（容差固定 `1e-6`，同 R/C++ 参考）：

1. 记当前损失 `ll` 与容差 `ntol = max(tol·0.1, ll·1e-16)`；
2. 循环：`prec = 10 · min(diff(mu0))`（10 倍最小间距），
   `collapsemix` 把间距 `≤ prec` 的相邻点**按权重加权平均**合并
   （`mu = (mu_i·pi_i + mu_j·pi_j)/(pi_i + pi_j)`，权重相加），
   再 `simplifymix` 删零权重点，重复直到没有可合并的对；
3. 若合并后损失 `nll ≤ ll + ntol` 则接受、继续下一轮收缩；
   否则**回退**到合并前的混合（break）。
4. 收尾再 `simplifymix` 一次。

作用：支撑集增长时把数值上不可区分的点合并掉，控制 `k` 的无谓
膨胀（t 族 β=5 的 AS-243 级数在相邻点上几乎线性相关）。

## 9. 结果汇总（`finish`）

- `pt`/`pr`：解出的支撑点/权重 **拼上固定分量**，整体 `sortmix`；
- `min_gradient`：在**最终密度**上对所有**支撑点**求 gain
  （`gradfunvec(resultpt_, …)` 取最小）——KKT 证书（支撑方向无负
  梯度；收敛解处应 `≥ -1e-4`，parity 门据此设门）；
- `grid_gain`：同一最终密度上对**全部网格点**求 gain 取最小（N-D
  正态族为全部 (G−1)^k 格中心，`npfc_famnd.h`）——网格级证书（精化根只
  认证其子区间，网格外方向另行跟踪）。依赖网格分辨率，可合法地略负
  （如 `npnormcvm` −8.2e-4），故对一维族只做**信息性**输出，不设门
  （docs/PERF.md §4c）；2-D 族另有门：`tests/verify_2d_same.py` 在
  `grid_gain < −1` 时失败——大负值是 FAST 目标按行主序误读 `(n, 2)`
  数据（损坏数据集）的特征，similarity 门抓不到它（docs/PERF.md
  §4a.3）；
- `ll`：最终损失的报告值（`lossfunction(dens) + extrafun()`）；
- `iter`、`convergence`、`flag`、`beta`、`family`。

## 10. `estpi0`（零处 point-mass 的比例）

目标：找 `sp`（`pi0fixed = {sp}`，零处 point-mass 权重），使假设
统计量命中阈值 `val`。流程：

1. `pi0fixed = {0}` 先跑一次完整 `computemixdist(tol, 100)`，得
   `minloss`；
2. 若纯 point-mass 的统计量 `hypofun(stat0, minloss) < val` → 直接
   返回 `pt = {0}, pr = {1}`（全质量在零处）；
3. 否则初始化 `sp0 = 当前拟合在 0 处的族密度 / point-mass 在 0 处的
   族密度`，`[lb, ub] = [0, 1]`，进入循环（guard 上限 1000）：
   - 固定 `pi0fixed = {sp}`，**以当前解为热启动**
     （`initpt_ = resultpt_`）跑 `computemixdist(1e-6, 100)`，
     记 `ll`，算残差 `h = hypofun(ll, minloss) - val`；
   - `|h| ≤ tol` 或区间宽 `≤ tol` → 收敛；
   - 按 `h` 的符号更新 bracket，下一次候选 = 三点
     `(lb, sp, ub)` 的**二次插值根**（Cramer 规则解 3×3，判别式
     负/越界时退回中点）；
   - **每轮做两次内层求解**（先评估中点、再评估插值点），所以
     一次 `estpi0` ≈ 15 次完整 `computemixdist`——这也是核列缓存的
     字节上限（默认 512 MiB，可配）存在的原因。
4. `estpi0_fast`（`fast=True` 显式启用，不逐位等同 legacy）：
   同样的 bracket/二次插值，但内层用可配的 `inner_tol`（比 1e-6
   松），且 `relax` 模式下当目标已从上方被小幅越过且 `sp0` 很小时
   提前停（省一半内层求解，统计量精度略降）。

注意：`estpi0` 的终点**不是自由极小**——它把 `sp` 钉在命中 `val`
上，所以 KKT 证书（`min_gradient`）对它不适用；parity 套件检查的是
统计量命中，不是梯度（tests/verify_cvmadcll.py、accept_kkt_023.py
均如此处理）。

## 11. 缓存层（为什么快）

三层缓存，全部**逐位等价**于重算（不改变算术，只省重复求值）：

1. **按 `mu` 的核列缓存**（`KernelColumnCache`，`npfc_kernels.h`）：
   每列 = 一整个数据列在核 `K(·, mu)` 下的值（`n` 个 double）。
   `column(mu)` 命中则 O(1)；未命中才算一次。可驱逐列有字节上限
   （默认 512 MiB；驱逐前裁剪，保证刚算的列不成为自己的受害者）；
   `pin(mu)` 钉住的列不驱逐（`prepare` 的网格矩阵、端点列）。
   第一遍迭代后，外层重复请求的向量（网格、当前支撑集、少量
   solver 内部点）几乎全命中。fresh 列的计数/耗时就是 PROFILE 行
   的 `freshcols=`/`freshms=`（t 族上这是主要成本，见 docs/PERF.md
   §1 的解读）。
2. **一次性网格矩阵**（`prepare(grid)`）：贵核在 run 开始时把
   数据×网格矩阵整体算好并灌进核列缓存（`insert`），此后网格
   扫描 = 查表。矩阵只依赖数据/beta/网格，`estpi0` 的 bisection
   全程复用。
3. **per-solve 不变量**（`prepare_solve(dens)`，0.2.2 方案 A）：
   外层 `dens` 在一次 `solvegrad` 内不变，于是 `fl = 1/(dens + pre)`
   这类"每个候选点都会用到"的数组只算一次；族用 `dens` 指针判
   命中，未命中原地重建（`finish()` 等无 prepare_solve 的路径
   走 `ensure_fl`）。

配合 0.2.3 的搜索放宽（零成本接受 / CNM 再校验 / 负增益早停），
fresh 列数从"每个候选区间都精化"降到"只在必须时精化"，这是
0.2.2 → 0.2.3 的主要提速来源（docs/PERF.md §4c 有 A/B 表）。

## 12. 多元正态族 `npnormND`（`npfc_famnd.h`）

R `npnorm2Dll` 的 N-D 泛化（0.3.0）：支撑点为 `k × d`（d 维均值，
d = `k` ≥ 2；绑定名 `npnorm2Dll` 保留为 d = 2 的别名）。结构同 §5
（solvegrad → computeweights → collapse → 收敛判据），差异：

- 搜索空间是 d 轴张量积网格：每轴一份 R 同款 1-D marginal
  （`gridpoints_npnorm`/`initial_npnorm`），逐格对格内 d 维均值跑
  有界 L-BFGS-B；
- 核 = d 维正态 `N(μ; x, beta)`：k = 2 保留手展开 2×2 Cholesky 快路径
  （与 R `densityND.h` 相同的因子取值与逐点运算序，~12 flops + 4 除法/点，
  无逐点 Eigen 调用，位级不变），k > 2 走 `dnormNDarray` 的 Eigen `LLT`
  通用路径（0.3.0 起移除了 R 同款 `NPFIC_2D_EXACT` 路径——验收契约
  已放宽为密度相似 + ll 不劣，不再要求逐位轨迹一致，docs/PERF.md §4a）；
- 权重子问题用 `LBFGSpp::LBFGSBSolver`（有界 LBFGS，`[0,1]` 约束，
  `max_iterations = max_linesearch = 100`）替代一维族的 NNLS；
- 逐格热启动 + `f_stop = 0` 负增益早停：上一轮每格收敛点作为本轮
  L-BFGS-B 起点（仍在格内时），负温热根/负网格点一次求值即接受
  （docs/PERF.md §4a.3）；
- gain 目标的**真方向导数**：每点梯度贡献按 `1/F_i`
  （`F_i = dens + precompute`）加权（`Σ_i (t_i/F_i)(μ − x_i)`，B 为
  k×k 权重矩阵）；R 移植的无权重形式是未归一化 KDE 的梯度，只有
  1/F 近似常数时才是 gain 方向导数；`LBFGSB.h` 对 `grad·d >= 0`
  （过期曲率）做 BFGS 历史重置 + 最速下降重启（docs/PERF.md §4a.4）；
- `grid_gain` = 最终混合下全部 (G−1)^d 格中心的最小 gain（`get_ans`
  计算，与一维族的网格级证书同义；可合法略负，parity 门在 < −1 时
  失败——docs/PERF.md §4a.3）。

## 13. 文件与符号速查

| 想看的 | 在哪 |
|--------|------|
| 外层主循环 | `npfc_engine.h` `MixSolver::computemixdist`（L681） |
| 支撑点搜索 d1/d0 | 同上 `solvegradd1`（L509）/ `solvegradd0`（L587） |
| Brent / 抛物精化 | 同上 `brmin`（L359）/ `dfmin`（L430） |
| collapse | 同上 `collapse`（L648） |
| estpi0 / estpi0_fast | 同上（L829 / L944） |
| 结果与证书 | 同上 `finish`（L1032） |
| Family 接口 | 同上（L69） |
| 族实现（正态/t/Poisson/CLL/AD/分箱） | `npfc_families.h`（`NpNormLL`/`NpTLL`/… 11 个类） |
| 多元正态族（N-D） | `npfc_famnd.h`（`NpNormND`；k = 2 即 R `npnorm2Dll`） |
| 核 + 列缓存 | `npfc_kernels.h`（`KernelColumnCache` L57） |
| NNLS / pnnlssum / pnnqp | `npfc_nnls.{h,cpp}` |
| 网格 / 初始分布 | `npfc_grid.h`（`whist`/`initial_npnorm`/`gridpoints_*`） |
| Python 前端 / 分箱 | `python/npfixedcomppy/npfc.py`（`computemixdist`/`estpi0`） |
| pybind11 入口 | `npfc_py.cpp`（每族一个函数） |

