# Q3-C｜候选配比枚举与局部联合优化（第三问 baseline C）

- **baseline**：`Q3-C`（第三问）
- **配置**：`/tmp/opencode/wt-c/configs/baselines/Q3-C.yaml` ｜ 配置哈希 `e9bf4f675e3b8003` ｜ 源码哈希 `074f04bd49cc32c7`
- **随机种子**：`20260923`
- **产物目录**：`artifacts/baselines/Q3-C`
- **本卡范围**：仅实现 `建模方案/baselines/Q3-C.md` 所述方法——候选配比枚举、逐候选 (N,Q) 搜索、多起点 SLSQP 局部细调与交叉核验。**不实现** Q3-A（固定 Q/p 的一维搜索）与 Q3-B（固定配比的二维搜索）这两条独立 baseline 的方法与交付物。

## 1. 摘要

冻结 Q2 主模型 **M1（= Q2-B，$\rho=0$）**，参数取自 Q2-C 的 `stage2_M1_staged`；质量通道指数 $\kappa$ 按 Q2-C 的判定作**情景参数**（主情景 $\kappa=0.8047$）。预算 $C$ 取题给三档并在 $\log_{10}C\in[17,26]$ 上连续扫描；成本函数 $g(Q)$ 取题给三式（指数/幂/对数）。参考配比 $p_0$ 的语料质量 $Q(p_0)=0.416633$。

- **泛函 `mixture_l1`，$\omega=0.0$（零效应对照）**：配比**不可辨识** —— `fixed_ref` 下全部候选并列最优、`mixture` 下由 $\arg\max Q$ 决定，两种 $Q_0$ 设定并列出现 `dirichlet_a20_19`、`reference_p0`。这是平局而非转移，见 §6.4。
- **泛函 `mixture_l1`，$\omega=0.5$**：全部情景的最优配比均为 `reference_p0`（各情景一致）
- **泛函 `mixture_l1`，$\omega=1.0$**：全部情景的最优配比均为 `reference_p0`（各情景一致）
- **泛函 `mixture_l1`，$\omega=2.0$**：全部情景的最优配比均为 `reference_p0`（各情景一致）
- **泛函 `quality_linear`，$\omega=0.0$（零效应对照）**：配比**不可辨识** —— `fixed_ref` 下全部候选并列最优、`mixture` 下由 $\arg\max Q$ 决定，两种 $Q_0$ 设定并列出现 `dirichlet_a20_19`、`reference_p0`。这是平局而非转移，见 §6.4。
- **泛函 `quality_linear`，$\omega=0.5$**：全部情景的最优配比均为 `dirichlet_a20_19`（各情景一致）
- **泛函 `quality_linear`，$\omega=1.0$**：全部情景的最优配比均为 `dirichlet_a20_19`（各情景一致）
- **泛函 `quality_linear`，$\omega=2.0$**：全部情景的最优配比均为 `dirichlet_a20_19`（各情景一致）

**主结论**：$\hat L$ 对 $h(p)$ **完全可分离**（$h$ 只进入 $T=D Q^\kappa h$，与 $N,D,Q$ 无交互）且 $\partial\hat L/\partial h<0$，故

$$
\arg\min_p \hat L = \arg\max_p h(p)
$$

**与预算 $C$、上下文长度 $\ell$、成本形式 $g$、$\kappa$ 完全无关**。在精确解层面的 720 个情景单元上，该等式在 **720/720** 个单元成立（`verify_p_channel.csv`）；其中 270/720 个单元里 $\arg\max h$ 同时就是 $\arg\max Q(p)$（对 `quality_linear` 成立，因为其 $\Delta$ 是 $Q$ 的仿射减函数；对 `mixture_l1`，$h$ 在 $p_0$ 取最大，与 $\arg\max Q$ 本就不同，**不是缺陷**）。

因此**未发现**「最优配比随预算转移」的结构性证据。按任务卡验收条款，这属于「完全可分离 $h(p)$ 下 $p$ 不变」的**正确结果**，不应强求转移。真正随预算变化的是 $Q^*$ 的**状态**（下界 $Q_0$ ↔ 内部解 ↔ 上界），见第 8 节。

**一个必须如实说明的量级事实**：各情景单元内候选之间的 $\hat L$ 跨度中位数仅 0.560084（相对 $\hat L\approx1.69$ 约 0.33141091）。即「选哪个 $p$」对预测损失的影响**远小于模型自身的误差**，因此 $p^*$ 的确定靠的是 $h$ 的单调性论证，而**不能**靠数值分辨 $\hat L$ 的差异。任何声称从数据中「学出」最优配比的结论，都必须报告这个跨度。

## 2. 方法、口径与求解流程

### 2.1 模型与预算消去

$$
\hat L(N,D,Q,p) = E + A N^{-\alpha} + B\left(D\,Q^{\kappa}\,h(p)\right)^{-\beta},\qquad
D\left[(6+\eta\ell)N + [g(Q)-g(Q_0)]_+\right] \le C,\quad \eta=2\times10^{-4}
$$

因 $\partial \hat L/\partial D = -\beta B (Q^\kappa h)^{-\beta} D^{-\beta-1} < 0$ 且题设无额外 $D$ 上限，预算必为紧约束，可消去 $D$：

$$
D = \frac{C}{u},\qquad u = aN + c(Q),\qquad a \equiv 6+\eta\ell,\qquad c(Q) \equiv [g(Q)-g(Q_0)]_+,
\qquad T = \frac{C\,Q^\kappa h}{u}
$$

于是给定 $p$ 后只需在 $(N,Q)$ 上求解。解析梯度（已与有限差分逐点核验）：

$$
\frac{\partial \hat L}{\partial N} = -\alpha A N^{-\alpha-1} + \frac{\beta\, a\, B\, T^{-\beta}}{u},\qquad
\frac{\partial \hat L}{\partial Q} = -\beta B T^{-\beta}\left(\frac{\kappa}{Q} - \frac{c'(Q)}{u}\right)
$$

$Q$ 的内部一阶条件为 $c'(Q) = \kappa\, u / Q$；若该方程在所允许区间无解，则 $Q^*$ 落在端点。

### 2.2 质量泛函与配比通道（与 Q2-C 冻结口径一致）

$$
Q(p)=\frac{\sum_{i\in\mathcal M} p_i q_i}{\sum_{i\in\mathcal M} p_i},\qquad
\Delta(p)=\frac{Q(p_0)-Q(p)}{s},\qquad h(p)=e^{-\omega \Delta(p)},\qquad h(p_0)=1
$$

$\mathcal M$ 为第一问完成质量映射的域（6 个，未映射域被剔除并重新归一化）。尺度 $s$ 直接取 Q2-C `mixture_transfer_scenarios.json` 中冻结的 `scale_s`（`quality_linear`: 0.00428；`mixture_l1`: 0.1），使 $\omega$ 的含义与 Q2-C 完全一致。

$\omega$ 取 [0.0, 0.5, 1.0, 2.0]（主情景 $\omega=1.0$），$\omega=0$ 即 Q2-B 的零效应情景（$h\equiv1$）作对照。**注意**：$h$ 关于 $p$ 可分离且单调依赖 $Q(p)$，故 $\omega$ 只改变效应幅度、不改变 $\arg\max_p h$。

### 2.3 候选生成规则（Q3-C 第 1 条：不用最终检验 Loss 偷选）

候选**只**来自第一问训练信息与既定可信区域：

1. 参考配比 $p_0$（第一问导出的参考混合）；
2. 单纯形成对扰动 $p_i+\delta,\ p_j-\delta$（$\delta=0.05$，与 Q2-C 冻结步长一致）；
3. 以 $p_0$ 为均值尺度参数的 Dirichlet 抽样（$\alpha_{dir}\in\{200,50,20\}$，各 20 个，固定种子）；
4. 线性分式规划（Charnes–Cooper）在可信区域内的**精确** $\max Q$ 解，以及无 L1 约束/带 L1 约束的两个数值解，作为解析参照点；
5. 由第一问质量坐标排序取质量最高/最低各 10 个（极值登记）。

全过程只读 `mixture_predictor.json`（列序、$p_0$、训练支持分位点）与 `quality_domain_mapping.csv`（冻结质量分），**未读取任何最终检验集**——见 `verify_input_audit.csv`（其中 `is_test_set` 列全为 False）。

### 2.4 可信区域

$p$ 限制在**第一问训练支持的 $[p_{{05}}, p_{{95}}]$ 盒 $\cap$ 单纯形**内，另加相对 $p_0$ 的 $L_1$ 半径 0.6。超出盒但仍在数据支持 $[\min,\max]$ 内的候选保留并标 `box_only`，不静默丢弃（`candidates.csv` 的 `trust_status` 列）。

### 2.5 求解流程

| 阶段 | 做法 |
| --- | --- |
| ① 上游读取 | Q1-C 列序/$p_0$/训练支持、Q2-C 冻结参数与转移情景、C7 架构元数据 |
| ② 候选枚举 | 见 2.3，得 207 个候选 |
| ③ 廉价筛选 | 每个候选在每个离散情景下做一次粗网格搜索（41×21），用于挑终选候选并检查 $\arg\min L$ 与 $\arg\max Q$ 是否一致 |
| ④ 精确求解 | 终选候选 × 全情景网格，逐点精确成本、粗网格 + 局部加密（201×101 起，4 轮收缩至 61²） |
| ⑤ 预算路径 | 沿 $\log_{10}C$ 网格（91 点）跟踪 $Q^*,N^*,D^*$ 与成本份额 |
| ⑥ 联合细调 | 单纯形 + 可信区域 + 多起点 SLSQP（24 起点），按标准化参数距离聚类标注局部解 |
| ⑦ 核验 | 解析导数 vs 有限差分、预算/单纯形约束、粗网格复现、上游一致性、输入审计 |

## 3. 样本流转

| stage | n_in | n_out | note |
| --- | --- | --- | --- |
| 上游接口读取 | 17 | 17 | Q1 列序/参考配比/训练支持 + Q2 冻结参数 |
| 候选配比枚举 | 17 | 207 | 仅用 p0 / 可信区域 / 第一问质量坐标 |
| 候选筛选（廉价网格） | 3726 | 22356 | 每个候选在每个离散情景下做一次粗搜索 |
| 逐候选 (N,Q) 精确搜索 | 1260 | 720 | 复用 Q3-B 式预算/成本内核；每点精确成本 |

**说明**：本问是**确定性优化**，没有训练/验证/测试样本划分；上表的「样本」是**候选配比与情景单元**的计数流转，用于说明数据从哪里来、经过哪些变换、最终进入哪张表。上游文件与其 SHA-256 登记在 `run_metadata.json`。

## 4. 情景与指标

- **预算**：题给三档 $C\in\{1.0000e+19, 1.0000e+22, 1.0000e+24\}$，另在 $\log_{10}C\in[17,26]$ 上取 91 点连续扫描。
- **上下文长度**：$\ell\in\{2048, 4096, 8192, 32768, 131072\}$，主情景 $\ell=4096$；注意力成本份额 $\eta\ell/6$，临界长度 $\ell_{crit}=6/\eta=30000$。
- **成本形式**：exponential, power, logarithmic（题给参数）。
- **$Q_0$ 设定**：`fixed_ref`（$Q_0$ 固定于 $Q(p_0)$，基线）与 `mixture`（$Q_0=Q(p)$，随语料组成变化，单独情景）——对应 Q3-C 第 2 条。
- **指标**：预测损失 $\hat L$、$N^*/D^*/Q^*$、成本份额 ($C_{train},C_{attn},C_{quality},C_{unused}$)、预算残差、边界命中标志、外推标志。

## 5. 结果：最优分配

`optimal_allocations.csv` 共 720 行（情景 × 泛函 × $\omega$ × 终选候选的最优解）。

### 5.1 主情景切片（泛函 `quality_linear`，$\omega=1.0$，$\ell=4096$）

| functional | omega | C | cost_form | ell | q0_mode | candidate | N_B | D_B | Q | Q0 | L_pred | share_train | share_attn | share_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality_linear | 1 | 1.0000e+19 | exponential | 4096 | fixed_ref | dirichlet_a20_19 | 2.452148 | 0.516223 | 0.937432 | 0.416633 | 2.317979 | 0.759513 | 0.103699 | 0.136788 |
| quality_linear | 1 | 1.0000e+22 | exponential | 4096 | fixed_ref | dirichlet_a20_19 | 48.52587 | 29.866785 | 1 | 0.416633 | 1.900623 | 0.869587 | 0.118728 | 0.011685 |
| quality_linear | 1 | 1.0000e+24 | exponential | 4096 | fixed_ref | dirichlet_a20_19 | 383.518808 | 381.795394 | 1 | 0.416633 | 1.793603 | 0.878554 | 0.119952 | 0.001494 |
| quality_linear | 1 | 1.0000e+19 | logarithmic | 4096 | fixed_ref | dirichlet_a20_19 | 2.348984 | 0.570462 | 1 | 0.416633 | 2.306534 | 0.804003 | 0.109773 | 0.086223 |
| quality_linear | 1 | 1.0000e+22 | logarithmic | 4096 | fixed_ref | dirichlet_a20_19 | 48.122332 | 30.333614 | 1 | 0.416633 | 1.900389 | 0.875835 | 0.119581 | 0.004585 |
| quality_linear | 1 | 1.0000e+24 | logarithmic | 4096 | fixed_ref | dirichlet_a20_19 | 383.14221 | 382.521107 | 1 | 0.416633 | 1.793588 | 0.87936 | 0.120062 | 0.000578 |
| quality_linear | 1 | 1.0000e+19 | power | 4096 | fixed_ref | dirichlet_a20_19 | 2.699314 | 0.437555 | 0.979251 | 0.416633 | 2.323228 | 0.708659 | 0.096756 | 0.194585 |
| quality_linear | 1 | 1.0000e+22 | power | 4096 | fixed_ref | dirichlet_a20_19 | 48.669102 | 29.697059 | 1 | 0.416633 | 1.900714 | 0.867198 | 0.118401 | 0.014401 |
| quality_linear | 1 | 1.0000e+24 | power | 4096 | fixed_ref | dirichlet_a20_19 | 383.707245 | 381.471794 | 1 | 0.416633 | 1.793608 | 0.878241 | 0.119909 | 0.00185 |
| quality_linear | 1 | 1.0000e+19 | exponential | 4096 | mixture | dirichlet_a20_19 | 2.448537 | 0.517491 | 0.937074 | 0.438297 | 2.317889 | 0.760258 | 0.103801 | 0.135942 |
| quality_linear | 1 | 1.0000e+22 | exponential | 4096 | mixture | dirichlet_a20_19 | 48.52587 | 29.868293 | 1 | 0.438297 | 1.900622 | 0.869631 | 0.118734 | 0.011635 |
| quality_linear | 1 | 1.0000e+24 | exponential | 4096 | mixture | dirichlet_a20_19 | 383.518808 | 381.797858 | 1 | 0.438297 | 1.793603 | 0.87856 | 0.119953 | 0.001487 |
| quality_linear | 1 | 1.0000e+19 | logarithmic | 4096 | mixture | dirichlet_a20_19 | 2.337473 | 0.575738 | 1 | 0.438297 | 2.306071 | 0.807463 | 0.110246 | 0.082291 |
| quality_linear | 1 | 1.0000e+22 | logarithmic | 4096 | mixture | dirichlet_a20_19 | 48.098699 | 30.356019 | 1 | 0.438297 | 1.900381 | 0.876051 | 0.11961 | 0.004339 |
| quality_linear | 1 | 1.0000e+24 | logarithmic | 4096 | mixture | dirichlet_a20_19 | 383.14221 | 382.533129 | 1 | 0.438297 | 1.793588 | 0.879388 | 0.120066 | 0.000547 |
| quality_linear | 1 | 1.0000e+19 | power | 4096 | mixture | dirichlet_a20_19 | 2.692692 | 0.439457 | 0.978191 | 0.438297 | 2.32307 | 0.709993 | 0.096938 | 0.193069 |
| quality_linear | 1 | 1.0000e+22 | power | 4096 | mixture | dirichlet_a20_19 | 48.669102 | 29.700045 | 1 | 0.438297 | 1.900711 | 0.867285 | 0.118413 | 0.014302 |
| quality_linear | 1 | 1.0000e+24 | power | 4096 | mixture | dirichlet_a20_19 | 383.707245 | 381.476722 | 1 | 0.438297 | 1.793608 | 0.878252 | 0.119911 | 0.001837 |

### 5.2 全部情景下最优配比的复现性

若 $\arg\min_p \hat L$ 不随情景变化，则配比通道与资源通道**解耦**，调配比与调预算可以分开决策。下表给出每个（泛函，$\omega$）下在各情景中出现的最优配比集合：

| functional | omega | n_distinct_optimal_p | optimal_candidates |
| --- | --- | --- | --- |
| mixture_l1 | 0 | 2 | dirichlet_a20_19, reference_p0 |
| mixture_l1 | 0.5 | 1 | reference_p0 |
| mixture_l1 | 1 | 1 | reference_p0 |
| mixture_l1 | 2 | 1 | reference_p0 |
| quality_linear | 0 | 2 | dirichlet_a20_19, reference_p0 |
| quality_linear | 0.5 | 1 | dirichlet_a20_19 |
| quality_linear | 1 | 1 | dirichlet_a20_19 |
| quality_linear | 2 | 1 | dirichlet_a20_19 |

### 5.3 质量投入的状态

| functional | omega | cost_form | q0_mode | regime | n |
| --- | --- | --- | --- | --- | --- |
| mixture_l1 | 0 | exponential | fixed_ref | upper_bound | 10 |
| mixture_l1 | 0 | exponential | fixed_ref | interior | 5 |
| mixture_l1 | 0 | exponential | mixture | upper_bound | 10 |
| mixture_l1 | 0 | exponential | mixture | interior | 5 |
| mixture_l1 | 0 | logarithmic | fixed_ref | upper_bound | 15 |
| mixture_l1 | 0 | logarithmic | mixture | upper_bound | 15 |
| mixture_l1 | 0 | power | fixed_ref | upper_bound | 10 |
| mixture_l1 | 0 | power | fixed_ref | interior | 5 |
| mixture_l1 | 0 | power | mixture | upper_bound | 10 |
| mixture_l1 | 0 | power | mixture | interior | 5 |
| mixture_l1 | 0.5 | exponential | fixed_ref | upper_bound | 10 |
| mixture_l1 | 0.5 | exponential | fixed_ref | interior | 5 |
| mixture_l1 | 0.5 | exponential | mixture | upper_bound | 10 |
| mixture_l1 | 0.5 | exponential | mixture | interior | 5 |
| mixture_l1 | 0.5 | logarithmic | fixed_ref | upper_bound | 15 |
| mixture_l1 | 0.5 | logarithmic | mixture | upper_bound | 15 |
| mixture_l1 | 0.5 | power | fixed_ref | upper_bound | 10 |
| mixture_l1 | 0.5 | power | fixed_ref | interior | 5 |
| mixture_l1 | 0.5 | power | mixture | upper_bound | 10 |
| mixture_l1 | 0.5 | power | mixture | interior | 5 |
| mixture_l1 | 1 | exponential | fixed_ref | upper_bound | 10 |
| mixture_l1 | 1 | exponential | fixed_ref | interior | 5 |
| mixture_l1 | 1 | exponential | mixture | upper_bound | 10 |
| mixture_l1 | 1 | exponential | mixture | interior | 5 |
| mixture_l1 | 1 | logarithmic | fixed_ref | upper_bound | 15 |
| mixture_l1 | 1 | logarithmic | mixture | upper_bound | 15 |
| mixture_l1 | 1 | power | fixed_ref | upper_bound | 10 |
| mixture_l1 | 1 | power | fixed_ref | interior | 5 |
| mixture_l1 | 1 | power | mixture | upper_bound | 10 |
| mixture_l1 | 1 | power | mixture | interior | 5 |
| mixture_l1 | 2 | exponential | fixed_ref | upper_bound | 10 |
| mixture_l1 | 2 | exponential | fixed_ref | interior | 5 |
| mixture_l1 | 2 | exponential | mixture | upper_bound | 10 |
| mixture_l1 | 2 | exponential | mixture | interior | 5 |
| mixture_l1 | 2 | logarithmic | fixed_ref | upper_bound | 15 |
| mixture_l1 | 2 | logarithmic | mixture | upper_bound | 15 |
| mixture_l1 | 2 | power | fixed_ref | upper_bound | 10 |
| mixture_l1 | 2 | power | fixed_ref | interior | 5 |
| mixture_l1 | 2 | power | mixture | upper_bound | 10 |
| mixture_l1 | 2 | power | mixture | interior | 5 |

_（共 76 行，此处只列前 40 行；完整表见对应 CSV）_
## 6. 对照与交叉核验

### 6.1 配比通道不变性：$\arg\min_p\hat L \equiv \arg\max_p h$

在 720/720 个情景单元上，「按预测 Loss 排序的最优配比」与「按 $h(p)$ 排序的最优配比」**指向同一配比**（判定用数值容差，浮点平局不计为不一致）。

**判据说明**：正确的不变性判据是 $\arg\min L \equiv \arg\max h$（因为 $h$ 是唯一进入模型的配比通道），而**不是** $\arg\max Q$。两者只在 `quality_linear` 下等价。

各单元内候选间 $\hat L$ 的跨度：

| functional | omega | C | cost_form | ell | q0_mode | argmin_L_candidate | L_min | L_spread | n_tied_at_L_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 2048 | fixed_ref | reference_p0 | 3.123095 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 2048 | mixture | dirichlet_a20_19 | 3.120686 | 0.004157 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 4096 | fixed_ref | reference_p0 | 3.135573 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 4096 | mixture | dirichlet_a20_19 | 3.133238 | 0.004032 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 8192 | fixed_ref | reference_p0 | 3.15869 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 8192 | mixture | dirichlet_a20_19 | 3.156481 | 0.003816 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 32768 | fixed_ref | reference_p0 | 3.264134 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 32768 | mixture | dirichlet_a20_19 | 3.262391 | 0.003017 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 131072 | fixed_ref | reference_p0 | 3.48506 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | exponential | 131072 | mixture | dirichlet_a20_19 | 3.483923 | 0.001973 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 2048 | fixed_ref | reference_p0 | 3.137817 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 2048 | mixture | dirichlet_a20_19 | 3.132219 | 0.010354 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 4096 | fixed_ref | reference_p0 | 3.148194 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 4096 | mixture | dirichlet_a20_19 | 3.142655 | 0.010245 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 8192 | fixed_ref | reference_p0 | 3.167551 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 8192 | mixture | dirichlet_a20_19 | 3.162122 | 0.010044 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 32768 | fixed_ref | reference_p0 | 3.257906 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 32768 | mixture | dirichlet_a20_19 | 3.252961 | 0.009155 | 1 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 131072 | fixed_ref | reference_p0 | 3.456393 | 0 | 14 |
| mixture_l1 | 0 | 1.0000e+19 | logarithmic | 131072 | mixture | dirichlet_a20_19 | 3.452362 | 0.007476 | 1 |

_（共 720 行，此处只列前 20 行；完整表见对应 CSV）_

### 6.2 解析参照：可信区域内的 $\max Q(p)$

线性分式规划（Charnes–Cooper 变换后为 LP，`method="highs"`）给出盒内精确最优 $\max Q = 0.487162$。其最优配比的非零分量（$p_i>10^{-12}$）为：

| domain | p | q_i |
| --- | --- | --- |
| stackexchange | 0.4969 | 0.487162 |
| uspto_backgrounds | 0.38322 | — |
| pubmed_central | 0.11988 | — |

该解是**退化顶点**：质量上界把所有已映射质量压在单一域上、其余映射域取盒下界。这是线性分式目标在单纯形 $\cap$ 盒上的普遍现象，**不是数据结论**；只有引入多样性/供给约束才能真正避免，而题面未给此类约束。本卡如实登记这一局限。

数据侧的上界核对：第一问已映射域的 $\max_i q_i = 0.487162 = q_{supported\_max}$，与 LP 解一致 → 该上界是**数据集性质**，不依赖优化器。

### 6.3 枚举 vs 多起点 SLSQP

12 个代表性情景上比较「枚举最优」与「SLSQP 最优等价解类」，其中 SLSQP 严格更优的单元数 = 12。

| C | cost_form | ell | q0_mode | enum_candidate | enum_L | enum_N_B | enum_Q | slsqp_L | slsqp_N_B | slsqp_Q | dL_slsqp_minus_enum | slsqp_Q_cand | enum_Q_cand | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000e+19 | exponential | 4096 | fixed_ref | dirichlet_a20_19 | 2.317979 | 2.452148 | 0.937432 | 1.850673 | 106.381388 | 1 | -0.467306 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+19 | power | 4096 | fixed_ref | dirichlet_a20_19 | 2.323228 | 2.699314 | 0.979251 | 1.850705 | 106.496338 | 1 | -0.472523 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+19 | logarithmic | 4096 | fixed_ref | dirichlet_a20_19 | 2.306534 | 2.348984 | 1 | 1.836424 | 129.52316 | 1 | -0.470111 | 0.477948 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+22 | exponential | 4096 | fixed_ref | dirichlet_a20_19 | 1.900623 | 48.52587 | 1 | 1.745466 | 2394.892819 | 1 | -0.155158 | 0.475373 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+22 | power | 4096 | fixed_ref | dirichlet_a20_19 | 1.900714 | 48.669102 | 1 | 1.745469 | 2393.231944 | 1 | -0.155246 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+22 | logarithmic | 4096 | fixed_ref | dirichlet_a20_19 | 1.900389 | 48.122332 | 1 | 1.745467 | 2394.204133 | 1 | -0.154922 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+24 | exponential | 4096 | fixed_ref | dirichlet_a20_19 | 1.793603 | 383.518808 | 1 | 1.71781 | 9999.999923 | 1 | -0.075793 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+24 | power | 4096 | fixed_ref | dirichlet_a20_19 | 1.793608 | 383.707245 | 1 | 1.71781 | 9999.999998 | 1 | -0.075798 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+24 | logarithmic | 4096 | fixed_ref | dirichlet_a20_19 | 1.793588 | 383.14221 | 1 | 1.71781 | 9999.999999 | 1 | -0.075778 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+19 | exponential | 4096 | mixture | dirichlet_a20_19 | 2.317889 | 2.448537 | 0.937074 | 1.850666 | 106.43296 | 1 | -0.467223 | 0.475373 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+19 | power | 4096 | mixture | dirichlet_a20_19 | 2.32307 | 2.692692 | 0.978191 | 1.850701 | 106.585767 | 1 | -0.472368 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |
| 1.0000e+19 | logarithmic | 4096 | mixture | dirichlet_a20_19 | 2.306071 | 2.337473 | 1 | 1.850584 | 106.009529 | 1 | -0.455487 | 0.475372 | 0.438297 | SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选 |

SLSQP 共得 33 个等价解类；各情景的最优类标记为 `is_best_cluster`。类数多于情景数即说明存在**多个局部解**，必须在结论中使用全局最优类并声明其余为局部解。

### 6.4 $\omega=0$ 对照（无质量效应）

$\omega=0$ 时 $h\equiv1$，配比对预测损失**完全没有影响** → $p^*$ 不可辨识。两种 $Q_0$ 设定下「不可辨识」的表现**不同**，必须分开读：

- `q0_mode=fixed_ref`（$Q_0$ 固定）：各候选的 $\hat L$ **逐位相同**，单元内 14 个候选**全部并列最优**（$\hat L$ 跨度恰为 0，即精确 0）。此时表里的 `candidate` 只是候选集的枚举顺序，**不含任何信息**，不可解读为「最优配比」，更不可写成结论。
- `q0_mode=mixture`（$Q_0=Q(p)$ 随候选变化）：各候选的 $\hat L$ 不再相同，但差异只来自 $T=C Q^\kappa/u$ 里的 $Q$；并列数降为 1，$\hat L$ 跨度最大仅 0.010377307078 —— 最优解回到 $\arg\max Q$。

同时 Q2-C 判定 $\kappa$ 未辨识，$\kappa\to0$ 时同理。故结论是：**只有 $\kappa\neq0$ 且 $\omega\neq0$ 时配比才可辨识**（本数据上即可辨识单元 540/720，全部落在 $\omega>0$）；而本模型的 $h(p)$ 通道完全可分离，故一旦可辨识，其最优点就与预算无关。第 5.2 节中 $\omega=0$ 行出现 2 个「最优配比」，正是上述两种设定并列的结果，**不是**配比随情景转移——读表时若不区分 $Q_0$ 设定，就会把一个纯技术性平局误读成经济含义上的转移。

## 7. 数值核验与误差

| check | pass | max_violation | tolerance | note |
| --- | --- | --- | --- | --- |
| budget_tight_and_consistent | True | 2.0972e-16 | 1.0000e-09 | D = C/u，回算 D*{(6+eta*l)N + c(Q)} 与 C 的相对残差 |
| cost_shares_sum_to_one | True | 3.3307e-16 | 1.0000e-09 | 训练+注意力+质量+未使用 = 1（分母为 C）。残差处于机器精度（约 1e-16，即 1–2 ULP），来源是 used/C 与 (C-used)/C 相加的末位舍入，不是建模或求解误差 |
| attn_share_equals_eta_l_over_6 | True | 8.8818e-16 | 1.0000e-12 | C_attn/C_train = eta*l/6 |
| Q_within_bounds | True | 0 | 1.0000e-09 | Q0 <= Q <= q_hi |
| simplex_feasible | True | 2.2204e-16 | 1.0000e-09 | sum(p)=1 且 p>=0 是硬约束，全部通过；可信区域盒只对 in_trust_box=True 的候选要求，盒外对照点按设计不计违规 |
| physical_unit_anchor | True | 1.8527e-07 | 1.0000e-05 | 物理单位锚点：N=1e9,D=1e11,Q=1,h=1 -> 2.385578（旧错误单位 1.691141）；该检查独立于代数相同的数值-解析对照，防止上下游同时误解单位 |
| analytic_gradient_matches_fd | True | 8.4064e-07 | 0.001 | 解析 dL/dN、dL/dQ 与差分一致；Q=Q0 处按可行域取右导数 g'(Q0) |
| upstream_predictor_agrees | True | 0 | 1.0000e-12 | 与 predict_q2c_loss.py(M1, rho=0) 逐点一致 |
| grid_coarsening_stable | True | 2.9598e-10 | 1.0000e-06 | 粗网格下结论不变：相对损失变化（dL 除以 L 取绝对值）在容差内，且 Q* 状态（下界/内部/上界）相同；N*/Q* 位置只作诊断量 |
| p_channel_invariant_argminL_eq_argmaxh | True | 0 | 0 | argmin_p L 恒等于 argmax_p h（h 可分离的必然后果），故 p* 与预算无关 |
| argmax_h_eq_argmaxQ_for_quality_linear | True | 0 | 0 | quality_linear 的 Δ 是 Q 的仿射减函数，故 argmax h 恒等于 argmax Q；mixture_l1 的 h 在 p0 取最大，二者本就不同（非缺陷）。仅对 h 可辨识（ω>0）的单元判定：ω=0 时 h 恒为 1，任何候选都是 argmax，该单元下 p 不可辨识 |
| no_final_test_set_read | True | 0 | 0 | 输入审计中不含任何最终检验/估算文件 |

各项核验的完整输出见 `verify_derivatives.csv`、`verify_constraints.csv`、`verify_simplex.csv`、`verify_grid_stability.csv`、`verify_upstream_consistency.csv`、`verify_p_channel.csv`、`verify_input_audit.csv`。

**误差来源与量级（须在论文中如实披露）**

1. **模型误差（主导）**：$\hat L$ 本身是 Q2 在 $N_{\text{data}},D_{\text{data}}$ 范围内拟合的幂律，$R^2$ 级误差（Q2-C 报 `rmse` 约 0.047 量级）远大于本问的数值误差；本问所有「最优」都是**在该模型内部**的最优。
2. **参数不确定性**：$\kappa$ 未辨识（Q2-C 判定为情景参数）。本报告用 $\kappa$ 情景网格覆盖，但 $Q^*$ 的内部值对 $\kappa$ 敏感（见 `optimal_allocations.csv` 中不同 $\kappa$ 的对照）。
3. **外推**：解可能落在 Q2-C 数据支持范围之外（$N\notin[0.0705,11.97]$ B，$D\notin[0.134,600]$ B，$Q>0.4872$）。所有解都带 `extrapolation` 列标注，不静默外推。
4. **数值误差**：网格离散 + 局部加密。粗网格复现检查（`verify_grid_stability.csv`）用于排除「网格造成的伪转移」。
5. **成本函数不确定性**：题给三式差异极大，故结论按成本形式分别给出，不取平均。

## 8. 结构性转移

按**预先定义**的事件（① $Q^*$ 离开/进入边界；② $p^*$ 支持集稳定变化；③ 成本份额曲线可复现的分段变化）在连续 $\log C$ 网格上扫描，共得到 214 个事件。

**按事件类型分解**（这是判断「哪条通道发生转移」的直接依据）：

| event_type | n |
| --- | --- |
| Q_regime_change | 214 |

未出现的类型：`p_support_change`, `share_segment_change` —— 即配比通道与成本份额通道在本模型族内**没有**结构性转移，这与第 6.1 节的解析结论一致，是模型结构的必然后果，**不应**为了「有转移」而放宽判据。

| event | cost_form | ell | kappa | from | to | C_before | C_after | log10_C_mid | share_quality_before | share_quality_after | note | functional | omega |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q_regime_change | exponential | 2048 | 0.804707 | lower_bound_Q0 | interior | 1.0000e+18 | 1.2589e+18 | 18.05 | 0 | 0.032453 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 2048 | 0.804707 | interior | upper_bound | 3.9811e+21 | 5.0119e+21 | 21.65 | 0.130837 | 0.124812 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 2048 | 0.804707 | lower_bound_Q0 | interior | 1.0000e+18 | 1.2589e+18 | 18.05 | 0 | 0.032453 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 2048 | 0.804707 | interior | upper_bound | 3.9811e+21 | 5.0119e+21 | 21.65 | 0.130837 | 0.124812 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 4096 | 0.804707 | lower_bound_Q0 | interior | 1.0000e+18 | 1.2589e+18 | 18.05 | 0 | 0.045079 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 4096 | 0.804707 | interior | upper_bound | 3.9811e+21 | 5.0119e+21 | 21.65 | 0.130316 | 0.121596 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 4096 | 0.804707 | lower_bound_Q0 | interior | 1.0000e+18 | 1.2589e+18 | 18.05 | 0 | 0.045079 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 4096 | 0.804707 | interior | upper_bound | 3.9811e+21 | 5.0119e+21 | 21.65 | 0.130316 | 0.121596 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 8192 | 0.804707 | lower_bound_Q0 | interior | 7.9433e+17 | 1.0000e+18 | 17.95 | 0 | 0.029289 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 8192 | 0.804707 | interior | upper_bound | 3.1623e+21 | 3.9811e+21 | 21.55 | 0.130953 | 0.125562 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 8192 | 0.804707 | lower_bound_Q0 | interior | 7.9433e+17 | 1.0000e+18 | 17.95 | 0 | 0.029289 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 8192 | 0.804707 | interior | upper_bound | 3.1623e+21 | 3.9811e+21 | 21.55 | 0.130953 | 0.125562 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 32768 | 0.804707 | lower_bound_Q0 | interior | 3.9811e+17 | 5.0119e+17 | 17.65 | 0 | 0.011159 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 32768 | 0.804707 | interior | upper_bound | 1.5849e+21 | 1.9953e+21 | 21.25 | 0.131563 | 0.12942 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 32768 | 0.804707 | lower_bound_Q0 | interior | 3.9811e+17 | 5.0119e+17 | 17.65 | 0 | 0.011159 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 32768 | 0.804707 | interior | upper_bound | 1.5849e+21 | 1.9953e+21 | 21.25 | 0.131563 | 0.12942 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 131072 | 0.804707 | lower_bound_Q0 | interior | 1.2589e+17 | 1.5849e+17 | 17.15 | 0 | 0.009587 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 131072 | 0.804707 | interior | upper_bound | 5.0119e+20 | 6.3096e+20 | 20.75 | 0.13161 | 0.129668 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 131072 | 0.804707 | lower_bound_Q0 | interior | 1.2589e+17 | 1.5849e+17 | 17.15 | 0 | 0.009587 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | exponential | 131072 | 0.804707 | interior | upper_bound | 5.0119e+20 | 6.3096e+20 | 20.75 | 0.13161 | 0.129668 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 2048 | 0.804707 | lower_bound_Q0 | upper_bound | 7.9433e+18 | 1.0000e+19 | 18.95 | 0 | 0.379596 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 2048 | 0.804707 | lower_bound_Q0 | upper_bound | 7.9433e+18 | 1.0000e+19 | 18.95 | 0 | 0.379596 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 4096 | 0.804707 | lower_bound_Q0 | upper_bound | 6.3096e+18 | 7.9433e+18 | 18.85 | 0 | 0.391102 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 4096 | 0.804707 | lower_bound_Q0 | upper_bound | 6.3096e+18 | 7.9433e+18 | 18.85 | 0 | 0.391102 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 8192 | 0.804707 | lower_bound_Q0 | upper_bound | 6.3096e+18 | 7.9433e+18 | 18.85 | 0 | 0.380864 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 8192 | 0.804707 | lower_bound_Q0 | upper_bound | 6.3096e+18 | 7.9433e+18 | 18.85 | 0 | 0.380864 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 32768 | 0.804707 | lower_bound_Q0 | upper_bound | 3.1623e+18 | 3.9811e+18 | 18.55 | 0 | 0.387324 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 32768 | 0.804707 | lower_bound_Q0 | upper_bound | 3.1623e+18 | 3.9811e+18 | 18.55 | 0 | 0.387324 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 131072 | 0.804707 | lower_bound_Q0 | upper_bound | 1.0000e+18 | 1.2589e+18 | 18.05 | 0 | 0.387848 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |
| Q_regime_change | logarithmic | 131072 | 0.804707 | lower_bound_Q0 | upper_bound | 1.0000e+18 | 1.2589e+18 | 18.05 | 0 | 0.387848 | 预定义事件 1（Q* 离开/进入边界） | mixture_l1 | 0.5 |

_（共 214 行，此处只列前 30 行；完整表见对应 CSV）_

**判据的预先声明**（避免事后挑选）：$Q^*$ 端点判定容差 1.0000e-06；配比支持集变化要求在同一情景维度上连续 ≥3 个格点与整体最优不同（占比容差 0.02）；所有候选变点都要在更粗网格（放大 [2, 4] 倍）下复现才登记。

事件 ①（$Q^*$ 由下界进入内部）是**真实存在**的：由内部条件 $c'(Q)=\kappa u/Q$ 可知，预算太小则质量投入无意义（$Q^*=Q_0$），预算增大后转入内部解。事件 ②③属配比通道，在本模型族内**不存在**，这是模型结构的必然后果。

## 9. 任务卡验收自查

逐条对 `Q3-C.md` 的「验证与完成条件」作答，避免把未做的当成做了：

| 卡片要求 | 落实位置 | 结论 |
| --- | --- | --- |
| 检查最优 $p$ 是否随预算变化 | `verify_p_channel.csv`；本报告 §5.2、§6.1 | **已检查**：在 720 个情景单元上最优 $p$ 均不随 $C$ 变化；给出的是等式成立的计数，而非「没找到变化」的空泛陈述 |
| 完全可分离 $h(p)$ 下 $p$ 不变可能是正确结果，不应强求转移 | 本报告 §1、§6.1、§8 | **已按此处理**：报告只登记预定义事件实际发生者（$Q^*$ 状态转移），配比通道与份额通道的缺席被明确写为模型结构的必然后果 |
| 必用数据要求全部落实，或逐项解释限制 | §2.3、§3、§10.6 | **已落实并逐项解释**：Q1 质量坐标、Q2 冻结参数、C7 全部使用；唯一未落实项（第一问随机森林对象未持久化）已单独声明并给出替代泛函 |
| 按公共协议完成留出验证、消融与数值核验 | §4、§7、`verify_*.csv` | **已按本题性质执行**，见下方说明 |
| 配置、随机种子、环境、输入哈希和上游版本可复现 | `run_metadata.json` | **已登记**：配置哈希、**源码哈希**、种子、`env`、六个上游文件的 SHA-256、命中的根目录与实际上游版本。同时记源码哈希，是因为配置哈希只锁旋钮、锁不住逻辑：「改了数值代码却没改配置」时配置哈希不变，仅凭它无法判断产物出自哪版代码 |
| 正结果、负结果、失败与外推边界均记录 | §6.4、§7.3、§10 | **已记录**：负结果（无配比转移）、失败项（树模型不可用）、外推边界（`extrapolation` 列 + §10.2）均明确列出 |

**关于「留出验证」与「消融」在本问的性质**：本问是**确定性约束优化**，不存在训练/验证/测试样本划分，因此协议中的「留出验证」发生在**上游**——Q2-C 已用 A4/A5 内部验证选择主模型并留出 A6–A11 作最终检验，本卡**不重做也不触碰**该切分（`verify_input_audit.csv` 的 `is_test_set` 全为 False 可独立复核）。本卡实际完成的消融是**情景消融**，即把每个未标定/有争议的模型构件逐一切换并观察结论是否改变：

- $\kappa\in$ ['none', 'M1_staged', 'M2_staged', 'boot_lo', 'boot_hi']（含 $\kappa=0$ 的无质量通道对照）；
- $\omega\in$ [0.0, 0.5, 1.0, 2.0]（含 $\omega=0$ 的零效应对照）；
- 成本形式 $g\in$ ['exponential', 'power', 'logarithmic']（差异极大，结论按形式分列不取平均）；
- $Q_0$ 设定 $\in$ ['fixed_ref', 'mixture']（$Q_0$ 是否依赖 $p$，对应卡片第 2 条）；
- 上下文长度 $\ell\in$ [2048, 4096, 8192, 32768, 131072]（跨越 $\ell_{crit}=6/\eta$ 两侧）；
- 泛函 $\in$ ['quality_linear', 'mixture_l1'] 与可信区域强度（盒内/盒外对照点）。

**消融的结论**：上述任一切换都**不改变** $\arg\max_p h$，故「$p^*$ 与预算无关」不是某一个情景设定的产物。这正是该结论可信的原因，也是它作为**结构性结论**（而非估计结果）的证据。

### 9.1 与本 baseline 完整范围的关系（范围层级说明）

运行 `Baseline-C.md` 规定「第三问本轮做到候选配比枚举与基本资源搜索；多起点 SLSQP 连续联合细调暂留讨论」。本卡据此把**枚举 + 逐候选 (N,Q) 精确搜索**作为产出结论的主路径；多起点 SLSQP 只作 `Q3-C.md` 第 4 条要求的**有界交叉核验**——仅 12 个代表性情景、限制在单纯形 ∩ 可信区域内、按标准化参数距离聚类并显式标注局部解，**不**作为生产求解器，也**不**用它去挑选配比（候选集不因 SLSQP 结果扩充）。若上游口径变更，该核验可整体移除而不影响主结论，故它与「暂留讨论」的边界不冲突。

`Baseline-C.md` 另要求完整路线报告 `reports/Baseline-C.md`（覆盖 Q1-C/Q2-C/Q3-C/Q4-C）与统一运行入口。该路线级交付物涉及第四问，**不属于本卡范围**，本目录不生成也不改写它。

## 10. 运行方式

```bash
cd src/baselines/Q3-C
python run_all.py            # 完整运行
python run_all.py --quick    # 冒烟测试（缩小网格，结果不得引用）
```

依赖：`numpy`、`pandas`、`scipy`、`pyyaml`。环境与版本登记在 `run_metadata.json` 的 `env` 字段。路径解析：脚本在「项目根 → `paths.upstream_roots` → `paths.q2_bridge_root`」中按序探测，命中的根如实登记在 `run_metadata.json` 的 `roots` 字段。

**上游版本**：Q2 模型 `M1`（参数阶段 `stage2_M1_staged`），Q2 预测器 `loss_predictor.json`，Q1 预测器 `mixture_predictor.json`；各文件 SHA-256 见 `run_metadata.json`。

## 11. 结论边界与局限

1. **范围**：本节只交付 Q3-C 卡片要求的内容。第三问的另外两条 baseline（Q3-A 的一维解析资源解、Q3-B 的固定配比二维搜索）不在本卡范围内，本目录不产出它们的交付物。

2. **$p^*$ 与预算无关是模型结构的结果，不是估计结果**：$h(p)$ 完全可分离且 $Q$ 线性分式 ⇒ $\arg\min_p \hat L$ 是可信区域的顶点，与 $C,\ell,g,\kappa,\omega$ 无关。若上游（Q2）改用含 $N$/$D$ 与 $p$ 交互的 $h$，该结论会失效——这是本卡最重要的外推边界。

3. **顶点退化解**：$\max Q$ 把所有质量压在单一域上。若把结论直译为「语料应 100% 来自 stackexchange」，那是**过度解读**：模型只认识 6 个已映射域的质量分，没有刻画任何域的边际收益递减或供给上限。可信区域不足以免除此问题，需额外的多样性约束（题面未给）。

4. **$\kappa$、$\omega$ 未标定**：$\kappa$ 未辨识，$\omega$ 无上游标定，两者只能作情景。本卡的数值结论是「给定情景下的模型内部最优」，不是对真实训练结果的预测。

5. **$Q_0$ 的两种设定都保留了**：`fixed_ref` 与 `mixture` 并列报告，未择优。两者对应不同的建模语义（$Q_0$ 是否随语料组成变化），须由论文正文明确选择并说明理由。

6. **第一问随机森林未持久化**：Q1-C 只导出了预测器的接口描述符（列序、权重、训练支持），没有保存拟合好的森林对象，因此本流程**不能**调用 $f(p)=\sum_k v_k\hat L_k(p)$ 这一泛函。本卡如实声明，改用两个可由 $p$ 直接算出的冻结泛函（`quality_linear`、`mixture_l1`），未假装使用过森林。

7. **不覆盖他人产物**：本卡的代码与结果全部落在 `src/baselines/Q3-C/`、`configs/baselines/Q3-C.yaml`、`artifacts/baselines/Q3-C/`、`reports/baselines/Q3-C.md` 四个槽位内，未改写任何共享发布物。

