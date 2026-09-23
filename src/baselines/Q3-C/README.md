# Q3-C 工作区：候选配比枚举与局部联合优化

本目录实现任务卡 `math_model_F/建模方案/baselines/Q3-C.md` 所述方法，对应
2026 中国研究生数学建模竞赛 F 题**问题三**：在算力预算约束下联合优化
$(N, D, Q, p)$，使预测损失最小。

## 本 baseline 的范围（重要）

**只实现 Q3-C 卡片所述内容**：

1. 从第一问训练信息与既定可信区域内**枚举候选配比** $p$，不用最终检验 Loss 偷选；
2. 对每个候选运行 Q3-B 式的预算/成本内核，搜索 $(N, Q)$；
3. 用**多起点 SLSQP** 在单纯形 + 可信区域内做局部联合细调，与枚举结果交叉核验并标注局部解；
4. 检查最优 $p$ 是否随预算变化，并按**预先定义**的判据识别结构性转移。

**不实现** Q3-A（固定 $Q/p$ 的一维资源搜索）与 Q3-B（固定配比的二维搜索）这两条独立
baseline 的交付物；本目录不写入、也不改动它们的任何文件。

### 范围层级：多起点 SLSQP 的定位

`Baseline-C.md` 规定「第三问本轮做到候选配比枚举与基本资源搜索；多起点 SLSQP 连续联合
细调暂留讨论」。据此，本目录把**枚举 + 逐候选 $(N,Q)$ 精确搜索**作为产出结论的主路径；
多起点 SLSQP 只作为 `Q3-C.md` 第 4 条所要求的**有界交叉核验**——仅在少数代表性情景上运行、
限制在单纯形 ∩ 可信区域内、按聚类显式标注局部解，既不作生产求解器，也不用于挑选候选
（候选集不因 SLSQP 结果扩充）。因此移除该项不影响任何主结论。

## 目录结构

```
q3_c/
├── README.md                                  本文件
├── configs/baselines/Q3-C.yaml                冻结全部可调旋钮（改任一项即改变配置哈希）
├── src/baselines/Q3-C/
│   ├── common.py       配置加载、双布局路径解析、哈希、落盘、环境登记
│   ├── model.py        Q3 主模型 M1(rho=0)：成本形式、预算反解、解析梯度、成本份额
│   ├── mixtures.py     第一问质量坐标、可信区域、两类泛函、LP(Charnes–Cooper)、SLSQP、候选枚举
│   ├── search.py       固定 p 的 (N,Q) 精确搜索 + 多起点联合 SLSQP
│   ├── scenarios.py    情景网格、预算路径、转移事件识别、成本份额分段、C7 依据
│   ├── verify.py       五类数值核验 + 输入审计
│   ├── report.py       报告生成
│   └── run_all.py      全流程入口
├── artifacts/baselines/Q3-C/                  全部交付物（见下）
└── reports/baselines/Q3-C.md                  报告
```

## 运行方式

```bash
cd src/baselines/Q3-C
python run_all.py                # 完整运行（约 12 分钟）
python run_all.py --quick        # 冒烟测试：缩小网格与情景，结果不得作为结论
python run_all.py --report-only  # 只从已落盘产物重渲染报告（秒级，不重算数值）
```

`--report-only` 用于**只改措辞、不动数值**的场合：报告是产物的视图，改一句话不该重跑
数值流程。它只读产物目录里的 CSV 重建上下文，并重新生成派生的
`cost_constraint_checks.csv`；报告里的配置哈希与源码哈希沿用 `run_metadata.json`
所记的值（即**产出这批产物的代码**），而非当前磁盘上的源码。

依赖：`numpy`, `pandas`, `scipy`, `pyyaml`, `pyarrow`。环境与版本登记在
`artifacts/baselines/Q3-C/run_metadata.json`。

## 输入与其解析

上游数据仓库 `math_model_F` 与本工作区**并列**，故 `common.py` 把「项目根」
（`q3_c`，存放 configs/artifacts/reports）与「上游根」（`math_model_F`，存放
`real_attachments` 与第一问产物）分开解析，候选根按 `paths.upstream_roots` 顺序探测。

| 输入 | 来源 | 用途 |
| --- | --- | --- |
| `quality_domain_mapping.csv` | Q1-C | 6 个已映射域的冻结质量坐标 $q_i$ |
| `mixture_predictor.json` | Q1-C | 列序、参考配比 $p_0$、训练支持范围 |
| `loss_predictor.json` | Q2-C | 冻结主模型参数 $E,A,\alpha,B,\beta$ |
| `predict_q2c_loss.py` | Q2-C | Q2→Q3 **可执行**接口，用于口径核验 |
| `model_architecture_metadata.csv` | C7 | 上下文长度情景的元数据依据 |

`artifacts/baselines/Q2-C/` 当前只在并列的 `q2_c` 工作区（及分支 `baseline-q2-c`）中，
故 `paths.q2_bridge_root` 显式登记为**只读**备用根。这是临时桥接：Q2-C 并入主仓库后
可删除该项，删除不改变任何结果。实际命中的根目录如实登记在 `run_metadata.json`。

## 交付物

| 文件 | 内容 |
| --- | --- |
| `optimization_scenarios.csv` | 优化情景表（预算 × 成本形式 × 上下文长度 × $Q_0$ 设定） |
| `optimal_allocations.csv` | 每个情景的最优 $(N,D,Q,p)$ 与成本份额 |
| `all_candidate_evaluations.csv` | 每个 (情景, 候选) 的完整评估，供配比通道不变性核验 |
| `candidates.csv` / `finalists.csv` | 候选配比全集与终选集 |
| `budget_paths.csv` | 沿连续 $\log_{10} C$ 网格的最优解轨迹 |
| `transition_candidates.csv` | 按预定义判据识别的结构性转移事件 |
| `cost_constraint_checks.csv` | 成本与约束检查表（交付物要求） |
| `verify_*.csv` | 五类核验明细（导数、约束、配比可行性、网格稳定性、上游一致） |
| `verify_input_audit.csv` | 实际读入文件的哈希与数据角色（证明未读最终检验集） |
| `screening*.csv`, `slsqp_*.csv`, `enum_vs_slsqp.csv` | 筛选、多起点聚类与交叉核验 |
| `q3_report.md` | 报告（同时写入 `reports/baselines/Q3-C.md`） |

## 核心结论与口径要点

- **主模型**：Q2-C 判定质量通道参数 $\kappa,\rho$ 未辨识，主模型退回 M1（$\rho=0$）：
  $\hat L = E + A N^{-\alpha} + B\,[D\,Q^{\kappa}h(p)]^{-\beta}$。$\kappa$ 与 $\omega$ 作情景网格处理。
- **预算紧约束**：$\partial \hat L/\partial D < 0$，故最优解取 $D = C/u$，
  其中 $u = (6+\eta\ell)N + [g(Q)-g(Q_0)]_+$，$\eta = 2\times10^{-4}$。
- **配比通道的解析结论**：$\hat L$ 对 $h(p)$ 完全可分离且 $\partial\hat L/\partial h < 0$，
  故 $\arg\min_p \hat L \equiv \arg\max_p h$，**与 $C,\ell,g,\kappa,\omega$ 无关**。
  因此 $p^\*$ 不随预算转移是正确结果，本流程不强行制造转移。
- **必须如实报告的量级**：候选间 $\hat L$ 的跨度仅约 $10^{-3}$，远低于模型自身误差。
  「哪个 $p$ 最优」由上述单调性论证给出，而不是靠数值分辨 $\hat L$ 的差异。

## 已知限制（详见报告「结论边界与局限」）

- LP 顶点把质量上界全部压在单一已映射域上（退化解）；可信区域单独不足以避免，
  需要多样性/供给约束——而题目未提供此类数据。
- $Q_0$ 的两种设定（固定基准 / 随配比变化）必须分别呈现，不能混同。
- 数值搜索边界与数据支持范围不一致处，一律在 `extrapolation` 列显式标注外推。
