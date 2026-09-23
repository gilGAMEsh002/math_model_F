# Q1 完整流程：数据质量评价 + 冲突消解 + 领域配比建模

本目录是**问题一的自包含完整流程**，由两处代码合并而成：

| 来源 | 内容 | 原位置 |
|---|---|---|
| `math_model_F` 仓库 `baseline-q1-c` 分支 | Q1-C 建模代码、配置、产物、报告、方案文档 | `D:\26mathmodel\math_model_F` |
| Step 1 预处理工程 | 原始 JSONL → 标准化质量矩阵 Z 的代码与产物 | `D:\26mathmodel\q1_preprocessing` |

原分支中 Q1-C 靠 `upstream.preprocessed_dir: ../q1_preprocessing/...` **跨目录**引用预处理产物，
两者缺一不可。本目录把上游并入同一根目录后，整条流程可以独立拷贝、独立运行。

## 一键运行

```bash
cd D:\26mathmodel\q1-c
python run_q1.py
```

| 步骤 | 脚本 | 作用 | 耗时 |
|---|---|---|---|
| `preprocess` | `src/preprocess/quality_preprocess.py` | 原始 JSONL → Z ∈ [0,1]^(N×22) | ~20 s |
| `verify` | `src/preprocess/verify_preprocess.py` | 不信任上一步产物，从原始数据重算核对（46 项） | ~5 s |
| `q1c` | `src/baselines/Q1-C/run_all.py` | 熵权质量分 + 冲突消解 + 随机森林配比回归 | ~93 s |

常用参数：`--skip-preprocess`（复用已有 parquet 只跑建模）、`--skip-verify`、`--only q1c`、`--list`。
某步失败会中止并提示单独重跑的 `--only` 命令。

> **控制台编码**：上游 Q1-C 报告规定的运行方式是 `PYTHONIOENCODING=utf-8`，本入口沿用该约定
> （子进程与其一致）。若中文显示为乱码，先执行 `chcp 65001` 切到 UTF-8 代码页再运行。

## 目录结构

```
q1-c\
├── run_q1.py                          一键入口（本目录新增）
├── src\
│   ├── preprocess\                    ← 原 q1_preprocessing\src\preprocess\
│   │   ├── quality_preprocess.py      Step 1 预处理（方向统一、MinMax、参考集冻结）
│   │   ├── verify_preprocess.py       独立验证，46 项
│   │   └── README.md                  预处理工程说明
│   └── baselines\Q1-C\                ← 原 baseline-q1-c 分支
│       ├── common.py                  IO / 指标 / 哈希 / 环境指纹
│       ├── quality.py                 熵权、去相关、冲突标签、短板消解、域聚合
│       ├── mixture.py                 行归一化、CV 选参、岭回归与森林、排序与扰动
│       ├── diagnostics.py             事后诊断：收益来源、复合口径、扰动标定
│       └── run_all.py                 端到端入口
├── configs\
│   ├── baselines\Q1-C.yaml            Q1-C 全部可调旋钮
│   └── quality_preprocess_params.json 预处理冻结参数
├── artifacts\
│   ├── q1\preprocessed\               Step 1 产物（3 个 parquet，43 MB）
│   └── baselines\Q1-C\                Q1-C 产物（25 个文件）
├── reports\
│   ├── q1_preprocess_report.md        Step 1 报告
│   └── baselines\Q1-C.md              Q1-C 报告（含结论边界）
├── real_attachments\                  原始数据附件（530 MB，2012 文件）
├── 建模方案\                          上游任务卡与实施计划
├── 数据说明_干净版.pdf、赛题 docx      题目原始材料
└── README_上游仓库原文.md             被替换掉的原仓库 README（保留溯源）
```

## 合并时改了什么

**代码一行没动。** 两个上游脚本都用「相对 `__file__`」解出根目录：

- `quality_preprocess.py` / `verify_preprocess.py`：`ROOT = 上两级（src/preprocess → 根）`
- `Q1-C/common.py:repo_root()`：`ROOT = 上三级（src/baselines/Q1-C → 根）`

因此搬迁后路径自动指向新根目录。**唯一改动是 `configs/baselines/Q1-C.yaml` 里上游的三条路径**，
去掉了 `../q1_preprocessing/` 前缀：

```yaml
upstream:
  preprocessed_dir:  artifacts/q1/preprocessed                     # 原 ../q1_preprocessing/artifacts/q1/preprocessed
  preprocess_params: configs/quality_preprocess_params.json        # 原 ../q1_preprocessing/configs/...
  preprocess_report: reports/q1_preprocess_report.md               # 原 ../q1_preprocessing/reports/...
```

配置里其余内容与分支逐字节一致。

> ⚠️ **这会让 `config_hash` 变化。** 原分支报告记录的是 `d15692258d17cbb2`；本副本运行后
> `artifacts/baselines/Q1-C/run_metadata.json` 里的哈希会不同。这是路径变更导致的，
> **不影响任何数值结果**——计算逻辑与随机种子都未变。

## 完整链条

```
real_attachments/A_data_value/slimpajama_quality_{signal_sample,extended}/*.jsonl.xz
   │
   ├─ src/preprocess/quality_preprocess.py
   │     字段筛选 → 缺失填充(A1中位数) → 列表型压缩 → dsir 对数变换
   │     → 方向统一(负向补转换 1-MinMax / 非单调 log空间梯形隶属度) → Z ∈ [0,1]^(N×22)
   │     参数冻结在 A1 上，原样套用到 A2/A3
   ▼
artifacts/q1/preprocessed/A{1,2,3}_processed.parquet   (51230 / 17523 / 203752)
   │
   ├─ src/baselines/Q1-C/quality.py
   │     子任务1：熵权法 w_j = (1-e_j)/Σ(1-e)  → Q_entropy = Z·w；等权对照
   │     子任务2：7 语义组 → c = max_g z_ig − min_g z_ig → τ=q90(eq 0.7238) → Q_resolved
   ▼
   ├─ src/baselines/Q1-C/mixture.py     （配比来自 real_attachments/A_data_value/regmix_tables）
   │     子任务3：17 域配比 → 13 域 Loss，随机森林 vs 岭回归，5 折 CV 选参
   │     三档评估：1M 绝对误差 / 60M·1B 跨尺度排序 / 10B·70B 仅外推诊断
   ▼
artifacts/baselines/Q1-C/*（25 个文件） + reports/baselines/Q1-C.md
```

## 复现验证

本副本已用 `python run_q1.py` 全流程重跑一遍，与分支自带的产物逐键比对：

| 产物 | 比对结果 |
|---|---|
| `A{1,2,3}_processed.parquet` | 形状、列序、id 对齐全部一致，**Z 最大绝对差 = 0.00e+00（逐位相同）** |
| `quality_entropy_weights.csv` / `quality_domain_aggregate.csv` | **完全一致** |
| `mixture_cv_grid.csv`（25 行） | 全部匹配，数值最大差 2.2e-16 |
| `mixture_metrics_same_scale_1m.csv`（140 行） | 全部匹配，数值最大差 2.2e-16 |
| `mixture_ranking_cross_scale.csv`（420 行） | 全部匹配，数值最大差 8.9e-16 |

差异仅剩浮点末位舍入与 `cv_seconds`（计时本身），**结论是数值上完全可复现**。
环境一致：Python 3.11.6 / sklearn 1.5.2 / pandas 3.0.3。

> 注意：`mixture_cv_grid.csv` 是按分数排序的，若有配置分数并列，行序会换位。
> 逐行比对会误报大差异，**必须按 `label` 主键合并后再比**。

## 已核实的数据事实

- `real_attachments` 与 `q1_preprocessing\real_attachments` **文件清单完全一致**（2012 文件 / 530 MB）。
- 三个 `*_processed.parquet` 均落在 `[0,1]`，域分布与 Q1-C 报告的样本流转一致。

### 分支报告中两处与产物不符的描述

这两处都是**报告文字写错，代码与产物本身没有问题**：

1. **报告 §10 的 `quality_scores.parquet` 行数**。报告写「37.9 MB，261,086 条文档级得分」。
   实际是 **272,505 行**——即 A1+A2+A3 拼接（**未去重**，带 `source_set` 列），文件大小 37.9 MB 是对的。
   261,086 是**去重后**的行数（见 `quality_sample_flow.csv` 的 `merged_rows`），只出现在域级聚合里。
   该文件本身正合赛题「须使用全量质量信号」的要求，只是报告把口径写成了去重后的数字。

2. **报告 §6.1 与 §7 的候选数**。报告写「7 个岭回归 α + 21 个森林配置 = 28 个候选」，
   而 `configs/baselines/Q1-C.yaml` 的网格是
   `n_estimators[300,600] × max_depth[null,8,12] × min_samples_leaf[1,2,4] × max_features[1.0]`
   = **18 个森林 + 7 个岭回归 = 25 个候选**，与 `mixture_cv_grid.csv` 的 25 行、以及
   `run_all.py` 自己打印的日志一致。选出的最优配置 `rf_n600_d12_l1_f1.0`（CV 0.38716）本身是对的。

## 依赖环境

Python 3.11 + numpy / pandas / scikit-learn / scipy / pyarrow / pyyaml。
原分支实测环境见 `reports/baselines/Q1-C.md` §2（sklearn 用 1.5.2，1.9.1 在本机有 DLL 加载问题）。
