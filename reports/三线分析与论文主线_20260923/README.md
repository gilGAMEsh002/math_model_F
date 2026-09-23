# 三线分析与论文主线

> **2026-09-24 更新（重要）**：本分析交付时发现的 B/C 第三问单位错配等问题**已在各自分支修复并全表重算**：
> - `baseline-b` 提交 `e498b2c`（见 `results/Q3_UNIT_FIX_NOTES.md`）
> - `baseline-q3-c` 提交 `7e7e590`（见 `reports/baselines/Q3-C_FIX_NOTES.md`）
>
> 本目录 `evidence/`、`new_branch_verification.json`、`c3_*` 等是**修复前的取证快照**，其中 B/C 第三问的
> 数值（如 1.691141 反例、lp_vertex_box 顶点推荐、启动阈值 19.6–21.1）**已失效**，仅作历史诊断记录。
> 论文与后续比较必须使用修复后的产物，不得引用本目录中的旧 B/C Q3 数值。

以完整上传后的 A、B、C 三问成果为准；C 分布于 `baseline-q1-c`、`baseline-q2-c`、`baseline-q3-c`，B 另有第四问。

1. [三条线分析报告](./01_三条线分析报告.md)：结果比较、适用范围、实现问题与整合建议。
2. [论文故事线与完整大纲](./02_论文故事线与完整大纲.md)：四问逻辑、候选公式、验证要求与最小实验矩阵。
3. [三线指标表](./q1_three_lines_verified.csv)：35 组已独立复核记录。
4. [关键问题核验](./new_branch_verification.json)：B/C 第三问单位反例、C 可信域检查及证据。

`evidence/` 为固定 Git 提交的小型结果、源码、配置与报告快照。A/B/C/C2/C3 的完整提交号及文件哈希见 `manifest.json`，不会随远程分支前进而静默改变。原 baseline 与原始附件未修改。

本次只核验已有产物与公式；未重新训练全模型，未修复或重跑 B/C 第三问。`c3_rescore_diagnostic.csv` 是对旧决策点按正确单位重评分的诊断，不能当作修复后的最优解。

在仓库根目录运行（需要 numpy、pandas、scipy、PyYAML、matplotlib）：

```bash
python3 reports/三线分析与论文主线_20260923/collect_evidence.py
python3 reports/三线分析与论文主线_20260923/verify_new_branches.py
python3 reports/三线分析与论文主线_20260923/plot_comparison.py
```

脚本仅写本目录。`verification.json` 对应 A/C1 既有结果；`new_branch_verification.json` 补充新上传 B/C2/C3 的检查。图中的 10B/70B 表为估算数据，不能解释为真实大规模验证。
