# Baseline C 汇总（完整三问）

- 分支：`baseline-c`
- 构成：`baseline-q2-c`（含 Q1-C + Q2-C）+ 合并 `baseline-q3-c`（已修复版）+ 从 `baseline-q1-c-fix` 引入的自包含预处理。
- 生成方式：**不重算**，直接整合各分支已提交且已修复的产物；本文件为索引与结论边界，各问细节见对应报告。

| 问 | 代码 | 配置 | 产物 | 报告 |
|---|---|---|---|---|
| 预处理 | `src/preprocess/quality_preprocess.py`、`verify_preprocess.py` | `configs/quality_preprocess_params.json` | `artifacts/q1/preprocessed/A{1,2,3}_processed.parquet` | `reports/q1_preprocess_report.md` |
| Q1-C | `src/baselines/Q1-C/` | `configs/baselines/Q1-C.yaml` | `artifacts/baselines/Q1-C/` | `reports/baselines/Q1-C.md` |
| Q2-C | `src/baselines/Q2-C/` | `configs/baselines/Q2-C.yaml` | `artifacts/baselines/Q2-C/` | `reports/baselines/Q2-C.md` |
| Q3-C | `src/baselines/Q3-C/` | `configs/baselines/Q3-C.yaml` | `artifacts/baselines/Q3-C/` | `reports/baselines/Q3-C.md` |

一键运行 Q1 全链：`python run_q1.py`（`--skip-preprocess`/`--skip-verify`/`--only q1c`）。
Q2/Q3：在对应目录运行 `python src/baselines/Q2-C/run_all.py`、`python src/baselines/Q3-C/run_all.py`。
本分支已自包含：Q1-C 的预处理路径、Q2-C 与 Q3-C 的上游接口均在本仓库内解析，无需外部 `../q1_preprocessing` 或桥接目录。

## 关键结果（1M 同尺度，13 域等权 Loss）

| 配比模型 | RMSE | R² |
|---|---:|---:|
| 岭回归（等权质量） | 0.22765 | 0.328 |
| 岭回归（无质量） | 0.22983 | 0.315 |
| 随机森林（无质量） | 0.11874 | 0.817 |
| 随机森林（熵权质量特征） | 0.11658 | 0.824 |
| 随机森林（proxyQ） | 0.11747–0.11750 | ~0.821 |
| 随机森林（逐目标，事后调参） | 0.04773 | 0.970 |

- 质量特征相对无质量基线改进约 1.8%（熵权），proxyQ 约 1.1%；**不能**据此声称质量因果效应。
- Q2-C：主模型推荐退回 M1（Q2-B, ρ=0），`kappa` 未辨识（M1_staged 0.8047；M2_staged κ=0.3821、ρ=0.5387）。来源偏移是嵌套 SSE 最大首段改善。
- Q3-C（修复后）：C=1e22、ℓ=4096、`quality_linear`、ω=1 时 Q*=1（上界），N*≈48.5B、D*≈29.9B、L≈1.9006；主表全部落在可信区域内（`in_trust_l1=True`）。

## P0 修复状态

- **Q3-C 单位错配已修复**：锚点 N=1e9,D=1e11,Q=1 → 2.385578（旧 1.691141）；主表不再混入可信区域外候选；新增真实 κ 消融 `kappa_ablation.csv`（5 个 κ × 6 代表情景）；`dL/dQ` 改为 Q≥Q0 的右导数。详见 `reports/baselines/Q3-C_FIX_NOTES.md`。
- **Q1-C 自包含化**：预处理已并入，`configs/baselines/Q1-C.yaml` 三条上游路径改为本目录相对路径（数值结果与旧分支逐位一致，仅浮点末位/行序差异）。
- Q2-C 未改：其 `loss_predictor.json` 默认参数为 M2、而推荐主模型为 M1/ρ=0；Q3-C 通过显式参数使用 M1，故本轮结论不受影响，但该接口默认值仍待修正。

## 结论边界（写论文时必须保留）

1. 质量分数坐标：C 使用熵权/合理映射后的 6 域 entropy 分，与 A/B 的 Q 不可直接比较或排名。
2. κ、ρ 未通过可辨识性判定，只能作情景参数；Q3 结论对 κ 不敏感（已用真实运行清单支撑）。
3. Q1-C 的随机森林优势**没有**以可执行预测器形式传递到 Q2/Q3；Q2/Q3 使用 `quality_linear`/`mixture_l1` 替代。
4. 替代表尺度的逐对归一化（`substitution_table`）属 P1，未修；不能据其做领域投入排序。
5. 10B/70B 表为估算/外推，仅作一致性诊断，不作真实大规模验证。
6. 复现环境：作者记录 sklearn 1.5.2；不同 sklearn 版本会带来末位差异（预测器差异 ~1e-6），建议固定版本。
