# R3a：评分坐标对齐与迁移假设审计

- 匹配记录数：51230；A 侧子进程：Z_A rows 51230

| 对齐 | Spearman |
|---|---:|
| S_A_equal vs S_Aprep_Cweight | 0.6874 |
| S_A_equal vs S_Cprep_equal | 0.8941 |
| S_A_equal vs S_C_entropy | 0.6994 |
| S_A_equal vs S_C_resolved | 0.7070 |
| S_A_equal vs S_C_entropy_recomputed | 0.6994 |

- A↔C 平均绝对差：0.2562
- 单调映射（isotonic，fit/验证各半）：{"method": "isotonic", "valid_spearman": 0.7015397492521693, "valid_rmse": 0.057053496012467717}

- 分解：预处理差异 = S_Cprep_equal−S_A_equal；赋权差异 = S_Aprep_Cweight−S_A_equal；冲突规则 = S_C_resolved−S_C_entropy。
- **B**：未在 A1 上应用（同源重建样本 + LFS 未 pull），记为 blocked/假设分支。

> **关键边界**：文本评分对齐 ≠ 与 B6–B8 的 `Q_score` 的真实训练收益锚点；缺共同观测时跨附件映射只作假设情景。
