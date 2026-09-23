# 轻量验收结果（E0-A）

- run_id: `20260923T165441Z_acceptance_2974389`
- 汇总: {'n': 14, 'pass': 12, 'fail': 0, 'blocked': 1, 'n_a': 1, 'critical_failures': []}

| id | check | status | expected | actual | 说明 |
|---|---|---|---|---|---|
| 1.C | C Q2-C vs Q3-C at physical anchor | **pass** | `{"q2": 2.385578, "q3": 2.385578}` | `{"q2": 2.385578185266307, "q3": 2.385578185266307}` | Q2-C 预测器（十亿坐标）与 Q3-C 模型（实际个数，内部换算） |
| 1.B | B Q2 params vs Q3-B loss at physical anchor | **pass** | `2.385578` | `2.385578178410373` | Q3-B loss() 修复后应与十亿坐标经典项一致 |
| 1.A | A Q2/Q3 loss at physical anchor (Q=1) | **pass** | `2.385578` | `2.3855781852651727` | A 全链同一 loss_predictor.json |
| 2.C | C cost recomputed from actual N,D; residual | **pass** | `"<=1e-9"` | `3.1086244689504383e-15` | 用实际个数回算 6ND+ηNDℓ+D·c(Q) 与预算 C 的残差 |
| 2.A | A cost_check presence + loss predictor units | **pass** | `"exists"` | `true` | A Q3 成本份额与约束残差见 artifacts/baselines/Q3-A/ |
| 2.B | B cost constraint max relative error | **pass** | `"<=1e-9"` | `2.22045e-16` | B Q3 影子成本回算 |
| 3.C-deriv | C right derivative at Q=Q0 vs forward FD | **pass** | `{"g_analytic": -0.09447806248413058, "g_fd": -0.09447798216504791}` | `"rel<=1e-3"` | 可行域 Q>=Q0 的右导数（旧实现会置零） |
| 3.C-trust | C main optimal table all in_trust_l1 | **pass** | `true` | `true` | rows=720; 区域外候选单列 optimal_allocations_outside_l1.csv |
| 3.B | B trust-region constraint | **n/a** | `"n/a"` | `"n/a"` | B 第三问不含 p 可信域约束（固定配比） |
| 4.C-predictor | C Q1 mixture predictor executable for arbitrary p | **blocked** | `true` | `false` | 仅描述符；未持久化森林系数/模型对象，无法对任意合法 p 调用 |
| 4.C-descriptor | C Q1 descriptor well-formed (17 in / 13 out / v / p0) | **pass** | `{"n_inputs": 17, "n_outputs": 13, "keys_ok": true}` | `{"n_inputs": 17, "n_outputs": 13, "keys_ok": true}` | 描述符协议检查 |
| 4.C-scale | Cross-question quality scale frozen (Q2-C calibration) | **pass** | `"exists"` | `{"q2": true, "q3": false}` | Q3-C 的 h(p) 尺度取自 Q2-C 冻结的 scale_s，避免逐候选重标定 |
| 5.planned | experiments.csv: executed=true has run_id; no done-without-run | **pass** | `{"bad_executed": [], "done_without_executed": []}` | `{"bad_executed": [], "done_without_executed": []}` | 共 5 条登记（含 R1/R2/R3，均为 pending） |
| 6.fix | fix evidence present (A tests / B test / C anchor+kappa) | **pass** | `{"A_tests": true, "B_unit_test": true, "C_unit_anchor": true, "C_kappa_ablation": true}` | `{"A_tests": true, "B_unit_test": true, "C_unit_anchor": true, "C_kappa_ablation": true}` | 代码已修≠全量重跑；三项状态在 decisions.md 分开登记 |

> 状态口径：pass 通过；fail 执行了但不符预期（实现问题）；blocked 无法执行（缺输入/未持久化），**不计为通过**；n/a 该线不适用。
