# R2b：来源差异 vs 质量机制（最小辨别实验）

- seed=20260924，bootstrap=2000 次/源
- 经典参数：`artifacts/baselines/Q2-A/loss_predictor.json`（B1 冻结，全源同一套，未按源重拟合）
- 坐标：`n=N_params_B=N/1e9`，`d=D_tokens_B=D/1e9`（十亿坐标）；`L0` 由 A 的 `scaling.py` 按路径导入
- 模型：M0=L0；M1=L0+δ；M2=L0+δ+γ(1−Q), γ≥0；M3=M2 但 γ 无约束（仅诊断）

## 每源结果

| 源 | n | RMSE M0 | RMSE M1 | RMSE M2 | δ(M1) | γ(M2) | γ 95%CI | 边界 | 残差corr(1−Q) |
|---|---|---|---|---|---|---|---|---|---|
| B6 | 360 | 0.2172 | 0.1374 | 0.0761 | 0.1682 | 0.3622 | [0.337, 0.389] | 否 | 0.8329 |
| B7 | 450 | 0.2073 | 0.1271 | 0.0731 | 0.1638 | 0.3620 | [0.336, 0.386] | 否 | 0.8182 |
| B8 | 1704 | 1.1825 | 0.6582 | 0.6582 | -0.9823 | 0.0000 | [0.000, 0.000] | 命中0边界 | -0.9669 |
| B6::B1_support | 168 | 0.2180 | 0.1370 | 0.0709 | 0.1696 | 0.3709 | [0.335, 0.404] | 否 | 0.8557 |
| B8::calibrated | 984 | 1.1825 | 0.6727 | 0.6727 | -0.9725 | 0.0000 | [0.000, 0.000] | 命中0边界 | -0.9816 |
| B8::extrapolated | 720 | 1.1823 | 0.6377 | 0.6377 | -0.9956 | 0.0000 | [0.000, 0.000] | 命中0边界 | -0.9468 |

## 判定

- **来源偏移是否主导**：`True` — On B8 (the only large-scale set) the additive source offset removes the dominant share of SSE and the non-negative quality term adds exactly nothing (gamma lands on the zero boundary). On the semi-synthetic B6/B7 the quality term adds more than the offset alone, but that family is not real quality evidence and its gamma has the opposite sign to B8, so no unified quality coefficient survives.
  - 注意：value=true is driven by B8 + the B6/B7-vs-B8 sign conflict. Within B6/B7 alone the offset does NOT dominate the fit improvement.
- **γ 是否可辨识（单一正系数）**：`False`
  - B8's gamma is censored at the zero boundary (unconstrained estimate is negative), so a positive quality coefficient is not identified there; B6/B7 are separable but semi-synthetic. A single unified gamma across sources is therefore NOT forced.

## 三段必须分开的结论

### 1. 评分定义敏感性
- 是什么：Sensitivity of the fitted quality coefficient to how the (single) text quality score is parameterised before entering as (1-Q): identity, within-source z-score, min-max, rank-uniform, square and sqrt.
- 支持：The sign and boundary behaviour of gamma are stable under these monotone reparameterisations for each source; B6/B7 stay positive, B8 stays on the zero boundary.
- **不支持**：The MAGNITUDE of gamma is not stable (e.g. identity vs z-score differ several-fold), and these data contain no second, independently defined quality score, so this does NOT test sensitivity to a genuinely different text-score definition or establish that the score content is correct.

### 2. 半合成可辨识性
- 是什么：Identifiability / separability of delta and gamma INSIDE the semi-synthetic quality families B6 and B7. Q varies within every (N,D) cell and 1-Q is orthogonal to L0, so the [1, 1-Q] design is well conditioned.
- 支持：Within B6/B7 the source offset and the quality term ARE separable, gamma is positive and its bootstrap CI excludes 0. This establishes that the additive quality form is identifiable and fittable on a semi-synthetic grid.
- **不支持**：It does NOT establish a real data-quality mechanism: B6/B7 are semi-synthetic, the effect may be present by construction, and the positive sign does not carry over to B8.

### 3. 真实质量证据
- 是什么：Evidence from B8, the large-scale set with calibrated (observed-style) and extrapolated rows, including a calibrated-only subset.
- 支持：Under the frozen classical model, a single additive source offset absorbs the bulk of the improvement, and no non-negative quality term is supported: gamma is censored to the zero boundary and its CI includes 0. This holds on the calibrated-only subset too.
- **不支持**：These data do NOT support a positive additive quality mechanism. The strong negative residual association (corr ~ -0.97) is an ASSOCIATION confounded with scale/classical misfit and data_type; it is not an estimated negative quality effect (the constraint prevents a negative gamma from being reported).

## 约束与注意事项

- A non-negative constraint that pushes gamma to the zero boundary (B8) is NOT a negative quality effect; it is a censored estimate and is reported as such.
- resid_corr_1mQ is a residual ASSOCIATION, not the fitted gamma effect. B8's ~-0.97 correlation means higher 1-Q co-occurs with larger residuals under L0, confounded with scale/classical misfit; it must not be read as an estimated negative quality mechanism.
- B6/B7 are semi-synthetic quality-supplementary families. A positive gamma there speaks to the functional form / separability, not to real-world data quality.
- B8 mixes calibrated (984 rows) and extrapolated (720 rows); the extrapolated rows lie far outside A's B1 support (N up to 700B vs B1 max ~12B), so classical L0 misfit dominates. Calibrated-only is reported separately and shows the same sign/boundary behaviour.
- The classical parameters are frozen from Q2-A/B1 for every source; no per-source classical refit is performed here (that would be a labelled alternative).
- B6/B7 and B8 are not pooled (protocol: opposite gamma sign, non-poolable).
- delta is a per-source additive offset, not a physical parameter.
- Because M2 includes the source offset delta, the fitted gamma differs slightly from Q2-A's frozen no-intercept gamma on B6 (0.36225 here vs 0.36328 in gamma_estimates.json); the offset is part of the R2b protocol and this difference is expected, not a discrepancy.
- No B quality-rebuild LFS data was available or used.
- Only one quality-score column exists in these CSVs; score_definition_sensitivity therefore tests monotone reparameterisations/standardisations, not a genuinely different score definition.

> `identifiable: false` 与“γ 命中零边界”均为允许结论；不强行给出跨源统一系数。
