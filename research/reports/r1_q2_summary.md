# R1-Q2 来源分层比较（source-stratified Q2 comparison）

- 生成: 2026-09-24T00:23:52Z
- 上游提交: A `c8b97cf4` / B `e498b2c6` / C `cf765761`
- 本文只复现既有产物、不重新拟合任何模型；每行数字带 artifact_path 与 commit。
- `fit_mode`：`refit_per_source_internal`=在该源上估计（样本内，**不是泛化证据**）；
  `fixed_model_cross_source`=用他处拟合的参数套用该源（唯一可作迁移证据的模式）。两者不得混用。

## 0. 长表（reports/r1_q2_sources.csv）

| line | model_form | source | fit_mode | nature | role | n | rmse | r2 |
|---|---|---|---|---|---|---|---|---|
| A | classical | B1 | refit_per_source_internal | 真实 | fit | 1176 | 0.0001465 | 1 |
| A | classical | B2 | fixed_model_cross_source | 半合成 | validation | 1029 | 1.243 | -5.073 |
| A | classical | B4 | fixed_model_cross_source | 真实 | validation | 57 | 0.2927 | 0.6045 |
| A | classical | B5 | fixed_model_cross_source | 真实 | validation | 44 | 0.1976 | 0.7306 |
| A | additive_quality | B6 | refit_per_source_internal | 半合成 | fit | 360 | 0.07607 | 0.6936 |
| A | additive_quality | B7 | refit_per_source_internal | 半合成 | fit | 450 | 0.07306 | 0.6694 |
| A | additive_quality | B8 | refit_per_source_internal | 半合成 | fit | 1704 | 0.1732 | 0.9308 |
| B | classical | B1 | refit_per_source_internal | 真实 | fit | 1176 | 0.0001465 | 1 |
| B | classical | B2 | refit_per_source_internal | 半合成 | fit | 1029 | 0.01076 | 0.9995 |
| B | classical | B4 | refit_per_source_internal | 真实 | fit | 57 | 0.08818 | 0.9641 |
| B | classical | B5 | refit_per_source_internal | 真实 | fit | 44 | 0.1317 | 0.8804 |
| B | classical | B6 | refit_per_source_internal | 半合成 | fit | 360 | 0.129 | 0.8603 |
| B | classical | B7 | refit_per_source_internal | 半合成 | fit | 450 | 0.1182 | 0.8795 |
| B | classical | B8 | refit_per_source_internal | 半合成 | fit | 1704 | 0.651 | 0.1284 |
| B | classical | B10 | refit_per_source_internal | 估算 | fit | 128 | 2.914e-05 | 1 |
| B | classical | B4 | fixed_model_cross_source | 真实 | validation | 57 | 0.2927 | 0.6045 |
| B | classical | B5 | fixed_model_cross_source | 真实 | validation | 44 | 0.1976 | 0.7306 |
| B | classical | B10 | fixed_model_cross_source | 估算 | validation | 128 | 0.001094 | 1 |
| B | classical | B2 | fixed_model_cross_source | 半合成 | validation | 1029 | 1.243 | -5.073 |
| B | classical | B3 | fixed_model_cross_source | 插值 | validation | 4000 | 0.003821 | 1 |
| B | effective_token | B6 | refit_per_source_internal | 半合成 | fit | 360 |  |  |
| B | effective_token | B7 | refit_per_source_internal | 半合成 | fit | 450 | 0.1147 |  |
| B | effective_token | B8 | refit_per_source_internal | 半合成 | fit | 1704 | 1.185 |  |
| C | classical | B1 | refit_per_source_internal | 真实 | fit | 1176 | 0.0001465 | 1 |
| C | classical | B10 | fixed_model_cross_source | 估算 | validation | 128 | 0.001094 | 1 |
| C | classical | B2 | fixed_model_cross_source | 半合成 | validation | 1029 | 1.243 | -5.073 |
| C | classical | B4 | fixed_model_cross_source | 真实 | validation | 57 | 0.2927 | 0.6045 |
| C | classical | B5_Chowdhery et al. 2022 | fixed_model_cross_source | 真实 | validation | 3 | 0.3304 | -2.545 |
| C | classical | B5_Hoffmann et al. 2022 | fixed_model_cross_source | 真实 | validation | 16 | 0.1633 | 0.7931 |
| C | classical | B5_Kaplan et al. 2020 | fixed_model_cross_source | 真实 | validation | 9 | 0.1437 | 0.7422 |
| C | classical | B5_Scao et al. 2022 | fixed_model_cross_source | 真实 | validation | 4 | 0.2074 | 0.4928 |
| C | classical | B5_Touvron et al. 2023 | fixed_model_cross_source | 真实 | validation | 7 | 0.2159 | -3.606 |
| C | classical | B5_Zhang et al. 2022 | fixed_model_cross_source | 真实 | validation | 5 | 0.2365 | -0.02402 |
| C | classical | B6 | fixed_model_cross_source | 半合成 | validation | 360 | 0.2172 | 0.6036 |
| C | classical | B7 | fixed_model_cross_source | 半合成 | validation | 450 | 0.2073 | 0.6296 |
| C | classical | B8_calibrated | fixed_model_cross_source | 半合成 | validation | 984 | 1.183 | -1.774 |
| C | classical | B8_extrapolated | fixed_model_cross_source | 半合成 | validation | 720 | 1.182 | -2.448 |
| C | dual_channel | B1 | refit_per_source_internal | 真实 | fit | 1176 | 0.0001465 | 1 |
| C | dual_channel | B6 | refit_per_source_internal | 半合成 | fit | 360 | 0.0675 | 0.9617 |
| C | dual_channel | B10 | fixed_model_cross_source | 估算 | validation | 128 | 0.001094 | 1 |
| C | dual_channel | B2 | fixed_model_cross_source | 半合成 | validation | 1029 | 1.243 | -5.073 |
| C | dual_channel | B4 | fixed_model_cross_source | 真实 | validation | 57 | 0.2927 | 0.6045 |
| C | dual_channel | B5_Chowdhery et al. 2022 | fixed_model_cross_source | 真实 | validation | 3 | 0.3304 | -2.545 |
| C | dual_channel | B5_Hoffmann et al. 2022 | fixed_model_cross_source | 真实 | validation | 16 | 0.1633 | 0.7931 |
| C | dual_channel | B5_Kaplan et al. 2020 | fixed_model_cross_source | 真实 | validation | 9 | 0.1437 | 0.7422 |
| C | dual_channel | B5_Scao et al. 2022 | fixed_model_cross_source | 真实 | validation | 4 | 0.2074 | 0.4928 |
| C | dual_channel | B5_Touvron et al. 2023 | fixed_model_cross_source | 真实 | validation | 7 | 0.2159 | -3.606 |
| C | dual_channel | B5_Zhang et al. 2022 | fixed_model_cross_source | 真实 | validation | 5 | 0.2365 | -0.02402 |
| C | dual_channel | B7 | fixed_model_cross_source | 半合成 | validation | 450 | 0.06501 | 0.9636 |
| C | dual_channel | B8_calibrated | fixed_model_cross_source | 半合成 | validation | 984 | 1.398 | -2.875 |
| C | dual_channel | B8_extrapolated | fixed_model_cross_source | 半合成 | validation | 720 | 1.31 | -3.234 |

共 51 行。

## 1. 固定模型跨源预测 vs 逐源重拟合

B 的同一批来源既有『B1 冻结参数直接外推』(q2_validation_sources.csv)，又有『每源独立重拟合』(q2_source_diagnosis.csv)，可直接对照：

| source | nature | n_fixed | rmse_fixed | rmse_refit |
|---|---|---|---|---|
| B2 | 半合成 | 1029 | 1.243 | 0.01076 |
| B4 | 真实 | 57 | 0.2927 | 0.08818 |
| B5 | 真实 | 44 | 0.1976 | 0.1317 |
| B10 | 估算 | 128 | 0.001094 | 2.914e-05 |

→ 逐源重拟合普遍把误差压低一个量级以上（B4 0.2927→0.0882、B5 0.1976→0.1317、B2 1.2431→0.01076、B10 1.094e-3→2.914e-5）。这说明固定模型跨源的误差大部分是**来源水平差**，而不是形状律失效。artifact: `line-b/results/q2_validation_sources.csv` / `line-b/results/q2_source_diagnosis.csv`。

A 只有一次拟合（B1），B2/B4/B5 均为 B1 冻结参数外推，没有逐源重拟合；C 的 M0 同样只在 B1 拟合。因此『跨源 vs 重拟合』的干净对照只有 B 提供。

## 2. 哪些模型形式在哪些源上被真正拟合

- A：`classical` 在 B1 拟合；`additive_quality` 的 γ 在 B6(主)/B7/B8 上拟合，**经典参数冻结自 B1**（`line-a/artifacts/baselines/Q2-A/gamma_estimates.json`）。
- B：`classical` 在 B1 拟合（`q2_scaling_parameters.json.classic`）；`effective_token` 的 κ 在 B6 拟合，并在 B7/B8 上重拟合（`q2_quality_alt_datasets.csv`）；h(p) 的 ω 无共同实验，列为情景、不可辨识。
- C：`classical`(M0) 在 B1 拟合；`dual_channel`(M2) 的 κ,ρ,δ 在 B1∪B6 拟合（`scaling_parameters.json.stages`）。M1(仅 κ) 只作为嵌套阶段存在，没有逐源验证表。

## 3. 来源偏移（source offset）是否解释了主要改进：是

C 的嵌套 SSE 分解在 B1∪B6 (n=1536) 上（artifact `line-c/artifacts/baselines/Q2-C/scaling_parameters.json`）：

| 阶段 | SSE | 相对上一步 | 占总降幅 |
|---|---|---|---|
| M0 无偏移 | 16.9891 | — | — |
| +δ 来源偏移 | 6.7998 | −10.1893 | 66.4%（占嵌套总降幅；相对 M0 为 59.98%） |
| +κ 质量(有效 token) | 3.4198 | −3.3801 | 22.0% |
| +ρ 质量(参数通道) | 1.6404 | −1.7793 | 11.6% |

嵌套 F 检验（**单看偏移**）：M0→M0+δ  F=2292.6, df=(1,1530), p=0；M0δ→M1  F=1511.2, p=0；M1→M2  F=1657.4, p=0。

来源偏移 δ 只有一个分组（质量族 B6/B7/B8 加 δ，B1 族不加），δ=0.168237；它恰好等于 B6 在 M0 下的偏差：C 表 B6 bias=-0.168237。也就是说 M0→M0+δ 的『主要改进』本质是减去质量族的一个水平差。

B 侧的同向证据：B4/B5/B2/B10 固定跨源 RMSE 远大于逐源重拟合（第 1 节）；B 诊断 verdict 对 B1/B2/B10 判为『公式生成（残差≈舍入量级）』、对 B4/B5/B6/B7/B8 判为『含真实散布』（`line-b/results/q2_source_diagnosis.csv`）。固定加入顺序下，偏移占嵌套总 SSE 降幅的 66%（相对 M0 的下降为 59.98%），是四步中最大的一步；该比例为顺序分解的描述量，不代表独立因果贡献。

## 4. 偏移之后质量项的额外增益与可辨识性

偏移之后：κ 再降 SSE 3.3801 (F=1511.2)，ρ 再降 1.7793 (F=1657.4)，p 均≈0——数字上极显著。

C 的 M2 staged 可辨识性判定：`dual_channel_identifiable=True`，κ=0.3821 CI95=[0.345,0.4185]，ρ=0.5387 CI95=[0.5168,0.5602]，profile=ok/ok，κ–ρ 相关=-0.656（`line-c/artifacts/baselines/Q2-C/scaling_parameters.json`）。

**但这个可辨识性只在半合成质量族内部成立**，不能当作真实新定律：
- B8 大模型质量集上 κ、ρ 双双命中下界、经典 M0 与七参 M2 给出相同 RMSE（`line-c/artifacts/baselines/Q2-C/quality_source_sensitivity.csv`：kappa=1e-19, rho=2.36e-19, boundary_hits={'kappa': 'lower', 'rho': 'lower'}），与 B6/B7 直接矛盾。
- C 自判：『质量双通道在 B6/B7 内部可辨识……但 (a) B1 这一真实观测上根本没有可用残差，(b) B8 的 κ=ρ=0 与 B6/B7 直接矛盾。故不得把 B6/B7 的拟合当作真实新定律。』
- C 的 Q 坐标同尺度映射声明为**未验证假设**；h(p) 的 ω **不可辨识**，只有情景网格。

## 5. 明确告警（caveats）

1. **A 的 γ 用 B6/B7/B8 拟合、经典参数冻结自 B1**；且 **B8 的 γ 符号相反**（γ=−2.0912，B6/B7 约 +0.363），不可池化。
2. **B 的质量项用同源重建集**（不同样本；`line-b/data/quality_rebuild/A*.jsonl.xz` 仍是 LFS 指针、未 `git lfs pull`），因此 B 的质量结果**不构成统一质量消融**，不能与 A/C 的质量项直接合并比较。
3. **κ/ρ 不可辨识为真实定律**：其可辨识性只在 B6/B7 半合成族内成立；真实 B1 对其零信息，B8 给出矛盾的零信息。ω 完全不可辨识。
4. 三条线 Q 坐标不同（A 22 维等权+冲突消解，B 组内均值/组间几何均值+同源重建 A1=50412，C 熵权 6 域），跨线 Q 数值不可直接排名或相减。
5. B1 虽标注真实，但 B/C 诊断判其残差≈4 位小数舍入（公式生成）；B3 为插值，B10 为估算——这些来源不得写成真实实验。
6. `fit_mode` 两类不得混用：样本内重拟合不能当作泛化证据，固定模型跨源才是迁移证据。
7. B 的 B6 `effective_token` RMSE 未持久化（仅 stdout），故该行 rmse 留空，仅给 κ 与 CI。
8. B9 在本轮产物中不存在（仅在 source_levels 文字中提及），未纳入长表。

## 6. 无法解析/缺失的输入

- `line-b/data/quality_rebuild/A*.jsonl.xz`：LFS 指针（134B），未 pull，无法复核 B 质量重建；不影响本表的既有数字。
- B6 `effective_token` 的 RMSE：B 未写入任何 results 文件（见上）。
- A/B 的表没有 `bias`/CI 列（B diagnosis、A scaling_validation）→ 对应单元格留空，不以 0 代替。

