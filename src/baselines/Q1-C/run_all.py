# -*- coding: utf-8 -*-
"""Q1-C 端到端运行入口。

用法（在仓库根目录 math_model_F 下）：
    python src/baselines/Q1-C/run_all.py

产出：
    artifacts/baselines/Q1-C/*.csv|parquet|json
    reports/baselines/Q1-C.md  （由本脚本给出运行摘要，正式报告单独撰写）

元数据按 00_公共约定与接口.md「每份结果的最小元数据」落盘：
    baseline_id、配置哈希、上游版本、数据来源及哈希、训练/验证/检验样本量、
    随机种子、软件环境、运行命令、指标定义、外推标记、区间生成方法。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import quality  # noqa: E402
from common import (METRIC_ORDER, Timer, artifacts_dir, config_hash, env_info,
                    git_rev, hash_if_exists, load_config, load_pairs,
                    preprocessed_dir, save_json, save_table, utc_now)  # noqa: E402
from diagnostics import run_diagnostics  # noqa: E402
from mixture import quality_lookup, run_mixture, short_target  # noqa: E402

RUN_COMMAND = "python src/baselines/Q1-C/run_all.py"


def build_predictor(cfg: dict, mix: dict, qres: dict) -> dict:
    """Q1→Q2 配比预测器接口（§5）：17 输入列序、13 输出列序、固定 v、参考 p0、训练支持范围。"""
    ic = cfg["mixture"]["input_cols"]
    tc = cfg["mixture"]["target_cols"]
    Xn = mixture_normalized(cfg)
    support = {}
    for j, c in enumerate(ic):
        v = Xn[:, j]
        support[c] = {"min": float(v.min()), "p05": float(np.quantile(v, 0.05)),
                      "p50": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
                      "max": float(v.max()), "mean": float(v.mean()),
                      "frac_zero": float((v <= 0).mean())}
    p0 = [float(x) for x in Xn.mean(axis=0)]
    best = mix["best"]
    best_row = mix["best_row"]
    return {
        "baseline_id": cfg["baseline_id"],
        "input_columns": ic,
        "input_units": "行归一化后的训练域配比，非负且和为 1",
        "n_inputs": len(ic),
        "output_columns": tc,
        "output_short_names": [short_target(c) for c in tc],
        "output_units": "验证交叉熵 Loss（越低越好）",
        "n_outputs": len(tc),
        "target_weights_v": [1.0 / len(tc)] * len(tc),
        "target_weights_note": "baseline 令 v 等权，且不随 p 改变（§1.4）",
        "p0_reference_mixture_mean": p0,
        "training_support": support,
        "row_normalization": "p_i := p_i / Σ_j p_j；A4 原始行和 0.996-1.003",
        "selected_model": {
            "overall_best": {"kind": best["kind"], "label": best["label"],
                             "cv_mean_target_rmse": float(best_row["cv_mean_target_rmse"]),
                             "cv_composite_rmse": float(best_row["cv_composite_rmse"])},
            "ridge": {"alpha": mix["best_params"]["ridge_alpha"]},
            "forest": mix["best_params"]["forest"],
            "selection": f"A4/A5 内部 {cfg['mixture']['cv_folds']} 折 CV，"
                         f"准则 {cfg['mixture']['cv_selection_metric']}；"
                         f"线性与森林分别取各自最优，避免最优超参被另一支覆盖",
        },
        "quality_feature": {
            "definition": "Q(p) = Σ_i p_i q_i，q_i 为训练域对应的质量域质量分",
            "modes": cfg["mixture"]["quality_feature"]["modes"],
            "primary_unmapped_policy": cfg["mixture"]["quality_feature"]["primary_unmapped_policy"],
            "caveat": "§1.5(7) Q(p) 与完整线性配比变量存在确定关系，不能把两类系数"
                      "都解释为独立效应",
        },
        "frozen_entropy_weights": dict(zip(METRIC_ORDER,
                                          [float(x) for x in qres["W"]["weight_entropy"].to_numpy()])),
        "extrapolation_flags": {
            "final_test": [s["name"] for s in cfg["splits"]["final_test"]],
            "diagnostic_only": [s["name"] for s in cfg["splits"]["extrapolation_diagnostic"]],
            "warning": "森林不作为远距离尺度外推公式；10B/70B 仅为估算表一致性诊断",
        },
    }


def mixture_normalized(cfg: dict) -> np.ndarray:
    from mixture import normalize_rows
    p = load_pairs(cfg)["train_1m"]
    return normalize_rows(p["X"].to_numpy(dtype=float))


def collect_input_hashes(cfg: dict) -> dict:
    h = {}
    for name in ("A1", "A2", "A3"):
        h[f"preprocessed/{name}_processed.parquet"] = hash_if_exists(
            os.path.join(preprocessed_dir(cfg), f"{name}_processed.parquet"))
    h["preprocess_params"] = hash_if_exists(
        os.path.abspath(os.path.join(cfg["_root"], cfg["upstream"]["preprocess_params"].replace("/", os.sep))))
    h["preprocess_report"] = hash_if_exists(
        os.path.abspath(os.path.join(cfg["_root"], cfg["upstream"]["preprocess_report"].replace("/", os.sep))))
    h["domain_mapping_guide.csv"] = hash_if_exists(
        os.path.abspath(os.path.join(cfg["_root"], cfg["paths"]["domain_mapping"].replace("/", os.sep))))
    pairs = load_pairs(cfg)
    reg = os.path.abspath(os.path.join(cfg["_root"], cfg["paths"]["regmix_dir"].replace("/", os.sep)))
    for _, spec in pairs.items():
        for k in ("mixture_file", "loss_file"):
            h[f"regmix/{spec[k]}"] = hash_if_exists(os.path.join(reg, spec[k]))
    h["config_yaml"] = hash_if_exists(cfg["_config_path"])
    return h


def main() -> int:
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    t_all = Timer()
    cfg = load_config()
    log(f"[Q1-C] 仓库根目录 {cfg['_root']}")
    log(f"[Q1-C] 配置 {cfg['_config_path']}  hash={config_hash(cfg)}  seed={cfg['seed']}")

    log("\n=== 第一部分：熵权质量分 / 共同冲突规则（与等权对照） ===")
    qres = quality.run_quality(cfg, log=log)
    qmap = quality_lookup(cfg, qres["mapping"])
    mapped = {k: v for k, v in qmap.items() if not pd.isna(v[0])}
    log(f"[quality] 可用质量特征的训练域 {len(mapped)}/{len(qmap)}："
        f"{sorted(k.replace('train_the_pile_', '') for k in mapped)}")
    save_table(qres["mapping"], cfg, "quality_domain_mapping.csv")

    log("\n=== 第二部分：随机森林配比回归（与线性对照） ===")
    mix = run_mixture(cfg, qmap, log=log)

    log("\n=== 第三部分：事后诊断（口径稳健性 / 收益来源 / 扰动标定） ===")
    diag = run_diagnostics(cfg, mix, log=log)
    save_table(diag["linearity"], cfg, "mixture_posthoc_target_linearity.csv")
    save_table(diag["composite"], cfg, "mixture_posthoc_composite_sensitivity.csv")
    save_table(diag["calibration"], cfg, "mixture_perturbation_calibration.csv")
    save_table(diag["stability"], cfg, "mixture_selection_stability.csv")

    log("\n=== 汇总 ===")
    abs1m = mix["absolute"]
    comp = abs1m[abs1m["target"] == "__composite_equal_v__"].sort_values("rmse")
    log("1M 综合目标（RMSE 升序，前 6）：")
    for _, r in comp.head(6).iterrows():
        log(f"   {r['variant']:26s} mode={r['mode']:8s} RMSE={r['rmse']:.4f} "
            f"MAE={r['mae']:.4f} R2={r['r2']:.4f}{'  <- train_mean 最低参照' if r['variant']=='train_mean' else ''}")

    rank = mix["ranking"]
    sub = rank[(rank["target"] == "__composite_equal_v__")]
    log("\n跨尺度排序（综合目标）：")
    for _, r in sub.sort_values(["dataset", "variant"]).iterrows():
        log(f"   {r['dataset']:10s} {r['variant']:26s} spearman={r['spearman']:.4f} "
            f"regret={r['regret']:.4f} topk1={r['topk1_hit']} rmse={r['rmse_uncalibrated']:.4f}")

    log("\n10B/70B 扰动稳健性：")
    log(mix["perturbation"].to_string(index=False))

    predictor = build_predictor(cfg, mix, qres)
    save_json(predictor, cfg, "mixture_predictor.json")

    pairs = load_pairs(cfg)
    meta = {
        "baseline_id": cfg["baseline_id"],
        "question": cfg["question"],
        "description": cfg["description"],
        "run_command": RUN_COMMAND,
        "run_utc": utc_now(),
        "wall_seconds": t_all.stop(),
        "config_path": cfg["_config_path"],
        "config_hash": config_hash(cfg),
        "seed": cfg["seed"],
        "git_rev": git_rev(cfg["_root"]),
        "environment": env_info(),
        "upstream": {
            "preprocessed_dir": preprocessed_dir(cfg),
            "version": cfg["upstream"]["preprocess_version"],
            "note": "复用上游冻结的标准化质量矩阵 Z∈[0,1]^(N×22)；未重新读取原始 JSONL",
        },
        "input_hashes_sha256": collect_input_hashes(cfg),
        "sample_sizes": {
            "quality_A1": 51230, "quality_A2": 17523, "quality_A3": 203752,
            "quality_merged_dedup": qres["summary"]["sample_flow"]["merged_rows"],
            "quality_dev": qres["summary"]["n_dev"], "quality_verify": qres["summary"]["n_verify"],
            "mixture_train": int(len(pairs["train_1m"]["X"])),
            **{k: int(len(v["X"])) for k, v in pairs.items() if k != "train_1m"},
        },
        "metric_definitions": {
            "rmse": "sqrt(mean((y-yhat)^2))",
            "mae": "mean(|y-yhat|)",
            "r2": "1 - SS_res/SS_tot（对留出集计算，可为负）",
            "spearman": "候选集内真实与预测 Loss 的秩相关",
            "topk_hit": "预测最优是否落入真实前 k",
            "regret": "预测最优配方的真实 Loss - 候选集真实最小 Loss",
            "composite": "13 个目标等权平均后的综合 Loss",
        },
        "interval_methods": {
            "domain_quality_ci95": "各域内文档独立的均值的正态近似区间（非 bootstrap，非情景包络）",
            "no_prediction_interval": "本 baseline 不输出预测区间；外推情景包络属第二问范围",
        },
        "posthoc_diagnostics": {
            "note": "第三部分为事后诊断，不参与选参、不接触最终测试集调参；"
                    "用于检验结论对复合口径与配比扰动的稳健性（报告 §6.3.1/§6.3.2/§6.5）",
            "artifacts": ["mixture_posthoc_target_linearity.csv",
                          "mixture_posthoc_composite_sensitivity.csv",
                          "mixture_perturbation_calibration.csv",
                          "mixture_selection_stability.csv"],
            "perturbation_input_displacement": "以各候选配比自身为中心构造 Dirichlet(α=p·conc)，"
                                               "故浓度可换算为输入空间位移；标定见 calibration 表",
        },
        "extrapolation_flags": {
            "real_holdout": [s["name"] for s in cfg["splits"]["final_test"]],
            "estimated_or_semi_synthetic": [s["name"] for s in cfg["splits"]["extrapolation_diagnostic"]],
        },
        "artifacts_dir": artifacts_dir(cfg),
    }
    save_json(meta, cfg, "run_metadata.json")

    log(f"\n[Q1-C] 完成，总耗时 {meta['wall_seconds']:.1f}s，产物目录 {meta['artifacts_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
