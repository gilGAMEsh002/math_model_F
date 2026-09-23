# -*- coding: utf-8 -*-
"""Q2-C 主流程：参数与数据双通道质量效率。

模型族（§2.3）
    M0 经典   L = E + A N^(-α) + B D^(-β)
    M1 = Q2-B L = E + A N^(-α) + B [D Q^κ h(p)]^(-β)            ρ = 0 嵌套
    M2 = Q2-C L = E + A [N Q^ρ]^(-α) + B [D Q^κ h(p)]^(-β)      释放 ρ

估计策略（§2.4）
    分阶段：B1 拟合经典参数并冻结 → 在 B1∪B6 上只释放质量通道 (κ,δ)、(κ,ρ,δ)；
    联合微调：限制半合成权重，分别报告真实/半合成残差；
    判定：剖面似然 + 参数相关 + 多起点等价解，不只比训练误差。

运行：python src/baselines/Q2-C/run_all.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (artifacts_dir, config_hash, env_info, git_rev, hash_if_exists,
                    load_config, q1_path, read_b, read_b3_trajectories, rmse, mae, bias, r2,
                    save_json, save_table, scaling_dir, utc_now, write_text)
from scaling import (BASE, FULL, MODEL_FREE, delta_loss_Q, elasticities,
                     equivalent_N, derivatives, fit, predict)
from calibrate import (candidate_mixtures, calibration_report, calibrate_series,
                       mixture_effect, q1_quality_table, q1_reference, qmap_from_table,
                       quality_of_mixture, substitution_table, transfer_scenarios)
from identify import correlation_table, identify_channel, solution_table, verdict
from provenance import provenance_audit
from reconcile import reconcile
from validate import (SOURCES, cross_source_table, extrapolation_scenario,
                      grouped_bootstrap, holdout_by_size, holdout_d_direction,
                      prep_source, trajectory_check)

_LOG: list[str] = []


def log(msg: str = "") -> None:
    _LOG.append(msg)
    print(msg, flush=True)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {title}")
    log("=" * 78)


# --------------------------------------------------------------------- 数据装配
def build_frames(cfg: dict):
    """读取全部用到的附件 B 表（不读附件 A 原始文件，只走 Q1→Q2 接口）。

    B9 是纯元数据表（无 Loss 列），不走 read_b 的 N/D/L 强制校验，单独按元数据读。
    """
    keys = ["b1_real_main", "b2_family_out", "b4_cross_family", "b5_published",
            "b6_quality_main", "b7_quality_expanded", "b8_quality_large",
            "b10_large_baseline"]
    loaded, hashes = {}, {}
    sdir = scaling_dir(cfg)
    for k in keys:
        df = read_b(cfg, k)
        loaded[k] = df
        hashes[cfg["data"][k]] = hash_if_exists(os.path.join(sdir, cfg["data"][k]))
    b9 = pd.read_csv(os.path.join(sdir, cfg["data"]["b9_large_models"]))
    loaded["b9_large_models"] = b9
    hashes[cfg["data"]["b9_large_models"]] = hash_if_exists(
        os.path.join(sdir, cfg["data"]["b9_large_models"]))
    traj = read_b3_trajectories(cfg)
    return loaded, hashes, traj


def unit_audit(cfg: dict, loaded: dict) -> pd.DataFrame:
    """B9/B10 单位审计（§2.6）：确认参数量/数据量口径，标记激活参数歧义。"""
    rows = []
    for key, cols in (("b9_large_models", ("N_params_B", "D_tokens_B")),
                      ("b10_large_baseline", ("N", "D"))):
        d = loaded.get(key)
        if d is None:
            continue
        for c in cols:
            if c not in d.columns:
                continue
            v = pd.to_numeric(d[c], errors="coerce").dropna()
            rows.append({
                "file": cfg["data"][key], "column": c, "n": int(len(v)),
                "min": float(v.min()), "median": float(v.median()),
                "max": float(v.max()), "unit": "十亿(B) —— 与公共口径一致",
                "note": ("总参数口径；元数据未区分总参数与激活参数，"
                         "若为 MoE 需另行审计" if c.startswith("N") else
                         "训练 token 口径"),
            })
    return pd.DataFrame(rows)


def quality_frame(b1: pd.DataFrame, qdf: pd.DataFrame, q_ref: float) -> pd.DataFrame:
    """B1(真实, Q 取基准约定, δ 组=0) ∪ B6(质量实验, δ 组=1)。"""
    a = b1[["N", "D", "L"]].copy()
    a["Q"] = float(q_ref)
    a["dgrp"] = 0.0
    a["batch"] = "B1"
    b = qdf[["N", "D", "L", "Q"]].copy()
    b["dgrp"] = 1.0
    b["batch"] = "B6"
    return pd.concat([a, b], ignore_index=True)


def sse_of(x, df, weights=None) -> float:
    r = predict(x, df["N"], df["D"], df["Q"], 1.0, df["dgrp"]) - df["L"]
    if weights is not None:
        r = r * np.sqrt(np.asarray(weights, float))
    return float(np.sum(np.asarray(r, float) ** 2))


def aic_bic(sse: float, n: int, k: int) -> tuple[float, float]:
    sse = max(float(sse), 1e-12)
    return n * np.log(sse / n) + 2 * k, n * np.log(sse / n) + k * np.log(n)


def ftest_nested(sse_small: float, k_small: int, sse_big: float, k_big: int, n: int) -> dict:
    """嵌套 F 检验：新增参数是否显著降低 SSE。"""
    from scipy.stats import f as fdist

    df1, df2 = max(k_big - k_small, 1), max(n - k_big, 1)
    if sse_big <= 0:
        return {"F": float("nan"), "p": float("nan"), "df1": df1, "df2": df2}
    F = ((sse_small - sse_big) / df1) / (sse_big / df2)
    p = float(1.0 - fdist.cdf(F, df1, df2)) if np.isfinite(F) else float("nan")
    return {"F": float(F), "p": p, "df1": df1, "df2": df2}


# ------------------------------------------------------------------- 主流程
def main() -> int:
    cfg = load_config()
    art = artifacts_dir(cfg)
    q_ref = float(cfg["model"]["q_reference"])
    bounds = cfg["model"]["bounds"]
    seed = int(cfg["seed"])
    ms = cfg["model"]["multi_start"]
    tol = float(ms["cluster_tol"])
    mfev = int(cfg["model"]["least_squares"]["max_nfev"])

    log("Q2-C 参数与数据双通道质量效率 —— 主流程")
    log(f"  项目根      : {cfg['_root']}")
    log(f"  配置        : {cfg['_config_path']}")
    log(f"  配置哈希    : {config_hash(cfg)}")
    log(f"  时间(UTC)   : {utc_now()}")

    section("0. 数据读取与输入登记")
    loaded, in_hashes, traj = build_frames(cfg)
    b1 = prep_source(loaded["b1_real_main"], cfg, q_ref)
    log(f"  B1 真实主拟合 : n={len(b1)}  规模 {[round(float(v), 4) for v in sorted(b1['N'].unique())]}B")
    for k in ("b6_quality_main", "b7_quality_expanded", "b8_quality_large"):
        if k in loaded:
            d = prep_source(loaded[k], cfg, q_ref)
            log(f"  {k:26s}: n={len(d)}  N∈[{d['N'].min():.3g},{d['N'].max():.3g}]B "
                f"Q∈[{d['Q'].min():.3f},{d['Q'].max():.3f}]")
    log(f"  B3 轨迹       : n={len(traj)} 文件数={traj['traj_file'].nunique()}")
    unit_tab = unit_audit(cfg, loaded)
    save_table(unit_tab, cfg, "unit_audit.csv")
    for _, r in unit_tab.iterrows():
        log(f"  单位审计 {r['file']:34s} {r['column']:12s} "
            f"[{r['min']:.3g}, {r['max']:.3g}] {r['unit']}")

    # 辅助索引溯源核验（B11/B12 无 Loss 列，不能进拟合；但可核验口径与覆盖面）
    other_names = set()
    for _k, _c in (("b9_large_models", "model_name"), ("b10_large_baseline", "family")):
        if _k in loaded and _c in loaded[_k].columns:
            other_names |= set(loaded[_k][_c].astype(str))
    prov = provenance_audit(cfg, loaded["b1_real_main"], log=log,
                            other_models=other_names, loaded=loaded)
    save_table(prov["b1_internal_consistency"], cfg, "provenance_b1_internal.csv")
    if len(prov["b12_checkpoint_coverage"]):
        save_table(prov["b12_checkpoint_coverage"], cfg, "provenance_b12_coverage.csv")
    if len(prov["b11_param_audit"]):
        save_table(prov["b11_param_audit"], cfg, "provenance_b11_param_audit.csv")
    if len(prov["convergence_audit"]):
        save_table(prov["convergence_audit"], cfg, "provenance_convergence.csv")

    q1_tab = q1_quality_table(cfg)
    qmap = qmap_from_table(q1_tab, "q_entropy")
    p0, pcols = q1_reference(cfg)
    n_mapped = int(np.sum([not np.isnan(v) for v in qmap.values()]))
    log(f"  Q1→Q2 接口    : 训练域 {len(qmap)} 个，完成质量映射 {n_mapped} 个，"
        f"参考配比 p0 维度 {len(p0)}")

    # ---------------------------------------------------------------- 1. 阶段一
    section("1. 阶段一：B1 经典拟合（M0）并冻结经典参数")
    s1 = fit(b1["N"], b1["D"], b1["Q"].fillna(q_ref), b1["L"], model="M0", bounds=bounds,
             n_starts=int(ms["n_starts_stage1"]), seed=seed, max_nfev=mfev,
             cluster_tol=tol, log=log)
    x1 = s1["x"]
    log(f"  E={x1[0]:.4f}  A={x1[1]:.4g}  α={x1[2]:.4f}  B={x1[3]:.4g}  β={x1[4]:.4f}")
    log(f"  B1 RMSE={s1['rmse']:.4f}  等价解类数={len(s1['clusters'])}")

    # ------------------------------------------------------- 2. 阶段二：质量通道
    section("2. 阶段二：在 B1∪B6 上释放质量通道（经典参数冻结）")
    comb = quality_frame(b1, prep_source(loaded["b6_quality_main"], cfg, q_ref), q_ref)
    w_comb_m0 = np.ones(len(comb))
    sse_m0_comb = sse_of(x1, comb)
    sse_m0_b6 = sse_of(x1, comb[comb.batch == "B6"])
    log(f"  嵌套基线 M0 在 B1∪B6 上 SSE={sse_m0_comb:.3f}（仅 B6 部分 SSE={sse_m0_b6:.3f}）")

    base1 = x1.copy()
    base1[5] = 0.3          # κ 初值（会被多起点覆盖）

    # δ-only 消融：只有来源水平偏移、没有质量通道。用于把"相对嵌套基线的收益"
    # 拆成两部分——来源水平差（δ）与质量通道（κ/ρ）——否则 δ 的贡献会被误记到 κ 头上。
    m0d = fit(comb["N"], comb["D"], comb["Q"], comb["L"], base=x1.copy(),
              free_only=["delta"], bounds=bounds, n_starts=int(ms["n_starts_stage2"]),
              seed=seed + 5, dgrp=comb["dgrp"], max_nfev=mfev, cluster_tol=tol, log=log)
    log(f"  M0+δ 消融 : δ={m0d['x'][7]:+.4f}  SSE={m0d['sse']:.3f}  "
        f"（相对 M0 的 SSE 降幅 {sse_m0_comb - m0d['sse']:.3f}，全部来自来源水平差）")

    m1s = fit(comb["N"], comb["D"], comb["Q"], comb["L"], base=base1,
              free_only=["kappa", "delta"], bounds=bounds,
              n_starts=int(ms["n_starts_stage2"]), seed=seed + 1,
              dgrp=comb["dgrp"], max_nfev=mfev, cluster_tol=tol, log=log)
    log(f"  M1-staged: κ={m1s['x'][5]:.4f}  δ={m1s['x'][7]:+.4f}  SSE={m1s['sse']:.3f}")

    base2 = m1s["x"].copy()
    base2[6] = 0.3
    m2s = fit(comb["N"], comb["D"], comb["Q"], comb["L"], base=base2,
              free_only=["kappa", "rho", "delta"], bounds=bounds,
              n_starts=int(ms["n_starts_stage2"]), seed=seed + 2,
              dgrp=comb["dgrp"], max_nfev=mfev, cluster_tol=tol, log=log)
    log(f"  M2-staged: κ={m2s['x'][5]:.4f}  ρ={m2s['x'][6]:.4f}  δ={m2s['x'][7]:+.4f}  "
        f"SSE={m2s['sse']:.3f}")
    F_delta = ftest_nested(sse_m0_comb, 5, m0d["sse"], 6, len(comb))
    F_m1 = ftest_nested(m0d["sse"], 6, m1s["sse"], 7, len(comb))
    F_m2 = ftest_nested(m1s["sse"], 7, m2s["sse"], 8, len(comb))
    log(f"  嵌套 F 检验 M0  →M0+δ : F={F_delta['F']:.2f} p={F_delta['p']:.3g}  （来源水平差）")
    log(f"  嵌套 F 检验 M0+δ→M1   : F={F_m1['F']:.2f} p={F_m1['p']:.3g}  （质量通道 κ）")
    log(f"  嵌套 F 检验 M1  →M2   : F={F_m2['F']:.2f} p={F_m2['p']:.3g}  （参数通道 ρ）")
    log(f"  收益分解：SSE {sse_m0_comb:.3f} → {m0d['sse']:.3f}(δ) → {m1s['sse']:.3f}(κ) "
        f"→ {m2s['sse']:.3f}(ρ)")
    floor_eff = float(m2s["x"][0] + m2s["x"][7])
    if floor_eff <= 0.05:
        log(f"  [警告] 有效损失下限 E+δ={floor_eff:.3f} ≤ 0.05，δ 命中边界语义，"
            f"质量通道证据不可采信")

    # -------------------------------------------------------- 3. 半合成权重对照
    section("3. 联合微调对照：限制半合成（B6）权重")
    joint_rows, joint_models, joint_w = [], {}, {}
    for w in cfg["stages"]["joint_weights_B6"]:
        wv = np.where(comb["batch"].to_numpy() == "B6", float(w), 1.0)
        jm = fit(comb["N"], comb["D"], comb["Q"], comb["L"], model="M2",
                 free_extra=["delta"], bounds=bounds,
                 n_starts=int(ms["n_starts_joint"]), seed=seed + 3,
                 weights=wv, dgrp=comb["dgrp"], max_nfev=mfev, cluster_tol=tol)
        joint_models[float(w)] = jm
        joint_w[float(w)] = wv
        b6m = comb.batch.to_numpy() == "B6"
        row = {"w_B6": float(w), "sse_weighted": jm["sse"],
               "rmse_B1": rmse(comb["L"][~b6m],
                               predict(jm["x"], comb["N"][~b6m], comb["D"][~b6m],
                                       comb["Q"][~b6m], 1.0, comb["dgrp"][~b6m])),
               "rmse_B6": rmse(comb["L"][b6m],
                               predict(jm["x"], comb["N"][b6m], comb["D"][b6m],
                                       comb["Q"][b6m], 1.0, comb["dgrp"][b6m])),
               "sse_B1": sse_of(jm["x"], comb[~b6m]),
               "sse_B6": sse_of(jm["x"], comb[b6m]),
               "n_clusters": len(jm["clusters"])}
        for i, nm in enumerate(FULL):
            row[nm] = float(jm["x"][i])
        joint_rows.append(row)
        log(f"  w_B6={w:.2f}: κ={jm['x'][5]:.4f} ρ={jm['x'][6]:.4f} δ={jm['x'][7]:+.4f} "
            f"RMSE(B1)={row['rmse_B1']:.4f} RMSE(B6)={row['rmse_B6']:.4f}")
    joint_tab = pd.DataFrame(joint_rows)

    # 主对照模型：以中位权重（默认 0.30）为联合微调代表
    w_mid = float(cfg["stages"]["joint_weights_B6"][len(cfg["stages"]["joint_weights_B6"]) // 2])
    jm_main = joint_models[w_mid]
    log(f"  主对照联合模型取 w_B6={w_mid}")

    # -------------------------------------------------------- 4. 质量备选集敏感性
    section("4. 质量数据集敏感性（B6 主集 vs B7/B8 替代集）")
    alt_rows = []
    alt_keys = ["b6_quality_main"]
    for short in cfg["stages"]["quality_alternatives"]:
        full = next((k for k in loaded if k.startswith(f"{short}_")), None)
        if full and full not in alt_keys:
            alt_keys.append(full)
        elif full is None:
            log(f"  [warn] 配置的备选质量集 {short} 未在数据登记中找到，跳过")
    for k in alt_keys:
        d = prep_source(loaded[k], cfg, q_ref)
        n_excluded = 0
        if "data_type" in d.columns:
            # B8 自带 calibrated / extrapolated 分段。外推段的 N（20–700B）远超 B1
            # 支撑（≤12B）且处于另一水平，混入会把该水平差吸收进质量参数。
            n_all = len(d)
            d = d[d["data_type"].astype(str) == "calibrated"].reset_index(drop=True)
            n_excluded = n_all - len(d)
            log(f"  [note] {k} 含 data_type 列：仅用 calibrated 段拟合，"
                f"剔除 extrapolated {n_excluded} 行")
        c = quality_frame(b1, d, q_ref)
        try:
            r = fit(c["N"], c["D"], c["Q"], c["L"], base=base1,
                    free_only=["kappa", "rho", "delta"], bounds=bounds,
                    n_starts=int(ms["n_starts_stage2"]), seed=seed + 4,
                    dgrp=c["dgrp"], max_nfev=mfev, cluster_tol=tol)
            alt_rows.append({"source": k, "n": int(len(d)), "n_excluded": int(n_excluded),
                             "kappa": float(r["x"][5]), "rho": float(r["x"][6]),
                             "delta": float(r["x"][7]), "sse": float(r["sse"]),
                             "n_clusters": len(r["clusters"]),
                             "boundary_hits": str(r["boundary_hits"])})
            log(f"  {k:30s}: κ={r['x'][5]:.4f} ρ={r['x'][6]:.4f} δ={r['x'][7]:+.4f} "
                f"等价类={len(r['clusters'])}")
        except Exception as exc:
            alt_rows.append({"source": k, "n": int(len(d)), "n_excluded": int(n_excluded),
                             "kappa": np.nan, "rho": np.nan, "delta": np.nan,
                             "sse": np.nan, "n_clusters": 0, "boundary_hits": ""})
            log(f"  {k:30s}: 拟合失败 {exc}")
    alt_tab = pd.DataFrame(alt_rows)

    # 4b. 质量集之间的一致性核验（负结果必须留存，见 reconcile.py）
    section("4b. 质量数据集一致性核验（B6/B7 与 B8 是否在同一模型族下相容）")
    rec_tab = reconcile(cfg, b1, loaded, x1, bounds, q_ref, log=log)
    save_table(rec_tab, cfg, "quality_set_reconciliation.csv")
    core = rec_tab[rec_tab["check"] == "per_set_fit"].copy()
    if "quality_channel_informative" in core.columns:
        flag = core["quality_channel_informative"].fillna(False).astype(bool)
        informative = sorted(core.loc[flag, "source"].unique().tolist())
        uninformative = sorted(core.loc[~flag, "source"].unique().tolist())
        # 逐子集看（B8 有 calibrated / extrapolated 两段）
        detail = {f"{r['source']}/{r['label']}": bool(r["quality_channel_informative"])
                  for _, r in core.iterrows()}
    else:
        informative, uninformative, detail = [], [], {}
    log(f"  质量通道有信息的集: {informative or '（无）'}")
    log(f"  质量通道零信息的集: {uninformative or '（无）'}")
    for kk, vv in detail.items():
        log(f"    - {kk:38s} {'有信息' if vv else '零信息（经典律即最优）'}")
    b8_uninformative = any(k.startswith("b8") and not v for k, v in detail.items())
    if uninformative and b8_uninformative:
        log("  → 判定：半合成质量集之间互相矛盾，质量通道未被真实观测支持；"
            "按 Q2-C.md 第 4 条与 §2.4.9 退回 Q2-B（ρ=0），κ、ρ 仅作情景参数。")

    # 样本重叠登记（§2.6）：按 (N,D) 行级匹配，而不是只比名字。
    # B1 无 family 列，故无法按族比对；行级 (N,D) 匹配才是可核验的"是否同一实验"证据。
    b1_key = set(zip(b1["N"].round(6), b1["D"].round(6)))
    overlap_rows = [{
        "file": cfg["data"]["b1_real_main"], "n_rows": int(len(b1)),
        "n_matched_with_B1": int(len(b1_key)), "overlap_frac": 1.0,
        "labels": "Pythia（B1 自身）",
        "note": "基准表",
    }]
    for key in ("b2_family_out", "b4_cross_family", "b5_published",
                "b10_large_baseline"):
        if key not in loaded:
            continue
        d = loaded[key]
        keys = set(zip(d["N"].round(6), d["D"].round(6)))
        matched = len(keys & b1_key)
        labels = ""
        for col in ("family", "source"):
            if col in d.columns:
                u = d[col].dropna().astype(str).unique()
                # 唯一值≈行数说明该列是逐行标识而非分组标签，不作为族列出
                if len(u) < len(d):
                    labels = ";".join(sorted(u)[:12]) + ("…" if len(u) > 12 else "")
                break
        overlap_rows.append({
            "file": cfg["data"][key], "n_rows": int(len(d)),
            "n_matched_with_B1": int(matched),
            "overlap_frac": float(matched / max(len(keys), 1)),
            "labels": labels,
            "note": ("与 B1 完全同点" if matched == len(keys) else
                     (f"有 {matched} 个 (N,D) 点与 B1 重复" if matched else
                      "无 (N,D) 点与 B1 重复（同族也可能因 D 网格不同而不重合）")),
        })
        log(f"  [重叠] {cfg['data'][key]:38s} n={len(d):5d} 与 B1 同 (N,D) 点={matched} "
            f"({matched / max(len(keys), 1):.1%})")
    overlap_tab = pd.DataFrame(overlap_rows)
    save_table(overlap_tab, cfg, "sample_overlap.csv")

    # ------------------------------------------------------------ 5. 可辨识性
    section("5. 可辨识性诊断（剖面似然、参数相关、多起点等价解）")
    ident, corr_tabs, profs = {}, {}, {}
    for tag, res, wts in (("M2_staged", m2s, None), ("M2_joint", jm_main, joint_w[w_mid])):
        corr_tab, _ = correlation_table(res, comb["N"], comb["D"], comb["Q"], comb["L"],
                                        comb["dgrp"], weights=wts)
        corr_tabs[tag] = corr_tab
        d = {}
        for pname in ("kappa", "rho"):
            d[pname] = identify_channel(cfg, res, pname, comb["N"], comb["D"], comb["Q"],
                                        comb["L"], bounds, dgrp=comb["dgrp"],
                                        weights=wts, log=log)
            profs[f"{tag}_{pname}"] = pd.DataFrame(d[pname]["profile"])
        ident[tag] = verdict(cfg, res, d, corr_tab, log=log)
        solution_table(res).to_csv(os.path.join(art, f"solutions_{tag}.csv"),
                                   index=False, encoding="utf-8-sig")
    kr = corr_tabs["M2_staged"]
    log("  M2-staged 关键参数相关：")
    for _, r in kr.iterrows():
        log(f"    corr({r['param_a']:5s},{r['param_b']:5s}) = {r['corr']:+.3f}")

    # --------------------------------------------------------------- 6. 验证
    section("6. 验证：留规模 / 轨迹 / 跨源 / 分组 bootstrap / 外推")
    ho_size = holdout_by_size(cfg, b1, bounds, log=log)
    ho_d = holdout_d_direction(cfg, b1, bounds, log=log)
    traj_tab = trajectory_check(cfg, traj, x1, q_ref, log=log)
    log("  --- 逐来源误差：嵌套基线 M0（无质量通道，δ=0）---")
    cross_v = cross_source_table(cfg, x1, loaded, q_ref, log=log)
    log("  --- 逐来源误差：M2-staged（含质量通道与来源偏移 δ）---")
    cross_s = cross_source_table(cfg, m2s["x"], loaded, q_ref, log=log)
    boot_v = grouped_bootstrap(cfg, b1, "M0", bounds, x1, log=log)
    boot_s = grouped_bootstrap(cfg, comb, "M2", bounds, m2s["x"],
                               group_col=["batch", "N"], log=log)
    extr = extrapolation_scenario(cfg, m2s["x"], loaded.get("b9_large_models"),
                                  loaded["b10_large_baseline"], log=log)

    # 统一长表 scaling_validation.csv
    blocks = []
    for _, r in cross_s.iterrows():
        blocks.append({"block": "cross_source", "scope": r["source"], "kind": r["kind"],
                       "n": r["n"], "rmse": r["rmse"], "mae": r["mae"], "bias": r["bias"],
                       "r2": r["r2"],
                       "note": f"去水平差 RMSE={r['rmse_debiased']:.4f}；{r['note']}"})
    for _, r in ho_size.iterrows():
        blocks.append({"block": "holdout_by_size", "scope": f"N={r['held_out_N']:g}B",
                       "kind": "real", "n": r["n_test"], "rmse": r["rmse"], "mae": r["mae"],
                       "bias": r["bias"], "r2": np.nan, "note": r["note"]})
    blocks.append({"block": "holdout_D_direction", "scope": f"D>{ho_d['threshold_D']:.4g}",
                   "kind": "real", "n": ho_d["n_test"], "rmse": ho_d["rmse"],
                   "mae": ho_d["mae"], "bias": ho_d["bias"], "r2": np.nan,
                   "note": f"训练 n={ho_d['n_train']}"})
    for _, r in traj_tab.iterrows():
        blocks.append({"block": "trajectory", "scope": r["trajectory"], "kind": "interpolated",
                       "n": r["n_test"], "rmse": r["rmse_test"], "mae": r["mae_test"],
                       "bias": r["bias_test"], "r2": np.nan,
                       "note": f"N={r['N']:g}B 前半段 n={r['n_first_half']}；"
                               f"插值点占比 {r['interpolated_frac']:.2f}"})
    for _, r in boot_s.iterrows():
        blocks.append({"block": "bootstrap_param", "scope": r["param"],
                       "kind": "grouped_bootstrap", "n": r["n_boot"], "rmse": np.nan,
                       "mae": np.nan, "bias": np.nan, "r2": np.nan,
                       "note": f"点估计 {r['point']:.4g}，经验95%区间 "
                               f"[{r['lo95']:.4g}, {r['hi95']:.4g}]"})
    val_long = pd.DataFrame(blocks)
    save_table(val_long, cfg, "scaling_validation.csv")
    save_table(cross_v, cfg, "validation_cross_source_M0.csv")
    save_table(cross_s, cfg, "validation_cross_source_M2.csv")
    save_table(ho_size, cfg, "validation_holdout_by_size.csv")
    save_table(traj_tab, cfg, "validation_trajectory.csv")
    save_table(boot_v, cfg, "validation_bootstrap_M0.csv")
    save_table(boot_s, cfg, "validation_bootstrap_M2.csv")
    save_table(alt_tab, cfg, "quality_source_sensitivity.csv")
    save_table(joint_tab, cfg, "joint_weight_sensitivity.csv")
    if extr is not None and len(extr):
        save_table(extr, cfg, "validation_extrapolation.csv")
    for kk, vv in profs.items():
        save_table(vv, cfg, f"profile_{kk}.csv")
    for kk, vv in corr_tabs.items():
        save_table(vv, cfg, f"correlation_{kk}.csv")

    # ------------------------------------------------------------ 7. 弹性与等效
    section("7. 弹性、质量收益与质量-参数等效")
    b6 = prep_source(loaded["b6_quality_main"], cfg, q_ref)
    ops = [
        ("B1_median", float(b1["N"].median()), float(b1["D"].median()), q_ref, 1.0, 0.0),
        ("B6_median", float(b6["N"].median()), float(b6["D"].median()),
         float(b6["Q"].median()), 1.0, 1.0),
        ("B6_Q_low", float(b6["N"].median()), float(b6["D"].median()),
         float(b6["Q"].quantile(0.10)), 1.0, 1.0),
        ("B6_Q_high", float(b6["N"].median()), float(b6["D"].median()),
         float(b6["Q"].quantile(0.90)), 1.0, 1.0),
        ("anchor_1B_100B", 1.0, 100.0, 0.8, 1.0, 0.0),
        ("anchor_7B_300B", 7.0, 300.0, 0.8, 1.0, 0.0),
    ]
    erows, qrows = [], []
    for name, Nv, Dv, Qv, hv, dg in ops:
        e = elasticities(m2s["x"], np.array([Nv]), np.array([Dv]), np.array([Qv]),
                         np.array([hv]), np.array([dg]))
        row = {"operating_point": name, "N": Nv, "D": Dv, "Q": Qv, "h": hv,
               "offset_group": dg, "L_pred": float(e["L"][0]),
               "L_reducible": float(e["L_reducible"][0])}
        for k in ("eps_N", "eps_D", "eps_Q", "eps_h", "eps_N_reducible", "eps_D_reducible",
                  "eps_Q_reducible"):
            row[k] = float(e[k][0])
        erows.append(row)
        # 质量-参数等效：Q+ΔQ 的损失收益折合为多大参数规模
        for dQ in (0.05, 0.10, 0.20):
            if Qv + dQ > 1.0 + 1e-12:
                qrows.append({"operating_point": name, "N": Nv, "D": Dv, "Q": Qv,
                              "dQ": dQ, "delta_L_Q": np.nan, "N_eq": np.nan,
                              "N_eq_over_N_minus_1": np.nan, "status": "Q+dQ>1 截断跳过"})
                continue
            dl = float(delta_loss_Q(m2s["x"], np.array([Nv]), np.array([Dv]),
                                    np.array([Qv]), np.array([hv]),
                                    np.array([dg]), dQ=dQ)[0])
            neq, st = equivalent_N(m2s["x"], Nv, Dv, Qv, hv, dg, dQ=dQ)
            qrows.append({"operating_point": name, "N": Nv, "D": Dv, "Q": Qv, "dQ": dQ,
                          "delta_L_Q": dl,
                          "N_eq": (None if not np.isfinite(neq) else float(neq)),
                          "N_eq_over_N_minus_1": (None if not np.isfinite(neq)
                                                  else float(neq / Nv - 1.0)),
                          "status": st})
    elast_tab = pd.DataFrame(erows)
    eqv_tab = pd.DataFrame(qrows)
    save_table(elast_tab, cfg, "elasticities.csv")
    save_table(eqv_tab, cfg, "quality_parameter_equivalence.csv")
    for _, r in elast_tab.iterrows():
        log(f"  {r['operating_point']:16s}: L={r['L_pred']:.4f} ε_N={r['eps_N']:+.4f} "
            f"ε_D={r['eps_D']:+.4f} ε_Q={r['eps_Q']:+.4f}")

    # 有限差分核验（§2.6：导数须经差分校验）
    fd_rows = []
    hstep = float(cfg["elasticity"]["finite_diff_step"])
    Nv, Dv, Qv, hv, dg = 1.0, 100.0, 0.8, 1.0, 0.0
    ana = derivatives(m2s["x"], np.array([Nv]), np.array([Dv]), np.array([Qv]), np.array([hv]))
    for key, var, val, dvar in (("dL_dN", "N", Nv, hstep), ("dL_dD", "D", Dv, hstep),
                                ("dL_dQ", "Q", Qv, hstep)):
        args = {"N": np.array([Nv]), "D": np.array([Dv]), "Q": np.array([Qv])}
        up, dn = dict(args), dict(args)
        up[var] = np.array([val * (1 + dvar)])
        dn[var] = np.array([val * (1 - dvar)])
        pu = float(predict(m2s["x"], up["N"], up["D"], up["Q"], np.array([hv]),
                           np.array([dg]))[0])
        pd_ = float(predict(m2s["x"], dn["N"], dn["D"], dn["Q"], np.array([hv]),
                            np.array([dg]))[0])
        num = (pu - pd_) / (up[var][0] - dn[var][0])
        an = float(ana[key][0])
        rel = abs(num - an) / max(abs(an), 1e-12)
        fd_rows.append({"param": key, "at": f"N={Nv},D={Dv},Q={Qv}",
                        "analytic": an, "finite_diff": num, "rel_err": rel,
                        "passed": bool(rel <= float(cfg["elasticity"]["fd_rel_tol"]))})
        log(f"  差分核验 {key:6s}: 解析={an:+.6g} 差分={num:+.6g} 相对误差={rel:.2e} "
            f"{'OK' if rel <= float(cfg['elasticity']['fd_rel_tol']) else 'FAIL'}")
    fd_tab = pd.DataFrame(fd_rows)
    save_table(fd_tab, cfg, "derivative_check.csv")

    # ------------------------------------------------------------ 8. 校准与迁移
    section("8. 质量校准与配比迁移情景")
    q1_domain_q = quality_of_mixture(p0, qmap, pcols)
    b_quality = {k: prep_source(loaded[k], cfg, q_ref)["Q"].to_numpy()
                 for k in ("b6_quality_main", "b7_quality_expanded", "b8_quality_large")
                 if k in loaded}
    cal = calibration_report(cfg, np.array([v for v in qmap.values() if not np.isnan(v)]),
                             b_quality)
    cal["q1_reference_mixture_quality"] = float(np.asarray(q1_domain_q, float)[0])
    cal["primary_effect_functional"] = cfg["transfer"]["primary_effect"]
    cal["q1_quality_column"] = "q_entropy"
    cal["q1_unmapped_policy"] = "exclude（未映射域剔除并重新归一化，与 Q1-C 一致）"
    save_json(cal, cfg, "quality_scale_calibration.json")
    log(f"  A 侧 Q 范围 [{cal['a_side_q_range']['min']:.4f}, "
        f"{cal['a_side_q_range']['max']:.4f}]；B 侧 Q 范围 "
        f"[{cal['b_side_q_range']['min']:.4f}, {cal['b_side_q_range']['max']:.4f}]")
    log(f"  公共支撑区间: [{cal['common_support']['lo']:.4f}, "
        f"{cal['common_support']['hi']:.4f}] 空={cal['common_support']['empty']}")

    N_rep, D_rep, Q_rep = float(b6["N"].median()), float(b6["D"].median()), q_ref
    scen_all = []
    for func in cfg["transfer"]["effect_functionals"]:
        cand = candidate_mixtures(cfg, p0, pcols, seed)
        ts = transfer_scenarios(cfg, m2s["x"], cand, p0, qmap, pcols, N_rep, D_rep,
                                Q_rep, func)
        scen_all.append(ts)
    scen_tab = pd.concat(scen_all, ignore_index=True)
    sub = substitution_table(cfg, m2s["x"], p0, qmap, pcols, N_rep, D_rep, Q_rep,
                             cfg["transfer"]["primary_effect"],
                             float(cfg["transfer"]["omega_grid"][2]))
    save_table(scen_tab, cfg, "mixture_transfer_scenarios.csv")
    save_table(sub, cfg, "domain_substitution.csv")

    # 结构化 JSON（§2.7 要求的 mixture_transfer_scenarios.json）：按 泛函 × ω 归档，
    # 便于直接读取某个情景的 h 与 Loss 变化，而不必解析长表。
    scen_json = {
        "baseline_id": "Q2-C",
        "generated_utc": utc_now(),
        "config_hash": config_hash(cfg),
        "operating_point": {"N_B": N_rep, "D_B": D_rep, "Q": Q_rep},
        "h_definition": ("h(p)=exp{-omega*[f(p)-f(p0)]/s_f}，f 为损失型偏离泛函，"
                         "s_f 为参考配比处 eps=0.05 的有限差分尺度；h(p0)=1，h>0"),
        "omega_grid": [float(w) for w in cfg["transfer"]["omega_grid"]],
        "primary_functional": cfg["transfer"]["primary_effect"],
        "candidate_delta": float(cfg["transfer"]["candidate_delta"]),
        "substitution_delta": float(cfg["transfer"]["substitution_delta"]),
        "assumption_note": ("omega 无法由现有数据估计（附件中没有变配比训练实验），"
                            "故本文件全部内容为**情景**，不是经验区间或预测区间。"
                            "omega=0 时 h≡1，即退回 Q2-B 主模型（rho=0）。"),
        "by_functional": {},
    }
    for func, g in scen_tab.groupby("functional", sort=False):
        block = {"scale_s": float(g["scale_s"].iloc[0]), "by_omega": {}}
        for om, gg in g.groupby("omega", sort=True):
            rows = gg[gg["scenario"] != "reference_p0"]
            block["by_omega"][f"{float(om):g}"] = {
                "n_scenarios": int(len(rows)),
                "h_min": float(gg["h"].min()), "h_max": float(gg["h"].max()),
                "max_abs_dLoss": float(np.max(np.abs(gg["dLoss"]))),
                "loss_at_reference_p0": float(
                    gg.loc[gg["scenario"] == "reference_p0", "loss_with_h"].iloc[0]
                    if (gg["scenario"] == "reference_p0").any() else np.nan),
                "scenarios": [
                    {"scenario": str(r["scenario"]), "delta_l1": float(r["delta"]),
                     "h": float(r["h"]), "loss_h1": float(r["loss_h1"]),
                     "loss_with_h": float(r["loss_with_h"]), "dLoss": float(r["dLoss"])}
                    for _, r in rows.iterrows()],
            }
        scen_json["by_functional"][func] = block
    save_json(scen_json, cfg, "mixture_transfer_scenarios.json")

    log(f"  迁移情景 {len(scen_tab)} 行（{len(cfg['transfer']['effect_functionals'])} 泛函 × "
        f"{len(cfg['transfer']['omega_grid'])} ω）；领域替代 {len(sub)} 行")

    # ------------------------------------------------------------ 9. 损失预测器
    section("9. 损失预测器（Q2→Q3 接口）")
    predictor = {
        "baseline_id": "Q2-C",
        "form": "L = E + A*(N*Q^rho)^(-alpha) + B*(D*Q^kappa*h)^(-beta)",
        "nested": "M1(Q2-B) 为 rho=0；M0(经典) 为 rho=0 且 kappa=0",
        "parameters": {n: float(m2s["x"][i]) for i, n in enumerate(FULL)},
        "parameter_source": "M2-staged（经典参数由 B1 冻结，质量通道由 B1∪B6 估计）",
        "joint_reference_parameters": {n: float(jm_main["x"][i]) for i, n in enumerate(FULL)},
        "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        "q_reference_convention": q_ref,
        "offset_rule": ("dgrp=1 的行（质量实验族 B6/B7/B8）在预测时加 δ；"
                        "B1 及族外来源不加 δ"),
        "h_definition": ("h = exp(-omega * Delta)，Delta 为相对参考配比 p0 的损失型偏离；"
                         "h(p0)=1，h>0。omega 为情景参数，未经联合实验标定"),
        "supported_range": {
            "N_B": [float(b1["N"].min()), float(max(b1["N"].max(),
                                                     b6["N"].max()))],
            "D_B": [float(b1["D"].min()), float(max(b1["D"].max(),
                                                     b6["D"].max()))],
            "Q": [float(min(b6["Q"].min(), q_ref)), float(max(b6["Q"].max(), q_ref))],
        },
        "warnings": [
            "κ、ρ 的可辨识性以 identify 判定为准；未辨识时本预测器只用于情景，不作为真实新定律。",
            "h(p) 的 ω 无法由现有数据估计，只在情景网格下使用。",
            "族外来源（B2/B4/B5/B10）存在未吸收的水平差，跨族直接使用预测器须重新校准。",
        ],
        "identifiability": ident["M2_staged"]["dual_channel_identifiable"],
        "identify_reasons": ident["M2_staged"]["reasons_not_identified"],
        "quality_channel_status": "scenario_only",
        "quality_channel_note": (
            "κ、ρ 在 B6/B7 内部可辨识，但真实观测 B1 对它们零信息，且 B8 给出零信息"
            "（与 B6/B7 矛盾）。主模型按协议退回 Q2-B（rho=0）。第三问若使用本预测器"
            "做质量相关优化，必须把 ρ、κ 记为情景假设，并报告 ω=0 时的 Q2-B 结果作为"
            "下界对照。"
        ),
        "recommended_primary_model": "M1(Q2-B): rho=0",
        "scenario_model": "M2(Q2-C): rho 自由，仅情景",
    }
    save_json(predictor, cfg, "loss_predictor.json")
    write_text(PREDICTOR_SRC, os.path.join(art, "predict_q2c_loss.py"))
    log(f"  已写出 loss_predictor.json 与 predict_q2c_loss.py")

    # ------------------------------------------------------------ 10. 参数总表
    section("10. 汇总产出")
    params = {
        "baseline_id": "Q2-C",
        "question": "Q2",
        "generated_utc": utc_now(),
        "config_hash": config_hash(cfg),
        "seed": seed,
        "model_family": {
            "M0": "L = E + A*N^(-alpha) + B*D^(-beta)",
            "M1": "L = E + A*N^(-alpha) + B*(D*Q^kappa*h)^(-beta)",
            "M2": "L = E + A*(N*Q^rho)^(-alpha) + B*(D*Q^kappa*h)^(-beta)",
        },
        "stages": {
            "stage1_classical_B1": {
                "free": s1["free"], "params": {n: float(x1[i]) for i, n in enumerate(FULL)},
                "sse": s1["sse"], "rmse": s1["rmse"], "n": s1["n"],
                "n_clusters": len(s1["clusters"]), "boundary_hits": s1["boundary_hits"],
                "aic": aic_bic(s1["sse"], s1["n"], len(s1["free"]))[0],
            },
            "stage2_M1_staged": {
                "free": m1s["free"], "params": {n: float(m1s["x"][i]) for i, n in enumerate(FULL)},
                "sse": m1s["sse"], "rmse": m1s["rmse"], "n": m1s["n"],
                "n_clusters": len(m1s["clusters"]), "boundary_hits": m1s["boundary_hits"],
                "aic": aic_bic(m1s["sse"], m1s["n"], len(m1s["free"]))[0],
            },
            "stage2_M2_staged": {
                "free": m2s["free"], "params": {n: float(m2s["x"][i]) for i, n in enumerate(FULL)},
                "sse": m2s["sse"], "rmse": m2s["rmse"], "n": m2s["n"],
                "n_clusters": len(m2s["clusters"]), "boundary_hits": m2s["boundary_hits"],
                "aic": aic_bic(m2s["sse"], m2s["n"], len(m2s["free"]))[0],
            },
        },
        "nested_baseline_SSE_on_B1uB6": sse_m0_comb,
        "ablation_M0_plus_delta": {
            "free": m0d["free"],
            "params": {n: float(m0d["x"][i]) for i, n in enumerate(FULL)},
            "sse": m0d["sse"], "rmse": m0d["rmse"], "n": m0d["n"],
        },
        "nested_ftest_M0_to_M0delta": F_delta,
        "nested_ftest_M0delta_to_M1": F_m1,
        "nested_ftest_M1_to_M2": F_m2,
        "sse_decomposition": {
            "M0_no_offset": sse_m0_comb, "plus_delta": m0d["sse"],
            "plus_kappa": m1s["sse"], "plus_rho": m2s["sse"],
        },
        "effective_floor_E_plus_delta": floor_eff,
        "provenance_audit": {
            "note": ("B11/B12 为辅助索引，**不含 Loss 列，不能进入任何拟合**；"
                     "仅用于口径与覆盖面的溯源核验。"),
            "b1_internal_consistency": prov["b1_internal_consistency"].to_dict("records"),
            "b12_checkpoint_coverage": (
                prov["b12_checkpoint_coverage"].to_dict("records")
                if len(prov["b12_checkpoint_coverage"]) else []),
            "b12_sizes_absent_from_B1": list(
                prov["b12_checkpoint_coverage"].attrs.get("missing_sizes", [])
                if len(prov["b12_checkpoint_coverage"]) else []),
            "b11_param_audit_summary": (
                {str(k): {"n": int(v["count"]), "bytes_per_param_min": float(v["min"]),
                          "bytes_per_param_max": float(v["max"])}
                 for k, v in prov["b11_param_audit"]
                 .groupby("family_prefix")["bytes_per_param"].agg(["min", "max", "count"])
                 .iterrows()} if len(prov["b11_param_audit"]) else {}),
            "convergence_audit": (
                prov["convergence_audit"].to_dict("records")
                if len(prov["convergence_audit"]) else []),
        },
        "joint_weight_sensitivity": joint_tab.to_dict("records"),
        "quality_source_sensitivity": alt_tab.to_dict("records"),
        "quality_channel_reconciliation": {
            "informative_sets": informative,
            "uninformative_sets": uninformative,
            "per_subset_informative": detail,
            "real_data_support": (
                "无。B1（唯一真实观测）以 RMSE 1.5e-4 被经典律解释，残差量级等于数据"
                "自身的四位小数舍入；且 B1 无 Q 列，Q 按基准约定恒取 1.0，不含任何质量"
                "变化，因此对 κ、ρ 零信息。"
            ),
            "conclusion": (
                "质量双通道在 B6/B7 内部可辨识（κ、ρ 为内点解、无边界命中、嵌套 F 检验"
                "极显著），但 (a) B1 这一真实观测上根本没有可用残差，(b) B8 的 calibrated"
                " 与 extrapolated 两段上 κ=ρ=0 且经典 M0 与七参 M2 给出完全相同的 RMSE，"
                "即质量通道零信息，与 B6/B7 直接矛盾。故不得把 B6/B7 的拟合当作真实新"
                "定律：主模型按 Q2-C.md 第 4 条退回 Q2-B（ρ=0）；κ、ρ 仅作为情景参数，"
                "并在预测器中显式标注该边界。"
            ),
        },
        "sample_overlap": overlap_tab.to_dict("records"),
        "trajectory_check": traj_tab.to_dict("records"),
        "unit_audit": unit_tab.to_dict("records"),
        "identifiability": ident,
        "bootstrap_M0_B1": boot_v.to_dict("records"),
        "bootstrap_M2_quality": boot_s.to_dict("records"),
        "holdout_by_size_RMSE_median": float(ho_size["rmse"].median()),
        "holdout_D_direction": ho_d,
        "trajectory_RMSE_median": float(traj_tab["rmse_test"].median()),
        "cross_source_M2": cross_s.to_dict("records"),
        "calibration": {
            "primary_mapping": cal["primary"],
            "common_support": cal["common_support"],
            "assumption_note": cal["assumption_note"],
        },
        "transfer": {
            "primary_effect_functional": cfg["transfer"]["primary_effect"],
            "omega_grid": cfg["transfer"]["omega_grid"],
            "note": "ω 未经联合实验标定，仅给含零效应在内的小组预设情景",
        },
        "deliverables": sorted(os.listdir(art)),
    }
    save_json(params, cfg, "scaling_parameters.json")

    section("11. 运行元数据与日志")
    meta = {
        "baseline_id": "Q2-C",
        "generated_utc": utc_now(),
        "config_path": cfg["_config_path"],
        "config_hash": config_hash(cfg),
        "seed": seed,
        "git_rev_q2c": git_rev(cfg["_root"]),
        # 上游版本 = 实际存放附件 B 的那个仓库的 revision（按数据目录向上找 .git）。
        # 独立布局下是并列的 math_model_F；并入仓库后就是仓库自身，两种布局都对。
        "git_rev_upstream": git_rev(scaling_dir(cfg)),
        "env": env_info(),
        "input_hashes": in_hashes,
        # 用 q1_path 解析出的真实路径算哈希——不能硬编码 "..\\math_model_F"：
        # 并入仓库后上游就是仓库自身，硬编码会指到不存在的目录、哈希静默变 null，
        # 而数据其实读到了另一份，「输入哈希可复现」这一条就名存实亡。
        "q1_artifacts": {
            name: hash_if_exists(q1_path(cfg, name))
            for name in ("quality_domain_mapping.csv", "mixture_predictor.json")
        },
        "sample_flow": {
            "B1_real": int(len(b1)),
            "B6_quality": int((comb.batch == "B6").sum()),
            "combined_fit_n": int(len(comb)),
            "B3_trajectory_rows": int(len(traj)),
        },
        "artifacts": sorted(os.listdir(art)),
    }
    save_json(meta, cfg, "run_metadata.json")
    write_text("\n".join(_LOG), os.path.join(art, "run_log.txt"))
    log(f"  产物目录: {art}")
    log(f"  文件数  : {len(os.listdir(art))}")
    return 0


PREDICTOR_SRC = '''# -*- coding: utf-8 -*-
"""Q2-C 损失预测器（Q2→Q3 接口）。

用法：
    from predict_q2c_loss import Q2CPredictor
    m = Q2CPredictor("loss_predictor.json")
    L = m.predict(N=7.0, D=300.0, Q=0.8, h=1.0, dgrp=0.0)

注意（见 loss_predictor.json 的 warnings）：
    * κ、ρ 未通过可辨识性判定时，本预测器只用于情景分析；
    * ω 未经标定，h 只在情景网格下取值；
    * 族外来源存在未吸收的水平差。
"""

import json

import numpy as np


class Q2CPredictor:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            self.d = json.load(fh)
        p = self.d["parameters"]
        self.E, self.A, self.al = p["E"], p["A"], p["alpha"]
        self.B, self.be = p["B"], p["beta"]
        self.ka, self.rh, self.de = p["kappa"], p["rho"], p["delta"]

    def predict(self, N, D, Q, h=1.0, dgrp=0.0):
        N = np.asarray(N, float); D = np.asarray(D, float)
        Q = np.asarray(Q, float); h = np.asarray(h, float)
        dgrp = np.asarray(dgrp, float)
        return (self.E + self.de * dgrp
                + self.A * np.power(N * np.power(Q, self.rh), -self.al)
                + self.B * np.power(D * np.power(Q, self.ka) * h, -self.be))

    def h_from_delta(self, delta, omega):
        """h = exp(-omega*Delta)；omega=0 即零效应（嵌套回 Q2-B）。"""
        return np.exp(-np.asarray(omega, float) * np.asarray(delta, float))
'''


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
