# -*- coding: utf-8 -*-
"""Q3-C 全流程入口。

    1. 读取上游接口（Q1-C 配比预测器与质量坐标、Q2-C 冻结主模型与转移情景、C7 元数据）；
    2. 在第一问训练信息与可信区域内**枚举候选配比**（不用最终检验 Loss 偷选）；
    3. 复用 Q3-B 式的预算/成本内核，对每个候选在各预算情景下搜索 (N, Q)；
    4. 用多起点 SLSQP 在单纯形 + 可信区域内做局部联合细调，与枚举结果交叉核验，
       标注局部解；
    5. 沿连续 log C 网格跟踪最优解与成本份额，按**预先定义**的事件识别结构性转移，
       并在更粗网格下复核；
    6. 数值核验（解析导数 vs 有限差分、预算/单纯形约束、网格与初值稳定性、
       与上游预测器逐点一致、输入审计）；
    7. 落盘交付物与报告。

运行：  python run_all.py            （在 src/baselines/Q3-C 下）
        python run_all.py --quick    （缩小网格与情景，仅用于冒烟测试）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (TeeLog, Timer, artifacts_dir, config_hash, env_info, git_rev,
                    hash_if_exists, load_config, q1_path, q2_path, save_json,
                    save_table, source_hash, utc_now, which_root, write_text)
from mixtures import (enumerate_candidates, frozen_scale, h_of_mixture,
                      lp_max_quality, loss_type_deviation, q1_quality_table,
                      q1_reference, qmap_from_table, quality_of_mixture,
                      slsqp_max_quality, trust_region)
from model import Q3Model
from report import build_report
from scenarios import (budget_path, c7_discrete_cases, c7_grounding,
                       classify_q_regime, detect_q_transitions, p_support_change,
                       scenario_grid, share_segments)
from search import joint_slsqp, search_fixed_p
import verify as V


# --------------------------------------------------------------------- 小工具
def cells_of(cfg: dict, ell_list=None, q0_ref: float = None) -> list[dict]:
    """主情景单元：预算 × 成本形式 × 上下文长度 × Q0 设定（kappa 取主情景）。"""
    k = cfg["model"]["kappa_primary"]
    kappa = float(cfg["model"]["kappa_scenarios"][k])
    out = []
    for C in cfg["budget"]["C_scenarios"]:
        for form in cfg["cost"]["g_forms"]:
            for ell in (ell_list if ell_list is not None else cfg["context"]["ell_scenarios"]):
                for q0m in cfg["quality"]["q0_modes"]:
                    out.append({"C": float(C), "cost_form": form, "ell": int(ell),
                                "q0_mode": q0m, "kappa": kappa, "Q0_ref": q0_ref})
    return out


def _flags(cfg, model, r) -> dict:
    """外推标注：解是否落在 Q2-C 数据支持范围之外（§3.1 要求显式标注）。"""
    n_sup = cfg["search"]["N_support_B"]; d_sup = cfg["search"]["D_support_B"]
    nB = r["N"] / 1e9; dB = r["D"] / 1e9
    n_low, n_high = nB < n_sup[0], nB > n_sup[1]
    d_low, d_high = dB < d_sup[0], dB > d_sup[1]
    q_high = r["Q"] > float(cfg["quality"]["q_supported_max"]) + 1e-12
    return {
        "N_in_support": bool(not n_low and not n_high),
        "D_in_support": bool(not d_low and not d_high),
        "extrapolation": ";".join(
            [t for t, ok in (("N_below_support", n_low), ("N_above_support", n_high),
                             ("D_below_support", d_low), ("D_above_support", d_high),
                             ("Q_above_A_supported", q_high)) if ok]),
    }


# --------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟测试：缩小网格与情景")
    ap.add_argument("--report-only", action="store_true",
                    help="不重算数值，仅从已落盘产物重建 ctx 并重渲染报告")
    args = ap.parse_args()

    cfg = load_config()
    if args.report_only:
        return report_only(cfg)
    if args.quick:
        cfg["search"]["N_gross_grid"] = 61
        cfg["search"]["Q_grid"] = 41
        cfg["search"]["refine_rounds"] = 2
        cfg["search"]["refine_points"] = 31
        cfg["search"]["slsqp"]["n_starts"] = 6
        cfg["budget"]["C_grid_log10"]["n"] = 13
        cfg["context"]["ell_scenarios"] = [4096, 32768]
        print("[quick] 已缩小网格与情景，结果仅供冒烟测试，不得作为结论。")

    adir = artifacts_dir(cfg)
    log = TeeLog(os.path.join(adir, run_log_name(args.quick)))
    t_all = Timer()
    log("=" * 78)
    log("Q3-C：算力约束下的资源联合优化 —— 候选配比枚举 + 逐候选资源搜索 + 联合细调")
    log(f"配置 {cfg['_config_path']}  hash={config_hash(cfg)}  quick={args.quick}")
    log("=" * 78)

    ctx: dict = {"cfg": cfg, "quick": bool(args.quick), "flow": [],
                 "config_hash": config_hash(cfg), "source_hash": source_hash()}

    def flow(stage: str, n_in, n_out, note: str = ""):
        ctx["flow"].append({"stage": stage, "n_in": int(n_in), "n_out": int(n_out),
                            "note": note})

    # ================================================================ 1. 上游
    with Timer() as t1:
        p0, cols, pred = q1_reference(cfg)
        tab = q1_quality_table(cfg)
        qmap = qmap_from_table(tab)
        lo, hi = trust_region(cfg, pred, cols)
        model = Q3Model(cfg)
        c7 = None
        try:
            from common import load_c7
            c7 = load_c7(cfg)
        except FileNotFoundError as e:
            log(f"[warn] C7 元数据不可用：{e}")
        functionals = list(cfg["mixtures"]["functionals"])
        s_by = {f: frozen_scale(cfg, f) for f in functionals}
        omegas = [float(x) for x in cfg["quality"]["omega_scenarios"]]
        om_pri = float(cfg["quality"]["omega_primary"])
        k_pri = cfg["model"]["kappa_primary"]
        kappa = float(cfg["model"]["kappa_scenarios"][k_pri])
    q_hi = float(cfg["quality"]["q_extended_max"])
    q0_ref = float(quality_of_mixture(p0[None, :], qmap, cols)[0])
    ctx.update({"p0": p0, "cols": cols, "pred": pred, "tab": tab, "qmap": qmap,
                "lo": lo, "hi": hi, "s_by": s_by, "functionals": functionals,
                "omegas": omegas, "omega_primary": om_pri, "kappa": kappa,
                "kappa_name": k_pri, "q0_ref": q0_ref, "q_hi": q_hi})
    log(f"[1] 上游：{len(cols)} 域；p0 参考质量 Q(p0)={q0_ref:.6f}；"
        f"kappa={kappa:.4f}({k_pri})；冻结尺度 s={ {k: round(v, 6) for k, v in s_by.items()} }")
    log(f"    已映射域质量：{[ (c.replace('train_the_pile_',''), round(qmap[c],4)) for c in cols if not np.isnan(qmap[c]) ]}")
    log(f"    可信区域盒（p05/p95）已建立；C7={'有' if c7 is not None else '无'}")
    flow("上游接口读取", len(cols), len(cols), "Q1 列序/参考配比/训练支持 + Q2 冻结参数")

    # ================================================================ 2. 候选枚举
    with Timer() as t2:
        cand = enumerate_candidates(cfg, p0, cols, qmap, lo, hi,
                                    s_by[functionals[0]], functionals[0], cfg["seed"])
        # 解析参考点：线性分式规划（Charnes–Cooper）的精确最优配比，必须进候选集
        lp = lp_max_quality(p0, qmap, cols, lo, hi)
        sl_nol1 = slsqp_max_quality(p0, qmap, cols, lo, hi, None, cfg["seed"])
        sl_l1 = slsqp_max_quality(p0, qmap, cols, lo, hi,
                                  float(cfg["mixtures"]["trust_l1_radius"]), cfg["seed"])
        extra = []
        for tag, src, obj in (("lp_vertex_box", "analytic_lp_charnes_cooper", lp),
                              ("slsqp_maxQ_nol1", "numeric_slsqp_reference", sl_nol1),
                              ("slsqp_maxQ_l1", "numeric_slsqp_reference", sl_l1)):
            if obj.get("ok") and obj.get("p") is not None:
                pv = np.clip(np.asarray(obj["p"], float), lo, hi)
                pv = pv / pv.sum()
                extra.append({"candidate": tag, "source": src, "in_trust_box": True,
                              "in_trust_l1": bool(np.abs(pv - p0).sum() <=
                                                  float(cfg["mixtures"]["trust_l1_radius"]) + 1e-12),
                              **dict(zip(cols, pv))})
        if extra:
            ex = pd.DataFrame(extra)
            ex["trust_status"] = np.where(ex["in_trust_l1"], "in", "box_only")
            ex["Q"] = quality_of_mixture(ex[cols].to_numpy(float), qmap, cols)
            ex["delta"] = loss_type_deviation(ex[cols].to_numpy(float), p0, qmap, cols,
                                              functionals[0], s_by[functionals[0]])
            cand = pd.concat([cand, ex], ignore_index=True)
        cand["Q"] = quality_of_mixture(cand[cols].to_numpy(float), qmap, cols)
        # 去重：LP 顶点与多起点 SLSQP 可能落在**同一点**（Q 只差 ~1e-16），
        # 标签不同但配比相同；若不去重，"最优配比是否随情景变化"的判据会被这类
        # 浮点平局误判为"不一致"。这里按配比向量合并，保留首个标签并登记别名。
        Pt = cand[cols].to_numpy(float)
        keep_idx, alias = [], []
        for i in range(len(cand)):
            hit = next((k for k in keep_idx
                        if np.allclose(Pt[i], Pt[k], rtol=0.0, atol=1e-9)), None)
            if hit is None:
                keep_idx.append(i); alias.append(str(cand["candidate"].iloc[i]))
            else:
                j = keep_idx.index(hit)
                if str(cand["candidate"].iloc[i]) not in alias[j]:
                    alias[j] += " | " + str(cand["candidate"].iloc[i])
        n_dedup = len(cand) - len(keep_idx)
        cand = cand.iloc[keep_idx].reset_index(drop=True)
        cand["aliases"] = alias
        cand["delta_ref"] = cand["delta"]
        cand["delta_mixture_l1"] = loss_type_deviation(
            cand[cols].to_numpy(float), p0, qmap, cols, "mixture_l1", s_by["mixture_l1"])
        cand["is_p0"] = np.isclose(cand[cols].to_numpy(float) - p0[None, :], 0).all(axis=1)
    save_table(cand, cfg, "candidates.csv")
    ctx["cand"] = cand
    log(f"[2] 候选枚举：{len(cand)} 个（去重 {n_dedup} 个重合点，含 {len(extra)} 个解析参考点）；"
        f"Q 范围 [{cand['Q'].min():.4f}, {cand['Q'].max():.4f}]；"
        f"可信区域 in/box_only/outside = "
        f"{(cand['trust_status'] == 'in').sum()}/"
        f"{(cand['trust_status'] == 'box_only').sum()}/"
        f"{(cand['trust_status'] == 'outside').sum()}")
    log(f"    LP(Charnes–Cooper) 盒内 max Q 精确解 = {lp.get('q_max')}; "
        f"SLSQP(无 L1) = {sl_nol1.get('q_max')}; SLSQP(L1 半径 {cfg['mixtures']['trust_l1_radius']}) = {sl_l1.get('q_max')}")
    ctx["lp"] = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                 for k, v in lp.items()}
    flow("候选配比枚举", len(cols), len(cand), "仅用 p0 / 可信区域 / 第一问质量坐标")

    # ================================================================ 3. 筛选
    Pc = cand[cols].to_numpy(float)
    screen_cells = cells_of(cfg, ell_list=[int(cfg["context"]["ell_primary"])],
                            q0_ref=q0_ref)
    n_scr_grid = (31, 17) if args.quick else (41, 21)
    scr_rows = []
    with Timer() as t3:
        for f in functionals:
            for om in [x for x in omegas if x > 0.0]:
                hv = h_of_mixture(Pc, p0, qmap, cols, f, om, s_by[f])
                for cell in screen_cells:
                    for i in range(len(cand)):
                        Q0i = q0_ref if cell["q0_mode"] == "fixed_ref" else float(cand["Q"].iloc[i])
                        r = search_fixed_p(model, float(hv[i]), Q0i, cell["cost_form"],
                                           cell["C"], cell["ell"], cell["kappa"],
                                           Q0i, q_hi, cfg, n_grid=n_scr_grid[0],
                                           q_grid=n_scr_grid[1], refine_rounds=0)
                        scr_rows.append({
                            "functional": f, "omega": om, "C": cell["C"],
                            "cost_form": cell["cost_form"], "ell": cell["ell"],
                            "q0_mode": cell["q0_mode"], "candidate": cand["candidate"].iloc[i],
                            "Q_cand": float(cand["Q"].iloc[i]), "h": float(hv[i]),
                            "N_B": r["N"] / 1e9, "Q": r["Q"], "L_pred": r["L"],
                            "q_at_lower": r["boundary_q_lower"],
                            "q_at_upper": r["boundary_q_upper"],
                        })
    scr = pd.DataFrame(scr_rows)
    save_table(scr, cfg, "screening.csv")
    agree = V.cell_ranking(scr, ["functional", "omega", "C", "cost_form", "ell", "q0_mode"],
                           lcol="L_pred")
    save_table(agree, cfg, "screening_cell_ranking.csv")
    n_Lh = int(agree["argmin_L_is_argmax_h"].sum())
    n_hQ = int(agree["argmax_h_is_argmax_Q"].sum())
    ctx["screen_agree"] = agree
    log(f"[3] 候选筛选：{len(scr)} 次廉价搜索（{n_scr_grid[0]}x{n_scr_grid[1]} 网格，"
        f"{len(screen_cells)} 个离散情景 x {len(cand)} 候选）。")
    log(f"    以**数值容差**判定（浮点平局不算不一致）：argmin L 落在 argmax h 上 "
        f"{n_Lh}/{len(agree)}；argmax h 同时是 argmax Q 的单元 {n_hQ}/{len(agree)}"
        f"（后者只对 quality_linear 成立：该泛函的 Δ 是 Q 的仿射减函数；"
        f"mixture_l1 的 h 在 p0 取最大，argmax Q 与之本就不同，非缺陷）。")
    log(f"    候选间 L 跨度的中位数 = {agree['L_spread'].median():.3e}"
        f"（该量级说明「哪个 p 最优」由 h 的单调性决定，而非靠数值分辨 L 的差异）")
    flow("候选筛选（廉价网格）", len(cand) * len(screen_cells), len(scr),
         "每个候选在每个离散情景下做一次粗搜索")

    # ================================================================ 4. 终选 + 全情景
    fin_tags = set()
    fin_tags.update(agree["argmin_L_candidate"].unique())
    fin_tags.update(agree["argmax_h_candidate"].unique())
    fin_tags.update(agree["argmax_Q_candidate"].unique())
    k_sel = int(cfg["mixtures"]["select_top_k"])
    fin_tags.update(cand.nlargest(k_sel, "Q")["candidate"])
    fin_tags.update(cand.nsmallest(k_sel, "Q")["candidate"])
    fin_tags.update(cand.loc[cand["source"].astype(str).str.contains("analytic"), "candidate"])
    fin_tags.update(cand.loc[cand["is_p0"], "candidate"])
    finals = cand[cand["candidate"].isin(fin_tags)].reset_index(drop=True)
    save_table(finals, cfg, "finalists.csv")
    ctx["finals"] = finals
    log(f"[4] 终选候选 {len(finals)} 个：{sorted(fin_tags)}")

    full_cells = cells_of(cfg, q0_ref=q0_ref)
    Pf = finals[cols].to_numpy(float)
    alloc = []
    alloc_all = []                       # 每个 (情景, 终选候选) 的完整评估，供不变性核验
    with Timer() as t4:
        for f in functionals:
            for om in omegas:
                hv = h_of_mixture(Pf, p0, qmap, cols, f, om, s_by[f])
                for cell in full_cells:
                    best = None
                    for i in range(len(finals)):
                        Q0i = q0_ref if cell["q0_mode"] == "fixed_ref" else float(finals["Q"].iloc[i])
                        r = search_fixed_p(model, float(hv[i]), Q0i, cell["cost_form"],
                                           cell["C"], cell["ell"], cell["kappa"],
                                           Q0i, q_hi, cfg)
                        r.update({"functional": f, "omega": om, "C": cell["C"],
                                  "cost_form": cell["cost_form"], "ell": cell["ell"],
                                  "q0_mode": cell["q0_mode"], "kappa": cell["kappa"],
                                  "candidate": finals["candidate"].iloc[i],
                                  "aliases": finals["aliases"].iloc[i],
                                  "source": finals["source"].iloc[i],
                                  "trust_status": finals["trust_status"].iloc[i],
                                  "in_trust_box": bool(finals["in_trust_box"].iloc[i]),
                                  "in_trust_l1": bool(finals["in_trust_l1"].iloc[i]),
                                  "h_true": float(hv[i]), "Q0": Q0i,
                                  "Q_cand": float(finals["Q"].iloc[i])})
                        alloc_all.append({k: v for k, v in r.items()
                                          if k not in ("N", "D")} |
                                         {"N_B": r["N"] / 1e9, "D_B": r["D"] / 1e9})
                        if best is None or r["L"] < best["L"]:
                            best = r
                    best.update(_flags(cfg, model, best))
                    best["regime"] = classify_q_regime(
                        best["Q"], best["Q0"], q_hi,
                        float(cfg["verify"]["transition"]["q_boundary_tol"]))
                    best["D_B"] = best["D"] / 1e9
                    best["N_B"] = best["N"] / 1e9
                    best.pop("D", None); best.pop("N", None)
                    best["L_pred"] = best.pop("L")
                    win = finals.index[finals["candidate"] == best["candidate"]][0]
                    best.update({c: float(finals[c].iloc[win]) for c in cols})
                    alloc.append(best)
    alloc_df = pd.DataFrame(alloc)
    # 列顺序：情景 → 最优配比标识 → 资源 → 成本 → 状态
    front = ["functional", "omega", "C", "cost_form", "ell", "q0_mode", "kappa",
             "candidate", "source", "trust_status", "in_trust_box", "in_trust_l1",
             "N_B", "D_B", "Q", "Q0", "h", "h_true", "L_pred", "c_quality", "u"]
    rest = [c for c in alloc_df.columns if c not in front]
    alloc_df = alloc_df[front + rest]
    save_table(alloc_df, cfg, "optimal_allocations.csv")
    alloc_all_df = pd.DataFrame(alloc_all)
    save_table(alloc_all_df, cfg, "all_candidate_evaluations.csv")
    ctx["alloc"] = alloc_df
    ctx["alloc_all"] = alloc_all_df
    log(f"[4] 全情景精确求解：{len(full_cells)} 个情景 x {len(finals)} 个终选 x "
        f"{len(functionals)} 泛函 x {len(omegas)} 个 omega = {len(alloc_df)} 个最优分配"
        f"（完整逐候选评估 {len(alloc_all_df)} 行已留存）")
    flow("逐候选 (N,Q) 精确搜索", len(finals) * len(full_cells),
         len(alloc_df), "复用 Q3-B 式预算/成本内核；每点精确成本")

    # 配比通道的 p* 复现性（核心结论：可分离 h 下 p* 与资源情景解耦）
    win_by_cell = alloc_df.groupby(["functional", "omega"])["candidate"].agg(
        lambda s: sorted(set(s)))
    ctx["p_winners"] = win_by_cell
    for (f, om), tags in win_by_cell.items():
        log(f"    [{f} | omega={om}] 各情景最优配比集合 = {tags}")

    # 核心核验：精确解层面 argmin L ≡ argmax h（⇒ p* 与预算/长度/成本/κ/ω 无关）
    v_inv = V.cell_ranking(alloc_all_df, ["functional", "omega", "C", "cost_form",
                                          "ell", "q0_mode"])
    save_table(v_inv, cfg, "verify_p_channel.csv")
    ctx["v_inv"] = v_inv
    n_Lh = int(v_inv["argmin_L_is_argmax_h"].sum())
    n_hQ = int(v_inv["argmax_h_is_argmax_Q"].sum())
    log(f"    精确解不变性：argmin L 落在 argmax h 上 {n_Lh}/{len(v_inv)}；"
        f"argmax h 同时为 argmax Q 的单元 {n_hQ}/{len(v_inv)}；"
        f"候选间 L 跨度中位数 {v_inv['L_spread'].median():.3e}")

    # ================================================================ 5. 预算路径
    paths = []
    with Timer() as t5:
        for f in functionals:
            for om in [om_pri] + [x for x in omegas if 0.0 < x != om_pri]:
                hv = h_of_mixture(Pf, p0, qmap, cols, f, om, s_by[f])
                # 该泛函下的“整体最优配比”由主情景单元（kappa 主情景、ω）确定
                sub = alloc_df[(alloc_df["functional"] == f) & (alloc_df["omega"] == om)]
                if sub.empty:
                    continue
                top = sub["candidate"].value_counts().idxmax()
                i = int(finals.index[finals["candidate"] == top][0])
                h_win = float(hv[i])
                Q0_fixed = q0_ref
                Q0_mix = float(finals["Q"].iloc[i])
                for form in cfg["cost"]["g_forms"]:
                    for ell in cfg["context"]["ell_scenarios"]:
                        for q0m, Q0c in (("fixed_ref", Q0_fixed), ("mixture", Q0_mix)):
                            d = budget_path(model, cfg, h_win,
                                            (lambda C, v=Q0c: v), form, ell, kappa,
                                            lambda **kw: search_fixed_p(model, cfg=cfg, **kw))
                            d["functional"] = f; d["omega"] = om
                            d["candidate"] = top; d["q0_mode"] = q0m
                            paths.append(d)
    path_df = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    save_table(path_df, cfg, "budget_paths.csv")
    ctx["paths"] = path_df
    log(f"[5] 预算路径：{len(path_df)} 行（{path_df.groupby(['functional','omega','cost_form','ell','q0_mode']).ngroups if len(path_df) else 0} 条曲线）")

    # ================================================================ 6. 转移识别
    trans = []
    for key, g in path_df.groupby(["functional", "omega", "cost_form", "ell", "q0_mode"]):
        for ev in detect_q_transitions(g.sort_values("log10_C"), cfg):
            ev.update({"functional": key[0], "omega": key[1]})
            trans.append(ev)
    # 事件 2：配比支持集变化（在完整分配表上按维度扫描）
    rec = alloc_df[(alloc_df["functional"] == functionals[0]) &
                   (alloc_df["omega"] == om_pri)] \
        .rename(columns={"L_pred": "L"}) \
        .melt(id_vars=["functional", "omega", "candidate", "L"],
              value_vars=["C", "ell"], var_name="dimension", value_name="x")
    ps = p_support_change(rec, cfg)
    for ev in ps:
        trans.append(ev)
    trans_df = pd.DataFrame(trans)
    if trans_df.empty:
        trans_df = pd.DataFrame([{
            "event": "none_detected",
            "note": ("在预定义判据下未发现结构性转移；这与 h(p) 可分离、"
                     "p* 由 argmax Q 决定的解析结论一致，不应据此强造转移"),
        }])
    save_table(trans_df, cfg, "transition_candidates.csv")
    ctx["trans"] = trans_df
    log(f"[6] 结构性转移：{len(trans)} 个预定义事件"
        f"{'（无）' if not trans else ''}")

    # ================================================================ 7. 联合 SLSQP
    sl_rows, sl_clusters = [], []
    rep_cells = ([c for c in full_cells
                  if c["ell"] == int(cfg["context"]["ell_primary"])
                  and c["q0_mode"] == "fixed_ref"])[:9] + \
                [c for c in full_cells
                 if c["ell"] == int(cfg["context"]["ell_primary"])
                 and c["q0_mode"] == "mixture"][:3]
    f_pri = "quality_linear"
    with Timer() as t6:
        for cell in rep_cells:
            h_fn = lambda P, f=f_pri, om=om_pri: h_of_mixture(
                P, p0, qmap, cols, f, om, s_by[f])
            out = joint_slsqp(model, p0, qmap, cols, lo, hi,
                              float(cfg["mixtures"]["trust_l1_radius"]), s_by[f_pri],
                              f_pri, om_pri, cell["q0_mode"], cell["cost_form"],
                              cell["C"], cell["ell"], cell["kappa"], cfg,
                              cfg["seed"], h_fn=h_fn, q0_ref=q0_ref)
            for s_ in out["solutions"]:
                sl_rows.append({"functional": f_pri, "omega": om_pri,
                                "C": cell["C"], "cost_form": cell["cost_form"],
                                "ell": cell["ell"], "q0_mode": cell["q0_mode"],
                                "start": s_["start"], "success": s_["success"],
                                "L_pred": s_["L"], "N_B": s_["N"] / 1e9, "Q": s_["Q"],
                                "h": s_["h"], "Q0": s_["Q0"], "Q_ge_Q0": s_["Q_ge_Q0"],
                                "n_iter": s_["n_iter"], "message": s_["message"],
                                "Q_cand": float(quality_of_mixture(s_["p"][None, :],
                                                                   qmap, cols)[0]),
                                **{c: float(v) for c, v in zip(cols, s_["p"])}})
            for cl in out["clusters"]:
                b = cl["best"]
                sl_clusters.append({
                    "functional": f_pri, "omega": om_pri, "C": cell["C"],
                    "cost_form": cell["cost_form"], "ell": cell["ell"],
                    "q0_mode": cell["q0_mode"], "cluster_id": cl["cluster_id"],
                    "n_members": len(cl["members"]), "members": str(cl["members"]),
                    "best_L": b["L"], "N_B": b["N"] / 1e9, "Q": b["Q"],
                    "Q_cand": float(quality_of_mixture(b["p"][None, :], qmap, cols)[0]),
                    "is_best_cluster": False,
                })
    sl_df = pd.DataFrame(sl_rows)
    cl_df = pd.DataFrame(sl_clusters)
    if not cl_df.empty:
        bi = cl_df.groupby(["C", "cost_form", "ell", "q0_mode"])["best_L"].idxmin()
        cl_df.loc[bi, "is_best_cluster"] = True
    save_table(sl_df, cfg, "slsqp_starts.csv")
    save_table(cl_df, cfg, "slsqp_clusters.csv")
    ctx["slsqp"] = sl_df; ctx["slsqp_clusters"] = cl_df
    log(f"[7] 联合 SLSQP：{len(rep_cells)} 个情景 x "
        f"{cfg['search']['slsqp']['n_starts']} 起点；"
        f"等价解类 {len(cl_df)} 个"
        f"{'（>情景数 ⇒ 存在多个局部解，已在表中标注）' if len(cl_df) > len(rep_cells) else ''}")

    # 枚举 vs SLSQP 交叉核验
    xchk = []
    for _, c in cl_df[cl_df["is_best_cluster"]].iterrows():
        m = alloc_df[(alloc_df["functional"] == c["functional"]) &
                     (alloc_df["omega"] == c["omega"]) & (alloc_df["C"] == c["C"]) &
                     (alloc_df["cost_form"] == c["cost_form"]) &
                     (alloc_df["ell"] == c["ell"]) & (alloc_df["q0_mode"] == c["q0_mode"])]
        if m.empty:
            continue
        m = m.iloc[0]
        xchk.append({
            "C": c["C"], "cost_form": c["cost_form"], "ell": c["ell"],
            "q0_mode": c["q0_mode"],
            "enum_candidate": m["candidate"], "enum_L": m["L_pred"],
            "enum_N_B": m["N_B"], "enum_Q": m["Q"],
            "slsqp_L": c["best_L"], "slsqp_N_B": c["N_B"], "slsqp_Q": c["Q"],
            "dL_slsqp_minus_enum": c["best_L"] - m["L_pred"],
            "slsqp_Q_cand": c["Q_cand"], "enum_Q_cand": m["Q_cand"] if "Q_cand" in m else np.nan,
            "verdict": ("SLSQP 未优于枚举（枚举已取到该族最优）"
                        if c["best_L"] >= m["L_pred"] - 1e-9
                        else "SLSQP 更优：枚举候选集未覆盖连续最优，须扩充候选"),
        })
    xchk_df = pd.DataFrame(xchk)
    save_table(xchk_df, cfg, "enum_vs_slsqp.csv")
    ctx["xchk"] = xchk_df
    log(f"[7] 交叉核验：{len(xchk_df)} 行；"
        f"SLSQP 优于枚举的单元数 = "
        f"{int((xchk_df['dL_slsqp_minus_enum'] < -1e-9).sum()) if len(xchk_df) else 0}")

    # ================================================================ 8. 核验
    log("[8] 数值核验")
    v_deriv = V.derivative_check(model, cfg, [
        {"N": 1e9, "Q": 0.3, "h": 1.0, "Q0": q0_ref, "form": "power",
         "C": 1e22, "ell": 4096, "kappa": kappa},
        {"N": 1e10, "Q": 0.45, "h": 1.2, "Q0": q0_ref, "form": "exponential",
         "C": 1e24, "ell": 32768, "kappa": kappa},
        {"N": 1e11, "Q": 0.2, "h": 0.8, "Q0": q0_ref, "form": "logarithmic",
         "C": 1e19, "ell": 2048, "kappa": kappa},
    ] + ([{"N": 1e10, "Q": 0.35, "h": 1.0, "Q0": q0_ref, "form": "power",
           "C": 1e22, "ell": 4096, "kappa": 0.0}] if kappa != 0.0 else []))
    save_table(v_deriv, cfg, "verify_derivatives.csv")
    log(f"    解析梯度 vs 有限差分：{int(v_deriv['pass'].sum())}/{len(v_deriv)} 通过"
        f"（最大相对偏差 {v_deriv[['dL_dN_rel_err','dL_dQ_rel_err']].to_numpy().max():.2e}）")

    v_cons = V.constraint_check(model, cfg, alloc_df)
    save_table(v_cons, cfg, "verify_constraints.csv")
    log(f"    预算残差最大 {v_cons['budget_rel_residual'].max():.2e}；"
        f"份额加和通过 {int(v_cons['pass_share_sum'].sum())}/{len(v_cons)}；"
        f"注意力份额公式通过 {int(v_cons['pass_attn_formula'].sum())}/{len(v_cons)}")

    v_simp = pd.DataFrame([V.simplex_check(finals[cols].to_numpy(float)[i], lo, hi,
                                           float(cfg["verify"]["simplex_tol"]),
                                           in_trust_box=bool(finals["in_trust_box"].iloc[i]))
                           for i in range(len(finals))])
    v_simp.insert(0, "candidate", finals["candidate"].values)
    v_simp.insert(1, "trust_status", finals["trust_status"].values)
    save_table(v_simp, cfg, "verify_simplex.csv")
    log(f"    配比可行性：单纯形 {int(v_simp['pass_simplex'].sum())}/{len(v_simp)}；"
        f"可信区域盒 {int(v_simp['pass_box'].sum())}/{len(v_simp)}"
        f"（盒外 {int((~v_simp['in_trust_box_declared']).sum())} 个为有意保留的反事实对照点）；"
        f"综合通过 {int(v_simp['pass'].sum())}/{len(v_simp)}")

    v_stab = V.grid_stability(model, cfg,
                              lambda **kw: search_fixed_p(model, cfg=cfg, **kw),
                              h=1.0, Q0=q0_ref, form="power",
                              C=float(cfg["budget"]["C_scenarios"][1]), ell=4096.0,
                              kappa=kappa)
    save_table(v_stab, cfg, "verify_grid_stability.csv")
    log(f"    网格粗化稳定性：{'全部稳定' if v_stab['stable'].all() else '存在不稳定，见表'}")

    v_up = V.upstream_consistency(cfg, model, kappa)
    save_table(v_up, cfg, "verify_upstream_consistency.csv")
    log(f"    与上游 M1(rho=0) 预测器最大相对偏差 {v_up['max_rel_diff'].iloc[0]:.2e} —— "
        f"{v_up['verdict'].iloc[0]}")

    # （p 通道不变性已在第 4 步用完整逐候选评估表核验，见 verify_p_channel.csv）
    log(f"    p 通道不变性（精确解）：argmin L 落在 argmax h 上 "
        f"{int(v_inv['argmin_L_is_argmax_h'].sum())}/{len(v_inv)} 个情景单元")

    v_in = V.input_audit(cfg)
    save_table(v_in, cfg, "verify_input_audit.csv")
    ctx["v_in"] = v_in
    log(f"    输入审计：{int(v_in['exists'].sum())}/{len(v_in)} 个文件可读；"
        f"涉及最终检验集的行数 = {int(v_in['is_test_set'].sum())}（应为 0）")

    v_flow = pd.DataFrame(ctx["flow"])
    save_table(v_flow, cfg, "sample_flow.csv")

    # C7 依据
    if c7 is not None:
        save_table(c7_grounding(cfg, c7), cfg, "c7_grounding.csv")
        save_table(c7_discrete_cases(cfg, c7), cfg, "c7_discrete_cases.csv")

    # 情景表 + 成本份额分段
    save_table(pd.DataFrame(scenario_grid(cfg)), cfg, "optimization_scenarios.csv")
    segs = []
    for key, g in path_df.groupby(["functional", "omega", "cost_form", "ell", "q0_mode"]):
        d = share_segments(g, cfg)
        d["functional"] = key[0]; d["omega"] = key[1]
        segs.append(d)
    if segs:
        save_table(pd.concat(segs, ignore_index=True), cfg, "share_segments.csv")

    checks = build_checks(cfg, v_cons, v_simp, v_deriv, v_up, v_stab, v_inv, v_in)
    save_table(checks, cfg, "cost_constraint_checks.csv")
    ctx["checks"] = checks
    log(f"[8] 检查表：{int(checks['pass'].sum())}/{len(checks)} 项通过"
        f"{'；未通过：' + ', '.join(checks.loc[~checks['pass'], 'check']) if (~checks['pass']).any() else ''}")

    # ================================================================ 9. 元数据与报告
    with Timer() as t7:
        meta = {
            "baseline_id": cfg["baseline_id"], "question": cfg["question"],
            "generated_utc": utc_now(), "quick_mode": bool(args.quick),
            "config_path": cfg["_config_path"], "config_hash": config_hash(cfg),
            # 源码哈希与配置哈希并列记录：配置哈希锁旋钮，源码哈希锁逻辑。
            # 只有配置哈希时，"改了代码没改配置"这一情形无法被读者察觉。
            "source_hash": source_hash(),
            "seed": cfg["seed"], "env": env_info(),
            "git_rev_q3c_workspace": git_rev(cfg["_root"]),
            "git_rev_upstream_repo": git_rev(
                os.path.abspath(os.path.join(cfg["_root"], "..", "math_model_F"))),
            "roots": {"project_root": cfg["_root"],
                      "q1_root": which_root(cfg, os.path.join(cfg["paths"]["q1_artifacts"],
                                                              cfg["upstream"]["q1_predictor_file"])),
                      "q2_root": which_root(cfg, os.path.join(cfg["paths"]["q2_artifacts"],
                                                              cfg["upstream"]["q2_predictor_file"])),
                      "c7_root": which_root(cfg, os.path.join(cfg["paths"]["c7_dir"],
                                                              cfg["context"]["c7_file"]))},
            "upstream_files_sha256": {
                r["file"]: r["sha256"] for _, r in v_in.iterrows() if r["exists"]},
            "upstream_versions": {
                "q1_predictor": cfg["upstream"]["q1_predictor_file"],
                "q2_predictor": cfg["upstream"]["q2_predictor_file"],
                "q2_model": cfg["upstream"]["q2_model"],
                "q2_param_stage": cfg["upstream"]["q2_param_stage"],
                "kappa_primary": {"name": k_pri, "value": kappa},
                "kappa_scenarios": cfg["model"]["kappa_scenarios"],
                "quality_channel_status": "scenario_only（Q2-C 判定 kappa/rho 未辨识）",
            },
            "sample_flow": ctx["flow"],
            "counts": {"candidates": int(len(cand)), "finalists": int(len(finals)),
                       "full_cells": int(len(full_cells)), "allocations": int(len(alloc_df)),
                       "budget_path_rows": int(len(path_df)),
                       "transitions": int(len(trans))},
            "timings_s": {"upstream": round(t1.elapsed, 2), "candidates": round(t2.elapsed, 2),
                          "screening": round(t3.elapsed, 2), "search": round(t4.elapsed, 2),
                          "budget_paths": round(t5.elapsed, 2), "slsqp": round(t6.elapsed, 2)},
            "caveats": [
                "kappa、rho 被 Q2-C 判定为未辨识，本问以情景网格处理；主情景取 stage2_M1_staged 的 kappa。",
                "omega 与尺度 s 不可分辨，s 直接取 Q2-C 冻结的 scale_s 以保证跨 baseline 可比。",
                "Q1-C 未持久化随机森林对象，故不能调用 f(p)=Σ v_k L̂_k(p)；改用两个可由 p 直接算出的冻结泛函。",
                "LP 顶点使质量上界全部压在单一已映射域上（退化解），可信区域单独不足以避免，须在结论边界中声明。",
                "本流程只读训练/接口文件，未读取任何最终检验集——见 verify_input_audit.csv。",
            ],
            "artifacts": sorted(os.listdir(adir)),
        }
        save_json(meta, cfg, "run_metadata.json")
    log(f"[9] 元数据已落盘；总耗时 {t_all.stop():.1f}s")

    # 报告
    try:
        rep = build_report(cfg, ctx)
        write_text(rep, os.path.join(adir, "q3_report.md"))
        write_text(rep, os.path.join(cfg["_root"], cfg["outputs"]["report"]))
        log(f"[9] 报告已写入 {os.path.join(cfg['_root'], cfg['outputs']['report'])}")
    except Exception as e:                       # 报告失败不掩盖数值结论
        log(f"[ERR] 报告生成失败：{type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())

    log("=" * 78)
    log("完成。产物目录：" + adir)
    log("=" * 78)
    log.close()
    return 0


def run_log_name(quick: bool) -> str:
    return "run_log_quick.txt" if quick else "run_log.txt"


# ------------------------------------------------- 成本/约束检查表（交付物之一）
_SHARE_COLS = ("share_train", "share_attn", "share_quality", "share_unused")


def _share_sum_residual(v_cons) -> float:
    """份额加和残差。若四个份额列都在，则对两种等价算法取**较大**者上报。

    两种算法都合法：核验时可直接对 `share_*` 求和，也可用 `share_sum` 列
    （= used/C 与 (C-used)/C 相加）。浮点下二者相差 1–2 ULP，末位取决于求和
    顺序。**取较大者**是为了避免"挑一个更好看的数"，而不是因为二者有实质分歧。
    该列在 `verify_constraints.csv` 中通常只有 `share_sum`，故缺列时退回单算法。
    """
    resid = float(np.abs(v_cons["share_sum"].to_numpy(float) - 1.0).max())
    if all(c in v_cons.columns for c in _SHARE_COLS):
        recomputed = np.abs(v_cons[list(_SHARE_COLS)].to_numpy(float).sum(axis=1) - 1.0)
        resid = max(resid, float(recomputed.max()))
    return resid


def build_checks(cfg: dict, v_cons, v_simp, v_deriv, v_up, v_stab, v_inv,
                 v_in) -> pd.DataFrame:
    """成本与约束检查表。**独立成函数**，使 `--report-only` 能用同一份代码从
    verify_*.csv 重建它——否则报告里这张表会停留在旧口径，与产物脱钩。

    每行记 pass / max_violation / tolerance：只写 pass 而不写量级，
    读者无法判断"通过"是余量很大还是勉强压线。
    """
    _ql = v_inv[v_inv["functional"] == "quality_linear"]
    return pd.DataFrame([
        {"check": "budget_tight_and_consistent", "pass": bool(v_cons["pass_budget"].all()),
         "max_violation": float(v_cons["budget_rel_residual"].max()),
         "tolerance": float(cfg["verify"]["budget_rel_tol"]),
         "note": "D = C/u，回算 D*{(6+eta*l)N + c(Q)} 与 C 的相对残差"},
        # 份额加和残差有两个同样合法的算法：核验时对份额列求和，或直接对成本分量
        # 求和再除以 C。两者在浮点下相差 1–2 ULP（把 used/C 与 (C-used)/C 相加，
        # 末位取决于求和顺序）。取二者中**较大**者上报，避免挑一个更好看的数；
        # 关键是要写明这是机器精度量级，不是建模误差。
        {"check": "cost_shares_sum_to_one", "pass": bool(v_cons["pass_share_sum"].all()),
         "max_violation": float(_share_sum_residual(v_cons)),
         "tolerance": 1e-9,
         "note": ("训练+注意力+质量+未使用 = 1（分母为 C）。残差处于机器精度"
                  "（约 1e-16，即 1–2 ULP），来源是 used/C 与 (C-used)/C 相加的"
                  "末位舍入，不是建模或求解误差")},
        {"check": "attn_share_equals_eta_l_over_6",
         "pass": bool(v_cons["pass_attn_formula"].all()),
         "max_violation": float(v_cons["attn_share_formula_check"].max()),
         "tolerance": 1e-12, "note": "C_attn/C_train = eta*l/6"},
        {"check": "Q_within_bounds", "pass": bool(v_cons["Q_ge_Q0"].all() and
                                                  v_cons["Q_le_upper"].all()),
         "max_violation": 0.0, "tolerance": 1e-9, "note": "Q0 <= Q <= q_hi"},
        {"check": "simplex_feasible", "pass": bool(v_simp["pass"].all()),
         "max_violation": float(v_simp[["sum_minus_1", "box_violation_max"]]
                                .abs().to_numpy().max()),
         "tolerance": float(cfg["verify"]["simplex_tol"]),
         "note": ("sum(p)=1 且 p>=0 是硬约束，全部通过；可信区域盒只对 "
                  "in_trust_box=True 的候选要求，盒外对照点按设计不计违规")},
        {"check": "analytic_gradient_matches_fd", "pass": bool(v_deriv["pass"].all()),
         "max_violation": float(v_deriv[["dL_dN_rel_err", "dL_dQ_rel_err"]]
                                .to_numpy().max()),
         "tolerance": float(cfg["verify"]["fd_rel_tol"]),
         "note": "解析 dL/dN、dL/dQ 与中心差分一致"},
        {"check": "upstream_predictor_agrees", "pass": bool(v_up["max_rel_diff"].iloc[0] < 1e-12),
         "max_violation": float(v_up["max_rel_diff"].iloc[0]), "tolerance": 1e-12,
         "note": "与 predict_q2c_loss.py(M1, rho=0) 逐点一致"},
        {"check": "grid_coarsening_stable", "pass": bool(v_stab["stable"].all()),
         "max_violation": float(v_stab["L_rel"].max()),
         "tolerance": float(cfg["verify"]["grid_l_rel_tol"]),
         "note": ("粗网格下结论不变：相对损失变化（dL 除以 L 取绝对值）在容差内，"
                  "且 Q* 状态（下界/内部/上界）相同；N*/Q* 位置只作诊断量")},
        {"check": "p_channel_invariant_argminL_eq_argmaxh",
         "pass": bool(v_inv["argmin_L_is_argmax_h"].all()),
         "max_violation": float((~v_inv["argmin_L_is_argmax_h"]).sum()), "tolerance": 0.0,
         "note": "argmin_p L 恒等于 argmax_p h（h 可分离的必然后果），故 p* 与预算无关"},
        {"check": "argmax_h_eq_argmaxQ_for_quality_linear",
         "pass": bool(_ql.loc[_ql["h_identifiable"], "argmax_h_is_argmax_Q"].all()),
         "max_violation": float((~_ql.loc[_ql["h_identifiable"],
                                          "argmax_h_is_argmax_Q"]).sum()),
         "tolerance": 0.0,
         "note": ("quality_linear 的 Δ 是 Q 的仿射减函数，故 argmax h 恒等于 argmax Q；"
                  "mixture_l1 的 h 在 p0 取最大，二者本就不同（非缺陷）。"
                  "仅对 h 可辨识（ω>0）的单元判定：ω=0 时 h 恒为 1，"
                  "任何候选都是 argmax，该单元下 p 不可辨识")},
        {"check": "no_final_test_set_read", "pass": bool(int(v_in["is_test_set"].sum()) == 0),
         "max_violation": float(int(v_in["is_test_set"].sum())), "tolerance": 0.0,
         "note": "输入审计中不含任何最终检验/估算文件"},
    ])


# ------------------------------------------------------ 仅重渲染报告（不重算）
def rebuild_ctx(cfg: dict) -> dict:
    """从已落盘产物重建 build_report 所需的 ctx，用于 `--report-only`。

    为什么要有这个模式：报告是**产物**的视图，改措辞不该重跑 12 分钟的数值流程。
    但必须保证重建出的 ctx 与实跑时一致，否则报告会与产物脱钩——所以这里只读
    产物目录里的 CSV，不重算任何数值：凡是报告里出现的数字，都来自跑过一次的产物。
    唯一需要重新解析的是上游元数据（列序、质量坐标、冻结尺度），它们是**只读输入**，
    不参与数值计算，重读不会引入漂移。
    """
    adir = artifacts_dir(cfg)

    def rd(name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(adir, name), encoding="utf-8-sig")

    p0, cols, pred = q1_reference(cfg)
    tab = q1_quality_table(cfg)
    qmap = qmap_from_table(tab)
    lo, hi = trust_region(cfg, pred, cols)
    functionals = list(cfg["mixtures"]["functionals"])
    k_pri = cfg["model"]["kappa_primary"]
    q0_ref = float(quality_of_mixture(p0[None, :], qmap, cols)[0])

    cand = rd("candidates.csv")
    alloc = rd("optimal_allocations.csv")
    # LP 解析参照不单独落盘：它等价于 candidates.csv 里那一行（并已核对 Q 上界），
    # 故从产物取回，而不是重跑一次 LP——保证报告数字与产物逐位一致。
    lpv = cand[cand["candidate"] == "lp_vertex_box"]
    lp = ({"q_max": float(lpv["Q"].iloc[0]),
           "p": [float(lpv[c].iloc[0]) for c in cols]} if len(lpv) else {})
    v_inv = rd("verify_p_channel.csv")
    # 检查表由 verify_*.csv 重建（同 build_checks），否则改口径后这张表会停留在旧版。
    checks = build_checks(cfg, rd("verify_constraints.csv"), rd("verify_simplex.csv"),
                          rd("verify_derivatives.csv"),
                          rd("verify_upstream_consistency.csv"),
                          rd("verify_grid_stability.csv"), v_inv,
                          rd("verify_input_audit.csv"))
    save_table(checks, cfg, "cost_constraint_checks.csv")
    # source_hash 取自 run_metadata.json（**产出这批产物的代码**），而非当前磁盘上的源码：
    # `--report-only` 恰恰常用于"改了措辞但数值未动"的场合，此时重新计算会把**现在**的
    # 源码哈希写进报告，让读者误以为产物出自这版代码。配置同理，直接沿用元数据里的值。
    _meta = {}
    try:
        with open(os.path.join(adir, "run_metadata.json"), encoding="utf-8") as fh:
            _meta = json.load(fh)
    except Exception:
        pass
    return {
        "cfg": cfg, "quick": False,
        "config_hash": _meta.get("config_hash", config_hash(cfg)),
        "source_hash": _meta.get("source_hash", source_hash()),
        "flow": rd("sample_flow.csv").to_dict("records"),
        "p0": p0, "cols": cols, "pred": pred, "tab": tab, "qmap": qmap,
        "lo": lo, "hi": hi, "functionals": functionals,
        "s_by": {f: frozen_scale(cfg, f) for f in functionals},
        "omegas": [float(x) for x in cfg["quality"]["omega_scenarios"]],
        "omega_primary": float(cfg["quality"]["omega_primary"]),
        "kappa": float(cfg["model"]["kappa_scenarios"][k_pri]), "kappa_name": k_pri,
        "q0_ref": q0_ref, "q_hi": float(cfg["quality"]["q_extended_max"]),
        "cand": cand, "finals": rd("finalists.csv"), "alloc": alloc,
        "alloc_all": None, "paths": rd("budget_paths.csv"),
        "trans": rd("transition_candidates.csv"), "checks": checks,
        "v_inv": v_inv, "v_in": rd("verify_input_audit.csv"),
        "xchk": rd("enum_vs_slsqp.csv"), "slsqp_clusters": rd("slsqp_clusters.csv"),
        "lp": lp,
        "p_winners": alloc.groupby(["functional", "omega"])["candidate"]
                           .agg(lambda s: sorted(set(s))),
    }


def report_only(cfg: dict) -> int:
    adir = artifacts_dir(cfg)
    log = TeeLog(os.path.join(adir, "run_log_report.txt"))
    log("从产物重建 ctx 并重渲染报告（不重算数值）")
    ctx = rebuild_ctx(cfg)
    rep = build_report(cfg, ctx)
    write_text(rep, os.path.join(adir, "q3_report.md"))
    write_text(rep, os.path.join(cfg["_root"], cfg["outputs"]["report"]))
    log("报告已写入 " + os.path.join(cfg["_root"], cfg["outputs"]["report"]))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
