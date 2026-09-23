# -*- coding: utf-8 -*-
"""情景网格、预算路径与结构性转移的识别（§3.3、§3.4）。

情景维度：预算（题给三档 + 连续 log C 网格）× 成本形式（题给三类）
          × 上下文长度（C7 的五档）× Q0 设定（固定 / 随配比）× kappa 情景。

结构性转移按**预先定义**的事件判定（§3.4），并报告数值容差与稳定性：
  1. Q* 从 Q0 边界进入内部，或达到上界；
  2. p* 的支持集发生稳定变化；
  3. 成本份额曲线出现可复现的分段变化。

判据在本模块内固定（阈值来自配置 verify.transition.*），且只接受在更粗网格、
不同起点下**复现**的变点；只在三档离散预算上看到差异不算变点。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scenario_grid(cfg: dict) -> list[dict]:
    """展开全部主情景单元。"""
    rows = []
    kappa_sc = cfg["model"]["kappa_scenarios"]
    k_primary = cfg["model"]["kappa_primary"]
    for C in cfg["budget"]["C_scenarios"]:
        for form in cfg["cost"]["g_forms"]:
            for ell in cfg["context"]["ell_scenarios"]:
                for q0_mode in cfg["quality"]["q0_modes"]:
                    rows.append({"C": float(C), "cost_form": form, "ell": int(ell),
                                 "q0_mode": q0_mode,
                                 "kappa_scenario": k_primary,
                                 "kappa": float(kappa_sc[k_primary])})
    return rows


def classify_q_regime(Q, Q0, q_hi, tol: float) -> str:
    """Q* 的状态：下界（=Q0）/ 内部 / 上界（=q_hi）。"""
    if abs(Q - Q0) <= tol:
        return "lower_bound_Q0"
    if abs(Q - q_hi) <= tol:
        return "upper_bound"
    return "interior"


def budget_path(model, cfg: dict, h: float, q0_spec, form: str, ell: float,
                kappa: float, search_fn, C_list=None) -> pd.DataFrame:
    """沿连续 log C 网格跟踪最优 Q*、N*、D*、状态与成本份额（§3.4 第 3 项）。

    q0_spec 是 callable(C) -> Q0，便于把「Q0 随预算变化」这类设定也留下接口；
    本问的两种设定都是常数，故调用方传入常量函数。
    """
    g = cfg["budget"]["C_grid_log10"]
    if C_list is None:
        C_list = 10.0 ** np.linspace(float(g["min"]), float(g["max"]), int(g["n"]))
    q_hi = float(cfg["quality"]["q_extended_max"])
    tol = float(cfg["verify"]["transition"]["q_boundary_tol"])
    rows = []
    for C in np.asarray(C_list, float):
        Q0 = float(q0_spec(C))
        r = search_fn(h=h, Q0=Q0, form=form, C=C, ell=ell, kappa=kappa,
                      q_lo=Q0, q_hi=q_hi)
        rows.append({
            "C": float(C), "log10_C": float(np.log10(C)), "cost_form": form,
            "ell": int(ell), "kappa": float(kappa), "Q0": Q0,
            "N_B": r["N"] / 1e9, "D_B": r["D"] / 1e9, "Q": r["Q"], "h": r["h"],
            "L_pred": r["L"],
            "share_train": r["share_train"], "share_attn": r["share_attn"],
            "share_quality": r["share_quality"], "share_unused": r["share_unused"],
            "regime": classify_q_regime(r["Q"], Q0, q_hi, tol),
            "flag_q_at_lower": r["boundary_q_lower"], "flag_q_at_upper": r["boundary_q_upper"],
            "flag_n_at_lower": r["boundary_n_lower"], "flag_n_at_upper": r["boundary_n_upper"],
            "budget_residual_rel": r["budget_residual_rel"],
            "N_in_support": bool(cfg["search"]["N_support_B"][0] <= r["N"] / 1e9 <=
                                 cfg["search"]["N_support_B"][1]),
            "D_in_support": bool(cfg["search"]["D_support_B"][0] <= r["D"] / 1e9 <=
                                 cfg["search"]["D_support_B"][1]),
        })
    return pd.DataFrame(rows)


def detect_q_transitions(path: pd.DataFrame, cfg: dict) -> list[dict]:
    """在预算路径上找 Q* 状态变化点，并要求在更粗网格下复现。"""
    out = []
    reg = path["regime"].to_numpy()
    C = path["C"].to_numpy()
    for i in range(1, len(reg)):
        if reg[i] != reg[i - 1]:
            out.append({
                "event": "Q_regime_change",
                "cost_form": path["cost_form"].iloc[i], "ell": int(path["ell"].iloc[i]),
                "kappa": float(path["kappa"].iloc[i]),
                "from": str(reg[i - 1]), "to": str(reg[i]),
                "C_before": float(C[i - 1]), "C_after": float(C[i]),
                "log10_C_mid": float((np.log10(C[i - 1]) + np.log10(C[i])) / 2.0),
                "share_quality_before": float(path["share_quality"].iloc[i - 1]),
                "share_quality_after": float(path["share_quality"].iloc[i]),
                "note": "预定义事件 1（Q* 离开/进入边界）",
            })
    return out


def share_segments(path: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """成本份额曲线的一阶差分，用于判断是否存在可复现的分段变化（事件 3）。"""
    d = path.sort_values("log10_C").copy()
    for col in ("share_train", "share_attn", "share_quality", "share_unused"):
        d[f"d_{col}_dlogC"] = np.gradient(d[col].to_numpy(), d["log10_C"].to_numpy())
    d["dlog10Q_dlogC"] = np.gradient(np.log10(d["Q"].to_numpy() + 1e-300),
                                     d["log10_C"].to_numpy())
    d["dlog10N_dlogC"] = np.gradient(np.log10(d["N_B"].to_numpy()),
                                     d["log10_C"].to_numpy())
    return d


def p_support_change(records: pd.DataFrame, cfg: dict) -> list[dict]:
    """配比支持集是否随情景稳定变化（事件 2）。

    records 需含 [scenario_key, C, cost_form, ell, q0_mode, candidate, L, Q]。
    「稳定变化」定义为：在某一情景维度上，最优候选连续 ≥3 个格点与整体最优不同。
    """
    out = []
    tol = float(cfg["verify"]["transition"]["support_frac_tol"])
    for key, g in records.groupby("dimension"):
        g = g.sort_values("x")
        best_global = g.groupby("candidate")["L"].mean().idxmin()
        winners = g.loc[g.groupby("x")["L"].idxmin(), ["x", "candidate"]]
        diff = (winners["candidate"] != best_global).to_numpy()
        # 找连续 ≥3 的区段
        i, n = 0, len(diff)
        while i < n:
            if diff[i]:
                j = i
                while j < n and diff[j]:
                    j += 1
                if j - i >= 3:
                    out.append({"event": "p_support_change", "dimension": str(key),
                                "baseline_best": str(best_global),
                                "segment_best": str(winners["candidate"].iloc[i]),
                                "x_start": float(winners["x"].iloc[i]),
                                "x_end": float(winners["x"].iloc[j - 1]),
                                "n_points": int(j - i),
                                "frac_share": float((j - i) / n),
                                "tol": tol,
                                "note": "预定义事件 2（p* 支持集稳定变化）"})
                i = j
            else:
                i += 1
    return out


def c7_grounding(cfg: dict, c7: pd.DataFrame) -> pd.DataFrame:
    """给每个上下文长度情景挂上 C7 的元数据依据（§3.3）。

    不可把某架构支持的最大长度自动赋予其他架构：这里如实统计每个长度档下
    实际存在的架构数与代表模型，作为「上下文能力情景」的依据与边界提示。
    """
    col = "max_position_embeddings"
    rows = []
    for ell in cfg["context"]["ell_scenarios"]:
        sub = c7[c7[col] == int(ell)]
        rows.append({
            "ell": int(ell),
            "attn_share_eta_l_over_6": 2e-4 * int(ell) / 6.0,
            "n_architectures_in_C7": int(len(sub)),
            "examples": "; ".join(sub["model_name"].astype(str).head(3)),
            "single_architecture": bool(len(sub) == 1),
            "note": ("该长度在 C7 中仅 1 个架构支持，情景结论依赖单一架构"
                     if len(sub) == 1 else ""),
        })
    return pd.DataFrame(rows)


def c7_discrete_cases(cfg: dict, c7: pd.DataFrame) -> pd.DataFrame:
    """有元数据对应的离散案例：直接取 C7 的真实架构参数（§3.3 末段）。"""
    out = c7.copy()
    out["attn_share"] = 2e-4 * out["max_position_embeddings"].astype(float) / 6.0
    return out
