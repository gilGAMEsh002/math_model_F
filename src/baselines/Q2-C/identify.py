# -*- coding: utf-8 -*-
"""Q2-C 可辨识性诊断（§2.4 第 9 条）：剖面似然、参数相关、多起点等价解。

任务卡的完成条件是"用留规模/轨迹检验和剖面误差判断双通道是否可辨识，
不只报告更低训练误差"。因此本模块不对"ρ 更小 SSE"作正面解读，而是给出：
    - ρ、κ 的剖面似然曲线与 95% 区间宽度；
    - ρ–κ、A–B 等关键参数对的相关；
    - 多起点是否落到多个等价解（同一 SSE 不同参数）；
    - 边界命中（κ=0 或 ρ=0 即嵌套，不构成新通道的证据）。

判定规则来自配置 identify.*，全部结论输出为结构化字段，供报告逐条引用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scaling import FULL, fit_jacobian, profile, rel_position

KEY_PAIRS = [("kappa", "rho"), ("A", "B"), ("alpha", "beta"),
             ("A", "alpha"), ("B", "beta"), ("rho", "alpha"), ("kappa", "beta")]


def correlation_table(res: dict, N, D, Q, L, dgrp=None,
                      weights=None) -> tuple[pd.DataFrame, np.ndarray]:
    """由最优点数值 Jacobian 得协方差与相关矩阵（渐近近似）。"""
    cov, sd, corr = fit_jacobian(res, N, D, Q, L, dgrp, weights)
    free = res["free"]
    rows = []
    for a, b in KEY_PAIRS:
        if a in free and b in free:
            ia, ib = free.index(a), free.index(b)
            rows.append({"param_a": a, "param_b": b,
                         "corr": float(corr[ia, ib]),
                         "sd_a": float(sd[ia]), "sd_b": float(sd[ib])})
    tab = pd.DataFrame(rows)
    tab["abs_corr"] = tab["corr"].abs()
    return tab, corr


def identify_channel(cfg: dict, res: dict, name: str, N, D, Q, L, bounds,
                     dgrp=None, weights=None, log=None) -> dict:
    """对单个通道参数（κ 或 ρ）做剖面似然与边界判定。"""
    ic = cfg["identify"]
    prof = profile(res, name, N, D, Q, L, bounds,
                   ci_level=float(ic["profile_ci_level"]),
                   dgrp=dgrp, weights=weights, log=log)
    val = float(res["x"][FULL.index(name)])
    rel = rel_position(name, val, bounds)
    at_boundary = rel < 0.02 or rel > 0.98
    status = str(prof.get("status", "ok"))
    finite_w = bool(np.isfinite(prof["ci95_width"]))
    wide = bool(finite_w and prof["ci95_width"] > float(ic["ci_width_flag"]))
    # 剖面在整段可行域内都低于阈值 → 数据对该参数没有任何约束
    unbounded = status != "ok"
    return {"param": name, "estimate": val,
            "ci95_lo": prof["ci95"][0], "ci95_hi": prof["ci95"][1],
            "ci95_width": prof["ci95_width"], "ci95_finite": finite_w,
            "weights_mismatch": bool(prof.get("weights_mismatch", False)),
            "ci_too_wide": wide, "profile_status": status, "unbounded": unbounded,
            "at_boundary": bool(at_boundary), "boundary_side": ("lower" if rel < 0.02 else
                                                               "upper" if rel > 0.98 else None),
            "profile_flatness": prof["flatness"], "profile": prof["rows"],
            "threshold_sse": prof["threshold"]}


def verdict(cfg: dict, res: dict, ident: dict, corr_tab: pd.DataFrame, log=None) -> dict:
    """综合判定双通道是否可辨识。"""
    ic = cfg["identify"]
    k = ident["kappa"]; r = ident["rho"]
    cr = corr_tab[(corr_tab.param_a == "kappa") & (corr_tab.param_b == "rho")]
    kr_corr = float(cr["corr"].iloc[0]) if len(cr) else float("nan")
    why = []
    for tag, ch, label in (("kappa", k, "κ"), ("rho", r, "ρ")):
        if ch.get("weights_mismatch"):
            why.append(f"{label} 的剖面加权口径与拟合不一致，该区间无效（诊断作废）")
        if not ch["ci95_finite"]:
            why.append(f"{label} 的剖面 95% 区间无法判定（status={ch['profile_status']}）")
        elif ch["unbounded"]:
            why.append(f"{label} 的剖面在可行域内未穿越阈值（{ch['profile_status']}）——"
                       f"数据对该参数无约束")
        if ch["at_boundary"]:
            why.append(f"{label} 命中{bk(ch['boundary_side'])}边界（嵌套证据，非新通道）")
        if ch["ci_too_wide"]:
            why.append(f"{label} 的 95% 区间过宽（宽度 {ch['ci95_width']:.3f} "
                       f"> {ic['ci_width_flag']}）")
    if np.isfinite(kr_corr) and abs(kr_corr) >= float(ic["corr_flag"]):
        why.append(f"ρ–κ 高度相关（corr={kr_corr:+.3f} ≥ {ic['corr_flag']}）")
    n_clusters = len(res["clusters"])
    if n_clusters > 1:
        why.append(f"多起点落在 {n_clusters} 个等价解（同一 SSE 不同参数）")
    identifiable = len(why) == 0
    out = {
        "dual_channel_identifiable": bool(identifiable),
        "kappa_estimate": k["estimate"], "rho_estimate": r["estimate"],
        "kappa_ci95": [k["ci95_lo"], k["ci95_hi"]], "rho_ci95": [r["ci95_lo"], r["ci95_hi"]],
        "kappa_ci95_width": k["ci95_width"], "rho_ci95_width": r["ci95_width"],
        "kappa_profile_status": k["profile_status"], "rho_profile_status": r["profile_status"],
        "kappa_rho_corr": kr_corr,
        "n_equivalent_solutions": n_clusters,
        "reasons_not_identified": why,
    }
    if log:
        log(f"  [identify] 双通道可辨识 = {identifiable}"
            + ("" if identifiable else "；原因: " + "；".join(why)))
    return out


def bk(side) -> str:
    return {"lower": "下", "upper": "上"}.get(side, "?")


def solution_table(res: dict) -> pd.DataFrame:
    """多起点解表：每个收敛解的参数、SSE 与是否落在主等价类。"""
    best = res["sols"][0]["x"]
    rows = []
    for i, s in enumerate(res["sols"]):
        row = {"solution": i, "sse": s["sse"], "rmse": float(np.sqrt(s["sse"] / res["n"]))}
        row.update({n: float(s["x"][FULL.index(n)]) for n in res["free"]})
        row["dist_to_best"] = float(np.sqrt(np.sum((s["x"][[FULL.index(n) for n in res["free"]]]
                                                    - best[[FULL.index(n) for n in res["free"]]]) ** 2)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sse").reset_index(drop=True)
