# -*- coding: utf-8 -*-
"""Q2-C 质量坐标校准与配比迁移情景（§2.4 第 4/5/7 条，§2.5 领域替代）。

1. 校准：第一问原始 Q → 模型 Q 的单调映射候选。最简方案（同尺度）必须标为
   未验证假设，并比较替代映射；报告 A 侧与 B 侧 Q 的公共支撑区间，以及无 Q
   列的基准约定（§2.4 第 5 条）。
2. 迁移：h(p) = exp(-ω·Δ(p))，h(p0)=1、h(p)>0。Δ 是相对参考配比 p0 的
   "损失型"偏离：优于 p0 的配比 Δ<0 → h>1，劣于则 Δ>0 → h<1。ω 缺少共同
   实验，只给含零效应在内的小组预设情景（§2.4 第 7 条）。
3. 领域替代：p_i 增 δ、p_j 减 δ 的单纯形可行扰动（§2.5 末段）。

输入仅来自 Q1→Q2 接口（quality_domain_mapping.csv、mixture_predictor.json）
与附件 B，不重复读取附件 A 原始文件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import load_q1_domain_mapping, load_q1_predictor
from scaling import predict


# --------------------------------------------------------------- 第一问质量坐标
def q1_quality_table(cfg: dict) -> pd.DataFrame:
    """第一问导出的 训练域 → 冻结质量分（Q1→Q2 接口）。"""
    m = load_q1_domain_mapping(cfg)
    need = {"mixture_col", "q_entropy", "q_equal", "q_resolved", "mapped"}
    miss = need - set(m.columns)
    if miss:
        raise ValueError(f"quality_domain_mapping.csv 缺列 {miss}")
    m = m.copy()
    m["mapped"] = m["mapped"].astype(str).str.lower().isin(("true", "1", "1.0"))
    return m


def qmap_from_table(tab: pd.DataFrame, which: str = "q_entropy") -> dict:
    """训练域列名 → 质量分；未映射域给 NaN（exclude 口径，与 Q1-C 一致）。"""
    out: dict[str, float] = {}
    for _, r in tab.iterrows():
        col = r["mixture_col"]
        if not isinstance(col, str) or not col:
            continue
        v = r[which]
        out[col] = float(v) if (bool(r["mapped"]) and pd.notna(v)) else float("nan")
    return out


def q1_reference(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """第一问配比预测器的参考配比 p0 与输入列序（17 维，冻结顺序）。"""
    pred = load_q1_predictor(cfg)
    cols = list(pred["input_columns"])
    p0 = np.asarray(pred["p0_reference_mixture_mean"], float)
    if len(cols) != len(p0):
        raise ValueError("p0 维度与输入列序不一致")
    return p0, cols


def quality_of_mixture(p: np.ndarray, qmap: dict, cols: list[str]) -> np.ndarray:
    """Q(p) = Σ_{已映射} p_i q_i / Σ_{已映射} p_i。未映射域剔除并重新归一化。"""
    p = np.atleast_2d(np.asarray(p, float))
    mask = np.array([not np.isnan(qmap.get(c, np.nan)) for c in cols], bool)
    if not mask.any():
        raise ValueError("没有任何域完成质量映射")
    q = np.array([qmap.get(c, np.nan) for c in cols], float)
    num = (p[:, mask] * q[mask][None, :]).sum(axis=1)
    den = p[:, mask].sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


# --------------------------------------------------------------------- 校准
def calibrate_series(q: np.ndarray, mapping: str, q_ref: float | None = None) -> np.ndarray:
    """把第一问原始 Q 单调映射到模型 Q 坐标（三种候选）。"""
    q = np.asarray(q, float)
    if mapping == "same_scale":
        return q.copy()
    if mapping == "minmax01":
        lo, hi = float(np.nanmin(q)), float(np.nanmax(q))
        return (q - lo) / (hi - lo) if hi > lo else np.zeros_like(q)
    if mapping == "reference_shift":
        ref = float(q_ref) if q_ref is not None else float(np.nanmax(q))
        return q / ref if ref > 0 else q.copy()
    raise ValueError(f"未知映射 {mapping}")


def calibration_report(cfg: dict, q1_domain_q: np.ndarray, b_quality: dict) -> dict:
    """校准假设报告：各映射下的坐标范围、A/B 侧公共支撑与未验证声明。"""
    q1_domain_q = np.asarray(q1_domain_q, float)
    rep = {"primary": cfg["calibration"]["primary_mapping"],
           "q1_domain_quality_source": "quality_domain_mapping.csv::q_entropy",
           "mappings": {}}
    for m in cfg["calibration"]["mappings"]:
        cq = calibrate_series(q1_domain_q, m)
        rep["mappings"][m] = {"q_min": float(np.nanmin(cq)), "q_max": float(np.nanmax(cq)),
                              "q_mean": float(np.nanmean(cq))}
    if b_quality:
        b_all = np.concatenate([np.asarray(v, float) for v in b_quality.values()])
        b_all = b_all[np.isfinite(b_all)]
    else:
        b_all = np.array([np.nan])
    bmin, bmax = float(np.nanmin(b_all)), float(np.nanmax(b_all))
    a_min, a_max = float(np.nanmin(q1_domain_q)), float(np.nanmax(q1_domain_q))
    lo, hi = max(a_min, bmin), min(a_max, bmax)
    rep["a_side_q_range"] = {"min": a_min, "max": a_max}
    rep["b_side_q_range"] = {"min": bmin, "max": bmax}
    rep["common_support"] = {"lo": float(lo), "hi": float(hi), "empty": bool(lo >= hi)}
    rep["assumption_note"] = (
        "同尺度（same_scale）为未验证假设：附件 A 的质量坐标与 B6-B8 的 Q_score "
        "来自不同提取流程，仅在某区间重叠；报告公共支撑区间，并以 minmax01 / "
        "reference_shift 作为替代单调映射做敏感性。"
    )
    rep["baseline_q_convention"] = (
        f"B1/B2/B4/B5/B10 无 Q 列，按约定取 Q=q_reference="
        f"{float(cfg['model']['q_reference']):.2f}。这是基准归一化约定，不代表这些"
        f"语料被实测为完美质量（§2.4 第 5 条）。"
    )
    return rep


# --------------------------------------------------------------------- 候选配比
def candidate_mixtures(cfg: dict, p0: np.ndarray, cols: list[str], seed: int) -> pd.DataFrame:
    """以 p0 为中心生成单纯形可行的候选配比情景（不依赖附件 A 原始文件）。"""
    p0 = np.asarray(p0, float)
    p0 = p0 / p0.sum()
    n = int(cfg["transfer"]["n_candidate_mixtures"])
    d = float(cfg["transfer"]["candidate_delta"])
    scheme = cfg["transfer"]["candidate_scheme"]
    rows = [{"scenario": "reference_p0", "i": -1, "j": -1, "delta": 0.0, **dict(zip(cols, p0))}]
    k = 0
    if scheme == "dirichlet":
        rng = np.random.default_rng(seed)
        for a in (200.0, 50.0, 20.0):
            for _ in range(max(n // 3, 1)):
                rows.append({"scenario": f"dirichlet_a{a:g}_{k}", "i": -1, "j": -1,
                             "delta": 0.0, **dict(zip(cols, rng.dirichlet(np.maximum(p0 * a, 1e-6))))})
                k += 1
    else:
        m = len(cols)
        short = [c.replace("train_the_pile_", "") for c in cols]
        stop = False
        for i in range(m):
            for j in range(m):
                if i == j or p0[j] < d:
                    continue
                p = p0.copy(); p[i] += d; p[j] -= d
                rows.append({"scenario": f"shift_{short[i]}_up__{short[j]}_down",
                             "i": i, "j": j, "delta": d, **dict(zip(cols, p))})
                k += 1
                if k >= n:
                    stop = True
                    break
            if stop:
                break
    return pd.DataFrame(rows)


def mixture_effect(cfg: dict, P: np.ndarray, p0: np.ndarray, qmap: dict,
                   cols: list[str], functional: str) -> tuple[np.ndarray, float]:
    """相对 p0 的损失型偏离 Δ(p)（Δ(p0)=0）与其固定尺度 s。"""
    P = np.atleast_2d(np.asarray(P, float))
    if functional == "quality_linear":
        Qp = quality_of_mixture(P, qmap, cols)
        Q0 = float(quality_of_mixture(p0[None, :], qmap, cols)[0])
        s = float(np.nanstd(Qp))
        s = s if s > 0 else 1.0
        return (Q0 - Qp) / s, s          # 质量低于 p0 → Δ>0（损失型）
    if functional == "mixture_l1":
        raw = np.abs(P - np.asarray(p0, float)[None, :]).sum(axis=1)
        nz = raw[raw > 0]
        s = float(nz.mean()) if nz.size else 1.0
        return raw / s, s
    raise ValueError(f"未知效应泛函 {functional}")


# --------------------------------------------------------------------- 迁移情景
def transfer_scenarios(cfg: dict, model_x: np.ndarray, cand: pd.DataFrame, p0: np.ndarray,
                       qmap: dict, cols: list[str], N: float, D: float, Q: float,
                       functional: str) -> pd.DataFrame:
    """ω 网格下的 h(p) 与预测 Loss 变化；ω=0 即零效应情景（嵌套回 Q2-B）。"""
    P = cand[cols].to_numpy(float)
    delta, s = mixture_effect(cfg, P, p0, qmap, cols, functional)
    rows = []
    for w in cfg["transfer"]["omega_grid"]:
        h = np.exp(-float(w) * delta)
        L1 = predict(model_x, np.full(len(P), N), np.full(len(P), D),
                     np.full(len(P), Q), h, np.zeros(len(P)))
        L0 = predict(model_x, np.full(len(P), N), np.full(len(P), D),
                     np.full(len(P), Q), 1.0, np.zeros(len(P)))
        for i in range(len(P)):
            rows.append({
                "functional": functional, "omega": float(w), "scale_s": float(s),
                "scenario": str(cand["scenario"].iloc[i]),
                "delta": float(delta[i]), "h": float(h[i]),
                "loss_h1": float(L0[i]), "loss_with_h": float(L1[i]),
                "dLoss": float(L1[i] - L0[i]),
                "N": float(N), "D": float(D), "Q": float(Q),
            })
    return pd.DataFrame(rows)


def substitution_table(cfg: dict, model_x: np.ndarray, p0: np.ndarray, qmap: dict,
                       cols: list[str], N: float, D: float, Q: float,
                       functional: str, omega: float) -> pd.DataFrame:
    """领域替代：p_i 增 δ、p_j 减 δ 的单纯形可行扰动下的 ΔQ 与 ΔLoss（§2.5）。"""
    d = float(cfg["transfer"]["substitution_delta"])
    Pn = np.vstack([p0])
    base_delta, s = mixture_effect(cfg, Pn, p0, qmap, cols, functional)
    h0 = float(np.exp(-omega * base_delta[0]))
    L0 = float(predict(model_x, np.array([N]), np.array([D]), np.array([Q]),
                       np.array([h0]), np.array([0.0]))[0])
    Q0 = float(quality_of_mixture(p0[None, :], qmap, cols)[0])
    rows = []
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j or p0[j] < d:
                continue
            p = p0.copy(); p[i] += d; p[j] -= d
            dd, _ = mixture_effect(cfg, np.vstack([p0, p]), p0, qmap, cols, functional)
            h = float(np.exp(-omega * dd[1]))
            Qp = float(quality_of_mixture(p[None, :], qmap, cols)[0])
            L = float(predict(model_x, np.array([N]), np.array([D]), np.array([Q]),
                              np.array([h]), np.array([0.0]))[0])
            rows.append({
                "domain_up": ci.replace("train_the_pile_", ""),
                "domain_down": cj.replace("train_the_pile_", ""),
                "delta": d, "Q_before": Q0, "Q_after": Qp, "dQ": Qp - Q0,
                "h1": h0, "h2": h, "dh_over_h": h / h0 - 1.0,
                "loss_h1_equiv": L0, "loss_with_h2": L,
                "dLoss": L - L0,
                "functional": functional, "omega": float(omega),
            })
    return pd.DataFrame(rows)
