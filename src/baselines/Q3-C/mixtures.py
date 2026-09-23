# -*- coding: utf-8 -*-
"""候选配比的生成、可信区域与质量泛函（Q3-C 第 1、2、4 条）。

三条硬规则，逐条对应任务卡：
  1. 候选**只**从第一问训练信息与既定可信区域生成。本模块只读：
       * Q1-C 的 `mixture_predictor.json`（列序、参考配比 p0、训练支持区间）
       * Q1-C 的 `quality_domain_mapping.csv`（域→冻结质量分）
     不读任何最终检验集（A6–A11）或估算表（A12–A15），因此不存在「用最终检验
     Loss 偷选配方」。runner 会把实际读入的文件哈希登记进 run_metadata，可复核。
  2. Q0 是否依赖 p 分两种设定：`fixed_ref`（基线，Q0 不随 p 变）与 `mixture`
     （Q0 = Q(p)，随语料组成变化，单独作情景）。
  3. 可信区域 = 第一问训练支持的 [p05, p95] 盒 ∩ 单纯形，另加相对 p0 的 L1 半径；
     超出者可保留但必须标为外推，不静默丢弃。

质量泛函（与 Q2-C 冻结口径一致，不重新拟合第一问预测器）：
    Q(p) = Σ_{已映射} p_i q_i / Σ_{已映射} p_i          （exclude 口径）
    Δ(p) = [Q(p0) - Q(p)] / s   （质量低于 p0 → Δ>0，是「损失型」偏离）
    h(p) = exp(-ω Δ(p))，h(p0)=1，h>0

尺度 s 直接取 Q2-C `mixture_transfer_scenarios.json` 中冻结的 `scale_s`，
使 ω 的含义与 Q2-C 完全一致（否则 ω 与 s 不可分辨，跨 baseline 比较会失真）。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from common import load_q1_domain_mapping, load_q1_predictor, q2_path


# --------------------------------------------------------------- 第一问质量坐标
def q1_quality_table(cfg: dict) -> pd.DataFrame:
    """第一问导出的 训练域 → 冻结质量分。"""
    m = load_q1_domain_mapping(cfg)
    need = {"mixture_col", "q_entropy", "mapped"}
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


def q1_reference(cfg: dict) -> tuple[np.ndarray, list[str], dict]:
    """第一问配比预测器的参考配比 p0、输入列序（17 维，冻结顺序）与描述符。"""
    pred = load_q1_predictor(cfg)
    cols = list(pred["input_columns"])
    p0 = np.asarray(pred["p0_reference_mixture_mean"], float)
    if len(cols) != len(p0):
        raise ValueError("p0 维度与输入列序不一致")
    return p0, cols, pred


def trust_region(cfg: dict, pred: dict, cols: list[str], quantiles=("p05", "p95")):
    """可信区域盒：逐域取第一问训练支持的分位点。返回 (lo, hi)。"""
    sup = pred["training_support"]
    lo_q, hi_q = quantiles
    lo = np.array([float(sup[c][lo_q]) for c in cols], float)
    hi = np.array([float(sup[c][hi_q]) for c in cols], float)
    return lo, hi


# ------------------------------------------------------------------- 质量泛函
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


def q1_rf_predict(p: np.ndarray, pred: dict) -> np.ndarray:
    """第一问随机森林的**未持久化**问题说明。

    Q1-C 只导出了预测器的接口描述符（列序、v、p0、训练支持），没有持久化拟合好的
    森林对象，因此本流程**不能**调用 f(p)=Σ_k v_k L̂_k(p) 这一泛函。这一点必须在
    报告里如实声明（不假装用过），并改用 Q2-C 冻结的两个可由 p 直接算出的泛函。
    """
    raise NotImplementedError(
        "Q1-C 未持久化森林模型；Q3-C 不重新拟合第一问预测器。"
        "请使用 quality_linear / mixture_l1 两个冻结泛函。")


def frozen_scale(cfg: dict, functional: str) -> float:
    """取 Q2-C 冻结的尺度 s（使 ω 的含义与 Q2-C 一致）。"""
    with open(q2_path(cfg, "mixture_transfer_scenarios.json"), "r", encoding="utf-8") as fh:
        js = json.load(fh)
    bf = js.get("by_functional", {})
    if functional in bf and "scale_s" in bf[functional]:
        return float(bf[functional]["scale_s"])
    raise KeyError(f"Q2-C 冻结的 mixture_transfer_scenarios.json 中没有 {functional} 的 scale_s")


def loss_type_deviation(P: np.ndarray, p0: np.ndarray, qmap: dict, cols: list[str],
                        functional: str, s: float) -> np.ndarray:
    """Δ(p)：相对 p0 的损失型偏离（优于 p0 为负）。"""
    P = np.atleast_2d(np.asarray(P, float))
    if functional == "quality_linear":
        Qp = quality_of_mixture(P, qmap, cols)
        Q0 = float(quality_of_mixture(p0[None, :], qmap, cols)[0])
        return (Q0 - Qp) / s
    if functional == "mixture_l1":
        raw = np.abs(P - np.asarray(p0, float)[None, :]).sum(axis=1)
        return raw / s
    raise ValueError(f"未知效应泛函 {functional}")


def h_of_mixture(P: np.ndarray, p0: np.ndarray, qmap: dict, cols: list[str],
                 functional: str, omega: float, s: float) -> np.ndarray:
    """h(p) = exp(-ω Δ(p))；ω=0 ⇒ h≡1（即 Q2-B 零效应情景）。"""
    return np.exp(-float(omega) * loss_type_deviation(P, p0, qmap, cols, functional, s))


# ------------------------------------------------- 质量上界的解析参考（线性分式）
def lp_max_quality(p0: np.ndarray, qmap: dict, cols: list[str],
                   lo: np.ndarray, hi: np.ndarray):
    """可信区域盒内 max Q(p) 的**精确**解（线性分式规划，Charnes–Cooper 变换）。

    max Σ_m q_m p_m / Σ_m p_m  s.t. p∈simplex, lo<=p<=hi
    令 t = 1/Σ_m p_m > 0、y = t·p，则等价于

        max Σ_m q_m y_m
        s.t. Σ_m y_m = 1,  Σ_all y_i = t,  lo_i t <= y_i <= hi_i t,  y,t >= 0

    这是标准 LP。Q(p) 是 p 的线性分式函数且 h(p) 单调依赖于 Q(p)，故本 LP 给出
    p 通道最优值的独立解析参照——用于核验枚举与 SLSQP 的结果，**不是** Q3-A 的
    解析资源解（那张卡不在本任务范围内）。
    """
    m = np.array([not np.isnan(qmap.get(c, np.nan)) for c in cols], bool)
    n = len(cols)
    # 变量顺序 [y_0..y_{n-1}, t]
    q_m = np.array([qmap.get(c, np.nan) for c in cols], float)
    c_obj = np.zeros(n + 1)
    c_obj[:n][m] = -q_m[m]                       # linprog 求最小，故取负

    A_eq, b_eq = [], []
    r = np.zeros(n + 1); r[:n][m] = 1.0; A_eq.append(r); b_eq.append(1.0)   # Σ_m y_m = 1
    r = np.zeros(n + 1); r[:n] = -1.0; r[n] = 1.0; A_eq.append(r); b_eq.append(0.0)  # Σy - t = 0

    A_ub, b_ub = [], []
    for i in range(n):                          # lo_i t - y_i <= 0
        r = np.zeros(n + 1); r[i] = -1.0; r[n] = lo[i]; A_ub.append(r); b_ub.append(0.0)
    for i in range(n):                          # y_i - hi_i t <= 0
        r = np.zeros(n + 1); r[i] = 1.0; r[n] = -hi[i]; A_ub.append(r); b_ub.append(0.0)

    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        return {"ok": False, "message": res.message, "q_max": np.nan, "p": None}
    y = res.x[:n]; t = res.x[n]
    p = y / t if t > 0 else np.full(n, np.nan)
    return {"ok": True, "message": res.message, "q_max": float(-res.fun),
            "p": p, "t": float(t)}


def slsqp_max_quality(p0: np.ndarray, qmap: dict, cols: list[str], lo: np.ndarray,
                      hi: np.ndarray, l1_radius: float | None, seed: int,
                      n_starts: int = 16):
    """盒 + 可选 L1 球内 max Q(p)，用多起点 SLSQP 数值求解。

    L1 半径约束 ‖p-p0‖_1 <= r 不是线性约束，无法并入上面的 LP；这里用数值解，
    并以「r=None 时应与 LP 一致」作为求解器的交叉核验。
    """
    n = len(cols)
    p0 = np.asarray(p0, float); p0 = p0 / p0.sum()
    cons = [{"type": "eq", "fun": lambda p: p.sum() - 1.0}]
    if l1_radius is not None:
        cons.append({"type": "ineq",
                     "fun": lambda p, r=float(l1_radius): r - np.abs(p - p0).sum()})
    bounds = [(float(a), float(b)) for a, b in zip(lo, hi)]
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_starts):
        if k == 0:
            x0 = p0.copy()
        else:
            # 以 p0 为中心的 LHS 扰动，再投影回盒内并归一化
            u = rng.random(n)
            step = (0.02 + 0.18 * rng.random()) if l1_radius is None else (l1_radius / 6.0)
            x0 = np.clip(p0 + (u - 0.5) * 2 * step, lo, hi)
            x0 = x0 / x0.sum()
        try:
            r = minimize(lambda p: -float(quality_of_mixture(p[None, :], qmap, cols)[0]),
                         x0, method="SLSQP", bounds=bounds, constraints=cons,
                         options={"maxiter": 400, "ftol": 1e-14})
        except Exception:
            continue
        if r.success and np.all(np.isfinite(r.x)):
            pv = np.clip(r.x, lo, hi); pv = pv / pv.sum()
            q = float(quality_of_mixture(pv[None, :], qmap, cols)[0])
            if best is None or q > best["q_max"]:
                best = {"q_max": q, "p": pv, "n_starts_ok": k + 1}
    if best is None:
        return {"ok": False, "q_max": np.nan, "p": None}
    best["ok"] = True
    return best


# ------------------------------------------------------------------- 候选生成
def _feasible(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(np.all(p >= lo - tol) and np.all(p <= hi + tol) and abs(p.sum() - 1.0) < 1e-9)


def enumerate_candidates(cfg: dict, p0: np.ndarray, cols: list[str], qmap: dict,
                         lo: np.ndarray, hi: np.ndarray, s: float,
                         functional: str, seed: int) -> pd.DataFrame:
    """枚举候选配比：参考配比、单纯形成对扰动、Dirichlet 抽样、训练信息排序选出的极值候选。

    全部候选只依赖 p0、可信区域与第一问质量坐标；不接触任何检验集 Loss。
    "out_of_trust" 列标记超出可信区域但仍在数据支持范围 [min,max] 的候选（外推登记）。
    """
    n = len(cols)
    short = [c.replace("train_the_pile_", "") for c in cols]
    d = float(cfg["mixtures"]["delta"])
    l1_r = float(cfg["mixtures"]["trust_l1_radius"])
    k_sel = int(cfg["mixtures"]["select_top_k"])

    rows = []

    def _add(tag: str, p: np.ndarray, note: str = ""):
        p = np.asarray(p, float)
        ps = p.sum()
        if ps <= 0:
            return
        p = p / ps
        in_box = bool(np.all(p >= lo - 1e-12) and np.all(p <= hi + 1e-12))
        in_l1 = bool(np.abs(p - p0).sum() <= l1_r + 1e-12)
        rows.append({"candidate": tag, "source": note, "in_trust_box": in_box,
                     "in_trust_l1": in_l1,
                     "trust_status": ("in" if (in_box and in_l1) else
                                      ("box_only" if in_box else "outside")),
                     **dict(zip(cols, p))})

    _add("reference_p0", p0, "first_question_reference")

    # 单纯形成对扰动 p_i + d, p_j - d（与 Q2-C 冻结步长一致）
    for i in range(n):
        for j in range(n):
            if i == j or p0[j] < d:
                continue
            p = p0.copy(); p[i] += d; p[j] -= d
            _add(f"shift_{short[i]}_up__{short[j]}_down", p, "simplex_perturbation")

    # Dirichlet 抽样（固定种子；只围绕 p0 的尺度参数，仍属训练信息）
    rng = np.random.default_rng(seed)
    for a in (200.0, 50.0, 20.0):
        for k in range(20):
            _add(f"dirichlet_a{a:g}_{k}", rng.dirichlet(np.maximum(p0 * a, 1e-6)),
                 "dirichlet_around_p0")

    # 由训练信息（第一问质量坐标）排序，取质量最高/最低各 K 个作为极值候选
    base = np.array([r[c] for r in rows for c in cols], float).reshape(len(rows), n)
    qq = quality_of_mixture(base, qmap, cols)
    order = np.argsort(-qq)          # Q 从大到小
    for rank, idx in enumerate(order[:k_sel]):
        rows[idx]["source"] += "|topQ"
        rows[idx]["select_rank"] = rank + 1
    for rank, idx in enumerate(order[-k_sel:]):
        rows[idx]["source"] += "|lowQ"
        rows[idx]["select_rank_low"] = rank + 1

    df = pd.DataFrame(rows)
    df["Q"] = quality_of_mixture(df[cols].to_numpy(float), qmap, cols)
    df["delta"] = loss_type_deviation(df[cols].to_numpy(float), p0, qmap, cols, functional, s)
    return df
