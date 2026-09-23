# -*- coding: utf-8 -*-
"""对给定候选配比的 (N, Q) 资源搜索，以及多起点 SLSQP 局部联合细调（Q3-C 第 3、4 条）。

第 3 条：复用 Q3-B 的预算与成本计算，对每个候选搜索 N、Q，并输出各候选误差与成本。
         —— 这里只实现**本问需要的那部分内核**：预算消去 D、粗网格 + 局部加密、
         每点精确计算成本。不产出 Q3-B 的独立交付物。
第 4 条：连续优化 p 时限制单纯形、可信区域和起点；与枚举结果交叉核验，标注局部解。

搜索变量用 log10(N) 与 Q 两维：
  * log10(N) 在 [log10 N_lo, log10 N_hi] 上取粗网格，随后围绕最优点按固定倍数收缩加密；
  * Q 在 [Q_lo, Q_hi] 上同样处理，Q_lo = Q0（只考虑质量提升，§3.1），
    Q_hi 取数据支持上界或 Q2 预测器支持上界（外推标记由调用方注入）。

所有点都用**精确**成本公式计算（不近似、不插值），并回算预算残差供 verify.py 核验。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from mixtures import quality_of_mixture


def _n_bounds(cfg: dict) -> tuple[float, float]:
    """N 的搜索边界（个）。显式 float() 以防 YAML 把 `1.0e4` 解析成字符串。"""
    lo, hi = cfg["search"]["N_search_B"]
    return float(lo) * 1e9, float(hi) * 1e9


# ------------------------------------------------------------------ 粗+细搜索
def _grid_axes(n_lo, n_hi, q_lo, q_hi, n_grid, q_grid):
    lN = np.linspace(np.log10(n_lo), np.log10(n_hi), int(n_grid))
    Q = np.linspace(q_lo, q_hi, int(q_grid))
    LN, QQ = np.meshgrid(lN, Q, indexing="ij")
    return LN, QQ


def search_fixed_p(model, h: float, Q0: float, form: str, C: float, ell: float,
                   kappa: float, q_lo: float, q_hi: float, cfg: dict,
                   n_lo: float = None, n_hi: float = None,
                   n_grid: int = None, q_grid: int = None,
                   refine_rounds: int = None, refine_points: int = None):
    """给定 h（即给定 p）与预算情景，搜索 (N,Q) 使预测 Loss 最小。

    返回最优解、预算/成本诊断、边界命中与求值次数。Q_lo 由调用方给（通常 = Q0）。

    n_grid/q_grid/refine_rounds/refine_points 默认取配置；显式传入用于
    (a) verify.grid_stability 的粗网格复现检查，(b) 候选筛选阶段的廉价配置。
    """
    sc = cfg["search"]
    n_lo_def, n_hi_def = _n_bounds(cfg)
    n_lo = n_lo_def if n_lo is None else float(n_lo)
    n_hi = n_hi_def if n_hi is None else float(n_hi)
    if q_hi <= q_lo:                      # 退化：质量无投入空间，固定 Q=Q0
        q_hi = q_lo + 1e-12

    n_grid = int(sc["N_gross_grid"]) if n_grid is None else int(n_grid)
    q_grid = int(sc["Q_grid"]) if q_grid is None else int(q_grid)
    rounds = int(sc["refine_rounds"]) if refine_rounds is None else int(refine_rounds)
    rpoints = int(sc["refine_points"]) if refine_points is None else int(refine_points)
    n_ev = 0
    best = None
    win_n = (np.log10(n_lo), np.log10(n_hi))
    win_q = (q_lo, q_hi)
    for rnd in range(rounds + 1):
        LN, QQ = _grid_axes(10 ** win_n[0], 10 ** win_n[1], win_q[0], win_q[1],
                            n_grid, q_grid)
        N = (10.0 ** LN).ravel(); Q = QQ.ravel()
        out = model.loss_at_budget(N, Q, np.full_like(N, float(h)), Q0, form, C, ell, kappa)
        L = out["L"]; n_ev += L.size
        k = int(np.argmin(L))
        if best is None or L[k] < best["L"]:
            best = {"L": float(L[k]), "N": float(N[k]), "Q": float(Q[k])}
        # 围绕当前最优收缩窗口（log10 N 与 Q 各自收缩）
        cn, cq = np.log10(N[k]), Q[k]
        hn = (win_n[1] - win_n[0]) / (2.0 * float(sc["refine_shrink"]))
        hq = (win_q[1] - win_q[0]) / (2.0 * float(sc["refine_shrink"]))
        win_n = (max(cn - hn, np.log10(n_lo)), min(cn + hn, np.log10(n_hi)))
        win_q = (max(cq - hq, q_lo), min(cq + hq, q_hi))
        n_grid = rpoints; q_grid = rpoints

    N, Q = best["N"], best["Q"]
    out = model.loss_at_budget(N, Q, float(h), Q0, form, C, ell, kappa, need_grad=True)
    D = float(out["D"][0]); L = float(out["L"][0])
    c = float(out["c"][0]); u = float(out["u"][0])
    shares = model.cost_shares(N, Q, Q0, h, form, C, ell)
    used = shares["used"]
    return {
        "L": L, "N": N, "D": D, "Q": Q, "h": float(h),
        "c_quality": c, "u": u,
        "budget_residual": float(C - used),
        "budget_residual_rel": float((C - used) / C),
        "boundary_n_lower": bool(np.isclose(np.log10(N), np.log10(n_lo), atol=1e-9)),
        "boundary_n_upper": bool(np.isclose(np.log10(N), np.log10(n_hi), atol=1e-9)),
        "boundary_q_lower": bool(np.isclose(Q, q_lo, atol=1e-12)),
        "boundary_q_upper": bool(np.isclose(Q, q_hi, atol=1e-12)),
        "n_evals": n_ev, **shares,
    }


# ------------------------------------------------- 多起点 SLSQP 联合细调（第 4 条）
def joint_slsqp(model, p0, qmap, cols, lo, hi, l1_radius, s, functional, omega,
                Q0_mode, form, C, ell, kappa, cfg, seed: int,
                h_fn=None, q0_ref: float = None):
    """在 (log10 N, Q, p) 上做多起点 SLSQP，限制单纯形、可信区域与起点。

    h_fn(p) 由调用方给出（把 p 映射到 h），便于与枚举使用完全相同的泛函。
    返回各起点解与按标准化距离聚类的「等价解类」，用于**标注局部解**。
    """
    sc = cfg["search"]["slsqp"]
    n = len(cols)
    n_lo, n_hi = _n_bounds(cfg)
    q_hi = float(cfg["quality"]["q_extended_max"])
    rng = np.random.default_rng(seed)
    PEN = 1e6

    def unpack(x):
        N = 10.0 ** x[0]
        Q = x[1]
        p = np.clip(x[2:], lo, hi)
        sp = p.sum()
        p = p / sp if sp > 0 else p0.copy()
        return N, Q, p

    def obj(x):
        N, Q, p = unpack(x)
        h = float(h_fn(p[None, :])[0])
        Q0 = float(q0_ref) if Q0_mode == "fixed_ref" else float(
            quality_of_mixture(p[None, :], qmap, cols)[0])
        o = model.loss_at_budget(np.array([N]), np.array([Q]), np.array([h]),
                                 Q0, form, C, ell, kappa)
        L = float(o["L"][0])
        # Q0 依赖 p 时用罚函数维持 Q >= Q0(p)（SLSQP 无法直接表达这种隐式下界）
        viol = max(0.0, Q0 - Q)
        return L + PEN * viol ** 2

    cons = [{"type": "eq", "fun": lambda x: x[2:].sum() - 1.0}]
    if l1_radius is not None:
        cons.append({"type": "ineq",
                     "fun": lambda x, r=float(l1_radius): r - np.abs(x[2:] - p0).sum()})
    # Q0 依赖 p 时 q0_ref 为 None：此时 Q 的下界由上面的罚函数隐式维持，
    # 变量下界退到一个很小的正数（Q>0 是模型定义域要求）。
    q_lo_var = 1e-3 if q0_ref is None else max(float(q0_ref), 1e-3)
    q_start0 = q_lo_var + 0.1 * (q_hi - q_lo_var)
    bounds = [(np.log10(n_lo), np.log10(n_hi)), (q_lo_var, q_hi)] + \
             [(float(a), float(b)) for a, b in zip(lo, hi)]

    sols = []
    for k in range(int(sc["n_starts"])):
        if k == 0:
            x0 = np.concatenate([[np.log10(1e9)], [q_start0], p0])
        else:
            u = rng.random(n)
            pp = np.clip(p0 + (u - 0.5) * 2 * (l1_radius / 4.0 if l1_radius else 0.05), lo, hi)
            pp = pp / pp.sum()
            x0 = np.concatenate([[np.log10(10 ** rng.uniform(np.log10(n_lo), np.log10(n_hi)))],
                                 [rng.uniform(q_lo_var, q_hi)], pp])
        try:
            r = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                         options={"maxiter": int(sc["maxiter"]), "ftol": float(sc["ftol"])})
        except Exception:
            continue
        N, Q, p = unpack(r.x)
        h = float(h_fn(p[None, :])[0])
        Q0 = float(q0_ref) if Q0_mode == "fixed_ref" else float(
            quality_of_mixture(p[None, :], qmap, cols)[0])
        o = model.loss_at_budget(np.array([N]), np.array([Q]), np.array([h]),
                                 Q0, form, C, ell, kappa)
        sols.append({"start": k, "success": bool(r.success), "L": float(o["L"][0]),
                     "N": float(N), "Q": float(Q), "h": h, "Q0": Q0,
                     "Q_ge_Q0": bool(Q >= Q0 - 1e-9),
                     "n_iter": int(getattr(r, "nit", -1)),
                     "p": p, "message": str(r.message)})

    # 按标准化参数距离聚类（与 Q2-C 的 cluster_tol 同一口径思路）
    tol = 0.15
    clusters = []
    for s_ in sorted(sols, key=lambda d: d["L"]):
        vec = np.concatenate([[np.log10(s_["N"]) / 3.0], [s_["Q"]], s_["p"]])
        placed = False
        for cl in clusters:
            ref = cl["ref"]
            d = np.linalg.norm((vec - ref) / (np.abs(ref) + 1e-6))
            if d < tol:
                cl["members"].append(s_["start"]); placed = True
                if s_["L"] < cl["best"]["L"]:
                    cl["best"] = s_; cl["ref"] = vec
                break
        if not placed:
            clusters.append({"ref": vec, "best": s_, "members": [s_["start"]]})
    for i, cl in enumerate(clusters):
        cl["cluster_id"] = i
        cl.pop("ref", None)
    return {"solutions": sols, "clusters": clusters,
            "n_ok": sum(1 for s_ in sols if s_["success"])}
