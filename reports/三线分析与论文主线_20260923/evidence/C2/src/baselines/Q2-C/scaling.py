# -*- coding: utf-8 -*-
"""Q2-C 广义标度律：模型族、多起点约束拟合、可辨识性诊断、导数与弹性。

模型族（§2.3）：
    M0 经典   L = E + A N^(-α) + B D^(-β)
    M1=Q2-B   L = E + A N^(-α) + B [D Q^κ h(p)]^(-β)            ρ = 0 嵌套
    M2=Q2-C   L = E + A [N Q^ρ]^(-α) + B [D Q^κ h(p)]^(-β)      释放 ρ

可选来源偏移 δ：对指定来源组的行给 E 加性偏移 δ，吸收不同语料/评测族的
不可约损失下限差（§2.4 第 6 条），避免把来源偏移误读为质量效应。

统一参数向量 x = [E, A, α, B, β, κ, ρ, δ]，自由度由 free 名单控制；
不在 free 中的参数取 base 值（κ/ρ 默认 0，即嵌套模型）。

全部拟合在原空间做非线性最小二乘（§2.2：「对加性幂律整体取 log 后做普通
线性回归并不等价」），log 空间仅作稳健性对照。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares

FULL = ["E", "A", "alpha", "B", "beta", "kappa", "rho", "delta"]
BASE = np.array([1.5, 0.5, 0.30, 0.5, 0.30, 0.0, 0.0, 0.0])

MODEL_FREE = {
    "M0": ["E", "A", "alpha", "B", "beta"],
    "M1": ["E", "A", "alpha", "B", "beta", "kappa"],
    "M2": ["E", "A", "alpha", "B", "beta", "kappa", "rho"],
}

# A、B 的区间跨度达 6 个数量级，其"距边界"必须按对数位置判断；
# 其余参数按线性位置判断。
LOG_SCALE = {"A", "B"}


def rel_position(name: str, val: float, bounds: dict) -> float:
    """参数在区间内的相对位置 [0,1]，A/B 走对数刻度。

    直接用 (v-lo)/(hi-lo) 会把量级中部的 A（如 0.35∈[1e-4,100]）误判为命中下边界。
    """
    lo, hi = bounds[name]
    if name in LOG_SCALE and lo > 0 and val > 0:
        return float(np.log(val / lo) / np.log(hi / lo))
    return float((val - lo) / max(hi - lo, 1e-12))


Q_FLOOR = 1e-9   # 质量坐标按定义满足 0<Q≤1；此下限只用于避免 Q=0 时幂运算溢出


def predict(x: np.ndarray, N, D, Q, h=1.0, dgrp=None) -> np.ndarray:
    """由 8 维完整参数向量预测 Loss。dgrp 为 1 的行额外加 δ。

    Q 会被夹到 [Q_FLOOR, ∞)：模型按定义针对正的 Quality 坐标（配置 units 写明
    0<Q≤1），Q=0 时 (N·Q^ρ)^(-α) 发散；夹紧只是防溢出，不代表允许 Q≤0 的输入。
    """
    E, A, al, B, be, ka, rh, de = (float(v) for v in x[:8])
    N = np.asarray(N, float)
    D = np.asarray(D, float)
    Q = np.maximum(np.asarray(Q, float), Q_FLOOR)
    h = np.asarray(h, float) if not np.isscalar(h) else np.full_like(N, float(h))
    h = np.maximum(h, Q_FLOOR)
    Neff = N * np.power(Q, rh)
    X = D * np.power(Q, ka) * h
    out = E + A * np.power(Neff, -al) + B * np.power(X, -be)
    if dgrp is not None:
        out = out + de * np.asarray(dgrp, float)
    return out


def terms(x: np.ndarray, N, D, Q, h=1.0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (参数项 U=A(N Q^ρ)^(-α), 数据项 V=B(D Q^κ h)^(-β))，供导数复用。"""
    E, A, al, B, be, ka, rh, de = (float(v) for v in x[:8])
    N = np.asarray(N, float); D = np.asarray(D, float)
    Q = np.maximum(np.asarray(Q, float), Q_FLOOR)
    h = np.asarray(h, float) if not np.isscalar(h) else np.full_like(N, float(h))
    h = np.maximum(h, Q_FLOOR)
    U = A * np.power(N * np.power(Q, rh), -al)
    V = B * np.power(D * np.power(Q, ka) * h, -be)
    return U, V


# ------------------------------------------------------------------ 解析导数
def derivatives(x, N, D, Q, h=1.0, dgrp=None) -> dict:
    """解析偏导（§2.5）。返回 dL/dN, dL/dD, dL/dQ, dL/dh。"""
    E, A, al, B, be, ka, rh, de = (float(v) for v in x[:8])
    N = np.asarray(N, float); D = np.asarray(D, float); Q = np.asarray(Q, float)
    h = np.asarray(h, float) if not np.isscalar(h) else np.full_like(N, float(h))
    U, V = terms(x, N, D, Q, h)
    return {
        "dL_dN": -al * U / N,
        "dL_dD": -be * V / D,
        "dL_dQ": -(al * rh * U + be * ka * V) / Q,
        "dL_dh": -be * V / h,
    }


def elasticities(x, N, D, Q, h=1.0, dgrp=None) -> dict:
    """弹性 ε_x = (∂L/∂x)(x/L)，并对可约损失 L-E-δ 另报一份（§2.5）。

    降低 Loss 的因素通常为负弹性；不得与相对可约损失的弹性混用。
    """
    E, A, al, B, be, ka, rh, de = (float(v) for v in x[:8])
    N = np.asarray(N, float); D = np.asarray(D, float); Q = np.asarray(Q, float)
    h = np.asarray(h, float) if not np.isscalar(h) else np.full_like(N, float(h))
    d = np.asarray(dgrp, float) if dgrp is not None else np.zeros_like(N)
    U, V = terms(x, N, D, Q, h)
    L = E + de * d + U + V
    Lred = np.maximum(L - E - de * d, 1e-12)
    return {
        "L": L,
        "L_reducible": L - E - de * d,
        "eps_N": -al * U / L,
        "eps_D": -be * V / L,
        "eps_Q": -(al * rh * U + be * ka * V) / L,
        "eps_h": -be * V / L,
        "eps_N_reducible": -al * U / Lred,
        "eps_D_reducible": -be * V / Lred,
        "eps_Q_reducible": -(al * rh * U + be * ka * V) / Lred,
    }


def delta_loss_Q(x, N, D, Q, h=1.0, dgrp=None, dQ=0.10) -> np.ndarray:
    """ΔL_Q = L(Q) - L(Q+dQ)，质量改善收益（§2.5）。Q+dQ 超 1 时按接口截断到 1。"""
    Q2 = np.minimum(np.asarray(Q, float) + dQ, 1.0)
    return predict(x, N, D, Q, h, dgrp) - predict(x, N, D, Q2, h, dgrp)


def equivalent_N(x, N, D, Q, h=1.0, dgrp=None, dQ=0.10):
    """同等等效参数规模 N_eq：使 L(N_eq, Q) = L(N, Q+dQ)。

    §2.5 闭式解 N_eq=[N^(-α) - ΔL_Q/A]^(-1/α) 在 M0/M1（N 通道不含 Q）成立；
    M2 的 N 通道含 Q，故统一用单调求根，闭式仅作 M0/M1 的交叉校验。

    返回 (N_eq, 状态)，状态 ∈ {ok, no_solution}。括号 ≤0 表示仅靠增大 N 无法
    实现同等质量收益（ΔL_Q ≥ A N^(-α)），返回 inf 并标注。
    """
    E, A, al, B, be, ka, rh, de = (float(v) for v in x[:8])
    N = float(N); Qv = float(Q); Dv = float(D)
    hv = float(h) if np.isscalar(h) else float(np.asarray(h, float).mean())
    Q2 = min(Qv + dQ, 1.0)
    d = float(dgrp) if dgrp is not None else 0.0
    target = float(predict(x, N, Dv, Q2, hv, d))

    def g(n):
        return float(predict(x, n, Dv, Qv, hv, d)) - target

    if g(N) <= 0:
        # Q 改善没有带来正收益（κ=ρ=0 或已饱和），无需增大参数
        return N, "ok"
    hi = N
    for _ in range(200):
        hi *= 1.6
        if g(hi) <= 0:
            break
        if hi > 1e12:
            return float("inf"), "no_solution"
    else:
        return float("inf"), "no_solution"
    return float(brentq(g, N, hi, xtol=1e-10, rtol=1e-12)), "ok"


# ------------------------------------------------------------------ 多起点拟合
def _lhs_starts(free, bounds, n, rng) -> np.ndarray:
    """Latin hypercube 起点；A、B 按对数均匀采样（量级跨度大）。"""
    k = len(free)
    out = np.empty((n, k))
    for j, name in enumerate(free):
        lo, hi = bounds[name]
        u = (rng.permutation(n) + rng.random(n)) / n
        if name in ("A", "B"):
            out[:, j] = np.exp(np.log(max(lo, 1e-8)) + u * (np.log(hi) - np.log(max(lo, 1e-8))))
        else:
            out[:, j] = lo + u * (hi - lo)
    return out


def fit(N, D, Q, L, model="M2", free_extra=(), free_only=None, bounds=None, n_starts=48,
        seed=0, base=None, weights=None, dgrp=None, max_nfev=20000, cluster_tol=0.15,
        log=None) -> dict:
    """多起点有界非线性最小二乘。

    free_only 给定时完全替换模型预设的自由参数名单（分阶段估计用：经典参数
    已由 B1 冻结在 base 中，只释放质量通道参数）；未列入 free 的参数一律取 base。

    返回 dict：x（8 维完整参数）、free、sse、rmse、n、starts（全部解）、
    clusters（等价解聚类）、boundary_hits。
    """
    N = np.asarray(N, float); D = np.asarray(D, float)
    Q = np.asarray(Q, float); L = np.asarray(L, float)
    w = np.ones_like(L) if weights is None else np.asarray(weights, float)
    base = BASE.copy() if base is None else np.asarray(base, float).copy()
    free = list(free_only) if free_only is not None else list(MODEL_FREE[model]) + list(free_extra)
    idx = [FULL.index(n) for n in free]
    lo = np.array([bounds[n][0] for n in free], float)
    hi = np.array([bounds[n][1] for n in free], float)

    def resid(z):
        x = base.copy()
        x[idx] = z
        return (predict(x, N, D, Q, 1.0, dgrp) - L) * np.sqrt(w)

    rng = np.random.default_rng(seed)
    starts = _lhs_starts(free, bounds, int(n_starts), rng)
    # 首个起点取经典文献量级，保证有确定性的基准解
    starts[0] = np.clip(np.array([base[i] for i in idx], float), lo, hi)

    sols = []
    for s in starts:
        try:
            r = least_squares(resid, np.clip(s, lo, hi), bounds=(lo, hi),
                              x_scale="jac", max_nfev=int(max_nfev))
        except Exception:
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        x = base.copy(); x[idx] = r.x
        sols.append({"x": x, "sse": float(np.sum(r.fun ** 2)), "z": r.x.copy()})
    if not sols:
        raise RuntimeError("所有起点均未收敛")
    sols.sort(key=lambda s: s["sse"])
    best = sols[0]
    nz = np.array([s["z"] for s in sols])
    span = np.maximum(hi - lo, 1e-12)
    dn = (nz - nz[0]) / span
    dist = np.sqrt((dn ** 2).sum(axis=1))

    # 单链接聚类：把落在同一局部解的起点归为一类（可辨识性证据）
    clusters, used = [], np.zeros(len(sols), bool)
    for i in range(len(sols)):
        if used[i]:
            continue
        grp = dist <= float(cluster_tol)
        grp[i] = True
        members = np.where(grp & ~used)[0]
        used |= grp
        clusters.append({"rep": sols[i]["x"].copy(), "sse": sols[i]["sse"],
                         "n": int(len(members))})
    bh = {}
    for name in free:
        rel = rel_position(name, float(best["x"][FULL.index(name)]), bounds)
        if rel < 0.02:
            bh[name] = "lower"
        elif rel > 0.98:
            bh[name] = "upper"
    res = {
        "x": best["x"], "free": free, "sse": best["sse"], "n": int(len(L)),
        "n_free": len(free),
        "rmse": float(np.sqrt(best["sse"] / len(L))),
        "n_solutions": int(len(sols)),
        "clusters": clusters,
        "boundary_hits": bh,
        "sols": sols,
    }
    if log:
        log(f"  [fit {model}] n={len(L)} free={free} SSE={best['sse']:.4f} "
            f"RMSE={res['rmse']:.4f} 解数={len(sols)} 等价类={len(clusters)}"
            + (f" 边界命中={bh}" if bh else ""))
    return res


def fit_jacobian(res: dict, N, D, Q, L, dgrp=None,
                 weights=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在最优点的数值 Jacobian → 协方差与参数相关矩阵（渐近近似）。

    weights 必须与拟合时一致，否则协方差与剖面阈值都不在同一口径上。
    """
    N = np.asarray(N, float); D = np.asarray(D, float)
    Q = np.asarray(Q, float); L = np.asarray(L, float)
    w = np.ones_like(L) if weights is None else np.asarray(weights, float)
    free, idx = res["free"], [FULL.index(n) for n in res["free"]]
    x = res["x"]

    def r(z):
        xx = x.copy(); xx[idx] = z
        return (predict(xx, N, D, Q, 1.0, dgrp) - L) * np.sqrt(w)

    z0 = np.array([x[i] for i in idx], float)
    J = np.empty((len(L), len(idx)))
    for j in range(len(idx)):
        h = max(1e-6, abs(z0[j]) * 1e-6)
        zp, zm = z0.copy(), z0.copy()
        zp[j] += h; zm[j] -= h
        J[:, j] = (r(zp) - r(zm)) / (2 * h)
    dof = max(len(L) - len(idx), 1)
    s2 = float(np.sum(r(z0) ** 2)) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(J.T @ J)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(sd, sd)
    corr = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), np.nan)
    return cov, sd, corr


def profile(res, name, N, D, Q, L, bounds, ci_level=0.95, grid=None, dgrp=None,
            weights=None, log=None, n_bisect=18, max_expand=60) -> dict:
    """剖面似然与 95% 区间。

    固定 name = v，重优化其余自由参数，得 SSE(v)。阈值用 F 检验
        SSE(v) <= SSE_min * (1 + F_{1,n-p,level} / (n-p))，  p = 自由参数个数。

    区间**由从最优点向外求根得到，而不是在网格上扫描**：n 大时阈值只比 SSE_min
    高约 0.25%，粗网格可能一个点都落不进阈值内，网格扫描会把"区间很窄"误报成
    "区间不可判定"(NaN)。求根先按几何步长向外扩张找到跨越阈值的点，再二分到
    机器精度；若扩张到参数边界仍未跨越，则这一侧被数据完全约束不了——这本身
    就是**参数未被辨识**的正面证据，作为 status 返回，而不是伪装成 NaN。

    status ∈ {ok, one_sided_lower_unbounded, one_sided_upper_unbounded,
              profile_below_threshold_to_bounds}
    """
    from scipy.stats import f as fdist

    N = np.asarray(N, float); D = np.asarray(D, float)
    Q = np.asarray(Q, float); L = np.asarray(L, float)
    w = np.ones_like(L) if weights is None else np.asarray(weights, float)
    free, idx = res["free"], [FULL.index(n) for n in res["free"]]
    j = free.index(name)
    lo, hi = bounds[name]
    o_idx = [i for i in idx if i != idx[j]]
    o_lo = np.array([bounds[FULL[i]][0] for i in o_idx], float)
    o_hi = np.array([bounds[FULL[i]][1] for i in o_idx], float)
    fixed_i = idx[j]
    z0 = np.clip(np.array([res["x"][i] for i in o_idx], float), o_lo, o_hi)

    def sse_at(v):
        g = float(min(max(v, lo), hi))

        def resid(z):
            x = res["x"].copy()
            x[fixed_i] = g
            x[o_idx] = z
            return (predict(x, N, D, Q, 1.0, dgrp) - L) * np.sqrt(w)

        try:
            r = least_squares(resid, z0, bounds=(o_lo, o_hi), x_scale="jac", max_nfev=8000)
            return float(np.sum(r.fun ** 2))
        except Exception:
            return float("nan")

    n, p = len(L), len(idx)
    dof = max(n - p, 1)
    est = float(min(max(res["x"][fixed_i], lo), hi))
    rows = [{"param": name, "value": est, "sse": sse_at(est)}]

    # 阈值基线必须与拟合时**同一加权口径**求得的 SSE 一致。若调用方传了 weights，
    # res["sse"] 是加权 SSE；若这里漏传 weights，则 sse_at(est) 会远大于它，阈值
    # 落在最优点的下方，二分退化成一个宽度≈网格精度的**假窄区间**。显式检出该不一致。
    sse_base = float(res["sse"])
    sse_est = float(rows[0]["sse"])
    mismatch = not (np.isfinite(sse_est)
                    and abs(sse_est - sse_base) <= 1e-6 * max(abs(sse_base), 1.0))
    sse_min = sse_est if np.isfinite(sse_est) else sse_base
    thr = sse_min * (1.0 + float(fdist.ppf(ci_level, 1, dof)) / dof)

    def bisect(a, b):
        """a: SSE<=thr 侧，b: SSE>thr 侧；返回阈值穿越点。"""
        for _ in range(int(n_bisect)):
            m = 0.5 * (a + b)
            s = sse_at(m)
            rows.append({"param": name, "value": float(m), "sse": s})
            if not np.isfinite(s) or s > thr:
                b = m
            else:
                a = m
        return 0.5 * (a + b)

    def side(sign):
        """从 EST 向外找阈值穿越点；返回 (root, hit_bound)。"""
        scale = max(abs(est) * 0.25, 0.02 * (hi - lo), 1e-9)
        step = scale / 4.0
        v_prev, v = est, est
        for _ in range(int(max_expand)):
            v = est + sign * step
            if v <= lo or v >= hi:
                v = lo if sign < 0 else hi
                s = sse_at(v)
                rows.append({"param": name, "value": float(v), "sse": s})
                if np.isfinite(s) and s > thr:
                    return bisect(v_prev, v), True
                return None, True          # 到边界仍未穿越 → 该侧无界
            s = sse_at(v)
            rows.append({"param": name, "value": float(v), "sse": s})
            if np.isfinite(s) and s > thr:
                return bisect(v_prev, v), False
            v_prev = v
            step *= 1.8
        return None, False

    lo_root, lo_bound = side(-1)
    hi_root, hi_bound = side(+1)
    if lo_root is None and hi_root is None:
        ci, status = (lo, hi), "profile_below_threshold_to_bounds"
    elif lo_root is None:
        ci, status = (lo, hi_root), "one_sided_lower_unbounded"
    elif hi_root is None:
        ci, status = (lo_root, hi), "one_sided_upper_unbounded"
    else:
        ci, status = (lo_root, hi_root), "ok"

    vals = np.array([r["sse"] for r in rows], float)
    finite = vals[np.isfinite(vals)]
    flat = (float((np.nanmax(finite) - np.nanmin(finite)) / max(sse_min, 1e-12))
            if finite.size else float("nan"))
    if mismatch and status == "ok":
        status = "sse_baseline_mismatch"
    if log:
        log(f"  [profile {name}] 95%CI=({ci[0]:.4g}, {ci[1]:.4g}) "
            f"宽度={ci[1] - ci[0]:.4g} 状态={status} SSE起伏比={flat:.3f}"
            + ("  [警告] 加权口径不一致" if mismatch else ""))
    return {"rows": rows, "ci95": ci, "ci95_width": float(ci[1] - ci[0]),
            "sse_min": sse_min, "sse_at_estimate": sse_est, "weights_mismatch": bool(mismatch),
            "threshold": thr, "flatness": flat, "status": status,
            "n_profile_points": len(rows)}
