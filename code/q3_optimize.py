# -*- coding: utf-8 -*-
"""
第三问：算力约束下的多维资源联合优化（Baseline B / Q3-B 固定配比二维搜索）。

模型（来自第二问 Q2-B）：
    L(N,D,Q) = E + A·N^(-α) + B·[D·Q^κ·h(p)]^(-β)     固定 p ⇒ h(p)=1
预算约束（题面三部分组成，同一 D）：
    C = 6ND + η·N·D·ℓ + D·[g(Q) - g(Q0)]_+  =  D·{ (6+ηℓ)N + [g(Q)-g(Q0)]_+ }
    η = 2×10⁻⁴,  ℓ_crit = 6/η = 30000

消去 D：  D(N,Q) = C / { (6+ηℓ)N + [g(Q)-g(Q0)]_+ }
在 (log N, Q) 上做粗网格 + 局部细化；另给 Q3-A 解析参照核验。

情景：三档预算 1e19 / 1e22 / 1e24 FLOPs；三类质量成本函数；
      C7 五档上下文长度 2048/4096/8192/32768/131072。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, save_csv, save_json

ETA = 2e-4
L_CRIT = 6 / ETA                     # 30000
BUDGETS = [1e19, 1e22, 1e24]
CONTEXTS = [2048, 4096, 8192, 32768, 131072]
COST_FUNCS = {
    "指数型": dict(kind="exp",   gamma=1e7,   lam=6.0,
                   g=lambda Q, g=1e7, l=6.0: g * np.exp(l * Q),
                   gp=lambda Q, g=1e7, l=6.0: g * l * np.exp(l * Q)),
    "幂函数型": dict(kind="power", gamma=5e9, lam=4.0,
                    g=lambda Q, g=5e9, l=4.0: g * Q ** l,
                    gp=lambda Q, g=5e9, l=4.0: g * l * Q ** (l - 1)),
    "对数渐进型": dict(kind="log", gamma=2e9, lam=10.0,
                     g=lambda Q, g=2e9, l=10.0: g * np.log1p(l * Q),
                     gp=lambda Q, g=2e9, l=10.0: g * l / (1 + l * Q)),
}
# 第二问估计的标度律参数（B1 经典 + B6 质量指数）
TH = dict(E=1.68979756, A=0.35398032, alpha=0.33997658,
          B=1.24030558, beta=0.27987813, kappa=1.0524)
Q0_DEFAULT = 0.5
N_MIN, N_MAX = 1e7, 1e12            # 1e7 ~ 1e12 参数（10M ~ 1T）
DATA_SUPPORT_N = (7.0542e7, 1.1965825e10)   # 数据支持范围（B1/B6）


def loss(N, D, Q, th=TH):
    return th["E"] + th["A"] * N ** (-th["alpha"]) + \
           th["B"] * (D * Q ** th["kappa"]) ** (-th["beta"])


def D_from_budget(N, Q, C, ell, cf, Q0):
    """由预算等式解出 D；分母必须为正。"""
    a = 6 + ETA * ell
    dg = max(cf["g"](Q) - cf["g"](Q0), 0.0)
    denom = a * N + dg
    return C / denom if denom > 0 else np.nan


def L_of_NQ(N, Q, C, ell, cf, Q0, th=TH):
    D = D_from_budget(N, Q, C, ell, cf, Q0)
    if not np.isfinite(D) or D <= 0:
        return np.inf, np.nan
    return loss(N, D, Q, th), D


def analytic_Nstar(C, ell, cf, Q0, th=TH, Q=None):
    """Q3-A 解析参照：固定 Q（且无质量开销，即 Q=Q0）时最优 N。

    由 dL/dN = 0：-αA·N^(-α-1) + βB·(Q^κ C/a)^(-β)·N^(β-1) = 0
        ⇒ N* = [ α·A·(Q^κ·C/a)^β / (β·B) ]^(1/(α+β)),  a = 6+ηℓ
    前提：预算紧约束、D 无额外上限、Q 固定不产生开销。
    """
    Q = Q0 if Q is None else Q
    a = 6 + ETA * ell
    num = th["alpha"] * th["A"] * (Q ** th["kappa"] * C / a) ** th["beta"]
    den = th["beta"] * th["B"]
    return (num / den) ** (1 / (th["alpha"] + th["beta"]))


def _min_over_N(Q, C, ell, cf, Q0, th):
    """固定 Q 时对 log N 做有界黄金分割极小化（损失面极平坦，1-D 有界搜索最稳）。"""
    lo, hi = np.log(N_MIN), np.log(N_MAX)

    def f(ln):
        v, _ = L_of_NQ(np.exp(ln), Q, C, ell, cf, Q0, th)
        return v if np.isfinite(v) else 1e6

    r = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-12))
    return float(np.exp(r.x)), float(r.fun)


def solve_2d(C, ell, cf, Q0, th=TH, n_grid=260, q_grid=200, refine=4, fix_Q=None):
    """(N, Q) 联合最优：外层 Q 网格，内层对 log N 有界极小化，再交替细化。

    该损失面对 N 极为平坦（β 小），普通二维 Nelder-Mead 会提前停止，
    故采用「Q 网格 + 一维精确极小化 + 坐标下降细化」的策略。
    fix_Q 不为 None 时把 Q 固定在该值（用于与解析参照对照）。
    """
    if fix_Q is not None:
        N, Lv = _min_over_N(fix_Q, C, ell, cf, Q0, th)
        D = D_from_budget(N, fix_Q, C, ell, cf, Q0)
        return dict(N=N, Q=float(fix_Q), D=float(D), L=float(Lv),
                    N_band_1e4=(N, N), N_band_ratio=1.0, grad_lnN=0.0, grad_Q=0.0,
                    kkt_ok=True)
    best = (np.inf, None, None)
    for _ in range(refine):
        qs = np.linspace(Q0, 1.0, q_grid)
        for Q in qs:
            N, Lv = _min_over_N(Q, C, ell, cf, Q0, th)
            if Lv < best[0]:
                best = (Lv, N, Q)
        # 在最优 Q 附近收紧网格
        span = (1.0 - Q0) / max(q_grid - 1, 1) * 8
        q_lo, q_hi = max(Q0, best[2] - span), min(1.0, best[2] + span)
        if q_hi - q_lo < 1e-9:
            break
        Q0_eff = q_lo
        q_grid = max(40, q_grid // 2)
        # 下一轮在 [q_lo, q_hi] 上搜索
        tmp = []
        for Q in np.linspace(q_lo, q_hi, 120):
            N, Lv = _min_over_N(Q, C, ell, cf, Q0, th)
            tmp.append((Lv, N, Q))
        Lv, N, Q = min(tmp)
        if Lv < best[0]:
            best = (Lv, N, Q)
        break
    Ns, Qs = float(best[1]), float(best[2])
    Ls, Ds = L_of_NQ(Ns, Qs, C, ell, cf, Q0, th)
    # 参数辨识宽度：ΔL 不超过 1e-4 的 N 区间（反映平坦度）
    lo, hi = np.log(N_MIN), np.log(N_MAX)
    lns = np.linspace(lo, hi, 4000)
    Lv = np.array([L_of_NQ(np.exp(l), Qs, C, ell, cf, Q0, th)[0] for l in lns])
    ok = Lv <= Ls + 1e-4
    band = (float(np.exp(lns[ok].min())), float(np.exp(lns[ok].max()))) if ok.any() else (Ns, Ns)
    # KKT / 一阶条件自检（数值梯度）
    h = 1e-6
    gN = (L_of_NQ(Ns * (1 + h), Qs, C, ell, cf, Q0, th)[0] -
          L_of_NQ(Ns * (1 - h), Qs, C, ell, cf, Q0, th)[0]) / (2 * h)
    qp = min(Qs + 1e-6, 1.0)
    qm = max(Qs - 1e-6, Q0)
    gQ = (L_of_NQ(Ns, qp, C, ell, cf, Q0, th)[0] -
          L_of_NQ(Ns, qm, C, ell, cf, Q0, th)[0]) / (qp - qm) if qp > qm else 0.0
    return dict(N=Ns, Q=Qs, D=float(Ds), L=float(Ls),
                N_band_1e4=band, N_band_ratio=band[1] / band[0],
                grad_lnN=gN, grad_Q=gQ,
                kkt_ok=bool(abs(gN) < 1e-6 or abs(np.log(Ns / N_MIN)) < 1e-6
                            or abs(np.log(Ns / N_MAX)) < 1e-6))


def cost_shares(N, D, Q, C, ell, cf, Q0):
    c_tr = 6 * N * D
    c_at = ETA * N * D * ell
    c_q = D * max(cf["g"](Q) - cf["g"](Q0), 0.0)
    tot = c_tr + c_at + c_q
    return dict(C_train=c_tr, C_attn=c_at, C_quality=c_q, C_total=tot,
                share_train=c_tr / C, share_attn=c_at / C, share_quality=c_q / C,
                budget_residual=(C - tot) / C)


def main():
    print("=" * 100)
    print("第三问 · 算力约束下的多维资源联合优化（Q3-B）")
    print(f"参数: E={TH['E']:.4f} A={TH['A']:.4f} α={TH['alpha']:.4f} "
          f"B={TH['B']:.4f} β={TH['beta']:.4f} κ={TH['kappa']:.4f}  Q0={Q0_DEFAULT}")
    print(f"η={ETA:g}, ℓ_crit=6/η={L_CRIT:.0f}")
    print("=" * 100)

    # ---------- 1. 解析参照核验（固定 Q=Q0，无质量开销 ⇒ 解析式严格成立） ----------
    chk = []
    for C in BUDGETS:
        for ell in CONTEXTS:
            cf = COST_FUNCS["指数型"]
            Ns = analytic_Nstar(C, ell, cf, Q0_DEFAULT)
            a = 6 + ETA * ell
            Ds = C / (a * Ns)
            La = loss(Ns, Ds, Q0_DEFAULT)
            # 关闭质量自由度（Q 固定为 Q0）后做数值搜索，与解析式对照
            num = solve_2d(C, ell, cf, Q0_DEFAULT, fix_Q=Q0_DEFAULT)
            chk.append(dict(C=C, ell=ell, N_analytic=Ns, N_numeric=num["N"],
                            D_analytic=Ds, D_numeric=num["D"],
                            L_analytic=La, L_numeric=num["L"],
                            rel_diff_N=abs(num["N"] / Ns - 1),
                            L_gap=num["L"] - La))
    chk_df = pd.DataFrame(chk)
    save_csv(chk_df, "q3_analytic_check.csv")
    print("\n[Q3-A 解析参照 vs 数值搜索]（固定 Q=Q0 且无质量开销，解析式严格成立）")
    print(chk_df[["C", "ell", "N_analytic", "N_numeric", "rel_diff_N", "L_gap"]].to_string(index=False))

    # ---------- 1b. 一般情形的一阶条件（KKT）自检 ----------
    kkt = []
    for C in BUDGETS:
        for cfname, cf in COST_FUNCS.items():
            for ell in [2048, 8192, 131072]:
                r = solve_2d(C, ell, cf, Q0_DEFAULT)
                kkt.append(dict(C=C, cost=cfname, ell=ell, N=r["N"], Q=r["Q"],
                                grad_lnN=r["grad_lnN"], grad_Q=r["grad_Q"],
                                Q_bound=("下界" if abs(r["Q"] - Q0_DEFAULT) < 1e-9 else
                                         "上界" if abs(r["Q"] - 1.0) < 1e-9 else "内部"),
                                N_band_ratio=r["N_band_ratio"]))
    kkt_df = pd.DataFrame(kkt)
    save_csv(kkt_df, "q3_kkt_check.csv")
    print("\n[一阶条件自检] |∂L/∂lnN| 应≈0；Q 在边界时梯度符号应指向可行域内")
    print(kkt_df.to_string(index=False))

    # ---------- 2. 主情景表：三档预算 × 三类成本 × 五档上下文 ----------
    rows = []
    for C in BUDGETS:
        for cfname, cf in COST_FUNCS.items():
            for ell in CONTEXTS:
                r = solve_2d(C, ell, cf, Q0_DEFAULT)
                sh = cost_shares(r["N"], r["D"], r["Q"], C, ell, cf, Q0_DEFAULT)
                Ns_an = analytic_Nstar(C, ell, cf, Q0_DEFAULT, Q=r["Q"])
                rows.append(dict(C=C, cost=cfname, ell=ell,
                                 c_attn_ratio=ETA * ell / 6,
                                 N=r["N"], D=r["D"], Q=r["Q"], L=r["L"],
                                 C_FLOPs_check=6 * r["N"] * r["D"] + ETA * r["N"] * r["D"] * ell
                                 + r["D"] * max(cf["g"](r["Q"]) - cf["g"](Q0_DEFAULT), 0),
                                 N_over_D=r["N"] / r["D"],
                                 tokens_per_param=r["D"] / r["N"],
                                 Q_at_bound=bool(abs(r["Q"] - Q0_DEFAULT) < 1e-9),
                                 Q_at_upper=bool(abs(r["Q"] - 1.0) < 1e-9),
                                 N_in_data_support=bool(DATA_SUPPORT_N[0] <= r["N"] <= DATA_SUPPORT_N[1]),
                                 N_analytic_fixedQ=Ns_an,
                                 **sh))
    main_df = pd.DataFrame(rows)
    save_csv(main_df, "q3_optimal_allocations.csv")
    print("\n[主情景 · 节选] 三档预算 × 三类成本（ℓ=8192）")
    sub = main_df[main_df.ell == 8192][["C", "cost", "N", "D", "Q", "L",
                                        "share_train", "share_quality", "share_attn", "Q_at_bound"]]
    print(sub.to_string(index=False))

    # ---------- 3. 预算连续路径与结构性转移识别 ----------
    path = []
    Cs = np.logspace(19, 25, 61)
    for cfname, cf in COST_FUNCS.items():
        for ell in [2048, 8192, 32768, 131072]:
            prev = None
            for C in Cs:
                r = solve_2d(C, ell, cf, Q0_DEFAULT, n_grid=150, q_grid=120, refine=3)
                sh = cost_shares(r["N"], r["D"], r["Q"], C, ell, cf, Q0_DEFAULT)
                path.append(dict(logC=float(np.log10(C)), C=C, cost=cfname, ell=ell,
                                 N=r["N"], D=r["D"], Q=r["Q"], L=r["L"],
                                 N_over_D=r["N"] / r["D"], tokens_per_param=r["D"] / r["N"],
                                 Q_at_bound=bool(abs(r["Q"] - Q0_DEFAULT) < 1e-6),
                                 **{k: v for k, v in sh.items() if k.startswith("share")}))
                prev = r
    path_df = pd.DataFrame(path)
    save_csv(path_df, "q3_budget_paths.csv")

    # 结构性转移：Q* 离开下界 / tokens-per-param 的对数斜率变号
    trans = []
    for (cfname, ell), g in path_df.groupby(["cost", "ell"]):
        g = g.sort_values("logC")
        qb = g["Q_at_bound"].to_numpy()
        cross = None
        for i in range(1, len(g)):
            if qb[i - 1] and not qb[i]:
                cross = float(g["logC"].iloc[i])
                break
        tpp = np.log10(g["tokens_per_param"].to_numpy())
        sl = np.gradient(tpp, g["logC"].to_numpy())
        trans.append(dict(cost=cfname, ell=ell,
                          Q_leaves_bound_at_logC=cross,
                          slope_tokens_per_param_start=float(sl[0]),
                          slope_tokens_per_param_end=float(sl[-1]),
                          Q_min=float(g["Q"].min()), Q_max=float(g["Q"].max()),
                          share_quality_min=float(g["share_quality"].min()),
                          share_quality_max=float(g["share_quality"].max()),
                          share_train_min=float(g["share_train"].min()),
                          share_train_max=float(g["share_train"].max())))
    tr_df = pd.DataFrame(trans)
    save_csv(tr_df, "q3_transition_candidates.csv")
    print("\n[结构性转移候选]")
    print(tr_df.to_string(index=False))

    # ---------- 4. 敏感性：κ / Q0 / α / β ----------
    sens = []
    base_ell, base_C = 8192, 1e22
    for tag, th2, q0 in [
        ("基准", TH, Q0_DEFAULT),
        ("κ=0.8", {**TH, "kappa": 0.8}, Q0_DEFAULT),
        ("κ=1.3", {**TH, "kappa": 1.3}, Q0_DEFAULT),
        ("Q0=0.3", TH, 0.3),
        ("Q0=0.7", TH, 0.7),
        ("α-10%", {**TH, "alpha": TH["alpha"] * 0.9}, Q0_DEFAULT),
        ("α+10%", {**TH, "alpha": TH["alpha"] * 1.1}, Q0_DEFAULT),
        ("β-10%", {**TH, "beta": TH["beta"] * 0.9}, Q0_DEFAULT),
        ("β+10%", {**TH, "beta": TH["beta"] * 1.1}, Q0_DEFAULT),
    ]:
        for cfname, cf in COST_FUNCS.items():
            r = solve_2d(base_C, base_ell, cf, q0, th=th2, n_grid=200, q_grid=150)
            sh = cost_shares(r["N"], r["D"], r["Q"], base_C, base_ell, cf, q0)
            sens.append(dict(scenario=tag, cost=cfname, N=r["N"], D=r["D"],
                             Q=r["Q"], L=r["L"], **{k: v for k, v in sh.items()
                                                    if k.startswith("share")}))
    sens_df = pd.DataFrame(sens)
    save_csv(sens_df, "q3_sensitivity.csv")
    print(f"\n[敏感性] C=1e22, ℓ=8192")
    print(sens_df[["scenario", "cost", "N", "D", "Q", "L"]].to_string(index=False))

    # ---------- 5. 约束与成本核验 ----------
    ver = []
    for _, r in main_df.iterrows():
        cf = COST_FUNCS[r["cost"]]
        c_tr = 6 * r["N"] * r["D"]
        c_at = ETA * r["N"] * r["D"] * r["ell"]
        c_q = r["D"] * max(cf["g"](r["Q"]) - cf["g"](r["Q0"] if "Q0" in r else Q0_DEFAULT), 0)
        tot = c_tr + c_at + c_q
        ver.append(dict(C=r["C"], cost=r["cost"], ell=r["ell"],
                        C_recomputed=tot, rel_error=abs(tot / r["C"] - 1),
                        within_budget=bool(tot <= r["C"] * (1 + 1e-9))))
    ver_df = pd.DataFrame(ver)
    save_csv(ver_df, "q3_constraint_check.csv")
    print(f"\n[约束核验] 最大相对误差={ver_df['rel_error'].max():.3e}  "
          f"全部不超预算={bool(ver_df['within_budget'].all())}")

    save_json(dict(
        model="L=E+A·N^(-α)+B·[D·Q^κ]^(-β),  D=C/{(6+ηℓ)N+[g(Q)-g(Q0)]_+}",
        parameters=TH, eta=ETA, ell_crit=L_CRIT,
        budgets=BUDGETS, contexts=CONTEXTS, cost_functions={k: {kk: vv for kk, vv in v.items()
                                                                if kk in ("kind", "gamma", "lam")}
                                                            for k, v in COST_FUNCS.items()},
        Q0=Q0_DEFAULT, data_support_N=DATA_SUPPORT_N,
        analytic_reference="N*=[αA(Q^κC/a)^β/(βB)]^(1/(α+β)), a=6+ηℓ",
    ), "q3_optimization_meta.json")
    print("\n完成：q3_*.csv / q3_optimization_meta.json")


if __name__ == "__main__":
    main()
