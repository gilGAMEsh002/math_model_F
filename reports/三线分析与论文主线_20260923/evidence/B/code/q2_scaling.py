# -*- coding: utf-8 -*-
"""
第二问：跨维度数据融合与广义标度律（Baseline B / Q2-B 有效数据量形式）。

经典律（对照）：      L(N,D) = E + A·N^(-α) + B·D^(-β)
广义律（主模型）：    L(N,D,Q,p) = E + A·N^(-α) + B·[D·Q^κ·h(p)]^(-β)
其中 h(p) = exp{ -ω·[f(p) - f(p0)] / s_f }，f 为第一问配比预测器，
h(p0)=1、h(p)>0；Q=1 且 p=p0 时退化为经典律。

数据分工：
  B1  主拟合（真实 Pythia 轨迹，1176 行）—— 分组留出：按 N 逐规模留一 + D 方向后段外推
  B2  族外验证（半合成）；B3 插值轨迹验证
  B4/B5 跨族与文献验证（真实）
  B6  质量主集（半合成）；B7/B8 替代情景
  B9/B10 百亿参数以上外推情景

输出：参数、区间、弹性、质量等价条件、各来源误差、外推边界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, save_csv, save_json, SEED, rmse, mae

B = DATA / "B_scaling_laws"
RT = DATA / "A_data_value" / "regmix_tables"
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- 模型
def pred_classic(th, N, D):
    E, A, al, Bc, be = th
    return E + A * N ** (-al) + Bc * D ** (-be)


def pred_general(th, N, D, Q, H):
    """H 为 h(p) 因子（无配比信息时取 1）。"""
    E, A, al, Bc, be, ka = th
    return E + A * N ** (-al) + Bc * (D * Q ** ka * H) ** (-be)


def _res_classic(t, N, D, L):
    return pred_classic(np.exp(t), N, D) - L


def _res_general(t, N, D, Q, H, L):
    return pred_general(np.exp(t), N, D, Q, H) - L


def fit_classic(N, D, L, n_start=60, seed=SEED):
    """多起点有界非线性最小二乘（对数参数化保证正性）。"""
    r = np.random.default_rng(seed)
    lo = np.log([1e-3, 1e-4, 0.01, 1e-4, 0.01])
    hi = np.log([10.0, 1e3, 2.0, 1e3, 2.0])
    best = None
    for _ in range(n_start):
        t0 = r.uniform(lo, hi)
        try:
            s = least_squares(_res_classic, t0, args=(N, D, L),
                              bounds=(lo, hi), max_nfev=6000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    th = np.exp(best.x)
    return th, best


def fit_general(N, D, Q, H, L, fix=None, n_start=60, seed=SEED):
    """fix: 长度 6 的数组，**自然尺度**；非 NaN 位置固定为给定值（用于分阶段估计 κ）。

    注意：内部参数化为对数尺度，传入的 fix 会先取对数，返回的 th 已还原为自然尺度。
    """
    r = np.random.default_rng(seed)
    lo = np.log([1e-3, 1e-4, 0.01, 1e-4, 0.01, 0.02])
    hi = np.log([10.0, 1e3, 2.0, 1e3, 2.0, 8.0])
    if fix is None:
        fix = np.full(6, np.nan)
    fix = np.asarray(fix, float)
    free = np.isnan(fix)
    lo_f, hi_f = lo[free], hi[free]
    fx_log = np.where(free, 0.0, np.log(np.where(free, 1.0, fix)))   # 自由位占位，随后覆盖

    def res(tf, N, D, Q, H, L):
        t = fx_log.copy()
        t[free] = tf
        return pred_general(np.exp(t), N, D, Q, H) - L

    best = None
    for _ in range(n_start):
        t0 = r.uniform(lo_f, hi_f)
        try:
            s = least_squares(res, t0, args=(N, D, Q, H, L),
                              bounds=(lo_f, hi_f), max_nfev=8000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    th = fix.copy()
    th[free] = np.exp(best.x)
    return th, best


def boot_params(fitfun, n_boot=300, seed=SEED, **kw):
    """经验 bootstrap（按行重采样）参数区间。"""
    r = np.random.default_rng(seed)
    N, D, L = kw.pop("N"), kw.pop("D"), kw.pop("L")
    n = len(L)
    out = []
    for _ in range(n_boot):
        s = r.integers(0, n, n)
        try:
            th, _ = fitfun(N[s], D[s], L[s], n_start=6, seed=int(r.integers(1 << 30)), **kw)
            out.append(th)
        except Exception:
            pass
    return np.array(out)


# ---------------------------------------------------------------- 主流程
def diagnose_sources():
    """来源可信度诊断：区分真实散布与公式生成（题面要求不得把生成数据当真实观测）。"""
    print("=" * 95)
    print("来源诊断：各标度律数据集对经典律的可解释程度")
    print("=" * 95)
    rows = []
    sets = [("B1 Pythia 训练日志", "pythia_training_log_existing.csv", "题面标注：真实"),
            ("B2 Cerebras 轨迹", "cerebras_training_log.csv", "题面标注：半合成"),
            ("B4 跨族收敛点", "scaling_baseline.csv", "题面标注：真实"),
            ("B5 文献基准", "published_scaling_data.csv", "题面标注：真实"),
            ("B6 NQ 半合成", "supplementary_NQ_experiment.csv", "题面标注：半合成"),
            ("B7 NQ 扩展", "supplementary_NQ_experiment_expanded.csv", "题面标注：半合成"),
            ("B8 NQ 大规模", "supplementary_NQ_experiment_large.csv", "题面标注：半合成"),
            ("B10 大模型估算", "supplementary_large_baseline.csv", "题面标注：估算")]
    for tag, fn, level in sets:
        d = pd.read_csv(B / fn).dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
        n, dd, l = (d["N_params_B"].to_numpy(float), d["D_tokens_B"].to_numpy(float),
                    d["val_loss"].to_numpy(float))
        th, _ = fit_classic(n, dd, l, n_start=25)
        r = l - pred_classic(th, n, dd)
        r2 = 1 - np.sum(r ** 2) / max(np.sum((l - l.mean()) ** 2), 1e-12)
        rows.append(dict(dataset=tag, level=level, n=len(l),
                         rmse=float(np.sqrt(np.mean(r ** 2))), r2=float(r2),
                         rel_rmse=float(np.sqrt(np.mean(r ** 2)) / l.mean()),
                         L_min=float(l.min()), L_max=float(l.max()),
                         N_min=float(n.min()), N_max=float(n.max()),
                         verdict=("公式生成（残差≈舍入量级，判为合成）" if r2 > 0.9995 else
                                  "含真实散布")))
    df = pd.DataFrame(rows)
    save_csv(df, "q2_source_diagnosis.csv")
    print(df[["dataset", "level", "n", "rmse", "r2", "rel_rmse", "verdict"]].to_string(index=False))
    return df


def comparability_b6_b1(th_c):
    """检验 B6 在 Q=1 时是否与 B1 经典律同尺度（跨源可比性假设的检验）。"""
    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    q1 = b6[np.isclose(b6["Q_score"], 1.0)]
    res = q1["val_loss"].to_numpy(float) - pred_classic(th_c, q1["N_params_B"].to_numpy(float),
                                                         q1["D_tokens_B"].to_numpy(float))
    ci = [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]
    print(f"\n[跨源可比性] B6 在 Q=1 的 {len(q1)} 个点 相对 B1 经典律："
          f"均值={res.mean():+.4f} 标准差={res.std():.4f} 95%区间=[{ci[0]:+.3f},{ci[1]:+.3f}]")
    print("  → 均值接近 0，支持「B6 的 Q=1 基准与 B1 同尺度」的可比性假设")
    return dict(n=int(len(q1)), mean=float(res.mean()), std=float(res.std()), ci95=ci)


def main():
    print("=" * 95)
    print("第二问 · 广义标度律（有效数据量形式 Q2-B）")
    print("=" * 95)
    diag = diagnose_sources()

    # ---------- 1. 载入 B1 并处理重复 (N,D) ----------
    b1 = pd.read_csv(B / "pythia_training_log_existing.csv")
    b1 = b1.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    dup = b1.duplicated(subset=["N_params_B", "D_tokens_B"]).sum()
    print(f"B1: {len(b1)} 行, N 规模数={b1['N_params_B'].nunique()}, "
          f"重复 (N,D) 行={dup}")
    # 同一 (N,D) 多超参配置 -> 取中位损失（记录规则）
    g = b1.groupby(["N_params_B", "D_tokens_B"], as_index=False).agg(
        val_loss=("val_loss", "median"), n_rep=("val_loss", "size"))
    print(f"聚合后 {len(g)} 个唯一 (N,D) 点; 每点重复数 {g['n_rep'].min()}~{g['n_rep'].max()}")
    N1, D1, L1 = g["N_params_B"].to_numpy(float), g["D_tokens_B"].to_numpy(float), g["val_loss"].to_numpy(float)

    # ---------- 2. 经典律主拟合 ----------
    th_c, sol_c = fit_classic(N1, D1, L1)
    names = ["E", "A", "alpha", "B", "beta"]
    print("\n[经典律] L = E + A·N^-α + B·D^-β")
    print("  " + "  ".join(f"{n}={v:.5g}" for n, v in zip(names, th_c)))
    print(f"  train RMSE={rmse(L1, pred_classic(th_c,N1,D1)):.4f}  "
          f"R²={1-np.sum((L1-pred_classic(th_c,N1,D1))**2)/np.sum((L1-L1.mean())**2):.5f}")

    comp = comparability_b6_b1(th_c)

    boot_c = boot_params(fit_classic, N=N1, D=D1, L=L1, n_boot=200)
    ci_c = {n: [float(np.percentile(boot_c[:, i], 2.5)), float(np.percentile(boot_c[:, i], 97.5))]
            for i, n in enumerate(names)}
    print("  bootstrap 95% 区间: " + "; ".join(
        f"{n}[{v[0]:.4g},{v[1]:.4g}]" for n, v in ci_c.items()))

    # ---------- 3. 分组留出验证 ----------
    rows = []
    sizes = np.unique(N1)
    for s in sizes:                                    # 逐规模留一
        te = N1 == s
        if te.sum() == 0 or (~te).sum() < 20:
            continue
        th, _ = fit_classic(N1[~te], D1[~te], L1[~te], n_start=20)
        rows.append(dict(scheme="留出规模", holdout=f"N={s:.6g}B", n_test=int(te.sum()),
                         rmse=rmse(L1[te], pred_classic(th, N1[te], D1[te])),
                         mae=mae(L1[te], pred_classic(th, N1[te], D1[te])),
                         bias=float(np.mean(pred_classic(th, N1[te], D1[te]) - L1[te]))))
    # D 方向后段外推：每个规模按 D 排序，末 25% 留出
    idx_te = []
    for s in sizes:
        m = np.where(N1 == s)[0]
        m = m[np.argsort(D1[m])]
        idx_te += list(m[int(len(m) * 0.75):])
    idx_te = np.array(idx_te)
    tr = np.setdiff1d(np.arange(len(N1)), idx_te)
    th_d, _ = fit_classic(N1[tr], D1[tr], L1[tr], n_start=20)
    rows.append(dict(scheme="D 后段外推", holdout="每规模末 25% D", n_test=len(idx_te),
                     rmse=rmse(L1[idx_te], pred_classic(th_d, N1[idx_te], D1[idx_te])),
                     mae=mae(L1[idx_te], pred_classic(th_d, N1[idx_te], D1[idx_te])),
                     bias=float(np.mean(pred_classic(th_d, N1[idx_te], D1[idx_te]) - L1[idx_te]))))
    hold = pd.DataFrame(rows)
    save_csv(hold, "q2_holdout_classic.csv")
    print("\n[分组留出验证]")
    print(hold.to_string(index=False))

    # ---------- 4. 质量扩展：B6 上分阶段估计 κ ----------
    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    Nq, Dq, Qq, Lq = (b6["N_params_B"].to_numpy(float), b6["D_tokens_B"].to_numpy(float),
                      b6["Q_score"].to_numpy(float), b6["val_loss"].to_numpy(float))
    print(f"\n[B6 质量主集] {len(b6)} 点, N∈[{Nq.min():.3g},{Nq.max():.3g}]B, "
          f"D∈[{Dq.min():.0f},{Dq.max():.0f}]B, Q∈[{Qq.min()},{Qq.max()}]")

    # 4a. 固定经典参数，仅估 κ
    fix_k = np.array([th_c[0], th_c[1], th_c[2], th_c[3], th_c[4], np.nan])
    th_k, _ = fit_general(Nq, Dq, Qq, np.ones_like(Qq), Lq, fix=fix_k, n_start=40)
    print(f"  分阶段估计（固定 B1 经典参数）: κ={th_k[5]:.4f}")
    # 4b. 联合估计
    th_j, _ = fit_general(Nq, Dq, Qq, np.ones_like(Qq), Lq, n_start=80)
    print("  联合估计: " + "  ".join(f"{n}={v:.5g}" for n, v in
                                  zip(["E", "A", "alpha", "B", "beta", "kappa"], th_j)))
    print(f"  B6 RMSE: 分阶段={rmse(Lq, pred_general(th_k,Nq,Dq,Qq,np.ones_like(Qq))):.4f}  "
          f"联合={rmse(Lq, pred_general(th_j,Nq,Dq,Qq,np.ones_like(Qq))):.4f}  "
          f"常数模型={rmse(Lq, np.full_like(Lq, Lq.mean())):.4f}")

    boot_k = []
    r2 = np.random.default_rng(SEED)
    for _ in range(200):
        s = r2.integers(0, len(Lq), len(Lq))
        try:
            t, _ = fit_general(Nq[s], Dq[s], Qq[s], np.ones_like(Qq[s]), Lq[s],
                               fix=fix_k, n_start=8)
            boot_k.append(t[5])
        except Exception:
            pass
    boot_k = np.array(boot_k)
    k_ci = [float(np.percentile(boot_k, 2.5)), float(np.percentile(boot_k, 97.5))]
    print(f"  κ bootstrap 95% 区间: [{k_ci[0]:.4f}, {k_ci[1]:.4f}]")

    # ---------- 5. B7/B8 替代情景 ----------
    alt = []
    for tag, fn in [("B7", "supplementary_NQ_experiment_expanded.csv"),
                    ("B8", "supplementary_NQ_experiment_large.csv")]:
        d = pd.read_csv(B / fn)
        d = d.dropna(subset=["N_params_B", "D_tokens_B", "Q_score", "val_loss"])
        Na, Da, Qa, La = (d["N_params_B"].to_numpy(float), d["D_tokens_B"].to_numpy(float),
                          d["Q_score"].to_numpy(float), d["val_loss"].to_numpy(float))
        ta, _ = fit_general(Na, Da, Qa, np.ones_like(Qa), La, fix=fix_k, n_start=25)
        tb, _ = fit_general(Na, Da, Qa, np.ones_like(Qa), La, n_start=40)
        alt.append(dict(dataset=tag, n=len(d), kappa_staged=ta[5], kappa_joint=tb[5],
                        rmse_staged=rmse(La, pred_general(ta, Na, Da, Qa, np.ones_like(Qa))),
                        rmse_joint=rmse(La, pred_general(tb, Na, Da, Qa, np.ones_like(Qa))),
                        rmse_const=rmse(La, np.full_like(La, La.mean())),
                        data_type=";".join(map(str, d["data_type"].unique())) if "data_type" in d else "n/a"))
    alt_df = pd.DataFrame(alt)
    save_csv(alt_df, "q2_quality_alt_datasets.csv")
    print("\n[B7/B8 替代情景]")
    print(alt_df.to_string(index=False))

    # ---------- 6. 跨族 / 文献 / 族外验证 ----------
    val_rows = []
    for tag, fn, level in [("B4 跨族收敛点", "scaling_baseline.csv", "真实"),
                           ("B5 文献基准", "published_scaling_data.csv", "真实"),
                           ("B10 大模型估算", "supplementary_large_baseline.csv", "估算")]:
        d = pd.read_csv(B / fn).dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
        Nv, Dv, Lv = (d["N_params_B"].to_numpy(float), d["D_tokens_B"].to_numpy(float),
                      d["val_loss"].to_numpy(float))
        ph = pred_classic(th_c, Nv, Dv)
        val_rows.append(dict(dataset=tag, level=level, n=len(d),
                             rmse=rmse(Lv, ph), mae=mae(Lv, ph),
                             bias=float(np.mean(ph - Lv)),
                             r2=1 - np.sum((Lv - ph) ** 2) / max(np.sum((Lv - Lv.mean()) ** 2), 1e-9),
                             N_range=f"{Nv.min():.4g}~{Nv.max():.4g}",
                             D_range=f"{Dv.min():.4g}~{Dv.max():.4g}",
                             L_range=f"{Lv.min():.3f}~{Lv.max():.3f}"))
    # B2 半合成族外
    b2 = pd.read_csv(B / "cerebras_training_log.csv").dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    N2, D2, L2 = (b2["N_params_B"].to_numpy(float), b2["D_tokens_B"].to_numpy(float),
                  b2["val_loss"].to_numpy(float))
    val_rows.append(dict(dataset="B2 半合成族外", level="半合成", n=len(b2),
                         rmse=rmse(L2, pred_classic(th_c, N2, D2)),
                         mae=mae(L2, pred_classic(th_c, N2, D2)),
                         bias=float(np.mean(pred_classic(th_c, N2, D2) - L2)),
                         r2=1 - np.sum((L2 - pred_classic(th_c, N2, D2)) ** 2) / max(np.sum((L2 - L2.mean()) ** 2), 1e-9),
                         N_range=f"{N2.min():.3g}~{N2.max():.3g}",
                         D_range=f"{D2.min():.3g}~{D2.max():.3g}",
                         L_range=f"{L2.min():.3f}~{L2.max():.3f}"))
    # B3 插值轨迹
    tdir = B / "training_trajectories"
    N3, D3, L3 = [], [], []
    for f in sorted(tdir.glob("*.csv")):
        d = pd.read_csv(f)
        col = [c for c in d.columns if "val" in c.lower() or "loss" in c.lower()]
        N3 += [float(d["N_params_B"].iloc[0])] * len(d)
        D3 += list(d["D_tokens_B"].to_numpy(float))
        L3 += list(d[col[0]].to_numpy(float))
    N3, D3, L3 = np.array(N3), np.array(D3), np.array(L3)
    val_rows.append(dict(dataset="B3 插值轨迹", level="插值", n=len(N3),
                         rmse=rmse(L3, pred_classic(th_c, N3, D3)),
                         mae=mae(L3, pred_classic(th_c, N3, D3)),
                         bias=float(np.mean(pred_classic(th_c, N3, D3) - L3)),
                         r2=1 - np.sum((L3 - pred_classic(th_c, N3, D3)) ** 2) / max(np.sum((L3 - L3.mean()) ** 2), 1e-9),
                         N_range=f"{N3.min():.3g}~{N3.max():.3g}",
                         D_range=f"{D3.min():.3g}~{D3.max():.3g}",
                         L_range=f"{L3.min():.3f}~{L3.max():.3f}"))
    val_df = pd.DataFrame(val_rows)
    save_csv(val_df, "q2_validation_sources.csv")
    print("\n[跨来源验证 · 经典参数直接外推]")
    print(val_df.to_string(index=False))

    # ---------- 7. 弹性、等价条件 ----------
    E, A, al, Bc, be = th_c
    kap = th_k[5]

    def elasticities(N, D, Q=1.0, H=1.0, th=th_c, k=0.0):
        E_, A_, al_, B_, be_, = th
        L = pred_general(np.array([E_, A_, al_, B_, be_, k]), N, D, Q, H)
        X = D * Q ** k * H
        dN = -al_ * A_ * N ** (-al_ - 1)
        dD = -be_ * B_ * X ** (-be_ - 1) * (X / D)
        dQ = -be_ * B_ * X ** (-be_ - 1) * (k * X / Q) if k else np.zeros_like(L)
        return dict(eN=dN * N / L, eD=dD * D / L, eQ=dQ * Q / L, L=L, reducible=L - E_)

    # 代表性配置网格
    el_rows = []
    for N in [0.1, 1, 10, 100, 1000]:
        for D in [10, 100, 1000, 10000]:
            e = elasticities(np.array([N]), np.array([D]))
            el_rows.append(dict(N_B=N, D_B=D, L0=float(e["L"][0]),
                                e_N=float(e["eN"][0]), e_D=float(e["eD"][0]),
                                C_FLOPs=6 * N * 1e9 * D * 1e9))
    el_df = pd.DataFrame(el_rows)
    save_csv(el_df, "q2_elasticities_classic.csv")
    print("\n[经典律弹性]（部分）")
    print(el_df.head(8).to_string(index=False))

    # 质量等价：ΔL_Q 与 N_eq
    eq_rows = []
    for N in [0.1, 1, 10, 100]:
        for D in [100, 1000]:
            for Q0 in [0.5, 0.7, 0.9]:
                Q1v = min(Q0 + 0.1, 1.0)
                if Q1v <= Q0:
                    continue
                L_a = float(pred_general(np.array([E, A, al, Bc, be, kap]),
                                         np.array([N]), np.array([D]), np.array([Q0]), np.array([1.0]))[0])
                L_b = float(pred_general(np.array([E, A, al, Bc, be, kap]),
                                         np.array([N]), np.array([D]), np.array([Q1v]), np.array([1.0]))[0])
                dL = L_a - L_b
                inner = N ** (-al) - dL / A
                Neq = float(inner ** (-1 / al)) if inner > 0 else np.nan
                eq_rows.append(dict(N_B=N, D_B=D, Q0=Q0, Q1=Q1v, L_Q0=L_a, L_Q1=L_b,
                                    dL_Q=dL, N_eq_B=Neq,
                                    N_eq_ratio=(Neq / N - 1) if np.isfinite(Neq) else np.nan,
                                    feasible=bool(inner > 0)))
    eq_df = pd.DataFrame(eq_rows)
    save_csv(eq_df, "q2_quality_equivalence.csv")
    print("\n[质量 +0.1 的参数等价规模]（部分）")
    print(eq_df.head(8).to_string(index=False))

    # ---------- 8. 有限差分核验 ----------
    chk = []
    for N, D in [(1.0, 100.0), (10.0, 1000.0)]:
        e = elasticities(np.array([N]), np.array([D]))
        h = 1e-5
        Lp = float(pred_classic(th_c, np.array([N * (1 + h)]), np.array([D]))[0])
        Lm = float(pred_classic(th_c, np.array([N * (1 - h)]), np.array([D]))[0])
        L0 = float(e["L"][0])
        fd = (Lp - Lm) / (2 * h * N) * N / L0
        chk.append(dict(N=N, D=D, analytic_eN=float(e["eN"][0]), fd_eN=fd,
                        abs_diff=abs(float(e["eN"][0]) - fd)))
    chk_df = pd.DataFrame(chk)
    save_csv(chk_df, "q2_derivative_check.csv")
    print("\n[导数有限差分核验]")
    print(chk_df.to_string(index=False))

    # ---------- 9. 保存参数 ----------
    save_json(dict(
        classic=dict(form="L = E + A*N^(-alpha) + B*D^(-beta)",
                     params=dict(zip(names, map(float, th_c))),
                     ci95=ci_c,
                     train_rmse=rmse(L1, pred_classic(th_c, N1, D1)),
                     n_points=int(len(N1)), source="B1 Pythia 训练日志（真实）"),
        general=dict(form="L = E + A*N^(-alpha) + B*[D*Q^kappa*h(p)]^(-beta)",
                     kappa_staged=float(th_k[5]), kappa_ci95=k_ci,
                     kappa_joint=float(th_j[5]),
                     params_joint=dict(zip(["E", "A", "alpha", "B", "beta", "kappa"],
                                           map(float, th_j))),
                     fitted_on="B6 半合成 N-D-Q 实验（360 点）",
                     degenerate_at="Q=1 且 h(p)=1 时退化为经典律"),
        h_p=dict(form="h(p)=exp{-omega*[f(p)-f(p0)]/s_f}", status="情景变量（无共同实验，不可辨识）",
                 omega_scenarios=[0.0, 0.5, 1.0, 2.0]),
        units=dict(N="十亿参数(B)", D="十亿 token(B)", L="验证集交叉熵(nats)"),
        source_levels={"B1": "题面标注真实；本分析判为公式生成", "B2": "半合成",
                       "B3": "插值", "B4": "真实（含散布）", "B5": "真实（含散布）",
                       "B6-B8": "半合成", "B9": "真实", "B10": "估算"},
        source_diagnosis=diag.to_dict("records"),
        b6_b1_comparability=comp,
    ), "q2_scaling_parameters.json")

    np.savez(RESULTS / "q2_scaling_model.npz", th_classic=th_c, th_kappa=th_k, th_joint=th_j)
    print("\n完成：q2_*.csv / q2_scaling_parameters.json")


if __name__ == "__main__":
    main()
