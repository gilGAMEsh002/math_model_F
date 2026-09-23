# -*- coding: utf-8 -*-
"""Q1-C 事后诊断：解释「结果为何如此」，并检验结论对口径与扰动的稳健性。

本模块回答三个报告 §6.3.1 / §6.3.2 / §6.5 提出的问题，全部数字由 run_all.py 可复现：

1. 森林的收益出现在哪些域、为什么  → 逐目标线性可解释度 vs 森林-岭回归差距（含偏相关）
2. 「森林优于岭回归」的幅度是否依赖复合口径 → 三种复合口径对照
3. 配比扰动在输入空间的真实量级，以及扰动后「最优配比」是否稳定

设计原则：只读冻结产物与已拟合模型，不重新选参、不接触最终测试集的选参环节。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import load_pairs, r2, rmse, spearman
from mixture import normalize_rows, predict_models, short_target


# ------------------------------------------------------------------ 工具
def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()


def partial_spearman(a, b, control) -> float:
    """控制 control 后排秩相关（三者的秩上做线性残差化）。"""
    ra, rb, rc = _rank(a), _rank(b), _rank(control)
    ra = ra - ra.mean(); rb = rb - rb.mean(); rc = rc - rc.mean()
    res_a = ra - (ra @ rc) / (rc @ rc) * rc
    res_b = rb - (rb @ rc) / (rc @ rc) * rc
    den = np.sqrt((res_a @ res_a) * (res_b @ res_b))
    return float(res_a @ res_b / den) if den > 0 else float("nan")


def dirichlet_draws(X: np.ndarray, conc: float, n_draws: int, rng) -> np.ndarray:
    """以每行配比自身为均值构造 Dirichlet 扰动（与 perturbation_robustness 同构）。"""
    draws = [rng.dirichlet(np.maximum(X[i] * float(conc), 1e-6), size=n_draws)
             for i in range(X.shape[0])]
    return np.vstack(draws).reshape(X.shape[0], n_draws, X.shape[1])


# ------------------------------------------------- 1. 逐目标线性度 vs 森林收益
def target_linearity(cfg: dict, mix: dict, pairs: dict) -> pd.DataFrame:
    """逐目标：线性可解释度（训练/留出）、真值标准差、两模型 RMSE 与差距。"""
    Xtr = normalize_rows(pairs["train_1m"]["X"].to_numpy(dtype=float))
    Ytr = pairs["train_1m"]["Y"].to_numpy(dtype=float)
    Xte = normalize_rows(pairs["same_scale_1m"]["X"].to_numpy(dtype=float))
    Yte = pairs["same_scale_1m"]["Y"].to_numpy(dtype=float)
    A = np.column_stack([Xtr, np.ones(len(Xtr))])
    B = np.column_stack([Xte, np.ones(len(Xte))])

    met = mix["absolute"]
    met = met[met["target"] != "__composite_equal_v__"]
    pv = met.pivot_table(index="target", columns="variant", values="rmse")

    rows = []
    for j, col in enumerate(cfg["mixture"]["target_cols"]):
        s = short_target(col)
        coef, *_ = np.linalg.lstsq(A, Ytr[:, j], rcond=None)
        rows.append({
            "domain": s,
            "linear_r2_train": r2(Ytr[:, j], A @ coef),
            "linear_r2_holdout": r2(Yte[:, j], B @ coef),
            "y_sd_train": float(Ytr[:, j].std(ddof=1)),
            "rmse_ridge": float(pv.loc[s, "ridge_none"]),
            "rmse_forest": float(pv.loc[s, "forest_none"]),
            "gap_forest_minus_ridge": float(pv.loc[s, "forest_none"] - pv.loc[s, "ridge_none"]),
        })
    df = pd.DataFrame(rows).sort_values("linear_r2_holdout").reset_index(drop=True)
    df["forest_wins"] = df["gap_forest_minus_ridge"] < 0
    return df


# --------------------------------------------- 2. 复合口径敏感性
def composite_sensitivity(cfg: dict, mix: dict, pairs: dict) -> pd.DataFrame:
    """三种都合理的复合口径下，各变体的综合误差与相对排名。

    口径 A：逐行先把 13 目标求均值 → 再算 RMSE（配置中的评价目标）
    口径 B：逐目标 RMSE 的等权平均
    口径 C：逐目标 RMSE 除以该目标真值标准差后再平均（去量纲）

    返回的 DataFrame 以 variant 为**列**（save_table 默认 index=False，留在索引里会被丢弃）。
    """
    Yte = pairs["same_scale_1m"]["Y"].to_numpy(dtype=float)
    met = mix["absolute"]
    pv = met[met["target"] != "__composite_equal_v__"].pivot_table(
        index="target", columns="variant", values="rmse")
    comp = met[met["target"] == "__composite_equal_v__"].set_index("variant")["rmse"]
    sds = pd.Series({short_target(c): Yte[:, j].std(ddof=1)
                     for j, c in enumerate(cfg["mixture"]["target_cols"])})

    out = pd.DataFrame({
        "composite_loss_rmse": comp,                       # 口径 A
        "mean_target_rmse": pv.mean(axis=0),               # 口径 B
        "sd_normalized_rmse": pv.div(sds, axis=0).mean(axis=0),  # 口径 C
    }).dropna(how="all").reset_index().rename(columns={"index": "variant"})
    for c in COMPOSITE_COLS:
        out[c + "_rank"] = out[c].rank().astype(int)
    return out


COMPOSITE_COLS = ["composite_loss_rmse", "mean_target_rmse", "sd_normalized_rmse"]


# --------------------------------------------- 3. 扰动标定 + 选择稳定性
def perturbation_diagnostics(cfg: dict, mix: dict, pairs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(输入位移标定表, 扰动后选择稳定性表)。

    输入空间位移让「浓度」可读；选择稳定性直接检验「最优配比」这一结论是否稳健。
    """
    model = mix["variants"]["forest_none"]["model"]
    pert = cfg["extrapolation"]["perturbation"]
    n_draws = int(pert["n_draws"])
    rng = np.random.default_rng(cfg["seed"])

    calib, stab = [], []
    for name in [s["name"] for s in cfg["splits"]["final_test"]] + \
                [s["name"] for s in cfg["splits"]["extrapolation_diagnostic"]]:
        X = normalize_rows(pairs[name]["X"].to_numpy(dtype=float))
        Y = pairs[name]["Y"].to_numpy(dtype=float)
        yc = Y.mean(axis=1)
        width = float(yc.max() - yc.min())
        base_full = predict_models(model, X)                 # (n_rows, n_targets)
        base_c = base_full.mean(axis=1)                      # 综合预测
        pick0 = int(np.argmin(base_c))
        is_extrap = name.startswith("est_")

        for conc in pert["concentrations"]:
            D = dirichlet_draws(X, conc, n_draws, rng)
            dp = np.abs(D - X[:, None, :])
            # 逐目标的预测位移 P (n_rows, n_draws, n_targets)；综合位移为其在目标维上的均值
            P = predict_models(model, D.reshape(-1, X.shape[1]))
            P = P.reshape(X.shape[0], n_draws, -1)
            shift = np.abs(P - base_full[:, None, :])        # 逐目标
            shift_c = np.abs(P.mean(axis=2) - base_c[:, None])   # 综合
            picks = [int(np.argmin(P[:, k, :].mean(axis=1))) for k in range(n_draws)]

            # 逐目标位移必须除以该目标自身的真值宽度，否则与综合宽度不同尺度；
            # 综合位移除以综合宽度，才与「能否分辨该尺度上的目标」直接对应。
            tw = Y.max(axis=0) - Y.min(axis=0)
            per_t = (shift.mean(axis=1) / np.maximum(tw, 1e-12)[None, :]).mean()
            calib.append({
                "dataset": name, "is_extrapolation": is_extrap, "concentration": float(conc),
                "n_draws": n_draws,
                "mean_abs_dp_per_dim": float(dp.mean()),
                "mean_l1_dp": float(dp.sum(axis=2).mean()),
                "mean_l2_dp": float(np.sqrt((dp ** 2).sum(axis=2)).mean()),
                "mean_abs_pred_shift_per_target": float(shift.mean()),
                "composite_pred_shift_mean": float(shift_c.mean()),
                "composite_pred_shift_p90": float(np.percentile(shift_c, 90)),
                "composite_pred_shift_max": float(shift_c.max()),
                "composite_true_width": width,
                "composite_shift_over_width": float(shift_c.mean() / width) if width > 0 else float("nan"),
                "composite_p90_over_width": float(np.percentile(shift_c, 90) / width)
                if width > 0 else float("nan"),
                "pertarget_shift_over_width": float(per_t),
            })
            stab.append({
                "dataset": name, "is_extrapolation": is_extrap, "concentration": float(conc),
                "unperturbed_pick": pick0,
                "true_best_row": int(np.argmin(yc)),
                "unperturbed_pick_is_true_best": bool(pick0 == int(np.argmin(yc))),
                "selection_unchanged_rate": float(np.mean(np.asarray(picks) == pick0)),
                "n_distinct_picks": int(len(set(picks))),
            })
    return pd.DataFrame(calib), pd.DataFrame(stab)


# ------------------------------------------------------------------ 入口
def run_diagnostics(cfg: dict, mix: dict, log=print) -> dict:
    pairs = load_pairs(cfg)
    lin = target_linearity(cfg, mix, pairs)
    comp = composite_sensitivity(cfg, mix, pairs)
    calib, stab = perturbation_diagnostics(cfg, mix, pairs)

    g = lin["gap_forest_minus_ridge"].to_numpy()
    log(f"[diag] 森林胜出 {int(lin['forest_wins'].sum())}/{len(lin)} 个目标；"
        f"Spearman(线性留出R², 差距)={spearman(lin['linear_r2_holdout'].to_numpy(), g):+.4f}  "
        f"偏Spearman(控制 y_sd)={partial_spearman(lin['linear_r2_holdout'].to_numpy(), g, lin['y_sd_train'].to_numpy()):+.4f}")
    c = comp.set_index("variant")
    adv = {k: 100.0 * (1.0 - c.loc["forest_none", k] / c.loc["ridge_none", k])
           for k in COMPOSITE_COLS}
    log("[diag] 森林优势（口径 A 综合Loss / B 逐目标均值 / C 去量纲）= "
        + " / ".join(f"{adv[k]:.1f}%" for k in COMPOSITE_COLS))
    log("[diag] 偏Spearman(线性R², 差距 | y_sd)="
        f"{partial_spearman(lin['linear_r2_holdout'].to_numpy(), g, lin['y_sd_train'].to_numpy()):+.4f}  "
        f"偏Spearman(y_sd, 差距 | 线性R²)="
        f"{partial_spearman(lin['y_sd_train'].to_numpy(), g, lin['linear_r2_holdout'].to_numpy()):+.4f}")
    for _, r in stab.iterrows():
        if r["concentration"] == cfg["extrapolation"]["perturbation"]["concentrations"][-1]:
            log(f"[diag] {r['dataset']}: 未扰动选择={r['unperturbed_pick']} "
                f"真值最优={r['true_best_row']} 命中={r['unperturbed_pick_is_true_best']} "
                f"最强扰动下保持={r['selection_unchanged_rate']:.1%}")
    return {"linearity": lin, "composite": comp, "calibration": calib, "stability": stab}
