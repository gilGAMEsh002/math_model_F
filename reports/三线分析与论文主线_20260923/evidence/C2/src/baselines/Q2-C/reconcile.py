# -*- coding: utf-8 -*-
"""质量数据集之间的一致性核验（Q2-C.md 第 4 条：保留失败证据）。

B6/B7/B8 三套"带质量标签"的数据集在同一个模型族下互相矛盾，本模块把这一
**负结果**做成可复现的流程内证据，而不是只在报告里叙述：

    A. 逐集独立拟合：不冻结任何参数，各自拟合 M2（7 参）与 M0（经典 5 参）。
       若某集上 M2 与 M0 的 RMSE 完全相同且 κ、ρ 落在下界 0，说明该集对质量
       通道"零信息"——经典律已经是它的最优描述。
    B. 方向检验：把 Q 做单调递减置换（Q → 2−Q，保持正值）后重拟合。若反向
       映射能大幅改善拟合，则原 Q 可能是反向刻度；若反向只是把 κ、ρ 顶到上界
       而误差几乎不降，说明两种取向都解释不了该集。
    C. 残差—质量相关性：以真实数据（B1）拟合出的经典律为基准，算各质量集
       残差与 Q 的相关系数及 Q 五分位残差均值，直接看单调方向是否一致。

判定口径与公共协议一致：半合成集内部的可辨识**不等于**真实新定律；只要真实
观测（B1）没有残差方差可供质量通道解释，且另一套质量集给出相反结论，主结论
就必须退回 Q2-B（ρ=0），κ、ρ 只作情景参数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import bias, mae, rmse
from scaling import fit, predict

QUALITY_SETS = ["b6_quality_main", "b7_quality_expanded", "b8_quality_large"]
REVERSE_MAP = ("reverse_2_minus_Q", lambda q: 2.0 - np.asarray(q, float))


def _subsets(d: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """B8 自带 data_type 分段，分开核验；其余整表作为一个子集。"""
    if "data_type" in d.columns:
        out = []
        for g, sub in d.groupby("data_type", dropna=False):
            out.append((str(g), sub.reset_index(drop=True)))
        return out
    return [("all", d)]


def per_set_fit(cfg: dict, d: pd.DataFrame, bounds: dict, seed: int,
                n_starts: int) -> list[dict]:
    """逐子集独立拟合 M2（7 参）与 M0（经典 5 参），不作任何参数冻结。"""
    rows = []
    for tag, s in _subsets(d):
        if len(s) < 10:
            continue
        rec = {"label": tag, "n": int(len(s)),
               "N_min": float(s["N"].min()), "N_max": float(s["N"].max()),
               "Q_min": float(s["Q"].min()), "Q_max": float(s["Q"].max()),
               "L_std": float(s["L"].std())}
        for mdl in ("M2", "M0"):
            try:
                r = fit(s["N"], s["D"], s["Q"], s["L"], model=mdl, bounds=bounds,
                        n_starts=n_starts, seed=seed, max_nfev=30000)
                key = "M2" if mdl == "M2" else "M0"
                rec[f"{key}_rmse"] = r["rmse"]
                rec[f"{key}_E"] = float(r["x"][0])
                rec[f"{key}_alpha"] = float(r["x"][2])
                rec[f"{key}_beta"] = float(r["x"][4])
                if key == "M2":
                    rec["M2_kappa"] = float(r["x"][5])
                    rec["M2_rho"] = float(r["x"][6])
                    rec["M2_boundary_hits"] = str(r["boundary_hits"])
                    rec["M2_n_clusters"] = len(r["clusters"])
            except Exception as exc:  # pragma: no cover
                rec[f"{mdl}_rmse"] = np.nan
                rec[f"{mdl}_note"] = str(exc)
        # 关键判据：M2 相对 M0 的 RMSE 改善
        if np.isfinite(rec.get("M2_rmse", np.nan)) and np.isfinite(rec.get("M0_rmse", np.nan)):
            rec["rmse_gain_M0_to_M2"] = rec["M0_rmse"] - rec["M2_rmse"]
            rec["r2_vs_constant"] = 1.0 - (rec["M0_rmse"] / max(rec["L_std"], 1e-12)) ** 2
            k_zero = (abs(rec.get("M2_kappa", 1.0)) < 1e-3 and abs(rec.get("M2_rho", 1.0)) < 1e-3)
            rec["quality_channel_informative"] = bool(rec["rmse_gain_M0_to_M2"] > 1e-6
                                                      and not k_zero)
        rows.append(rec)
    return rows


def orientation_test(cfg: dict, d: pd.DataFrame, base_x: np.ndarray, bounds: dict,
                     seed: int, n_starts: int) -> list[dict]:
    """原 Q 与反向映射 (2-Q) 的对照，检验 Q 是否为反向刻度。"""
    rows = []
    for tag, s in _subsets(d):
        if len(s) < 10:
            continue
        for lab, fn in (("original_Q", lambda q: np.asarray(q, float)), REVERSE_MAP):
            Q = fn(s["Q"])
            try:
                r = fit(s["N"], s["D"], Q, s["L"], base=base_x,
                        free_only=["E", "kappa", "rho"], bounds=bounds,
                        n_starts=n_starts, seed=seed, max_nfev=30000)
                rows.append({"label": tag, "orientation": lab, "n": int(len(s)),
                             "E": float(r["x"][0]), "kappa": float(r["x"][5]),
                             "rho": float(r["x"][6]), "rmse": r["rmse"],
                             "boundary_hits": str(r["boundary_hits"])})
            except Exception as exc:  # pragma: no cover
                rows.append({"label": tag, "orientation": lab, "n": int(len(s)),
                             "kappa": np.nan, "rho": np.nan, "rmse": np.nan,
                             "boundary_hits": "", "note": str(exc)})
    return rows


def residual_quality_corr(d: pd.DataFrame, base_x: np.ndarray, q_ref: float,
                          n_bins: int = 5) -> list[dict]:
    """以经典律为基准的残差 与 Q 的相关性 + Q 分位残差均值（看单调方向）。"""
    rows = []
    for tag, s in _subsets(d):
        if len(s) < 10:
            continue
        base = predict(base_x, s["N"], s["D"], np.full(len(s), float(q_ref)))
        res = np.asarray(s["L"], float) - base
        cc = float(np.corrcoef(np.asarray(s["Q"], float), res)[0, 1]) if len(s) > 2 else np.nan
        try:
            bins = pd.qcut(np.asarray(s["Q"], float), n_bins, duplicates="drop")
            means = [float(v) for v in pd.Series(res).groupby(bins, observed=True).mean()]
        except Exception:
            means = []
        rows.append({"label": tag, "n": int(len(s)), "corr_Q_residual": cc,
                     "residual_mean": float(np.mean(res)),
                     "residual_std": float(np.std(res)),
                     "residual_min": float(np.min(res)), "residual_max": float(np.max(res)),
                     "quintile_residual_means_lowQ_to_highQ": means,
                     "monotone_direction": ("Q↑→残差↓（质量提升降损失）" if cc < -0.5 else
                                            "Q↑→残差↑（与质量假设相反）" if cc > 0.5 else
                                            "无明显单调关系")})
    return rows


def reconcile(cfg: dict, b1: pd.DataFrame, loaded: dict, base_x: np.ndarray, bounds: dict,
              q_ref: float, log=None) -> pd.DataFrame:
    """汇总三套质量集的一致性核验，输出长表（每行一个 子集×核验项）。"""
    seed = int(cfg["seed"])
    n_starts = int(cfg["model"]["multi_start"]["n_starts_stage2"])
    rows = []
    for key in QUALITY_SETS:
        if key not in loaded:
            continue
        d = loaded[key].copy()
        if "Q" not in d.columns:
            continue
        d = d[np.isfinite(pd.to_numeric(d["Q"], errors="coerce"))].reset_index(drop=True)
        for fam, recs in (("per_set_fit", per_set_fit(cfg, d, bounds, seed, n_starts)),
                          ("orientation_test", orientation_test(cfg, d, base_x, bounds,
                                                                seed, n_starts)),
                          ("residual_vs_quality", residual_quality_corr(d, base_x, q_ref))):
            for r in recs:
                rows.append({"source": key, "check": fam, **r})
                if log and fam == "per_set_fit":
                    log(f"  [核验 {key}/{r['label']}] n={r['n']} "
                        f"M0 RMSE={r.get('M0_rmse', float('nan')):.4f} "
                        f"M2 RMSE={r.get('M2_rmse', float('nan')):.4f} "
                        f"κ={r.get('M2_kappa', float('nan')):.4f} "
                        f"ρ={r.get('M2_rho', float('nan')):.4f} "
                        f"{'通道有信息' if r.get('quality_channel_informative') else '通道零信息'}")
                if log and fam == "residual_vs_quality":
                    log(f"  [核验 {key}/{r['label']}] corr(Q, 残差)={r['corr_Q_residual']:+.3f} "
                        f"→ {r['monotone_direction']}")
    return pd.DataFrame(rows)
