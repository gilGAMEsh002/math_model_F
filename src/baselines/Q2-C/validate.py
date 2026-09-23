# -*- coding: utf-8 -*-
"""Q2-C 验证（§2.2、§2.6）：分组留出、轨迹检验、跨源误差、分组 bootstrap、外推情景。

要点（公共协议）：
    - 固定步距检查点高度相关，不能随机拆行后把误差称为独立泛化误差 → 一律整组留出；
    - 真实观测 / 半合成 / 插值 / 估算 分层报告，不把表格样本数当独立实验数；
    - 区分「经验分组 bootstrap 区间」与「情景包络」，不与未来观测预测区间混称。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import bias, mae, r2, rmse
from scaling import BASE, FULL, MODEL_FREE, fit, predict

# 来源登记表。offset=1 的行属于"质量实验族"（与 B1 不同语料/评测族），加性 δ 只对
# 它们生效；offset=0 的族外来源不加 δ——它们的水平差如实落在残差偏差里，因为没有
# 任何机制保证 δ 可跨族外推。
# split 指定时按该列分组逐组报告：B8 自带 data_type(calibrated/extrapolated)，
# 混在一起会把外推段的水平差吸收进质量参数（实测会导致 κ=ρ=0 且 δ 大幅偏移）。
SOURCES = [
    {"id": "B1", "key": "b1_real_main", "kind": "real", "offset": 0.0, "split": None,
     "note": "真实观测（Pythia 训练日志）"},
    {"id": "B2", "key": "b2_family_out", "kind": "semi_synthetic", "offset": 0.0,
     "split": None, "note": "半合成（Cerebras；D 最大 2050B 远超 B1 的 300B）"},
    {"id": "B4", "key": "b4_cross_family", "kind": "mixed", "offset": 0.0, "split": None,
     "note": "跨族（12 个模型族，含与 B1 重叠的 Pythia）"},
    {"id": "B5", "key": "b5_published", "kind": "literature", "offset": 0.0, "split": "source",
     "note": "文献值（6 个来源，评测口径各异）"},
    {"id": "B6", "key": "b6_quality_main", "kind": "semi_synthetic_quality", "offset": 1.0,
     "split": None, "note": "半合成质量主集（δ 组）"},
    {"id": "B7", "key": "b7_quality_expanded", "kind": "semi_synthetic_quality", "offset": 1.0,
     "split": None, "note": "半合成质量扩展集（δ 组）"},
    {"id": "B8", "key": "b8_quality_large", "kind": "semi_synthetic_quality", "offset": 1.0,
     "split": "data_type",
     "note": "半合成质量大模型集（calibrated 与 extrapolated 分开，δ 组）"},
    {"id": "B10", "key": "b10_large_baseline", "kind": "estimated", "offset": 0.0,
     "split": None, "note": "估算 Loss（N=100–10000B，外推情景）"},
]


def prep_source(df: pd.DataFrame, cfg: dict, q_ref: float) -> pd.DataFrame:
    """统一 Q 与来源偏移组：无 Q 列按基准约定取 q_ref（§2.4 第 5 条）。"""
    d = df.copy()
    if "Q" not in d.columns or d["Q"].isna().all():
        d["Q"] = float(q_ref)
        d["Q_is_convention"] = True
    else:
        d["Q"] = d["Q"].fillna(float(q_ref))
        d["Q_is_convention"] = False
    for c in ("N", "D", "L"):
        d = d[np.isfinite(pd.to_numeric(d[c], errors="coerce"))]
    return d.reset_index(drop=True)


# --------------------------------------------------------------- 分组留出（B1）
def holdout_by_size(cfg: dict, df: pd.DataFrame, bounds: dict, log=None) -> pd.DataFrame:
    """按模型规模整组留出：每组用其余规模重拟合，预测被留出组。"""
    groups = sorted(df["N"].unique())
    rows = []
    for g in groups:
        tr = df[df["N"] != g]
        te = df[df["N"] == g]
        try:
            res = fit(tr["N"], tr["D"], tr["Q"], tr["L"], model="M0", bounds=bounds,
                      n_starts=24, seed=cfg["seed"], max_nfev=8000)
        except Exception as exc:  # pragma: no cover
            rows.append({"held_out_N": float(g), "n_test": int(len(te)),
                         "rmse": np.nan, "mae": np.nan, "bias": np.nan, "note": str(exc)})
            continue
        p = predict(res["x"], te["N"], te["D"], te["Q"], 1.0, None)
        rows.append({"held_out_N": float(g), "n_test": int(len(te)),
                     "rmse": rmse(te["L"], p), "mae": mae(te["L"], p),
                     "bias": bias(te["L"], p), "note": ""})
    out = pd.DataFrame(rows)
    if log:
        good = out["rmse"].dropna()
        log(f"  [holdout-by-size] {len(out)} 组，RMSE 中位 {good.median():.4f} "
            f"最大 {good.max():.4f}")
    return out


def holdout_d_direction(cfg: dict, df: pd.DataFrame, bounds: dict, log=None) -> dict:
    """D 方向：前段训练、后段检验（§2.2）。"""
    q = float(cfg["validation"]["b1_d_quantile"])
    thr = float(df["D"].quantile(q))
    tr = df[df["D"] <= thr]
    te = df[df["D"] > thr]
    res = fit(tr["N"], tr["D"], tr["Q"], tr["L"], model="M0", bounds=bounds,
              n_starts=32, seed=cfg["seed"], max_nfev=12000)
    p = predict(res["x"], te["N"], te["D"], te["Q"], 1.0, None)
    out = {"threshold_D": thr, "n_train": int(len(tr)), "n_test": int(len(te)),
           "rmse": rmse(te["L"], p), "mae": mae(te["L"], p), "bias": bias(te["L"], p)}
    if log:
        log(f"  [holdout-D] D>{thr:.3g} 检验 n={len(te)} RMSE={out['rmse']:.4f} "
            f"bias={out['bias']:+.4f}")
    return out


# ----------------------------------------------------------------- 轨迹检验（B3）
def trajectory_check(cfg: dict, traj: pd.DataFrame, x_global: np.ndarray, q_ref: float,
                     log=None) -> pd.DataFrame:
    """B3：D 方向插值一致性校检（须标注 interpolated 性质）。

    B3 每条轨迹只有**单一 N**，却有 500 个 D 点（B1 在同一 N 上只有 147 点），
    且标注为 interpolated。在单一 N 上重拟合 5 个经典参数不可辨识（A 与 α 只以
    A·N^(-α) 的乘积出现），因此这里**冻结 B1 全局参数**，只评估插值点的复现误差：
    前 50% 点作为参考，后 50% 作为留出。误差量级若与 B1 的舍入噪声同阶，说明 B3
    是该幂律的确定性加密插值，不构成独立实验证据。
    """
    frac = float(cfg["validation"]["trajectory_train_frac"])
    rows = []
    for fn, t in traj.groupby("traj_file"):
        t = t.sort_values("D").reset_index(drop=True)
        cut = int(len(t) * frac)
        tr, te = t.iloc[:cut], t.iloc[cut:]
        if "Q" not in te.columns:
            te = te.assign(Q=float(q_ref))
        if "Q" not in tr.columns:
            tr = tr.assign(Q=float(q_ref))
        p_tr = predict(x_global, tr["N"], tr["D"], tr["Q"], 1.0, None)
        p_te = predict(x_global, te["N"], te["D"], te["Q"], 1.0, None)
        n_interp = int(pd.to_numeric(te.get("interpolated",
                                            pd.Series(0, index=te.index)),
                                     errors="coerce").fillna(0).sum())
        rows.append({"trajectory": fn, "N": float(t["N"].iloc[0]),
                     "n_first_half": int(len(tr)), "n_test": int(len(te)),
                     "rmse_test": rmse(te["L"], p_te), "mae_test": mae(te["L"], p_te),
                     "bias_test": bias(te["L"], p_te),
                     "rmse_first_half": rmse(tr["L"], p_tr),
                     "n_interpolated_test": n_interp,
                     "interpolated_frac": float(n_interp / max(len(te), 1))})
    out = pd.DataFrame(rows)
    if log:
        log(f"  [trajectory] {len(out)} 条轨迹，留出段 RMSE 中位 "
            f"{out['rmse_test'].median():.2e}（B1 舍入噪声量级 ~1.5e-4）；"
            f"插值点占比 {out['interpolated_frac'].min():.2f}–"
            f"{out['interpolated_frac'].max():.2f}")
    return out


# ----------------------------------------------------------------- 跨源误差表
def cross_source_table(cfg: dict, model_x: np.ndarray, loaded: dict, q_ref: float,
                       log=None) -> pd.DataFrame:
    """逐来源 RMSE / MAE / 偏差 / R²，标注数据性质；带 split 的来源分组报告。"""
    rows = []
    for s in SOURCES:
        if s["key"] not in loaded:
            continue
        full = prep_source(loaded[s["key"]], cfg, q_ref)
        if not len(full):
            continue
        if s["split"] and s["split"] in full.columns:
            groups = [(f"{s['id']}_{g}", sub) for g, sub in
                      full.groupby(s["split"], dropna=False)]
        else:
            groups = [(s["id"], full)]
        for sid, d in groups:
            if not len(d):
                continue
            p = predict(model_x, d["N"], d["D"], d["Q"], 1.0,
                        np.full(len(d), float(s["offset"])))
            # 去水平差 RMSE：先把残差的均值扣掉再算 RMSE。用于把"整族 Loss 口径/水平
            # 不可比"（体现在 bias）与"形状不匹配"（体现在去偏后的 RMSE）分开——
            # 族外来源偏差很大时，只看 RMSE 无法判断是个常数水平差还是幂律指错。
            p_db = p - float(np.mean(p - np.asarray(d["L"], float)))
            rows.append({
                "source": sid, "kind": s["kind"], "note": s["note"],
                "file": d["source_file"].iloc[0], "offset_group": float(s["offset"]),
                "n": int(len(d)), "rmse": rmse(d["L"], p),
                "rmse_debiased": rmse(d["L"], p_db),
                "mae": mae(d["L"], p), "bias": bias(d["L"], p), "r2": r2(d["L"], p),
                "N_min": float(d["N"].min()), "N_max": float(d["N"].max()),
                "D_min": float(d["D"].min()), "D_max": float(d["D"].max()),
                "L_min": float(d["L"].min()), "L_max": float(d["L"].max()),
                "Q_min": float(d["Q"].min()), "Q_max": float(d["Q"].max()),
                "Q_is_convention": bool(d["Q_is_convention"].all()),
            })
    out = pd.DataFrame(rows).sort_values("source").reset_index(drop=True)
    if log:
        for _, r in out.iterrows():
            log(f"  [source {r['source']:24s}] n={r['n']:5d} RMSE={r['rmse']:.4f} "
                f"(去水平差 {r['rmse_debiased']:.4f}) bias={r['bias']:+.4f} "
                f"N≤{r['N_max']:.4g}B kind={r['kind']}")
    return out


# --------------------------------------------------------------- 分组 bootstrap
def grouped_bootstrap(cfg: dict, df: pd.DataFrame, model: str, bounds: dict,
                      base_x: np.ndarray, group_col="N", log=None) -> pd.DataFrame:
    """按模型规模整组重抽的经验 bootstrap（不是未来观测预测区间）。

    每次重抽后从全数据最优解热启动（单起点），保证可计算性；区间为参数的经验分位。
    group_col 可为列名或列名列表（合并数据上按 来源批 × 规模 联合分组，避免把
    同一规模下的两个来源当作可独立重抽的单元）。
    """
    n = int(cfg["validation"]["bootstrap"]["n"])
    rng = np.random.default_rng(cfg["seed"])
    gcols = [group_col] if isinstance(group_col, str) else list(group_col)
    key = df[gcols].astype(str).agg("|".join, axis=1)
    groups = list(pd.unique(key))
    has_off = "dgrp" in df.columns
    # δ 在 bootstrap 中固定为点估计（δ 由"来源族"这一个对比识别，重抽规模组
    # 并不能提供关于它的额外信息）；区间只反映其余自由参数的经验不确定性。
    free = list(MODEL_FREE[model])
    idx = [FULL.index(p) for p in free]
    cols = []
    for _ in range(n):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        parts = [df[key == groups[i]] for i in pick]
        s = pd.concat(parts, ignore_index=True)
        try:
            r = fit(s["N"], s["D"], s["Q"], s["L"], model=model, bounds=bounds,
                    n_starts=1, seed=int(rng.integers(1 << 31)),
                    base=base_x, dgrp=(s["dgrp"] if has_off else None),
                    max_nfev=4000, cluster_tol=0.15)
            cols.append(r["x"][idx])
        except Exception:
            continue
    A = np.array(cols, float)
    rows = []
    for j, name in enumerate(free):
        v = A[:, j]
        rows.append({"param": name, "n_boot": int(len(v)),
                     "point": float(base_x[FULL.index(name)]),
                     "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                     "lo95": float(np.percentile(v, 2.5)),
                     "hi95": float(np.percentile(v, 97.5))})
    out = pd.DataFrame(rows)
    if log:
        r = out[out.param == "rho"]
        if len(r):
            log(f"  [bootstrap] n={len(A)} ρ 经验区间 [{r['lo95'].iloc[0]:.3f}, "
                f"{r['hi95'].iloc[0]:.3f}]")
    return out


# ----------------------------------------------------------------- 外推情景
def extrapolation_scenario(cfg: dict, model_x: np.ndarray, b9: pd.DataFrame,
                           b10: pd.DataFrame, log=None) -> pd.DataFrame:
    """B9/B10：>10B 参数区的预测与估算值对照（情景，不是经验区间）。"""
    lo = float(cfg["validation"]["b10_extrapolation_min_B"])
    d = prep_source(b10, cfg, float(cfg["model"]["q_reference"]))
    d = d[d["N"] >= lo].copy()
    if not len(d):
        return pd.DataFrame()
    d["L_pred"] = predict(model_x, d["N"], d["D"], d["Q"], 1.0, np.zeros(len(d)))
    d["resid"] = d["L_pred"] - d["L"]
    out = d[["family", "N", "D", "L", "L_pred", "resid"]].copy() if "family" in d.columns \
        else d[["N", "D", "L", "L_pred", "resid"]].copy()
    if log:
        log(f"  [extrapolation] N≥{lo:g}B n={len(out)} RMSE="
            f"{rmse(out['L'], out['L_pred']):.4f} bias={bias(out['L'], out['L_pred']):+.4f}")
    return out
