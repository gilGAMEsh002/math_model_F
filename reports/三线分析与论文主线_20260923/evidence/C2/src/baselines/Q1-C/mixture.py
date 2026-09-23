# -*- coding: utf-8 -*-
"""Q1-C 第二部分：17 维配比 → 13 个验证域 Loss 的回归（随机森林 vs 线性对照）。

对应任务卡 Q1-C.md 第 3、4 条与 01_问题分析与Baseline实施计划.md §1.4/§1.6：
  - 随机森林逐目标或多输出预测 13 个 Loss；
  - 只在训练集（A4/A5）内部搜索树数量/最大深度/叶子最小样本量小网格，
    严禁用 A6-A11 最终检验集调参；
  - 1M 报告逐目标与综合目标的 RMSE/MAE/R²，并给出训练均值预测作为最低参照；
  - 60M/1B 报告 Spearman 排序相关、Top-k 选方质量与后悔值，同时如实报告绝对误差；
  - 10B/70B 只用于估算表一致性与扰动稳健性诊断，不作为真实泛化证据；
  - 记录拟合时间与模型复杂度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from common import (Timer, load_pairs, mae, regret, r2, rmse, save_table, spearman,
                    topk_hit)


def short_target(c: str) -> str:
    return c.replace("metric/the_pile_", "").replace("_val_loss", "")


# ------------------------------------------------------------------ 设计矩阵
def normalize_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return X / s


def quality_lookup(cfg: dict, mapping: pd.DataFrame) -> dict:
    """配比列 -> (等权域质量 q_equal, 熵权域质量 q_entropy)。未映射域取 NaN。"""
    out = {}
    for _, r in mapping.iterrows():
        col = r["mixture_col"]
        if isinstance(col, str) and col.startswith("train_the_pile_"):
            out[col] = (r["q_equal"], r["q_entropy"])
    return out


def add_quality_feature(X: np.ndarray, cols: list[str], qmap: dict,
                        which: str, unmapped: str) -> np.ndarray:
    """构造质量特征 Q(p) = Σ_i p_i q_i。

    11 个训练域在 A16 中无同名质量域（mapping_type=inferred）。
      exclude（主口径）：q_i := 0，分母只取已完成映射的域，即
                         Q(p) = Σ_{mapped} p_i q_i / Σ_{mapped} p_i（显式缺失，不虚构取值）；
                         若某行 Σ_{mapped} p_i = 0，回退为全局已映射域质量均值。
      proxy（敏感性）：未映射域取已映射域质量均值作中性代理。
    注意 §1.5 第 7 条：Q(p) 与完整线性配比变量存在确定关系，不能把两类系数都解释为独立效应。
    """
    idx = {c: i for i, c in enumerate(cols)}
    mapped = [c for c in cols if c in qmap and not pd.isna(qmap[c][0 if which == "equal" else 1])]
    qi = np.array([qmap[c][0 if which == "equal" else 1] for c in mapped], dtype=float)
    qmap_all = np.array([qmap.get(c, (np.nan, np.nan))[0 if which == "equal" else 1]
                         for c in cols], dtype=float)
    fallback = float(np.nanmean(qmap_all))

    if unmapped == "proxy":
        q_all = np.where(np.isnan(qmap_all), fallback, qmap_all)
        Q = X @ q_all
    else:  # exclude
        pm = X[:, [idx[c] for c in mapped]]
        den = pm.sum(axis=1)
        num = pm @ qi
        Q = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), fallback)
    return Q.reshape(-1, 1)


def build_variant_X(cfg: dict, Xr: np.ndarray, cols: list[str], qmap: dict,
                    mode: str, drop_ref: bool = False) -> tuple[np.ndarray, list[str]]:
    names = list(cols)
    Xd = Xr
    if drop_ref:
        keep = [i for i, c in enumerate(cols) if c != cfg["mixture"]["drop_reference_col"]]
        Xd = Xr[:, keep]
        names = [cols[i] for i in keep]
    if mode == "none":
        return Xd, names
    qf = cfg["mixture"]["quality_feature"]
    pol = (qf["primary_unmapped_policy"] if mode == "equal" else qf["primary_unmapped_policy"])
    Q = add_quality_feature(Xd, names, qmap, mode, pol)
    return np.hstack([Xd, Q]), names + [f"Q_{mode}"]


# ------------------------------------------------------------------ 模型
def fit_ridge(X, Y, alpha, fit_intercept=True):
    m = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    m.fit(X, Y)
    return m


def fit_forest(X, Y, params, seed):
    m = RandomForestRegressor(random_state=seed, **params)
    m.fit(X, Y)
    return m


def predict_models(model, X: np.ndarray) -> np.ndarray:
    """统一预测接口：多输出单模型 / 逐目标模型列表 → 一律返回 (n, n_targets)。"""
    if isinstance(model, list):
        return np.column_stack([np.asarray(m.predict(X)).ravel() for m in model])
    return np.asarray(model.predict(X)).reshape(len(X), -1)


def forest_complexity(models, n_targets: int) -> dict:
    """记录模型复杂度：树数、总节点数、平均深度、叶子数。"""
    if isinstance(models, list):
        tops = [t for m in models for t in m.estimators_]
        n_models = len(models)
    else:
        tops = list(models.estimators_)
        n_models = 1
    nodes = sum(t.tree_.node_count for t in tops)
    leaves = sum(int(np.sum(t.tree_.children_left == -1)) for t in tops)
    depths = [t.tree_.max_depth for t in tops]
    return {
        "n_models": n_models, "n_trees_total": len(tops),
        "trees_per_model": len(tops) // max(n_models, 1),
        "total_nodes": int(nodes), "total_leaves": int(leaves),
        "mean_nodes_per_tree": float(nodes / max(len(tops), 1)),
        "max_depth_mean": float(np.mean(depths)), "max_depth_max": int(np.max(depths)),
        "n_outputs": n_targets,
    }


# ------------------------------------------------------------------ CV 选参
def cv_select(cfg: dict, X: np.ndarray, Y: np.ndarray, cands: list[dict],
              log=print) -> tuple[pd.DataFrame, dict]:
    """在训练集内部做 K 折交叉验证，返回 (各候选的 CV 结果, 最优候选)。

    选参准则（配置冻结）：13 个目标「逐目标 RMSE 的均值」最小。
    """
    kf = KFold(n_splits=cfg["mixture"]["cv_folds"], shuffle=True,
               random_state=cfg["mixture"]["cv_seed"])
    folds = list(kf.split(X))
    rows = []
    for cand in cands:
        t = Timer()
        per_target = []
        comp_rmse, comp_r2 = [], []
        with t:
            for tr, te in folds:
                if cand["kind"] == "ridge":
                    m = fit_ridge(X[tr], Y[tr], cand["params"]["alpha"],
                                  cfg["mixture"]["ridge"]["fit_intercept"])
                else:
                    m = fit_forest(X[tr], Y[tr], cand["params"], cfg["seed"])
                P = m.predict(X[te])
                per_target.append([rmse(Y[te][:, j], P[:, j]) for j in range(Y.shape[1])])
                comp_rmse.append(rmse(Y[te].mean(axis=1), P.mean(axis=1)))
                comp_r2.append(r2(Y[te].mean(axis=1), P.mean(axis=1)))
        pt = np.array(per_target)  # folds × targets
        rec = {
            "kind": cand["kind"], "label": cand["label"],
            "params": str(cand["params"]),
            "cv_mean_target_rmse": float(pt.mean()),
            "cv_composite_rmse": float(np.mean(comp_rmse)),
            "cv_composite_r2": float(np.mean(comp_r2)),
            "cv_seconds": t.elapsed,
        }
        for j in range(Y.shape[1]):
            rec[f"cv_rmse_{short_target(cfg['mixture']['target_cols'][j])}"] = float(pt[:, j].mean())
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("cv_mean_target_rmse").reset_index(drop=True)
    best = df.iloc[0]
    best_cand = next(c for c in cands if c["label"] == best["label"])
    log(f"[mixture] CV 选参完成：最优 {best['label']} "
        f"mean_target_rmse={best['cv_mean_target_rmse']:.4f} "
        f"composite_rmse={best['cv_composite_rmse']:.4f}")
    return df, best_cand


def best_per_kind(cv_df: pd.DataFrame, cands: list[dict]) -> dict:
    """分别取线性与森林各自的最优候选。

    cv_df 已按 cv_mean_target_rmse 升序排列，故每类首行即该类最优。
    两支必须分别取参：线性对照的意义在于「同一输入、不同函数形式」，
    不能被森林的最优超参覆盖，否则消融的差异无法归因。
    """
    out = {}
    for kind in ("ridge", "forest"):
        sub = cv_df[cv_df["kind"] == kind]
        if sub.empty:
            continue
        lab = sub.iloc[0]["label"]
        out[kind] = next(c for c in cands if c["label"] == lab)
    return out


def make_candidates(cfg: dict) -> list[dict]:
    # label 必须逐个唯一：best_per_kind 依赖 label 回查候选，重复 label 会导致选参错位
    cands = [{"kind": "ridge", "label": f"ridge_a{a:g}", "params": {"alpha": a}}
             for a in cfg["mixture"]["ridge"]["alpha_grid"]]
    f = cfg["mixture"]["forest"]
    for ne in f["n_estimators"]:
        for md in f["max_depth"]:
            for ml in f["min_samples_leaf"]:
                for mf in f["max_features"]:
                    cands.append({
                        "kind": "forest",
                        "label": f"rf_n{ne}_d{md}_l{ml}_f{mf}",
                        "params": {"n_estimators": ne, "max_depth": md,
                                   "min_samples_leaf": ml, "max_features": mf, "n_jobs": f["n_jobs"]},
                    })
    return cands


# ------------------------------------------------------------------ 评价
def evaluate_absolute(Y: np.ndarray, P: np.ndarray, target_cols: list[str],
                      extra: dict) -> pd.DataFrame:
    """逐目标 + 综合目标的 RMSE/MAE/R²（1M 同尺度检验用）。"""
    rows = []
    for j, c in enumerate(target_cols):
        rows.append({**extra, "target": short_target(c), "n": len(Y),
                     "rmse": rmse(Y[:, j], P[:, j]), "mae": mae(Y[:, j], P[:, j]),
                     "r2": r2(Y[:, j], P[:, j]),
                     "y_mean": float(Y[:, j].mean()), "y_std": float(Y[:, j].std(ddof=1))})
    yc, pc = Y.mean(axis=1), P.mean(axis=1)
    rows.append({**extra, "target": "__composite_equal_v__", "n": len(Y),
                 "rmse": rmse(yc, pc), "mae": mae(yc, pc), "r2": r2(yc, pc),
                 "y_mean": float(yc.mean()), "y_std": float(yc.std(ddof=1))})
    return pd.DataFrame(rows)


def evaluate_ranking(Y: np.ndarray, P: np.ndarray, target_cols: list[str],
                     topk: list[int], extra: dict) -> pd.DataFrame:
    """跨尺度排序评价：Spearman、Top-k 命中、后悔值（Loss 越低越好）。"""
    rows = []
    ks = [k for k in topk if k < len(Y)]

    def one(y, p, name):
        rec = {**extra, "target": name, "n": len(y), "spearman": spearman(y, p),
               "regret": regret(y, p, lower_is_better=True),
               # 无技能参照：均匀随机选方的期望后悔值 E[y]-min(y)。
               # train_mean 因预测为常数，np.argmin 恒取第 0 行，其 regret 不是无技能基准。
               "expected_regret_random_pick": float(np.mean(y) - np.min(y)),
               "rmse_uncalibrated": rmse(y, p), "mae_uncalibrated": mae(y, p),
               "y_min": float(np.min(y)), "y_max": float(np.max(y)),
               "y_range": float(np.max(y) - np.min(y))}
        for k in ks:
            rec[f"topk{k}_hit"] = bool(topk_hit(y, p, k, lower_is_better=True))
        return rec

    for j, c in enumerate(target_cols):
        rows.append(one(Y[:, j], P[:, j], short_target(c)))
    rows.append(one(Y.mean(axis=1), P.mean(axis=1), "__composite_equal_v__"))
    return pd.DataFrame(rows)


def perturbation_robustness(cfg: dict, model, Xr: np.ndarray, base_P: np.ndarray,
                            seed: int) -> pd.DataFrame:
    """对配比施加 Dirichlet 扰动，观察预测漂移（10B/70B 一致性诊断用）。

    每行以自身配比 p 为均值、concentration 为总强度构造 Dirichlet 扰动；
    衡量预测漂移的绝对幅度，以及相对该目标在估算表上标准差的幅度。
    漂移过大说明森林在插值域内的局部不稳定，需与「不作为外推公式」的结论一并报告。
    """
    rng = np.random.default_rng(seed)
    pert = cfg["extrapolation"]["perturbation"]
    n_draws = int(pert["n_draws"])
    scale = np.maximum(base_P.std(axis=0, ddof=1), 1e-9)
    rows = []
    for conc in pert["concentrations"]:
        draws = []
        for i in range(Xr.shape[0]):
            alpha = np.maximum(Xr[i] * float(conc), 1e-6)
            draws.append(rng.dirichlet(alpha, size=n_draws))
        Xall = np.vstack(draws)                      # (n_rows*n_draws, n_features)
        Pall = model.predict(Xall)
        n = Xr.shape[0]
        d = np.abs(Pall.reshape(n, n_draws, -1) - base_P[:, None, :])
        rows.append({
            "concentration": float(conc), "n_draws": n_draws,
            "n_perturbed_rows": int(n * n_draws),
            "mean_abs_pred_shift": float(d.mean()),
            "max_abs_pred_shift": float(d.max()),
            "mean_shift_over_target_sd": float((d / scale[None, None, :]).mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 主流程
def run_mixture(cfg: dict, qmap: dict, log=print) -> dict:
    mcfg = cfg["mixture"]
    tcols = mcfg["target_cols"]
    ic = cfg["seed"]

    pairs = load_pairs(cfg)
    tr = pairs["train_1m"]
    Xtr = normalize_rows(tr["X"].to_numpy(dtype=float))
    Ytr = tr["Y"].to_numpy(dtype=float)
    cols = list(tr["X"].columns)
    log(f"[mixture] 训练集 {Xtr.shape} → {Ytr.shape}；行和(归一化前) "
        f"[{tr['X'].to_numpy().sum(axis=1).min():.4f}, {tr['X'].to_numpy().sum(axis=1).max():.4f}]")

    # ---- CV 选参（只用训练集内部）
    cands = make_candidates(cfg)
    log(f"[mixture] 候选 {len(cands)} 个（ridge {len(mcfg['ridge']['alpha_grid'])} + "
        f"forest {len(cands)-len(mcfg['ridge']['alpha_grid'])}），{mcfg['cv_folds']} 折内部 CV")
    cv_df, best = cv_select(cfg, Xtr, Ytr, cands, log=log)
    best_by_kind = best_per_kind(cv_df, cands)
    best_ridge_alpha = float(best_by_kind["ridge"]["params"]["alpha"])
    best_forest_params = dict(best_by_kind["forest"]["params"])
    log(f"[mixture] 分类最优：ridge alpha={best_ridge_alpha}；"
        f"forest {best_forest_params}")

    # ---- 训练均值参照
    ybar = Ytr.mean(axis=0)
    cv_df = cv_df.copy()

    # ---- 主变体与交叉消融
    modes = mcfg["quality_feature"]["modes"]
    qf_pol = mcfg["quality_feature"]["primary_unmapped_policy"]
    variants: dict[str, dict] = {}
    for mode in modes:
        Xm, names = build_variant_X(cfg, Xtr, cols, qmap, mode)
        for kind in ("ridge", "forest"):
            label = f"{kind}_{mode}"
            t = Timer()
            if kind == "ridge":
                with t:
                    model = fit_ridge(Xm, Ytr, best_ridge_alpha,
                                      mcfg["ridge"]["fit_intercept"])
                complexity = {"n_features": Xm.shape[1],
                              "n_params": int(Xm.shape[1] + 1)}
            else:
                with t:
                    model = fit_forest(Xm, Ytr, best_forest_params, ic)
                complexity = forest_complexity(model, len(tcols))
            variants[label] = {"model": model, "X": Xm, "names": names, "mode": mode,
                               "kind": kind, "fit_seconds": t.elapsed,
                               "complexity": complexity,
                               "train_composite_r2": r2(Ytr.mean(axis=1),
                                                        model.predict(Xm).mean(axis=1))}
            log(f"[mixture] 拟合 {label}: {t.elapsed:.2f}s  训练综合R²="
                f"{variants[label]['train_composite_r2']:.4f}")

    # 逐目标随机森林（任务卡「逐目标或多输出」的另一支）
    fp = best_forest_params
    Xb, names_b = build_variant_X(cfg, Xtr, cols, qmap, "none")
    t = Timer()
    with t:
        per_target_models = [fit_forest(Xb, Ytr[:, j], fp, ic) for j in range(len(tcols))]
    variants["forest_pertarget_none"] = {
        "model": per_target_models, "X": Xb, "names": names_b, "mode": "none",
        "kind": "forest_pertarget", "fit_seconds": t.elapsed,
        "complexity": forest_complexity(per_target_models, len(tcols)),
        "train_composite_r2": float("nan"),
    }
    Ptr_pt = predict_models(per_target_models, Xb)
    variants["forest_pertarget_none"]["train_composite_r2"] = r2(Ytr.mean(axis=1),
                                                                Ptr_pt.mean(axis=1))
    log(f"[mixture] 逐目标森林 13 个模型拟合 {t.elapsed:.2f}s")

    # 去掉一个配比变量的敏感性（严格匹配 Q1-A 的 16 变量设定）
    Xd, names_d = build_variant_X(cfg, Xtr, cols, qmap, "none", drop_ref=True)
    t = Timer()
    with t:
        m_drop = (fit_ridge(Xd, Ytr, best_ridge_alpha, mcfg["ridge"]["fit_intercept"])
                  if best["kind"] == "ridge" else fit_forest(Xd, Ytr, fp, ic))
    variants["best_drop_ref16"] = {"model": m_drop, "X": Xd, "names": names_d, "mode": "none",
                                   "kind": best["kind"], "fit_seconds": t.elapsed,
                                   "complexity": {"n_features": Xd.shape[1]},
                                   "train_composite_r2": r2(Ytr.mean(axis=1),
                                                            m_drop.predict(Xd).mean(axis=1))}

    # 未映射域代理口径的敏感性
    Xp2, names_p = build_variant_X(cfg, Xtr, cols, qmap, "entropy")
    qprob = add_quality_feature(Xp2[:, :-1], names_p[:-1], qmap, "entropy",
                               mcfg["quality_feature"]["proxy_unmapped_policy"])
    Xp2 = np.hstack([Xp2[:, :-1], qprob])
    t = Timer()
    with t:
        m_proxy = (fit_ridge(Xp2, Ytr, best_ridge_alpha, mcfg["ridge"]["fit_intercept"])
                   if best["kind"] == "ridge" else fit_forest(Xp2, Ytr, fp, ic))
    variants["best_entropy_proxyQ"] = {"model": m_proxy, "X": Xp2, "names": names_p,
                                       "mode": "entropy_proxy", "kind": best["kind"],
                                       "fit_seconds": t.elapsed,
                                       "complexity": {"n_features": Xp2.shape[1]},
                                       "train_composite_r2": r2(Ytr.mean(axis=1),
                                                                m_proxy.predict(Xp2).mean(axis=1))}

    # ---- 预测所有检验集与外推表
    # 数据集名取自配置的 splits 段：final_test 为真实留出，extrapolation 仅作诊断
    final_order = [s["name"] for s in cfg["splits"]["final_test"]]
    SAME_SCALE = final_order[0]          # 同尺度检验集（1M），报告绝对误差
    FINAL = set(final_order)
    preds_long, abs_rows, rank_rows, extrap_rows, pred_meta = [], [], [], [], {}
    keep_for_dump = set(variants.keys())
    for name, spec in pairs.items():
        if name == "train_1m":
            continue
        Xh = normalize_rows(spec["X"].to_numpy(dtype=float))
        Yh = spec["Y"].to_numpy(dtype=float)
        for label in keep_for_dump:
            v = variants[label]
            # 变体可能使用了不同特征数（drop_ref / proxy），此处按其自身列构造
            Xv = build_variant_X(cfg, normalize_rows(spec["X"].to_numpy(dtype=float)),
                                 cols, qmap,
                                 v["mode"].replace("_proxy", ""),
                                 drop_ref=(label == "best_drop_ref16"))[0]
            if label == "best_entropy_proxyQ":
                Xv = np.hstack([Xv[:, :-1],
                                add_quality_feature(Xv[:, :-1], names_p[:-1], qmap, "entropy",
                                                    mcfg["quality_feature"]["proxy_unmapped_policy"])])
            model = v["model"]
            P = predict_models(model, Xv)
            mode_short = v["mode"]
            base = {"dataset": name, "kind": spec["kind"], "variant": label, "mode": mode_short}
            if name in FINAL:
                if name == SAME_SCALE:
                    abs_rows.append(evaluate_absolute(Yh, P, tcols, base))
                rank_rows.append(evaluate_ranking(Yh, P, tcols, mcfg["topk"], base))
            else:
                r = evaluate_ranking(Yh, P, tcols, mcfg["topk"], base)
                r["note"] = "estimated/semi_synthetic — 仅一致性诊断"
                extrap_rows.append(r)
            pred_meta[f"{name}|{label}"] = float(np.mean(np.abs(P - Yh)))
            for j, c in enumerate(tcols):
                preds_long.append(pd.DataFrame({
                    "dataset": name, "variant": label, "row_index": spec["index"],
                    "target": short_target(c), "y_true": Yh[:, j], "y_pred": P[:, j]}))
            # 综合目标
            preds_long.append(pd.DataFrame({
                "dataset": name, "variant": label, "row_index": spec["index"],
                "target": "__composite_equal_v__", "y_true": Yh.mean(axis=1), "y_pred": P.mean(axis=1)}))

    # ---- 训练均值最低参照
    for name, spec in pairs.items():
        if name == "train_1m":
            continue
        Yh = spec["Y"].to_numpy(dtype=float)
        P = np.tile(ybar, (len(Yh), 1))
        base = {"dataset": name, "kind": spec["kind"], "variant": "train_mean", "mode": "-"}
        if name == SAME_SCALE:
            abs_rows.append(evaluate_absolute(Yh, P, tcols, base))
        if name in FINAL:
            rank_rows.append(evaluate_ranking(Yh, P, tcols, mcfg["topk"], base))
        else:
            r = evaluate_ranking(Yh, P, tcols, mcfg["topk"], base)
            r["note"] = "estimated/semi_synthetic — 仅一致性诊断"
            extrap_rows.append(r)

    abs_df = pd.concat(abs_rows, ignore_index=True)
    rank_df = pd.concat(rank_rows, ignore_index=True)
    extrap_df = pd.concat(extrap_rows, ignore_index=True)

    # ---- 扰动稳健性（10B/70B，最佳变体）
    pert_rows = []
    for name in ("est_10b", "est_70b"):
        spec = pairs[name]
        v = variants[f"{best['kind']}_none"]
        Xv = build_variant_X(cfg, normalize_rows(spec["X"].to_numpy(dtype=float)),
                             cols, qmap, "none")[0]
        P = v["model"].predict(Xv)
        pr = perturbation_robustness(cfg, v["model"], Xv, P, ic)
        pr.insert(0, "dataset", name)
        pr.insert(1, "variant", f"{best['kind']}_none")
        pert_rows.append(pr)
    pert_df = pd.concat(pert_rows, ignore_index=True)

    # ---- 复杂度与时间表
    comp_rows = [{"variant": k, "kind": v["kind"], "mode": v["mode"],
                  "fit_seconds": v["fit_seconds"],
                  "train_composite_r2": v["train_composite_r2"],
                  **{f"cx_{kk}": vv for kk, vv in v["complexity"].items()}}
                 for k, v in variants.items()]
    comp_df = pd.DataFrame(comp_rows)

    # ---- 落盘
    save_table(pd.concat(preds_long, ignore_index=True), cfg, "mixture_predictions.csv")
    save_table(cv_df, cfg, "mixture_cv_grid.csv")
    save_table(abs_df, cfg, "mixture_metrics_same_scale_1m.csv")
    save_table(rank_df, cfg, "mixture_ranking_cross_scale.csv")
    save_table(extrap_df, cfg, "mixture_extrapolation_consistency.csv")
    save_table(pert_df, cfg, "mixture_perturbation_robustness.csv")
    save_table(comp_df, cfg, "mixture_complexity_time.csv")

    return {"variants": variants, "best": best, "best_row": cv_df.iloc[0].to_dict(),
            "cv": cv_df, "absolute": abs_df,
            "ranking": rank_df, "extrapolation": extrap_df, "perturbation": pert_df,
            "complexity": comp_df, "ybar": ybar,
            "best_params": {"ridge_alpha": best_ridge_alpha,
                            "forest": best_forest_params},
            "cols": cols, "target_cols": tcols}
