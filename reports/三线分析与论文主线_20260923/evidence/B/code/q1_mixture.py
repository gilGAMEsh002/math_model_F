# -*- coding: utf-8 -*-
"""
第一问（第三部分）：17 域配比 p 与 13 个验证域交叉熵损失的定量关系。

模型（Baseline B / Q1-B）：
    线性混料（对照，Q1-A 口径）:  L_k(p) = Σ_i a_ki p_i
    二次混料（主模型，Scheffé 规范形）: L_k(p) = Σ_i a_ki p_i + Σ_{i<j} b_kij p_i p_j
    两者均加岭惩罚，无截距（单纯形约束下截距冗余），惩罚强度由 A4/A5 内 5 折 CV 选择。

数据：A4/A5 训练（512）；A6-A11 检验（1M/60M/1B）；A12-A15 外推一致性诊断。
输出：逐域与综合评价目标 L_eval = Σ v_k L_k (v 等权) 的 RMSE/MAE/R²/Spearman，以及 Top-k 后悔值。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, LOSS_DOMAINS, TRAIN_DOMAINS, save_csv, save_json, SEED,
                    rmse, mae, r2, spearman)

RT = DATA / "A_data_value" / "regmix_tables"

GROUPS = {
    "train": ("train_mixture_1m.csv", "train_pile_loss_1m.csv"),
    "test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
    "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
    "test_1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
    "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
    "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv"),
}


def load(split):
    mf, lf = GROUPS[split]
    M = pd.read_csv(RT / mf)
    L = pd.read_csv(RT / lf)
    assert list(M["index"]) == list(L["index"]), f"{split} index 不对应"
    P = M.drop(columns=["index"]).to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)          # 行归一化到单纯形
    Y = L.drop(columns=["index"]).to_numpy(float)
    return P, Y, M["index"].to_numpy(), list(L.columns[1:])


def design(P, quad=True):
    """混料设计矩阵：线性项 + （可选）两两交互项。无截距、无平方项。"""
    cols = [P]
    if quad:
        inter = np.column_stack([P[:, i] * P[:, j] for i, j in combinations(range(P.shape[1]), 2)])
        inter = inter * P.shape[1]      # 尺度对齐，使岭惩罚对各列量级公平
        cols.append(inter)
    return np.hstack(cols)


def design_names(quad=True):
    names = list(TRAIN_DOMAINS)
    if quad:
        names += [f"{TRAIN_DOMAINS[i]}*{TRAIN_DOMAINS[j]}"
                  for i, j in combinations(range(len(TRAIN_DOMAINS)), 2)]
    return names


def fit_ridge(X, Y, alpha):
    """逐目标岭回归（多输出）。"""
    m = Ridge(alpha=alpha, fit_intercept=False)
    m.fit(X, Y)
    return m


def cv_alpha(X, Y, alphas, k=5):
    """5 折 CV 选惩罚强度：以各域标准化 RMSE 的均值最小为目标。"""
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    scale = Y.std(axis=0)
    score = np.zeros(len(alphas))
    for tr, va in kf.split(X):
        for ai, a in enumerate(alphas):
            pred = fit_ridge(X[tr], Y[tr], a).predict(X[va])
            score[ai] += np.mean(np.abs(pred - Y[va]) / scale)
    return alphas[int(np.argmin(score))], score / k


def evaluate(y, yhat, v=None):
    """逐域与综合指标。v 为 13 域评价权重（默认等权）。"""
    n, k = y.shape
    if v is None:
        v = np.ones(k) / k
    per = []
    for j in range(k):
        per.append(dict(domain=LOSS_DOMAINS[j], rmse=rmse(y[:, j], yhat[:, j]),
                        mae=mae(y[:, j], yhat[:, j]), r2=r2(y[:, j], yhat[:, j]),
                        spearman=spearman(y[:, j], yhat[:, j])))
    comp_y, comp_p = y @ v, yhat @ v
    comp = dict(domain="__综合(eval)__", rmse=rmse(comp_y, comp_p), mae=mae(comp_y, comp_p),
                r2=r2(comp_y, comp_p), spearman=spearman(comp_y, comp_p))
    return pd.DataFrame(per + [comp])


def regret(y, yhat, v=None):
    """Top-k 后悔值：预测最优配方的真实 Loss − 候选集真实最小 Loss。"""
    n, k = y.shape
    if v is None:
        v = np.ones(k) / k
    ty, py = y @ v, yhat @ v
    out = {}
    for kk in (1, 5, 10):
        idx = np.argsort(py)[:kk]                    # 预测最好的 k 个
        true_best = ty.min()
        out[f"regret_top{kk}"] = float(np.mean(ty[idx] - true_best))
        out[f"hit_top{kk}"] = bool(np.argmin(ty) in set(idx.tolist()))
    out["oracle_min"] = float(ty.min())
    out["mean_loss"] = float(ty.mean())
    out["baseline_regret"] = float(ty.mean() - ty.min())   # 随机选配方的后悔值
    return out


def main():
    print("=" * 90)
    print("第一问 · 领域配比建模（二次混料岭回归）")
    print("=" * 90)

    Ptr, Ytr, _, loss_cols = load("train")
    n, k = Ytr.shape
    print(f"训练集 A4/A5: p{Ptr.shape} L{Ytr.shape}  域数={k}")

    Xlin, Xquad = design(Ptr, False), design(Ptr, True)
    alphas = np.logspace(-4, 3, 22)
    a_lin, _ = cv_alpha(Xlin, Ytr, alphas)
    a_quad, cv_curve = cv_alpha(Xquad, Ytr, alphas)
    print(f"CV 选定惩罚: 线性 alpha={a_lin:.4g}  二次 alpha={a_quad:.4g}")

    mdl_lin = fit_ridge(Xlin, Ytr, a_lin)
    mdl_quad = fit_ridge(Xquad, Ytr, a_quad)

    # ---- 训练集 in-sample / CV 表现 ----
    cv_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    v = np.ones(k) / k
    for name, X, a in [("线性混料", Xlin, a_lin), ("二次混料", Xquad, a_quad)]:
        oof = np.zeros_like(Ytr)
        for tr, va in kf.split(X):
            oof[va] = fit_ridge(X[tr], Ytr[tr], a).predict(X[va])
        cv_rows.append(dict(model=name, alpha=a,
                            cv_rmse=rmse(Ytr @ v, oof @ v),
                            cv_mae=mae(Ytr @ v, oof @ v),
                            cv_r2=r2(Ytr @ v, oof @ v),
                            cv_spearman=spearman(Ytr @ v, oof @ v)))
    save_csv(pd.DataFrame(cv_rows), "q1_mixture_cv.csv")
    print(pd.DataFrame(cv_rows).to_string(index=False))

    # ---- 各检验集与外推集评价 ----
    all_eval, all_regret, calib_rows = [], [], []
    for split in ["train", "test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]:
        P, Y, idx, _ = load(split)
        for name, mdl, quad in [("线性混料", mdl_lin, False), ("二次混料", mdl_quad, True)]:
            X = design(P, quad)
            yhat = mdl.predict(X)
            ev = evaluate(Y, yhat, v)
            ev.insert(0, "split", split)
            ev.insert(1, "model", name)
            ev.insert(2, "variant", "原始盲测")
            all_eval.append(ev)
            rg = regret(Y, yhat, v)
            rg.update(split=split, model=name, n=len(Y))
            all_regret.append(rg)

            # 跨尺度尺度偏移诊断：检验集一半拟合仿射校准 a+b*pred，另一半评估
            if split not in ("train",):
                rng2 = np.random.default_rng(SEED)
                perm = rng2.permutation(len(Y))
                half = len(Y) // 2
                ca, ce = perm[:half], perm[half:]
                yh2 = yhat.copy()
                for j in range(k):
                    A = np.column_stack([np.ones(len(ca)), yhat[ca, j]])
                    coef, *_ = np.linalg.lstsq(A, Y[ca, j], rcond=None)
                    yh2[ce, j] = coef[0] + coef[1] * yhat[ce, j]
                ev2 = evaluate(Y[ce], yh2[ce], v)
                ev2.insert(0, "split", split)
                ev2.insert(1, "model", name)
                ev2.insert(2, "variant", "尺度校准后")
                all_eval.append(ev2)
                calib_rows.append(dict(
                    split=split, model=name,
                    raw_rmse=float(rmse(Y[ce] @ v, yhat[ce] @ v)),
                    cal_rmse=float(rmse(Y[ce] @ v, yh2[ce] @ v)),
                    raw_bias=float(np.mean(yhat[ce] @ v - Y[ce] @ v)),
                    raw_spearman=spearman(Y[ce] @ v, yhat[ce] @ v),
                    cal_spearman=spearman(Y[ce] @ v, yh2[ce] @ v)))
        lin_r = [e for e in all_eval if e.iloc[0]['split'] == split
                 and e.iloc[1]['model'] == "线性混料" and e.iloc[2]['variant'] == "原始盲测"]
        quad_r = [e for e in all_eval if e.iloc[0]['split'] == split
                  and e.iloc[1]['model'] == "二次混料" and e.iloc[2]['variant'] == "原始盲测"]
        quad_c = [e for e in all_eval if e.iloc[0]['split'] == split
                  and e.iloc[1]['model'] == "二次混料" and e.iloc[2]['variant'] == "尺度校准后"]
        tail = f"  (二次校准后 {quad_c[0].iloc[-1]['rmse']:.4f})" if quad_c else ""
        print(f"[{split}] n={len(Y)}  线性RMSE={lin_r[0].iloc[-1]['rmse']:.4f}  "
              f"二次RMSE={quad_r[0].iloc[-1]['rmse']:.4f}{tail}")

    ev_df = pd.concat(all_eval, ignore_index=True)
    save_csv(ev_df, "q1_mixture_eval.csv")
    save_csv(pd.DataFrame(calib_rows), "q1_mixture_calibration.csv")
    print("\n--- 跨尺度尺度偏移诊断 ---")
    print(pd.DataFrame(calib_rows).to_string(index=False))
    rg_df = pd.DataFrame(all_regret)
    save_csv(rg_df, "q1_mixture_regret.csv")
    print("\n--- 后悔值 / Top-k ---")
    print(rg_df[["split", "model", "n", "regret_top1", "regret_top5", "regret_top10",
                 "baseline_regret"]].to_string(index=False))

    # ---- 系数稳定性（Bootstrap）----
    rng = np.random.default_rng(SEED)
    B = 200
    coefs = np.zeros((B, Xquad.shape[1], k))
    for b in range(B):
        s = rng.integers(0, n, n)
        coefs[b] = fit_ridge(Xquad[s], Ytr[s], a_quad).coef_.T
    coef_mean, coef_sd = coefs.mean(0), coefs.std(0)
    names = design_names(True)
    a_coef = pd.DataFrame(coef_mean[:17], index=TRAIN_DOMAINS, columns=LOSS_DOMAINS)
    a_sd = pd.DataFrame(coef_sd[:17], index=TRAIN_DOMAINS, columns=LOSS_DOMAINS)
    a_coef.to_csv(RESULTS_PATH("q1_mixture_linear_coef.csv"), float_format="%.6g")
    a_sd.to_csv(RESULTS_PATH("q1_mixture_linear_coef_sd.csv"), float_format="%.6g")

    b_coef = pd.DataFrame(coef_mean[17:], index=names[17:], columns=LOSS_DOMAINS)
    b_sd = pd.DataFrame(coef_sd[17:], index=names[17:], columns=LOSS_DOMAINS)
    b_ratio = (b_coef.abs() / (b_sd + 1e-12))
    b_coef.to_csv(RESULTS_PATH("q1_mixture_inter_coef.csv"), float_format="%.6g")
    b_sd.to_csv(RESULTS_PATH("q1_mixture_inter_coef_sd.csv"), float_format="%.6g")
    n_sig = int((b_ratio > 2).sum().sum())
    print(f"\n交互项系数: 共 {b_ratio.size} 个, |b|/sd > 2 的 {n_sig} 个 "
          f"({100*n_sig/b_ratio.size:.1f}%)")

    # ---- 保存预测器与元数据 ----
    np.savez(RESULTS / "q1_mixture_model.npz",
             coef_lin=mdl_lin.coef_, coef_quad=mdl_quad.coef_,
             alpha_lin=a_lin, alpha_quad=a_quad)
    save_json(dict(
        model="二次混料岭回归 (Scheffé 规范形)",
        formula="L_k(p)=sum_i a_ki p_i + sum_{i<j} b_kij p_i p_j",
        n_train=int(n), n_linear=17, n_interaction=136,
        alpha_linear=float(a_lin), alpha_quadratic=float(a_quad),
        target_domains=LOSS_DOMAINS, input_domains=TRAIN_DOMAINS,
        no_loss_domains=["nih_exporter", "enron_emails", "europarl", "philpapers"],
        weights_v="equal (1/13)", source="A4/A5 train, A6-A11 test, A12-A15 extrapolation",
        cv=cv_rows,
    ), "q1_mixture_meta.json")

    print("\n完成：q1_mixture_*.csv / q1_mixture_model.npz / q1_mixture_meta.json")


def RESULTS_PATH(name):
    from common import RESULTS as R
    return R / name


if __name__ == "__main__":
    main()
