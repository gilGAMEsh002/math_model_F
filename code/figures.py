# -*- coding: utf-8 -*-
"""
论文配图：统一风格（SimSun 中文、统一调色板、DPI≥330），全部输出到 figures/。
用法：python code/figures.py [q1|q2|q3|q4|all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (RESULTS, FIGURES, DATA, savefig, panel_label, PALETTE,
                    GROUP_COLORS, QUALITY_COST_COLORS, CTX_COLORS,
                    TRAIN_DOMAINS, LOSS_DOMAINS)

import seaborn as sns

QGROUPS = {
    "教育价值": ["fineweb_edu", "dsir_books", "dsir_wiki", "dsir_math", "qurater", "reason"],
    "可读性":   ["read", "clean", "fluency_en", "ue"],
    "专业性":   ["prof", "mwl", "fuw"],
    "低噪声":   ["ad", "fna", "t2", "t3", "up", "term", "num", "wc", "ns"],
}
METRIC_CN = {
    "fineweb_edu": "教育价值", "fluency_en": "流畅度", "clean": "洁净度", "read": "可读性",
    "reason": "推理性", "prof": "专业性", "dsir_books": "DSIR书籍", "dsir_wiki": "DSIR百科",
    "dsir_math": "DSIR数学", "qurater": "质量评分", "ad": "无广告", "wc": "词数",
    "ns": "句数", "ue": "一元熵", "fuw": "唯一词比", "fna": "非字母词比",
    "t2": "2-gram重复", "t3": "3-gram重复", "up": "大写行比", "term": "终止标点比",
    "num": "数字字符比", "mwl": "平均词长",
}
# 中文组名 → 配色（与 common.GROUP_COLORS 的英文键对应）
GCOLOR_CN = {"教育价值": "#2E5C8A", "可读性": "#4E8A6B",
             "专业性": "#E0A33E", "低噪声": "#C8553D"}
rng = np.random.default_rng(20260101)


# ============================================================ Q1
def fig_q1_all():
    res = pd.read_parquet(RESULTS / "q1_quality_scores.parquet")
    dom = pd.read_csv(RESULTS / "q1_domain_quality.csv")
    lam = pd.read_csv(RESULTS / "q1_lambda_grid.csv")
    cv = pd.read_csv(RESULTS / "q1_quality_augmented_cv.csv")
    rg = pd.read_csv(RESULTS / "q1_mixture_regret.csv")
    ev = pd.read_csv(RESULTS / "q1_mixture_eval.csv")
    mp = pd.read_csv(RESULTS / "q1_domain_mapping.csv")

    # ---- 图 1：22 指标相关聚类热图 ----
    zmet = pd.read_parquet(RESULTS / "q1_metric_z.parquet")
    gcol = {m: g for g, ms in QGROUPS.items() for m in ms}
    sub = zmet.sample(min(8000, len(zmet)), random_state=1)
    Z = sub[list(gcol)].copy()
    Z.columns = [METRIC_CN[m] for m in gcol]
    cg = sns.clustermap(Z.corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                        figsize=(10.2, 9.4), annot=False,
                        row_colors=[GCOLOR_CN[gcol[m]] for m in gcol],
                        col_colors=[GCOLOR_CN[gcol[m]] for m in gcol],
                        cbar_pos=(0.03, 0.83, 0.035, 0.13),
                        dendrogram_ratio=(0.14, 0.14), linewidths=0.4)
    cg.ax_heatmap.tick_params(labelsize=8.4)
    cg.ax_heatmap.set_xlabel(""); cg.ax_heatmap.set_ylabel("")
    cg.figure.legend(handles=[Patch(facecolor=v, label=k) for k, v in GCOLOR_CN.items()],
                     loc="upper right", bbox_to_anchor=(1.02, 1.06), fontsize=9.5, frameon=False)
    cg.figure.suptitle("22 个质量指标的相关结构与语义分组", y=1.015, fontsize=12)
    cg.figure.savefig(FIGURES / "fig_q1_metric_corr.png", dpi=330, bbox_inches="tight")
    cg.figure.savefig(FIGURES / "fig_q1_metric_corr.pdf", bbox_inches="tight")
    plt.close(cg.figure); print("[fig] fig_q1_metric_corr")

    # ---- 图 2：质量分布（1×3 多面板）----
    order = dom.sort_values("Q_mean")._domain.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))
    d2 = res[res._domain.isin(order)]
    sns.violinplot(data=d2, x="_domain", y="Q", order=order, ax=axes[0],
                   palette="crest", inner="quartile", cut=0, linewidth=0.7)
    axes[0].set_xticklabels(order, rotation=32, ha="right")
    axes[0].set_xlabel(""); axes[0].set_ylabel("综合质量 Q")
    axes[0].set_title("各域质量分布")
    panel_label(axes[0], "(a)")
    for i, dd in enumerate(order):
        v = res.loc[res._domain == dd, "Q"].to_numpy()
        xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
        axes[1].plot(xs, ys, lw=1.6, color=PALETTE[i % len(PALETTE)], label=dd)
    axes[1].set_xlabel("综合质量 Q"); axes[1].set_ylabel("累积概率")
    axes[1].set_title("经验累积分布")
    axes[1].legend(fontsize=7.6, ncol=2, frameon=False)
    panel_label(axes[1], "(b)")
    axes[2].hist(res.conflict, bins=70, color="#2E5C8A", alpha=0.85, edgecolor="white", lw=0.3)
    tau = float(np.percentile(res.conflict, 75))
    axes[2].axvline(tau, color="#C8553D", ls="--", lw=1.6, label=f"冲突阈值 τ={tau:.3f}")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("冲突强度 $c=\\max_g z_g-\\min_g z_g$"); axes[2].set_ylabel("样本数（对数）")
    axes[2].set_title("质量冲突强度分布")
    axes[2].legend(fontsize=8.6, frameon=False)
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q1_quality_dist")

    # ---- 图 3：分组质量雷达图 + 域级柱状 ----
    grp_cols = [c for c in res.columns if c.startswith("z_") and not c.endswith("价值") and not c.endswith("性") and not c.endswith("噪声")]
    gm = res.groupby("_domain")[grp_cols].mean()
    fig = plt.figure(figsize=(13.6, 5.2))
    ang = np.linspace(0, 2 * np.pi, len(grp_cols), endpoint=False).tolist()
    ang += ang[:1]
    ax = fig.add_subplot(1, 2, 1, polar=True)
    for i, dname in enumerate(order):
        v = gm.loc[dname, grp_cols].tolist(); v += v[:1]
        ax.plot(ang, v, lw=1.8, marker="o", ms=3.4, color=PALETTE[i % len(PALETTE)], label=dname)
        ax.fill(ang, v, alpha=0.06, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([c.replace("z_", "") for c in grp_cols], fontsize=10)
    ax.set_ylim(0, 1); ax.set_title("各域四维质量画像", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=7.8, frameon=False)
    panel_label(ax, "(a)", dx=-0.16, dy=1.02)
    ax2 = fig.add_subplot(1, 2, 2)
    y = np.arange(len(order))
    ax2.barh(y, dom.set_index("_domain").loc[order, "Q_mean"],
             color=[PALETTE[i % len(PALETTE)] for i in range(len(order))], alpha=0.9, height=0.62)
    ax2.errorbar(dom.set_index("_domain").loc[order, "Q_mean"], y,
                 xerr=[dom.set_index("_domain").loc[order, "Q_mean"] -
                       dom.set_index("_domain").loc[order, "Q_lo"],
                       dom.set_index("_domain").loc[order, "Q_hi"] -
                       dom.set_index("_domain").loc[order, "Q_mean"]],
                 fmt="none", ecolor="#333", capsize=3, lw=1.2)
    ax2.set_yticks(y); ax2.set_yticklabels(order)
    ax2.set_xlabel("域级平均质量 Q（误差棒为 2.5%–97.5% 分位区间）")
    ax2.set_title("域级质量与不确定区间")
    panel_label(ax2, "(b)")
    savefig(fig, "fig_q1_domain_quality")

    # ---- 图 4：冲突结构 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    hb = axes[0].hexbin(res.z_教育价值, res.z_低噪声, gridsize=46, cmap="viridis",
                        bins="log", mincnt=1)
    lims = [0, 1]
    axes[0].plot(lims, lims, "w--", lw=1.4)
    axes[0].set_xlabel("教育价值组得分"); axes[0].set_ylabel("低噪声组得分")
    axes[0].set_title("教育价值 vs 噪声控制")
    plt.colorbar(hb, ax=axes[0], label="样本数（对数）")
    panel_label(axes[0], "(a)")
    cr = dom.set_index("_domain").loc[order, "conflict_rate"]
    axes[1].barh(np.arange(len(order)), cr.values * 100,
                 color=[PALETTE[i % len(PALETTE)] for i in range(len(order))], alpha=0.9, height=0.62)
    axes[1].set_yticks(np.arange(len(order))); axes[1].set_yticklabels(order)
    axes[1].set_xlabel("冲突样本占比 (%)"); axes[1].set_title("各域冲突率")
    panel_label(axes[1], "(b)")
    ct = pd.read_csv(RESULTS / "q1_conflict_types.csv")
    axes[2].barh(np.arange(len(ct)), ct["占比"] * 100, color="#E0A33E", alpha=0.9, height=0.6)
    axes[2].set_yticks(np.arange(len(ct)))
    axes[2].set_yticklabels(ct["类型"], fontsize=8.6)
    axes[2].set_xlabel("样本占比 (%)"); axes[2].set_title("五类典型冲突构成")
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q1_conflict")

    # ---- 图 5：配比模型表现 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    cvp = cv[cv.model.str.contains("二次混料") | cv.model.str.contains("纯质量")]
    lbl = ["二次混料", "二次混料+Q(p)", "纯质量模型"]
    x = np.arange(len(cv))
    axes[0].bar(x - 0.2, cv.cv_rmse, 0.4, label="CV RMSE", color="#2E5C8A", alpha=0.92)
    axes[0].bar(x + 0.2, cv.cv_mae, 0.4, label="CV MAE", color="#E0A33E", alpha=0.92)
    axes[0].set_xticks(x); axes[0].set_xticklabels(lbl, rotation=14, ha="right", fontsize=8.4)
    axes[0].set_ylabel("交叉验证误差"); axes[0].legend(frameon=False, fontsize=8.6)
    axes[0].set_title("配比模型交叉验证（1M 训练集）")
    panel_label(axes[0], "(a)")
    e = ev[(ev.split == "test_1m") & (ev.domain == "__综合(eval)__")]
    axes[1].bar(np.arange(len(e)), e.spearman, 0.5,
                color=["#2E5C8A", "#C8553D", "#4E8A6B"][:len(e)], alpha=0.92)
    axes[1].set_xticks(np.arange(len(e)))
    axes[1].set_xticklabels([s.replace("二次混料+Q(p)", "二次+Q(p)").replace("线性混料", "线性")
                             for s in e.model][:len(e)], fontsize=8.6)
    axes[1].set_ylabel("Spearman 排序相关"); axes[1].set_ylim(0, 0.85)
    axes[1].set_title("1M 检验集排序能力")
    panel_label(axes[1], "(b)")
    for nm, c, mk in [("二次混料", "#2E5C8A", "o"), ("二次混料+Q(p)", "#C8553D", "s")]:
        g = rg[(rg.model == nm) & (rg.split != "train")]
        axes[2].plot(range(len(g)), g.regret_top5, marker=mk, lw=1.8, ms=6,
                     color=c, label=nm)
    axes[2].set_xticks(range(len(g)))
    axes[2].set_xticklabels(g.split.tolist(), rotation=20, ha="right", fontsize=8.4)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Top-5 后悔值（对数）"); axes[2].legend(frameon=False, fontsize=8.6)
    axes[2].set_title("选方后悔值")
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q1_mixture_perf")

    # ---- 图 6：交互项系数热图 + λ 敏感性 ----
    ic = pd.read_csv(RESULTS / "q1_mixture_inter_coef.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    top = ic.abs().mean(axis=1).sort_values(ascending=False).head(22).index
    M = ic.loc[top, LOSS_DOMAINS]
    sns.heatmap(M.T, cmap="RdBu_r", center=0, ax=axes[0], cbar_kws=dict(label="交互系数 $b_{kij}$"),
                linewidths=0.35, linecolor="white")
    axes[0].set_xticklabels([s.replace("*", "×") for s in top], rotation=60, ha="right", fontsize=7.6)
    axes[0].set_yticklabels(LOSS_DOMAINS, fontsize=8.0, rotation=0)
    axes[0].set_title("交互项系数（按平均绝对强度取前 22）")
    panel_label(axes[0], "(a)", dx=-0.10)
    axes[1].plot(lam.lam, lam.Q_mean, "o-", color="#2E5C8A", lw=2, ms=6, label="Q 均值")
    ax1b = axes[1].twinx()
    ax1b.plot(lam.lam, lam.rank_corr_with_base, "s--", color="#C8553D", lw=1.8, ms=5.5,
              label="与 λ=0 的秩相关")
    ax1b.set_ylabel("与 $\\lambda$=0 的 Spearman 秩相关", color="#C8553D")
    ax1b.tick_params(axis="y", colors="#C8553D")
    axes[1].set_xlabel("冲突消解强度 $\\lambda$"); axes[1].set_ylabel("质量分均值", color="#2E5C8A")
    axes[1].tick_params(axis="y", colors="#2E5C8A")
    axes[1].set_title("冲突消解强度的影响")
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.4, loc="center right")
    panel_label(axes[1], "(b)")
    savefig(fig, "fig_q1_interaction")


# ============================================================ Q2
def fig_q2_all():
    diag = pd.read_csv(RESULTS / "q2_source_diagnosis.csv")
    val = pd.read_csv(RESULTS / "q2_validation_sources.csv")
    el = pd.read_csv(RESULTS / "q2_elasticities_classic.csv")
    eq = pd.read_csv(RESULTS / "q2_quality_equivalence.csv")
    par = json.loads((RESULTS / "q2_scaling_parameters.json").read_text(encoding="utf-8"))
    th = par["classic"]["params"]
    E, A, al, Bc, be = th["E"], th["A"], th["alpha"], th["B"], th["beta"]
    kap = par["general"]["kappa_staged"]

    from common import DATA
    B = DATA / "B_scaling_laws"
    b1 = pd.read_csv(B / "pythia_training_log_existing.csv")

    # ---- 图 7：经典律拟合（2×2）----
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    for i, (n, c) in enumerate(sorted(b1.groupby("N_params_B").size().items())):
        g = b1[b1.N_params_B == n].sort_values("D_tokens_B")
        axes[0, 0].plot(g.D_tokens_B, g.val_loss, lw=1.5, color=PALETTE[i % len(PALETTE)],
                        label=f"N={n:.3g}B")
    axes[0, 0].set_xscale("log"); axes[0, 0].set_xlabel("训练 Token 数 D（B）")
    axes[0, 0].set_ylabel("验证交叉熵损失")
    axes[0, 0].set_title("B1 训练轨迹（8 个规模）")
    axes[0, 0].legend(fontsize=7.4, ncol=2, frameon=False)
    panel_label(axes[0, 0], "(a)")
    Dg = np.logspace(np.log10(b1.D_tokens_B.min()), np.log10(b1.D_tokens_B.max()), 200)
    for i, n in enumerate(sorted(b1.N_params_B.unique())):
        axes[0, 1].plot(Dg, E + A * n ** (-al) + Bc * Dg ** (-be), lw=1.6,
                        color=PALETTE[i % len(PALETTE)])
        g = b1[b1.N_params_B == n]
        axes[0, 1].scatter(g.D_tokens_B, g.val_loss, s=6, color=PALETTE[i % len(PALETTE)], alpha=0.6)
    axes[0, 1].set_xscale("log"); axes[0, 1].set_xlabel("D（B）"); axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title(f"经典律拟合 $L=E+AN^{{-\\alpha}}+BD^{{-\\beta}}$")
    panel_label(axes[0, 1], "(b)")
    pred = E + A * b1.N_params_B ** (-al) + Bc * b1.D_tokens_B ** (-be)
    r = b1.val_loss - pred
    axes[1, 0].scatter(pred, r, s=5, alpha=0.35, color="#2E5C8A")
    axes[1, 0].axhline(0, color="#C8553D", lw=1.4)
    axes[1, 0].set_xlabel("拟合值"); axes[1, 0].set_ylabel("残差")
    axes[1, 0].set_title(f"残差图（max|r|={np.abs(r).max():.1e}）")
    panel_label(axes[1, 0], "(c)")
    axes[1, 1].barh(np.arange(len(diag)), diag.r2,
                    color=["#C8553D" if v > 0.9995 else "#2E5C8A" for v in diag.r2],
                    alpha=0.9, height=0.62)
    axes[1, 1].set_yticks(np.arange(len(diag)))
    axes[1, 1].set_yticklabels(diag.dataset, fontsize=8.2)
    axes[1, 1].set_xlabel("对经典律的 $R^2$"); axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].axvline(0.9995, color="#C8553D", ls="--", lw=1.3)
    axes[1, 1].set_title("来源诊断：红色为公式生成（$R^2\\to1$）")
    panel_label(axes[1, 1], "(d)")
    savefig(fig, "fig_q2_scaling_fit")

    # ---- 图 8：质量效应 κ ----
    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for i, (n, d) in enumerate([(0.07, 10), (1.0, 100), (11.97, 600)]):
        g = b6[(b6.N_params_B == n) & (b6.D_tokens_B == d)].sort_values("Q_score")
        if len(g) < 3:
            continue
        axes[0].plot(g.Q_score, g.val_loss, "o-", color=PALETTE[i], ms=5.5, lw=1.8,
                     label=f"N={n}B, D={d}B")
        qq = np.linspace(g.Q_score.min(), 1, 100)
        axes[0].plot(qq, E + A * n ** (-al) + Bc * (d * qq ** kap) ** (-be), "--",
                     color=PALETTE[i], lw=1.4)
    axes[0].set_xlabel("数据质量 Q"); axes[0].set_ylabel("验证损失")
    axes[0].legend(fontsize=8.4, frameon=False)
    axes[0].set_title(f"质量—损失关系（虚线为 $\\kappa$={kap:.2f} 拟合）")
    panel_label(axes[0], "(a)")
    ks = np.linspace(0.0, 2.5, 300)
    Nq, Dq, Qq, Lq = (b6.N_params_B.to_numpy(float), b6.D_tokens_B.to_numpy(float),
                      b6.Q_score.to_numpy(float), b6.val_loss.to_numpy(float))
    sse = [np.mean((E + A * Nq ** (-al) + Bc * (Dq * Qq ** k) ** (-be) - Lq) ** 2) for k in ks]
    axes[1].plot(ks, sse, lw=2, color="#2E5C8A")
    axes[1].axvline(kap, color="#C8553D", ls="--", lw=1.6, label=f"最优 $\\kappa$={kap:.3f}")
    axes[1].axvline(1.0, color="#4E8A6B", ls=":", lw=1.6, label="$\\kappa$=1（等量折算）")
    axes[1].set_xlabel("质量指数 $\\kappa$"); axes[1].set_ylabel("均方误差")
    axes[1].set_title("κ 的剖面似然"); axes[1].legend(fontsize=8.4, frameon=False)
    panel_label(axes[1], "(b)")
    axes[2].barh(np.arange(len(val)), val.r2,
                 color=["#4E8A6B" if v > 0.8 else "#E0A33E" if v > 0.3 else "#C8553D"
                        for v in val.r2], alpha=0.92, height=0.6)
    axes[2].set_yticks(np.arange(len(val)))
    axes[2].set_yticklabels([f"{a}\n({b[:2]})" for a, b in zip(val.dataset, val.level)],
                            fontsize=7.8)
    axes[2].set_xlabel("$R^2$（经典参数直接外推）"); axes[2].set_xlim(-0.2, 1.05)
    axes[2].axvline(0, color="#333", lw=1)
    axes[2].set_title("跨来源泛化能力")
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q2_quality_kappa")

    # ---- 图 9：弹性与等价条件 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    Nv = np.logspace(-1, 3, 90); Dv = np.logspace(1, 5, 90)
    NN, DD = np.meshgrid(Nv, Dv, indexing="ij")
    LL = E + A * NN ** (-al) + Bc * DD ** (-be)
    ee = -al * A * NN ** (-al) / LL
    im = axes[0].pcolormesh(NN, DD, ee, cmap="mako_r", shading="auto")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("参数量 N（B）"); axes[0].set_ylabel("数据量 D（B）")
    axes[0].set_title("Loss 对 N 的弹性 $\\varepsilon_N$")
    plt.colorbar(im, ax=axes[0])
    panel_label(axes[0], "(a)")
    eD = -be * Bc * DD ** (-be) / LL
    im = axes[1].pcolormesh(NN, DD, eD, cmap="mako_r", shading="auto")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("N（B）"); axes[1].set_ylabel("D（B）")
    axes[1].set_title("Loss 对 D 的弹性 $\\varepsilon_D$")
    plt.colorbar(im, ax=axes[1])
    panel_label(axes[1], "(b)")
    for q0, c in zip([0.5, 0.7, 0.9], ["#2E5C8A", "#E0A33E", "#C8553D"]):
        g = eq[(eq.D_B == 100) & (eq.Q0 == q0)]
        if len(g):
            axes[2].plot(g.N_B, g.N_eq_ratio * 100, "o-", color=c, lw=1.9, ms=6,
                         label=f"$Q_0$={q0}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("基准参数量 N（B）"); axes[2].set_ylabel("所需规模增量 $N_{eq}/N-1$ (%)")
    axes[2].set_title("质量提升 0.1 的等效规模（D=100B）")
    axes[2].legend(frameon=False, fontsize=8.6)
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q2_elasticity")


# ============================================================ Q3
def fig_q3_all():
    alloc = pd.read_csv(RESULTS / "q3_optimal_allocations.csv")
    paths = pd.read_csv(RESULTS / "q3_budget_paths.csv")
    sens = pd.read_csv(RESULTS / "q3_sensitivity.csv")
    tr = pd.read_csv(RESULTS / "q3_transition_candidates.csv")

    # ---- 图 10：最优配置随预算 ----
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    for cf, c in QUALITY_COST_COLORS.items():
        nm = {"exp": "指数型", "power": "幂函数型", "log": "对数渐进型"}[cf]
        g = paths[(paths.cost == nm) & (paths.ell == 8192)].sort_values("logC")
        axes[0, 0].plot(g.logC, g.N, lw=2.2, color=c, label=nm)
        axes[0, 1].plot(g.logC, g.D, lw=2.2, color=c, label=nm)
        axes[1, 0].plot(g.logC, g.Q, lw=2.2, color=c, label=nm)
        axes[1, 1].plot(g.logC, g.tokens_per_param, lw=2.2, color=c, label=nm)
    for ax, yl, tt in [(axes[0, 0], "最优参数量 $N^*$", "(a) 最优规模随预算"),
                       (axes[0, 1], "最优数据量 $D^*$（B tokens）", "(b) 最优数据量随预算"),
                       (axes[1, 0], "最优数据质量 $Q^*$", "(c) 最优质量随预算"),
                       (axes[1, 1], "Token/参数比 $D^*/N^*$", "(d) 训练配比随预算")]:
        ax.set_yscale("log"); ax.set_xlabel("$\\log_{10} C$ (FLOPs)"); ax.set_ylabel(yl)
        ax.set_title(tt)
    axes[0, 0].legend(frameon=False, fontsize=8.6)
    for ax in axes.ravel():
        ax.axvline(np.log10(1e19), color="#999", ls=":", lw=1.2)
        ax.axvline(np.log10(1e22), color="#999", ls=":", lw=1.2)
        ax.axvline(np.log10(1e24), color="#999", ls=":", lw=1.2)
    for i, ax in enumerate(axes.ravel()):
        panel_label(ax, "abcd"[i])
    savefig(fig, "fig_q3_budget_paths")

    # ---- 图 11：成本份额 + 上下文影响 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
    g = paths[(paths.cost == "指数型") & (paths.ell == 8192)].sort_values("logC")
    axes[0].stackplot(g.logC, g.share_train * 100, g.share_quality * 100, g.share_attn * 100,
                      labels=["训练 6ND", "质量提升 $C_Q$", "注意力 $C_{attn}$"],
                      colors=["#2E5C8A", "#C8553D", "#E0A33E"], alpha=0.88)
    axes[0].set_xlabel("$\\log_{10} C$"); axes[0].set_ylabel("预算份额 (%)")
    axes[0].set_title("成本份额随预算演化（指数型，$\\ell$=8192）")
    axes[0].legend(frameon=False, fontsize=8.2, loc="lower left")
    axes[0].set_ylim(0, 100)
    panel_label(axes[0], "(a)")
    sub = alloc[(alloc.C == 1e22) & (alloc.cost == "指数型")].sort_values("ell")
    x = np.arange(len(sub))
    axes[1].bar(x, sub.share_attn * 100, 0.55, color=[CTX_COLORS[e] for e in sub.ell], alpha=0.95)
    axes[1].axhline(50, color="#C8553D", ls="--", lw=1.4, label="注意力占一半")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{int(e/1024)}K" if e >= 1024 else str(e)
                                                    for e in sub.ell], fontsize=8.6)
    axes[1].set_xlabel("上下文长度 $\\ell$"); axes[1].set_ylabel("注意力成本份额 (%)")
    axes[1].set_title("上下文长度挤压可用预算（$C$=1e22）")
    axes[1].legend(frameon=False, fontsize=8.4)
    panel_label(axes[1], "(b)")
    sub2 = alloc[(alloc.ell == 8192)].pivot_table(index="C", columns="cost", values="Q")
    for cf, c in QUALITY_COST_COLORS.items():
        nm = {"exp": "指数型", "power": "幂函数型", "log": "对数渐进型"}[cf]
        axes[2].scatter(sub2.index, sub2[nm], s=110, color=c, label=nm, zorder=3)
    axes[2].axhline(0.5, color="#333", ls=":", lw=1.3, label="$Q_0$=0.5（下界）")
    axes[2].axhline(1.0, color="#C8553D", ls="--", lw=1.3, label="Q 上界")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("算力预算 $C$（FLOPs）"); axes[2].set_ylabel("最优质量 $Q^*$")
    axes[2].set_title("质量投入的三段式结构")
    axes[2].legend(frameon=False, fontsize=8.0)
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q3_cost_shares")

    # ---- 图 12：优化曲面 + 敏感性龙卷风 ----
    from q3_optimize import COST_FUNCS, Q0_DEFAULT, TH, D_from_budget, loss, ETA
    fig = plt.figure(figsize=(13.8, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cf = COST_FUNCS["指数型"]
    ln = np.linspace(7.5, 11.5, 90); qq = np.linspace(0.5, 1.0, 70)
    LN, QQ = np.meshgrid(ln, qq, indexing="ij")
    Nn = 10 ** LN
    Dd = 1e22 / ((6 + ETA * 8192) * Nn + np.maximum(cf["g"](QQ) - cf["g"](Q0_DEFAULT), 0))
    LL = loss(Nn, Dd, QQ)
    ax.plot_surface(LN, QQ, LL, cmap="viridis", alpha=0.92, linewidth=0, antialiased=True)
    i, j = np.unravel_index(np.argmin(LL), LL.shape)
    ax.scatter([LN[i, j]], [QQ[i, j]], [LL[i, j]], color="#C8553D", s=90, marker="*",
               depthshade=False, label="最优点")
    ax.set_xlabel("$\\log_{10} N$"); ax.set_ylabel("$Q$"); ax.set_zlabel("预测损失", labelpad=-2)
    ax.set_title("$(N,Q)$ 目标曲面（$C$=1e22, $\\ell$=8192）", pad=2)
    ax.legend(fontsize=8.6, loc="upper left")
    ax.view_init(24, -128)
    base = sens[sens.scenario == "基准"].set_index("cost")
    ax2 = fig.add_subplot(1, 2, 2)
    sc = sorted(sens.scenario.unique())
    sc = [s for s in sc if s != "基准"]
    ys = np.arange(len(sc))
    for cf_n, c in QUALITY_COST_COLORS.items():
        nm = {"exp": "指数型", "power": "幂函数型", "log": "对数渐进型"}[cf_n]
        delta = [(sens[(sens.scenario == s) & (sens.cost == nm)].N.iloc[0] /
                  base.loc[nm, "N"] - 1) * 100 for s in sc]
        ax2.barh(ys + (list(QUALITY_COST_COLORS).index(cf_n) - 1) * 0.26, delta, 0.24,
                 color=c, alpha=0.95, label=nm)
    ax2.axvline(0, color="#333", lw=1.2)
    ax2.set_yticks(ys); ax2.set_yticklabels(sc, fontsize=8.8)
    ax2.set_xlabel("最优规模 $N^*$ 相对基准的变化 (%)")
    ax2.set_title("参数敏感性")
    ax2.legend(frameon=False, fontsize=8.0)
    panel_label(ax2, "(b)")
    savefig(fig, "fig_q3_opt_surface")


# ============================================================ Q4
def fig_q4_all():
    q = pd.read_csv(RESULTS / "q4_quarterly_summary.csv")
    dec = pd.read_csv(RESULTS / "q4_contribution_decomposition.csv")
    br = pd.read_csv(RESULTS / "q4_bridge_parameters.csv")
    fc = pd.read_csv(RESULTS / "q4_frontier_forecasts.csv")
    md = pd.read_csv(RESULTS / "q4_model_metadata_matches.csv", low_memory=False)
    c8v = pd.read_csv(RESULTS / "q4_c8_vs_c1.csv")

    md["sub_date"] = pd.to_datetime(md["Submission Date"], errors="coerce")
    md = md.dropna(subset=["sub_date", "#Params (B)", "Average ⬆️"])

    # ---- 图 13：能力演进时间线 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
    pre = md[~md.Type.astype(str).str.contains("chat", case=False, na=False)]
    cha = md[md.Type.astype(str).str.contains("chat", case=False, na=False)]
    axes[0].scatter(pre.sub_date, pre["Average ⬆️"], s=6, alpha=0.28, color="#2E5C8A",
                    label="pretrained")
    axes[0].scatter(cha.sub_date, cha["Average ⬆️"], s=6, alpha=0.32, color="#C8553D",
                    label="chat/finetuned")
    qq = md.assign(q=md.sub_date.dt.to_period("Q")).groupby("q")["Average ⬆️"].quantile(0.9)
    axes[0].plot([p.to_timestamp(how="end") for p in qq.index], qq.values, "k-o", lw=2.2,
                 ms=6, label="0.9 分位前沿")
    axes[0].set_ylabel("综合能力得分"); axes[0].legend(fontsize=8.0, frameon=False)
    axes[0].set_title("能力演进（按提交日期）")
    axes[0].tick_params(axis="x", rotation=22)
    panel_label(axes[0], "(a)")
    for c, nm in [("share_scale", "规模扩张"), ("share_nonscale", "非规模技术进步")]:
        g = dec[dec.tag == "固定组成(≥7B预训练)"]
        axes[1].bar(np.arange(len(g)) + (0 if c == "share_scale" else 0.38), g[c] * 100, 0.36,
                    color="#2E5C8A" if c == "share_scale" else "#C8553D", alpha=0.93, label=nm)
    g = dec[dec.tag == "固定组成(≥7B预训练)"]
    axes[1].set_xticks(np.arange(len(g)) + 0.19)
    axes[1].set_xticklabels([f"{a}→{b}" for a, b in zip(g.q_to, g.q_to)], fontsize=7.6, rotation=18)
    axes[1].axhline(0, color="#333", lw=1.1)
    axes[1].set_ylabel("对 ΔS 的贡献占比 (%)"); axes[1].legend(fontsize=8.2, frameon=False)
    axes[1].set_title("贡献分解（固定组成子集）")
    panel_label(axes[1], "(b)")
    axes[2].scatter(c8v.bbh_macro, c8v.BBH, s=5, alpha=0.22, color="#2E5C8A")
    b, a = np.polyfit(c8v.bbh_macro, c8v.BBH, 1)
    xs = np.linspace(c8v.bbh_macro.min(), c8v.bbh_macro.max(), 50)
    axes[2].plot(xs, a + b * xs, color="#C8553D", lw=2,
                 label=f"C1 = {a:.1f} + {b:.3f}·C8\n$\\rho$={c8v.bbh_macro.corr(c8v.BBH):.4f}")
    axes[2].set_xlabel("C8 逐任务宏平均 BBH"); axes[2].set_ylabel("C1 排行榜 BBH")
    axes[2].set_title("逐任务聚合与榜单口径的差异")
    axes[2].legend(fontsize=8.2, frameon=False)
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q4_timeline")

    # ---- 图 14：Loss-Benchmark 桥接 ----
    from common import DATA
    c6 = pd.read_csv(DATA / "C_efficiency_evolution" / "loss_benchmark_bridge_expanded.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for lv, c, mk in [("High", "#C8553D", "o"), ("Medium", "#2E5C8A", "s")]:
        g = c6[c6.Loss_Comparability.str.startswith(lv)]
        axes[0].scatter(g.Val_Loss, g.LB_Average, s=34, alpha=0.75, color=c, marker=mk,
                        label=f"{lv} 可比 (n={len(g)})")
    for lv, c in [("High", "#C8553D"), ("Medium", "#2E5C8A")]:
        r = br[(br.dataset == "C6") & (br.level == lv)]
        if len(r):
            xs = np.linspace(1.6, 2.9, 40)
            axes[0].plot(xs, r.a.iloc[0] + r.b.iloc[0] * xs, color=c, lw=1.8, ls="--")
    axes[0].set_xlabel("验证交叉熵损失"); axes[0].set_ylabel("排行榜综合得分")
    axes[0].set_title("Loss–Benchmark 桥接（虚线为分层拟合）")
    axes[0].legend(fontsize=8.0, frameon=False)
    panel_label(axes[0], "(a)")
    axes[1].bar(np.arange(len(br)), br.r2, 0.5,
                color=["#C8553D", "#2E5C8A", "#E0A33E", "#4E8A6B"][:len(br)], alpha=0.92)
    axes[1].set_xticks(np.arange(len(br)))
    axes[1].set_xticklabels([f"{d}\n{l}" for d, l in zip(br.dataset, br.level)], fontsize=7.8)
    axes[1].set_ylabel("$R^2$"); axes[1].set_title("桥接拟合优度（普遍偏弱）")
    panel_label(axes[1], "(b)")
    loo = pd.read_csv(RESULTS / "q4_bridge_validation.csv")
    for lv, c in zip(loo.level.unique(), ["#C8553D", "#2E5C8A"]):
        g = loo[loo.level == lv]
        axes[2].scatter(g.S_true, g.S_pred, s=30, alpha=0.72, color=c, label=f"{lv} 留一")
    lim = [0, 50]
    axes[2].plot(lim, lim, "k--", lw=1.3)
    axes[2].set_xlabel("真实得分"); axes[2].set_ylabel("留一预测得分")
    axes[2].set_title("桥接留一模型验证"); axes[2].legend(fontsize=8.2, frameon=False)
    panel_label(axes[2], "(c)")
    savefig(fig, "fig_q4_bridge")

    # ---- 图 15：前沿预测 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    pre = md[~md.Type.astype(str).str.contains("chat", case=False, na=False)]
    axes[0].scatter(pre.sub_date, pre["Average ⬆️"], s=5, alpha=0.22, color="#9DB4C8")
    qq = md.assign(q=md.sub_date.dt.to_period("Q")).groupby("q")["Average ⬆️"].quantile(0.9)
    tq = [p.to_timestamp(how="end") for p in qq.index]
    axes[0].plot(tq, qq.values, "o-", color="#2E5C8A", lw=2.2, ms=6, label="历史 0.9 分位前沿")
    S0 = fc.S0.iloc[0]
    t0 = pd.Timestamp("2025-03-13")
    colors = {"增长延续 g": "#C8553D", "增长放缓 0.5g": "#E0A33E", "算力近乎不变 0.1g": "#4E8A6B"}
    for sc in fc.scenario.unique():
        g = fc[fc.scenario == sc].sort_values("months")
        ts = [t0] + [t0 + pd.Timedelta(days=30.44 * m) for m in g.months]
        vs = [S0] + g.S_pred.tolist()
        axes[0].plot(ts, vs, "o--", color=colors[sc], lw=2.0, ms=6, label=sc)
        axes[0].fill_between([t0] + [t0 + pd.Timedelta(days=30.44 * m) for m in g.months],
                             [S0] + g.S_ci_lo.tolist(), [S0] + g.S_ci_hi.tolist(),
                             color=colors[sc], alpha=0.16)
    axes[0].axvline(t0, color="#333", ls=":", lw=1.4)
    axes[0].axhline(float(fc.S_obs_max.iloc[0]), color="#888", ls="-.", lw=1.3,
                    label="样本内观测上限")
    axes[0].set_ylabel("开源模型综合能力前沿"); axes[0].legend(fontsize=7.8, frameon=False)
    axes[0].set_title("前沿预测与 95% 区间")
    axes[0].tick_params(axis="x", rotation=22)
    panel_label(axes[0], "(a)")
    x = np.arange(len(fc))
    axes[1].bar(x, fc.S_pred - fc.S0, 0.55,
                color=[colors[s] for s in fc.scenario], alpha=0.93)
    axes[1].errorbar(x, fc.S_pred - fc.S0,
                     yerr=[fc.S_pred - fc.S_ci_lo, fc.S_ci_hi - fc.S_pred],
                     fmt="none", ecolor="#333", capsize=4, lw=1.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s}\n+{m}月" for s, m in zip(fc.scenario, fc.months)], fontsize=7.6)
    axes[1].set_ylabel("相对当前前沿的提升（分）")
    axes[1].set_title("不同算力情景下的能力增量")
    panel_label(axes[1], "(b)")
    savefig(fig, "fig_q4_forecast")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in [("q1", fig_q1_all), ("q2", fig_q2_all),
                  ("q3", fig_q3_all), ("q4", fig_q4_all)]:
        if which in ("all", k):
            print(f"--- {k} ---")
            fn()
    print("全部图完成 ->", FIGURES)
