# -*- coding: utf-8 -*-
"""
第一问（第三部分收尾）：跨体系关联与质量增强的配比模型。

两个域体系不同（质量信号 7 域 vs 配方 17 域，A16 给出参考映射），据此：
  1. 建立「配方域 → 质量域」映射：direct / near_direct 直接采用；
     13 个 inferred 域无同名质量域，按文本体裁给出**显式代理**并标注不确定性。
  2. 计算配比层面的综合质量  Q(p) = Σ_i p_i · q_i
  3. 质量增强配比模型：L_k(p) = Σ a_ki p_i + Σ b_kij p_i p_j + c_k · Q(p)
     与不含 Q 的二次混料模型在同一训练/检验切分下比较，判断引入 Q 是否带来
     留出增益；不把不可靠的域映射当作已验证事实。
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
from common import DATA, RESULTS, TRAIN_DOMAINS, LOSS_DOMAINS, save_csv, save_json, SEED, rmse, mae, r2, spearman
from q1_mixture import load, design, fit_ridge, cv_alpha, evaluate, regret, RT

# 13 个无同名质量域的代理指派（按文本体裁），必须与论文中说明一致
PROXY = {
    "dm_mathematics":   ("arxiv",        "长篇技术/公式文本，与 arxiv 体裁最接近"),
    "pubmed_central":   ("arxiv",        "学术论文全文，与 arxiv 体裁接近"),
    "pubmed_abstracts": ("arxiv",        "学术摘要，与 arxiv 体裁接近"),
    "philpapers":       ("arxiv",        "学术论文，与 arxiv 体裁接近"),
    "nih_exporter":     ("arxiv",        "科研项目文本，与 arxiv 体裁接近"),
    "uspto_backgrounds":("arxiv",        "专利背景技术描述，与 arxiv 体裁接近"),
    "freelaw":          ("wikipedia",    "正式书面语，参考 wikipedia 的规范性"),
    "europarl":         ("wikipedia",    "议会记录，正式书面语，参考 wikipedia"),
    "ubuntu_irc":       ("stackexchange","技术问答/聊天，参考 stackexchange"),
    "hackernews":       ("stackexchange","技术社区讨论，参考 stackexchange"),
    "enron_emails":     ("commoncrawl",  "非正式日常文本，参考 commoncrawl"),
}


def build_mapping(dom_quality):
    """返回 17×N 的映射表。"""
    g = pd.read_csv(DATA / "A_data_value" / "domain_mapping_guide.csv")
    qmap = dict(zip(dom_quality["_domain"], dom_quality["Q_mean"]))
    rows = []
    for _, r in g.iterrows():
        md, qd, mt = r["mixture_domain"], r["quality_domain"], r["mapping_type"]
        if mt in ("direct", "near_direct") and qd in qmap:
            rows.append(dict(mixture_domain=md, quality_domain=qd, mapping_type=mt,
                             q=qmap[qd], note=r["note"]))
        else:
            pd_, why = PROXY.get(md, ("arxiv", "无代理，默认取 arxiv"))
            rows.append(dict(mixture_domain=md, quality_domain=pd_,
                             mapping_type="inferred_proxy", q=qmap.get(pd_, np.nan),
                             note=f"无同名质量域；按体裁代理到 {pd_}：{why}"))
    return pd.DataFrame(rows)


def main():
    print("=" * 100)
    print("第一问 · 跨体系关联与质量增强配比模型")
    print("=" * 100)
    dom_quality = pd.read_csv(RESULTS / "q1_domain_quality.csv")
    mp = build_mapping(dom_quality)
    save_csv(mp, "q1_domain_mapping.csv")
    print("\n[配方域 → 质量域 映射]")
    print(mp[["mixture_domain", "quality_domain", "mapping_type", "q"]].to_string(index=False))

    qvec = mp.set_index("mixture_domain").loc[TRAIN_DOMAINS, "q"].to_numpy(float)
    print(f"\n17 域质量向量 q: 均值={qvec.mean():.4f} 范围=[{qvec.min():.4f}, {qvec.max():.4f}]")

    # ---- 训练/检验：与 q1_mixture 完全相同的切分 ----
    Ptr, Ytr, _, _ = load("train")
    Xq = design(Ptr, True)
    k = Ytr.shape[1]
    v = np.ones(k) / k
    alphas = np.logspace(-4, 3, 22)
    a_quad, _ = cv_alpha(Xq, Ytr, alphas)

    Qp_tr = Ptr @ qvec                                     # Q(p) = Σ p_i q_i
    # 关键诊断：Q(p) 是 p 的线性函数，与线性设计块完全共线——
    # 因此「二次混料 + Q(p)」在无惩罚时与二次混料等价，c_k 不可单独辨识。
    Xlin_tr = design(Ptr, False)
    b_col = np.linalg.lstsq(Xlin_tr, Qp_tr, rcond=None)[0]
    collin_r2 = 1 - np.sum((Qp_tr - Xlin_tr @ b_col) ** 2) / np.sum((Qp_tr - Qp_tr.mean()) ** 2)
    print(f"\n[共线性诊断] 用 17 个线性配比项回归 Q(p) 的 R² = {collin_r2:.10f} "
          f"（=1 表示完全共线，c_k 不可单独辨识）")

    XqQ = np.hstack([Xq, (Qp_tr * len(TRAIN_DOMAINS))[:, None]])
    a_q, _ = cv_alpha(XqQ, Ytr, alphas)
    # 简约「纯质量」模型：每个目标域 2 个参数，L_k = a_k + c_k·Q(p)
    # （必须带截距，否则退化为把 Q 直接缩放到 Loss 量级）
    qs = Qp_tr * len(TRAIN_DOMAINS)
    Xonly = np.column_stack([np.ones(len(qs)), qs])
    a_only, _ = cv_alpha(Xonly, Ytr, alphas)
    print(f"CV 惩罚: 二次混料 alpha={a_quad:.4g}; 二次+Q alpha={a_q:.4g}; "
          f"纯质量 alpha={a_only:.4g}")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = []
    for nm, X, a, npar in [("二次混料（136+17 项/域）", Xq, a_quad, 153),
                           ("二次混料+Q(p)（共线，仅作正则化对照）", XqQ, a_q, 154),
                           ("纯质量模型 L=a+c·Q(p)（1 项/域）", Xonly, a_only, 1)]:
        oof = np.zeros_like(Ytr)
        for tr, va in kf.split(X):
            oof[va] = fit_ridge(X[tr], Ytr[tr], a).predict(X[va])
        cv.append(dict(model=nm, alpha=a, n_par_per_domain=npar,
                       cv_rmse=rmse(Ytr @ v, oof @ v),
                       cv_mae=mae(Ytr @ v, oof @ v), cv_r2=r2(Ytr @ v, oof @ v),
                       cv_spearman=spearman(Ytr @ v, oof @ v)))
    cv_df = pd.DataFrame(cv)
    save_csv(cv_df, "q1_quality_augmented_cv.csv")
    print("\n[是否引入 Q 的留出比较]")
    print(cv_df.to_string(index=False))

    # 纯质量模型的 c_k（应为负：Q 越高 Loss 越低）
    mdl_only = fit_ridge(Xonly, Ytr, a_only)
    a_only_int = mdl_only.coef_[:, 0]                    # 截距（13 目标）
    c_only = mdl_only.coef_[:, 1] / len(TRAIN_DOMAINS)   # Q(p) 斜率
    coef_simple = pd.DataFrame(dict(domain=LOSS_DOMAINS, a_k=a_only_int, c_k=c_only))
    save_csv(coef_simple, "q1_quality_only_coef.csv")
    print("\n[纯质量模型系数]（c_k<0 表示该域质量越高 Loss 越低）")
    print(coef_simple.to_string(index=False))
    print(f"  c_k < 0 的域: {int((c_only < 0).sum())}/{len(c_only)}")

    # ---- 各检验集 ----
    rows = []
    for split in ["test_1m", "test_60m", "test_1B"]:
        P, Y, _, _ = load(split)
        Qp = P @ qvec
        for nm, X in [("二次混料", design(P, True)),
                      ("二次混料+Q(p)", np.hstack([design(P, True),
                                                  (Qp * len(TRAIN_DOMAINS))[:, None]]))]:
            a = a_quad if nm == "二次混料" else a_q
            mdl = fit_ridge(Xq if False else design(Ptr, True) if nm == "二次混料" else XqQ,
                            Ytr, a)
            yh = mdl.predict(X)
            ev = evaluate(Y, yh, v).iloc[-1]
            rows.append(dict(split=split, model=nm, rmse=ev.rmse, mae=ev.mae,
                             r2=ev.r2, spearman=ev.spearman,
                             **{f"r_{kk}": vv for kk, vv in regret(Y, yh, v).items()
                                if kk.startswith("regret")}))
    ev_df = pd.DataFrame(rows)
    ev_df["note"] = np.where(ev_df.split == "test_1m", "同尺度（1M），RMSE 可比",
                             "跨尺度：Loss 绝对水平不同，只看排序指标")
    save_csv(ev_df, "q1_quality_augmented_eval.csv")
    print("\n[检验集表现]（60M/1B 的 Loss 绝对水平与 1M 不同，绝对误差不可比，只看 Spearman/后悔值）")
    print(ev_df[["split", "model", "rmse", "spearman", "r_regret_top1",
                 "r_regret_top5"]].to_string(index=False))

    # ---- Q(p) 的系数与显著性（bootstrap）----
    rng = np.random.default_rng(SEED)
    coefs = np.zeros((300, k))
    n = len(Ytr)
    for b in range(300):
        s = rng.integers(0, n, n)
        coefs[b] = fit_ridge(XqQ[s], Ytr[s], a_q).coef_[:, -1]   # 13 个输出 × 最后一列(Q)
    c_mean, c_sd = coefs.mean(0), coefs.std(0)
    qcoef = pd.DataFrame(dict(domain=LOSS_DOMAINS, c_mean=c_mean, c_sd=c_sd,
                              t_ratio=c_mean / (c_sd + 1e-12),
                              significant=np.abs(c_mean / (c_sd + 1e-12)) > 2))
    save_csv(qcoef, "q1_quality_coefficient.csv")
    print("\n[Q(p) 的逐域系数（bootstrap 300 次）]")
    print(qcoef.to_string(index=False))
    print(f"  |t|>2 的域: {int(qcoef.significant.sum())}/{len(qcoef)}")

    save_json(dict(
        mapping="17 配方域 → 7 质量域（A16 参考 + 体裁代理）",
        direct_or_near=[r.mixture_domain for _, r in mp.iterrows()
                        if r.mapping_type in ("direct", "near_direct")],
        proxy_domains={r.mixture_domain: r.quality_domain for _, r in mp.iterrows()
                       if r.mapping_type == "inferred_proxy"},
        Q_of_p="Q(p) = Σ_i p_i · q_i（域质量线性折算，属未验证的建模假设）",
        augmented_model="L_k(p)=Σa_ki p_i+Σb_kij p_i p_j+c_k·Q(p)",
        caveat="13 个域无同名质量域，q 值为体裁代理；不得据此宣称域级因果效应",
    ), "q1_integration_meta.json")
    print("\n完成：q1_domain_mapping.csv / q1_quality_augmented_*.csv")


if __name__ == "__main__":
    main()
