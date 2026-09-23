# -*- coding: utf-8 -*-
"""
第一问（第一、二部分）：数据质量评价与质量冲突消解（Baseline B / Q1-B）。

流程：
  1. 22 个指标 → 标量（多维列表型先压缩）
  2. 方向统一为「越高越好」（负向指标补转换 x→1-x）
  3. 在开发参考集上估计稳健归一化参数，应用于其他集合
  4. 语义分组 → 组内算术平均 z_g → 组间几何平均（体现短板）得 Q_base
  5. 冲突强度 c = max_g z_g − min_g z_g；消解 Q = (1−λ)Q_base + λ·min_g z_g
  6. 域级聚合（文档等权为主、词数加权对照）与置信区间
  7. 敏感性：ε 截断、λ 网格、标量化方式、扩展集迁移

数据：A1 抽样集（7 域，27 字段，含 content）；
      A2/A3 扩展集（arxiv/github，24 字段，无 content）——同源重建。
"""
from __future__ import annotations

import itertools
import json
import lzma
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, DATA, save_csv, save_json, SEED

QR = Path(__file__).resolve().parent.parent / "data" / "quality_rebuild"
rng = np.random.default_rng(SEED)
EPS = 1e-3                     # 几何平均的零值截断
LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TAU_Q = 0.75                   # 冲突阈值分位（在开发参考集上确定）

# 22 指标的字段名（HF 原始拼写含 punctution 拼写差异）
F = dict(
    fineweb_edu="fineweb_edu",
    fluency_en="fluency_en",
    clean="modernbert_cleanliness",
    read="modernbert_readability",
    reason="modernbert_reasoning",
    prof="modernbert_professionalism",
    dsir_books="dsir_books", dsir_wiki="dsir_wiki", dsir_math="dsir_math",
    qurater="qurater", ad="ad_en",
    wc="rps_doc_word_count", ns="rps_doc_num_sentences", ue="rps_doc_unigram_entropy",
    fuw="rps_doc_frac_unique_words", fna="rps_doc_frac_no_alph_words",
    t2="rps_doc_frac_chars_top_2gram", t3="rps_doc_frac_chars_top_3gram",
    up="rps_lines_uppercase_letter_fraction",
    term="rps_lines_ending_with_terminal_punctution_mark",
    num="rps_lines_numerical_chars_fraction", mwl="rps_doc_mean_word_length",
)

GROUPS = {
    "教育价值": ["fineweb_edu", "dsir_books", "dsir_wiki", "dsir_math", "qurater", "reason"],
    "可读性":   ["read", "clean", "fluency_en", "ue"],
    "专业性":   ["prof", "mwl", "fuw"],
    "低噪声":   ["ad", "fna", "t2", "t3", "up", "term", "num", "wc", "ns"],
}
# 负向指标（原始越大越差）→ 补转换
NEGATIVE = ["ad", "t2", "t3", "up", "fna", "num"]
# 「适宜区间」指标：过短过长都不好
BANDED = ["wc", "ns"]


# ---------------------------------------------------------------- 标量化
def _softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scalarize(o, mode="argmax"):
    """把一个样本的 22 个指标压成 22 个标量。mode: argmax | expect。"""
    v = {}
    v["fineweb_edu"] = float(np.atleast_1d(o[F["fineweb_edu"]])[0])
    for key, col in [("clean", F["clean"]), ("read", F["read"]),
                     ("reason", F["reason"]), ("prof", F["prof"])]:
        a = np.asarray(o[col], float)
        if mode == "argmax":
            v[key] = float(np.argmax(a)) / (len(a) - 1)          # 6 类 → [0,1]
        else:
            p = _softmax(a)
            v[key] = float((p * np.arange(len(a))).sum() / (len(a) - 1))
    p = _softmax(np.asarray(o[F["fluency_en"]], float))
    v["fluency_en"] = float(p[-1])                                # 流畅类概率
    p = _softmax(np.asarray(o[F["ad"]], float))
    v["ad"] = float(p[-1])                                        # 官方顺序 [has_ad, no_ad]
    q = np.asarray(o[F["qurater"]], float)
    v["qurater"] = float(np.mean((q - q.mean()) / (q.std() + 1e-9)))   # 4 维各标准化后平均
    for key in ["dsir_books", "dsir_wiki", "dsir_math"]:
        v[key] = float(o[F[key]])
    for key in ["wc", "ns", "ue", "fuw", "fna", "t2", "t3", "up", "term", "num", "mwl"]:
        v[key] = float(o[F[key]])
    return v


# ---------------------------------------------------------------- 归一化
def fit_norm(df, cols):
    """在开发参考集上估计稳健归一化参数（1%/99% 分位截尾后的 min-max）。"""
    par = {}
    for c in cols:
        x = df[c].to_numpy(float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        par[c] = (float(lo), float(hi) if hi > lo else float(lo + 1))
    return par


def apply_norm(df, par):
    out = pd.DataFrame(index=df.index)
    for c, (lo, hi) in par.items():
        out[c] = np.clip((df[c].to_numpy(float) - lo) / (hi - lo), 0, 1)
    # 负向指标补转换
    for c in NEGATIVE:
        if c in out:
            out[c] = 1.0 - out[c]
    # 适宜区间：w=1 时最好，做成倒 U 型效用
    for c in BANDED:
        if c in out:
            out[c] = 1.0 - np.abs(out[c] - 0.35) / max(0.35, 0.65)
            out[c] = np.clip(out[c], 0, 1)
    return out


def composite(z, lam, eps=EPS):
    """组内平均 → 组间几何平均（短板）→ 凸组合消解。"""
    gs = pd.DataFrame({g: z[[c for c in cols if c in z.columns]].mean(axis=1)
                       for g, cols in GROUPS.items()})
    gs = gs.clip(lower=0)
    q_base = np.exp(np.log(gs + eps).mean(axis=1)) - eps      # 组间几何平均
    q_base = np.clip(q_base, 0, 1)
    gmin = gs.min(axis=1)
    c_i = gs.max(axis=1) - gmin
    Q = (1 - lam) * q_base + lam * gmin
    return Q.to_numpy(float), q_base.to_numpy(float), gs, c_i.to_numpy(float)


# ---------------------------------------------------------------- 载入
def load_a1(limit=None):
    p = QR / "A1_sample.jsonl.xz"
    recs = []
    with lzma.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rec = scalarize(o)
            rec["_domain"] = o.get("_source_domain")
            rec["_id"] = o.get("id")
            rec["_ntok"] = len(str(o.get("content", "")).split())
            recs.append(rec)
            if limit and len(recs) >= limit:
                break
    return pd.DataFrame(recs)


def load_ext(name):
    p = QR / f"{name}.jsonl.xz"
    if not p.exists() or p.stat().st_size < 1000:
        return None
    recs = []
    with lzma.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rec = scalarize(o)
            rec["_domain"] = "arxiv" if "arxiv" in name else "github"
            rec["_id"] = o.get("id")
            rec["_ntok"] = np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


def main():
    print("=" * 100)
    print("第一问 · 数据质量评价与冲突消解（Q1-B：组内平均 + 组间几何平均）")
    print("=" * 100)

    a1 = load_a1()
    print(f"A1 抽样集: {a1.shape}, 域分布 {a1._domain.value_counts().to_dict()}")

    cols = [c for c in a1.columns if not c.startswith("_")]
    par = fit_norm(a1, cols)                     # 归一化参数在 A1 上冻结
    z1 = apply_norm(a1, par)

    Q0v, qb0, gs0, c0 = composite(z1, lam=0.0)
    tau = float(np.percentile(c0, TAU_Q * 100))
    print(f"冲突阈值 τ = {tau:.4f}（开发集 {TAU_Q:.2f} 分位）")

    # λ 网格
    lam_rows = []
    for lam in LAM_GRID:
        Q, qb, gs, c = composite(z1, lam=lam)
        lam_rows.append(dict(lam=lam, Q_mean=float(Q.mean()), Q_std=float(Q.std()),
                             Q_p10=float(np.percentile(Q, 10)),
                             Q_p90=float(np.percentile(Q, 90)),
                             rank_corr_with_base=float(pd.Series(Q).corr(
                                 pd.Series(qb0), method="spearman"))))
    lam_df = pd.DataFrame(lam_rows)
    save_csv(lam_df, "q1_lambda_grid.csv")
    print("\n[λ 消解网格]")
    print(lam_df.to_string(index=False))

    LAM = 0.3                                    # 主用消解强度（小规模预设网格内）
    Q, q_base, gs, c_i = composite(z1, lam=LAM)
    res = a1[["_id", "_domain", "_ntok"]].copy()
    res["Q"] = Q
    res["Q_base"] = q_base
    res["conflict"] = c_i
    res["is_conflict"] = c_i > tau
    for g in GROUPS:
        res[f"z_{g}"] = gs[g].to_numpy()
    save_csv(res.head(2000), "q1_quality_scores_head.csv")
    res.to_parquet(RESULTS / "q1_quality_scores.parquet")
    print(f"[parquet] {RESULTS/'q1_quality_scores.parquet'}  {res.shape}")
    # 逐指标方向统一后的得分（供绘图与指标级相关分析）
    zsave = z1.copy()
    zsave.insert(0, "_domain", a1["_domain"].to_numpy())
    zsave.to_parquet(RESULTS / "q1_metric_z.parquet")
    save_json({k: list(v) for k, v in par.items()}, "q1_norm_params.json")
    print(f"[parquet] {RESULTS/'q1_metric_z.parquet'}  {zsave.shape}")

    # 域级聚合
    dom = res.groupby("_domain").agg(
        n=("Q", "size"), Q_mean=("Q", "mean"), Q_std=("Q", "std"),
        Q_median=("Q", "median"),
        Q_lo=("Q", lambda s: np.percentile(s, 2.5)),
        Q_hi=("Q", lambda s: np.percentile(s, 97.5)),
        conflict_rate=("is_conflict", "mean"), conflict_mean=("conflict", "mean"),
        Q_base_mean=("Q_base", "mean"))
    # 词数加权对照
    wsum = res.groupby("_domain")._ntok.sum()
    dom["Q_wordweighted"] = (res.assign(w=res._ntok).groupby("_domain")
                             .apply(lambda d: np.average(d.Q, weights=d.w + 1e-9)
                                    if d.w.notna().all() and d.w.sum() > 0 else np.nan))
    dom["token_share"] = wsum / wsum.sum()
    dom = dom.reset_index()
    save_csv(dom, "q1_domain_quality.csv")
    print("\n[域级质量]")
    print(dom[["_domain", "n", "Q_mean", "Q_median", "Q_lo", "Q_hi",
               "conflict_rate", "Q_wordweighted"]].to_string(index=False))

    # 冲突成因剖析：五类典型冲突
    ct = res.copy()
    lab = np.where(ct.is_conflict, "冲突", "不冲突")
    ct["组"] = lab
    prof = ct.groupby("组")[[f"z_{g}" for g in GROUPS] + ["Q", "Q_base", "conflict"]].mean()
    save_csv(prof.reset_index(), "q1_conflict_profile.csv")
    print("\n[冲突/非冲突画像]")
    print(prof.to_string(index=False))

    # 冲突类型：教育价值高但噪声分低 等
    types = []
    for g_hi, g_lo, tag in [("z_教育价值", "z_低噪声", "教育价值高、噪声控制差"),
                            ("z_可读性", "z_专业性", "可读性好、专业性弱"),
                            ("z_专业性", "z_可读性", "专业性强、可读性弱"),
                            ("z_教育价值", "z_可读性", "教育价值高、可读性弱"),
                            ("z_低噪声", "z_教育价值", "干净但信息量低")]:
        m = (ct[g_hi] > ct[g_hi].quantile(0.75)) & (ct[g_lo] < ct[g_lo].quantile(0.25))
        types.append(dict(类型=tag, 样本数=int(m.sum()), 占比=float(m.mean()),
                          平均Q=float(ct.loc[m, "Q"].mean()) if m.any() else np.nan))
    types_df = pd.DataFrame(types)
    save_csv(types_df, "q1_conflict_types.csv")
    print("\n[主要冲突类型]")
    print(types_df.to_string(index=False))

    # 敏感性：ε、标量化方式
    sens = []
    for eps in [1e-4, 1e-3, 1e-2]:
        Qa, _, _, _ = composite(z1, lam=LAM, eps=eps)
        sens.append(dict(参数="ε 截断", 取值=eps,
                         与主结果相关=float(pd.Series(Qa).corr(pd.Series(Q), method="spearman")),
                         Q均值=float(Qa.mean())))
    z_exp = apply_norm(a1.assign(**{
        c: a1[c] for c in cols}), par)          # 占位，稍后替换
    # 用 softmax 期望重新标量化
    a1b = load_a1()
    recs = []
    with lzma.open(QR / "A1_sample.jsonl.xz", "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            recs.append(scalarize(o, mode="expect"))
    zb = apply_norm(pd.DataFrame(recs), par)
    Qb, _, _, _ = composite(zb, lam=LAM)
    sens.append(dict(参数="标量化", 取值="softmax期望",
                     与主结果相关=float(pd.Series(Qb).corr(pd.Series(Q), method="spearman")),
                     Q均值=float(Qb.mean())))
    # 等权算术平均对照（Q1-A 口径）
    Qarith = z1.mean(axis=1).to_numpy(float)
    sens.append(dict(参数="合成方式", 取值="等权算术平均",
                     与主结果相关=float(pd.Series(Qarith).corr(pd.Series(Q), method="spearman")),
                     Q均值=float(Qarith.mean())))
    Qgeom_all = np.exp(np.log(z1.clip(lower=0) + EPS).mean(axis=1)) - EPS
    sens.append(dict(参数="合成方式", 取值="全指标几何平均",
                     与主结果相关=float(pd.Series(Qgeom_all).corr(pd.Series(Q), method="spearman")),
                     Q均值=float(Qgeom_all.mean())))
    sens_df = pd.DataFrame(sens)
    save_csv(sens_df, "q1_quality_sensitivity.csv")
    print("\n[质量评分敏感性]")
    print(sens_df.to_string(index=False))

    # 扩展集迁移检验
    ext_rows = []
    for nm in ["A2_arxiv", "A3_github"]:
        d = load_ext(nm)
        if d is None:
            ext_rows.append(dict(数据集=nm, 状态="未就绪（下载中）", n=0))
            continue
        ze = apply_norm(d, par)
        Qe, qbe, gse, ce = composite(ze, lam=LAM)
        de = pd.DataFrame(dict(_domain=d._domain, Q=Qe, conflict=ce))
        for dom_nm, gg in de.groupby("_domain"):
            base = dom[dom._domain == dom_nm]
            ext_rows.append(dict(
                数据集=nm, 状态="完成", 域=dom_nm, n=len(gg),
                Q_mean_ext=float(gg.Q.mean()), Q_mean_A1=float(base.Q_mean.iloc[0])
                if len(base) else np.nan,
                conflict_rate_ext=float((gg.conflict > tau).mean()),
                conflict_rate_A1=float(base.conflict_rate.iloc[0]) if len(base) else np.nan))
    ext_df = pd.DataFrame(ext_rows)
    save_csv(ext_df, "q1_extension_transfer.csv")
    print("\n[扩展集迁移检验]")
    print(ext_df.to_string(index=False))

    # 质量字典
    dic = [dict(指标=k, 分组=g, 处理="; ".join({
        "fineweb_edu": "取单元素，1%/99% 分位截尾 min-max",
        "fluency_en": "softmax 取流畅类概率",
        "qurater": "4 维各标准化后平均",
        "dsir_books": "稳健归一化（正向）", "dsir_wiki": "稳健归一化（正向）",
        "dsir_math": "稳健归一化（正向）",
        "ad": "softmax 取无广告类概率（已正向）",
        "wc": "倒 U 型适宜区间效用", "ns": "倒 U 型适宜区间效用",
    }.get(k, "稳健归一化 + 负向补转换" if k in NEGATIVE else "稳健归一化（正向）")))
        for g, ks in GROUPS.items() for k in ks]
    save_csv(pd.DataFrame(dic), "q1_quality_metric_dictionary.csv")

    save_json(dict(
        model="Q = (1-λ)·GeoMean_g(组内均值) + λ·min_g(组内均值)",
        groups={g: ks for g, ks in GROUPS.items()},
        lambda_main=LAM, lambda_grid=LAM_GRID, eps=EPS,
        tau=tau, tau_quantile=TAU_Q,
        norm="1%/99% 分位截尾 min-max（参数在 A1 上冻结）",
        negative_metrics=NEGATIVE, banded_metrics=BANDED,
        scalerization="modernbert 四指标 argmax/5（敏感性对照：softmax 期望）",
        n_A1=int(len(a1)), domains=a1._domain.value_counts().to_dict(),
        source="A1/A2/A3 按数据说明来源（SlimPajama-Meta-rater）同源重建",
    ), "q1_quality_meta.json")
    print("\n完成：q1_*.csv / q1_quality_scores.parquet / q1_quality_meta.json")


if __name__ == "__main__":
    main()
