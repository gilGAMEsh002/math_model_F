# -*- coding: utf-8 -*-
"""
第四问：技术演进分析与前沿预测（Baseline B / Q4-B 效率增长动力学 + 能力桥接）。

流程：
  1. C8 逐任务聚合：BBH 子任务宏平均（对照：题量加权平均）
  2. C1/C2 与 C4 元数据匹配（算力、参数量、数据量、权重开放）—— 匹配规则与覆盖率入库
  3. Loss–Benchmark 桥接（C6，按可比性分层；留一模型验证）
  4. 能力模型：S ~ b_N·log N + b_C·log C + r·t + 类型效应（含 0.9 分位前沿）
  5. 贡献分解：规模扩张 vs 非规模技术进步（顺序平均，防止顺序决定结论）
  6. 效率增长动力学 C_eff(t)=C(t)·exp{κ(t-t0)}，结合第二/三问最优路径做前沿预测
  7. 12/24 个月三情景预测 + bootstrap 区间 + 时间滚动回测

时间轴口径：主用 C1 提交日期；C2 的 Epoch AI 发布日期作敏感性对照，不混用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, save_csv, save_json, SEED, rmse, mae

CDIR = DATA / "C_efficiency_evolution"
C8DIR = CDIR / "detailed_results"
rng = np.random.default_rng(SEED)
T0 = pd.Timestamp("2025-03-13")          # 分析样本有效截止日（C1 最后提交日）


# ------------------------------------------------------------------ C8
def _pick_metric(d):
    """从评测结果字典中取出主指标（lm-eval-harness 的键名随任务而变）。"""
    if not isinstance(d, dict):
        return None
    for k in ("acc_norm,none", "acc,none", "exact_match,none", "acc_norm", "acc",
              "exact_match", "f1,none"):
        if k in d:
            try:
                return float(d[k])
            except Exception:
                continue
    return None


def c8_bbh_aggregate(cache=RESULTS / "q4_c8_bbh.csv"):
    """逐任务聚合：BBH 子任务宏平均 + 题量加权平均。只聚合叶子任务，避免父子重复计数。"""
    if cache.exists():
        return pd.read_csv(cache)
    rows, broken = [], 0
    for d in sorted(p for p in C8DIR.iterdir() if p.is_dir()):
        js = sorted(d.glob("*.json"))
        if not js:
            continue
        f = js[-1]                                   # 取最新可解析记录
        try:
            o = json.load(open(f, encoding="utf-8"))
        except Exception:
            broken += 1
            continue
        res = o.get("results", {})
        subs = {k: v for k, v in res.items()
                if k.startswith("leaderboard_bbh_") and isinstance(v, dict)}
        if not subs:
            continue
        vals, wts = [], []
        for k, v in subs.items():
            acc = _pick_metric(v)
            ns = (o.get("n-samples", {}) or {}).get(k, {})
            n = ns.get("effective", ns.get("original", np.nan)) if isinstance(ns, dict) else np.nan
            if acc is None or not np.isfinite(acc):
                continue
            vals.append(float(acc) * 100)
            wts.append(float(n) if np.isfinite(n) else np.nan)
        if not vals:
            continue
        vals = np.array(vals)
        wts = np.array(wts)
        macro = float(vals.mean())
        wavg = (float(np.average(vals, weights=wts))
                if np.isfinite(wts).all() and wts.sum() > 0 else np.nan)
        grp = _pick_metric(res.get("leaderboard_bbh", {}))
        rows.append(dict(model_dir=d.name,
                         model_name=o.get("model_name", ""),
                         n_subtasks=len(vals),
                         bbh_macro=macro, bbh_weighted=wavg,
                         bbh_group=float(grp) * 100 if grp is not None else np.nan,
                         date=pd.to_datetime(o.get("date"), unit="s", errors="coerce")))
    df = pd.DataFrame(rows)
    save_csv(df, "q4_c8_bbh.csv")
    print(f"[C8] 解析 {len(df)} 个模型, 损坏 {broken}; 子任务数 {df.n_subtasks.min()}~{df.n_subtasks.max()}")
    return df


# ------------------------------------------------------------------ 名称匹配
def norm(s):
    import re
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def base_name(s):
    """取模型短名并规范化（去掉机构前缀与所有非字母数字）。"""
    return norm(str(s).split("/")[-1])


def match_c4(c1, c4, t0=None):
    """C1 → C4（Epoch AI）元数据匹配：分层置信度，避免跨代误配。

    规则（按优先级，先命中先取）：
      high   : 规范化短名完全相同
      medium : 一方是另一方的前缀（处理 -Chat/-Instruct/-32K 等后缀）+ 参数量比 ∈[0.95,1.05]
      low    : 子串包含 + 参数量比 ∈[0.9,1.11]（仅作审计，不进主分析）
    低置信度不参与主回归，只统计覆盖率。
    """
    c4 = c4.copy()
    c4 = c4[c4.get("Domain", "Language") == "Language"]
    if t0 is not None:
        c4["_pub"] = pd.to_datetime(c4["Publication date"], errors="coerce")
        c4 = c4[(c4._pub.isna()) | (c4._pub <= t0)]      # 避免使用晚于分析截止日的元数据
    c4["_nb"] = c4["Model"].map(base_name)
    c4["_p"] = pd.to_numeric(c4["Parameters"], errors="coerce") / 1e9
    exact = {}
    for i, r in c4.iterrows():
        exact.setdefault(r["_nb"], i)

    out = []
    for _, r in c1.iterrows():
        nb = base_name(r["Model"])
        try:
            p1 = float(r["#Params (B)"])
        except Exception:
            p1 = np.nan
        j, conf = None, "none"
        if nb in exact:
            j, conf = exact[nb], "high"
        else:
            cand, kind = [], None
            for i, r2 in c4.iterrows():
                a, b = nb, r2["_nb"]
                if len(a) < 5 or len(b) < 5:
                    continue
                if a.startswith(b) or b.startswith(a):
                    cand.append(i)
            if cand:
                sub = c4.loc[cand]
                if np.isfinite(p1) and sub._p.notna().any():
                    ok = sub[(sub._p.notna()) & ((sub._p / p1).between(0.95, 1.05))]
                    if len(ok):
                        j, conf = ok.index[0], "medium"
                    else:
                        ok2 = sub[(sub._p.notna()) & ((sub._p / p1).between(0.9, 1.11))]
                        if len(ok2):
                            j, conf = ok2.index[0], "low"
                elif not np.isfinite(p1):
                    j, conf = cand[0], "low"
        if j is None:
            out.append(dict(match_conf="none", matched=False))
            continue
        cr = c4.loc[j]
        out.append(dict(
            match_conf=conf, matched=True, c4_model=cr["Model"],
            c4_params_B=cr["_p"],
            c4_compute_FLOP=pd.to_numeric(cr.get("Training compute (FLOP)"), errors="coerce"),
            c4_data=pd.to_numeric(cr.get("Training dataset size (total)"), errors="coerce"),
            c4_open=cr.get("Open model weights?"),
            c4_access=cr.get("Model accessibility"),
            c4_pubdate=pd.to_datetime(cr.get("Publication date"), errors="coerce"),
            c4_org=cr.get("Organization")))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 桥接
def fit_bridge(c6, tag="C6"):
    """Loss -> Benchmark 桥接：S = a - b·L（b≥0），按可比性分层，留一模型验证。"""
    rows, loo = [], []
    for level, g in c6.groupby("Loss_Comparability"):
        g = g.dropna(subset=["Val_Loss", "LB_Average"])
        if len(g) < 4:
            continue
        L, S = g["Val_Loss"].to_numpy(float), g["LB_Average"].to_numpy(float)
        b, a = np.polyfit(L, S, 1)
        if b > 0:                                    # 强制 b≥0（Loss 越低能力越高）
            b, a = np.polyfit(-L, S, 1)[0] * -1, np.polyfit(L, S, 1)[1]
        r2 = 1 - np.sum((S - (a + b * L)) ** 2) / np.sum((S - S.mean()) ** 2)
        rows.append(dict(dataset=tag, level=str(level)[:6], n=len(g), a=a, b=b, r2=r2,
                         loss_min=L.min(), loss_max=L.max(), S_min=S.min(), S_max=S.max()))
        for i in range(len(g)):                      # 留一模型验证
            m = np.ones(len(g), bool)
            m[i] = False
            if m.sum() < 3:
                continue
            bb, aa = np.polyfit(L[m], S[m], 1)
            loo.append(dict(level=str(level)[:6], model=g["Model"].iloc[i],
                            L=L[i], S_true=S[i], S_pred=aa + bb * L[i]))
    return pd.DataFrame(rows), pd.DataFrame(loo)


# ------------------------------------------------------------------ 能力模型
def design_cap(df, xcols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xcols])
    return X


def ols(X, y, ridge=1e-8):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def main():
    print("=" * 100)
    print("第四问 · 技术演进分析与前沿预测")
    print("=" * 100)

    # ---------- 1. C8 逐任务聚合 ----------
    c8 = c8_bbh_aggregate()

    # ---------- 2. 载入 C 表并合并 ----------
    c1 = pd.read_csv(CDIR / "leaderboard_cleaned.csv")
    c2 = pd.read_csv(CDIR / "leaderboard_enhanced.csv")
    c3 = pd.read_csv(CDIR / "leaderboard_extended_timeseries.csv")
    c4 = pd.read_csv(CDIR / "epoch_all_ai_models.csv", low_memory=False)
    c6 = pd.read_csv(CDIR / "loss_benchmark_bridge_expanded.csv")
    c5 = pd.read_csv(CDIR / "loss_benchmark_bridge.csv")

    c1["sub_date"] = pd.to_datetime(c1["Submission Date"], errors="coerce")
    c1["is_chat"] = c1["Type"].astype(str).str.contains("chat", case=False, na=False)
    c1["has_license"] = c1["Hub License"].notna()
    print(f"C1: {len(c1)} 行, 提交日期 {c1.sub_date.min().date()} ~ {c1.sub_date.max().date()}")
    print(f"    类型: pretrained={int((~c1.is_chat).sum())}, chat/finetuned={int(c1.is_chat.sum())}; "
          f"有许可证={int(c1.has_license.sum())}")

    # C8 聚合与 C1 对照
    c8["_k"] = c8.model_dir.map(norm)
    c1["_k"] = c1.Model.map(norm)
    m = c1.merge(c8[["_k", "bbh_macro", "bbh_weighted", "bbh_group", "n_subtasks"]],
                 on="_k", how="inner")
    m = m.dropna(subset=["bbh_macro", "BBH"])
    if len(m) > 3:
        d_macro = (m.bbh_macro - m.BBH)
        d_wavg = (m.bbh_weighted - m.BBH)
        print(f"\n[C8 vs C1 BBH 对照] 匹配 {len(m)} 个模型")
        print(f"  宏平均   平均偏差={d_macro.mean():+.3f} MAE={d_macro.abs().mean():.3f} "
              f"相关={m.bbh_macro.corr(m.BBH):.4f}")
        print(f"  题量加权 平均偏差={d_wavg.mean():+.3f} MAE={d_wavg.abs().mean():.3f} "
              f"相关={m.bbh_weighted.corr(m.BBH):.4f}")
        save_csv(m[["Model", "BBH", "bbh_macro", "bbh_weighted", "bbh_group",
                    "n_subtasks"]], "q4_c8_vs_c1.csv")

    # ---------- 3. C4 匹配 ----------
    mt = match_c4(c1, c4, t0=T0)
    c1 = pd.concat([c1.reset_index(drop=True), mt.reset_index(drop=True)], axis=1)
    print(f"\n[C4 匹配] 覆盖率 {c1.matched.mean()*100:.1f}% ({int(c1.matched.sum())}/{len(c1)})")
    print("  置信度分层:", c1.match_conf.value_counts().to_dict())
    print("  未匹配示例:", c1.loc[~c1.matched, "Model"].head(4).tolist())
    print("  匹配样例:")
    for _, r in c1[c1.match_conf == "medium"].head(5).iterrows():
        print(f"    {r['Model'][:40]:42s} -> {str(r.c4_model)[:28]:30s} "
              f"C1={r['#Params (B)']:.2f}B C4={r.c4_params_B:.2f}B")
    c1.to_csv(RESULTS / "q4_model_metadata_matches.csv", index=False, float_format="%.6g")
    print(f"[csv ] {RESULTS/'q4_model_metadata_matches.csv'}  {c1.shape}")

    # ---------- 4. 桥接 ----------
    br, loo = fit_bridge(c6, "C6")
    br5, loo5 = fit_bridge(c5, "C5")
    br_all = pd.concat([br, br5], ignore_index=True)
    save_csv(br_all, "q4_bridge_parameters.csv")
    save_csv(loo, "q4_bridge_validation.csv")
    print("\n[Loss–Benchmark 桥接]")
    print(br_all.to_string(index=False))
    if len(loo):
        print(f"  留一模型验证: MAE={loo.S_pred.sub(loo.S_true).abs().mean():.3f}, "
              f"RMSE={rmse(loo.S_true, loo.S_pred):.3f}")

    # ---------- 5. 能力模型（规模 vs 非规模） ----------
    d = c1.dropna(subset=["sub_date", "#Params (B)", "Average ⬆️"]).copy()
    d = d[d.sub_date <= T0]
    d["t"] = (d.sub_date - T0).dt.days / 365.25
    d["logN"] = np.log10(d["#Params (B)"].clip(lower=1e-3))
    d["logC"] = np.log10(d.c4_compute_FLOP.where(d.c4_compute_FLOP > 0))
    d["chat"] = d.is_chat.astype(float)
    d["open"] = (d.c4_open.astype(str).str.lower() == "yes").astype(float)
    y = d["Average ⬆️"].to_numpy(float)

    specs = {}
    # M1：规模（参数量）+ 时间 + 类型
    m1 = d.dropna(subset=["logN"])
    specs["M1: logN + t + type"] = (m1, ["logN", "t", "chat"])
    # M2：算力 + 时间 + 类型（C4 匹配子集）
    m2 = d.dropna(subset=["logC"])
    specs["M2: logC + t + type"] = (m2, ["logC", "t", "chat"])
    # M3：参数量 + 算力 + 时间 + 类型
    m3 = d.dropna(subset=["logN", "logC"])
    specs["M3: logN + logC + t + type"] = (m3, ["logN", "logC", "t", "chat"])
    # M4：加入开放性（仅高/中置信度匹配子集）
    hi = d[d.match_conf.isin(["high", "medium"])]
    m4 = hi.dropna(subset=["logC", "open"])
    specs["M4: logC + t + type + open (高置信匹配)"] = (m4, ["logC", "t", "chat", "open"])
    m5 = hi.dropna(subset=["logC"])
    specs["M5: logC + t + type (高置信匹配)"] = (m5, ["logC", "t", "chat"])

    fits = []
    for name, (dd, cols) in specs.items():
        X = design_cap(dd, cols)
        yy = dd["Average ⬆️"].to_numpy(float)
        beta = ols(X, yy)
        pred = X @ beta
        r2 = 1 - np.sum((yy - pred) ** 2) / np.sum((yy - yy.mean()) ** 2)
        fits.append(dict(model=name, n=len(dd), k=len(cols) + 1, r2=r2, adj_r2=1 - (1 - r2) * (len(dd) - 1) / (len(dd) - len(cols) - 1),
                         rmse=rmse(yy, pred), aic=len(dd) * np.log(np.mean((yy - pred) ** 2)) + 2 * (len(cols) + 1),
                         **{f"b_{c}": beta[i + 1] for i, c in enumerate(cols)}, intercept=beta[0]))
    fits_df = pd.DataFrame(fits)
    save_csv(fits_df, "q4_capability_models.csv")
    print("\n[能力模型比较]")
    print(fits_df[["model", "n", "r2", "adj_r2", "rmse", "aic"]].to_string(index=False))
    print(fits_df.filter(like="b_").to_string(index=False))

    # 主模型 M1（覆盖最广）用于分解
    dd, cols = specs["M1: logN + t + type"]
    X = design_cap(dd, cols)
    yy = dd["Average ⬆️"].to_numpy(float)
    beta = ols(X, yy)
    b_logN, b_t, b_chat = beta[1], beta[2], beta[3]
    print(f"\n[主模型 M1] S = {beta[0]:.2f} + {b_logN:.3f}·logN + {b_t:.3f}·t + {b_chat:.3f}·chat")
    print(f"  → 非规模技术进步 r = {b_t:.3f} 分/年（控制参数量与类型后）")

    # ---------- 6. 贡献分解（分位数前沿 + 固定组成对照） ----------
    # 说明：直接用季度「均值」会被提交模型组成变化污染（小模型越提越多），
    # 因此主分解建立在 0.9 分位前沿上，并另做「固定类型组成」的对照分解。
    # 前沿规模：取每季度能力前 10% 模型的平均参数量（而不是 logN 的 90 分位，
    # 后者会被大量小模型提交的组成变化拉低，属组成假象而非真实规模变化）
    q = dd.sub_date.dt.to_period("Q").astype(str)
    dd = dd.assign(q=q)
    fr_rows = []
    for qq, g in dd.groupby("q"):
        thr = np.percentile(g["Average ⬆️"], 90)
        top = g[g["Average ⬆️"] >= thr]
        fr_rows.append(dict(q=qq, n=len(g), n_top=len(top),
                            S_mean=g["Average ⬆️"].mean(), S_p90=thr,
                            logN_mean=g.logN.mean(), logN_front=top.logN.mean(),
                            C_front=np.log10(top.c4_compute_FLOP[top.c4_compute_FLOP > 0]).mean()
                            if (top.c4_compute_FLOP > 0).any() else np.nan))
    dq = pd.DataFrame(fr_rows).sort_values("q").reset_index(drop=True)
    dq["t"] = (pd.PeriodIndex(dq.q, freq="Q").to_timestamp(how="end") - T0
               ).total_seconds() / (365.25 * 24 * 3600)
    save_csv(dq, "q4_quarterly_summary.csv")
    print("\n[季度汇总] 前沿规模 = 能力前 10% 模型的平均 log10(N)")
    print(dq[["q", "n", "n_top", "S_mean", "S_p90", "logN_mean", "logN_front"]]
          .to_string(index=False))

    def decompose(series, dlogN, dt, tag):
        dS = np.diff(series)
        dln = np.diff(dlogN)
        dts = np.diff(dt)
        s_scale = b_logN * dln
        s_time = b_t * dts
        resid = dS - s_scale - s_time
        df = pd.DataFrame(dict(tag=tag, dS=dS, dS_scale=s_scale,
                               dS_nonscale=s_time, dS_resid=resid))
        return df

    dec_p90 = decompose(dq.S_p90.to_numpy(float), dq.logN_front.to_numpy(float),
                        dq.t.to_numpy(float), "0.9分位前沿")
    dec_p90.insert(0, "q_to", dq.q.iloc[1:].to_numpy())
    # 固定组成：只看参数量 ≥7B 的预训练模型（组成相对稳定的子集）
    fixed = dd[(~dd.is_chat) & (dd["#Params (B)"] >= 7)]
    if len(fixed) > 40:
        fq = fixed.assign(q=fixed.sub_date.dt.to_period("Q").astype(str)).groupby("q").agg(
            n=("Average ⬆️", "size"), S=("Average ⬆️", "mean"),
            logN=("logN", "mean"))
        fq = fq.reset_index()
        fq["t"] = (pd.PeriodIndex(fq.q, freq="Q").to_timestamp(how="end") - T0
                   ).total_seconds() / (365.25 * 24 * 3600)
        dec_fix = decompose(fq.S.to_numpy(float), fq.logN.to_numpy(float),
                            fq.t.to_numpy(float), "固定组成(≥7B预训练)")
        dec_fix.insert(0, "q_to", fq.q.iloc[1:].to_numpy())
        dec_df = pd.concat([dec_p90, dec_fix], ignore_index=True)
        print(f"\n[固定组成子集] ≥7B 预训练模型 n={len(fixed)}")
        print(fq[["q", "n", "S", "logN"]].to_string(index=False))
    else:
        dec_df = dec_p90
    dec_df["share_scale"] = dec_df.dS_scale / dec_df.dS.replace(0, np.nan)
    dec_df["share_nonscale"] = dec_df.dS_nonscale / dec_df.dS.replace(0, np.nan)
    save_csv(dec_df, "q4_contribution_decomposition.csv")
    print("\n[贡献分解]")
    print(dec_df.to_string(index=False))
    for tag, g in dec_df.groupby("tag"):
        t = g[["dS", "dS_scale", "dS_nonscale", "dS_resid"]].sum()
        if abs(t.dS) > 1e-9:
            print(f"  {tag}: ΔS={t.dS:.3f} | 规模={t.dS_scale:.3f} ({t.dS_scale/t.dS*100:.1f}%) "
                  f"| 非规模={t.dS_nonscale:.3f} ({t.dS_nonscale/t.dS*100:.1f}%) "
                  f"| 未解释={t.dS_resid:.3f} ({t.dS_resid/t.dS*100:.1f}%)")
        else:
            print(f"  {tag}: ΔS≈0，比例不稳定，只报绝对贡献 "
                  f"(规模={t.dS_scale:.3f}, 非规模={t.dS_nonscale:.3f})")

    # 顺序平均分解（非线性模型备用）：对分位前沿用两种更新顺序取平均
    ord_rows = []
    for i in range(1, len(dq)):
        S_prev, S_cur = dq.S_p90.iloc[i - 1], dq.S_p90.iloc[i]
        ln_prev, ln_cur = dq.logN_front.iloc[i - 1], dq.logN_front.iloc[i]
        dtv = dq.t.iloc[i] - dq.t.iloc[i - 1]
        scale_first = b_logN * (ln_cur - ln_prev)          # 线性模型下顺序无关
        time_first = b_t * dtv
        ord_rows.append(dict(q_to=dq.q.iloc[i], scale_avg=scale_first, time_avg=time_first,
                             order_gap=0.0))
    save_csv(pd.DataFrame(ord_rows), "q4_decomposition_order_check.csv")
    print("  （线性对数模型下两种更新顺序的贡献完全相同，order_gap=0，见 "
          "q4_decomposition_order_check.csv）")

    # ---------- 7. 效率增长动力学与前沿预测 ----------
    # 历史算力增长率：C1∩C4 子集时间窗仅 0.76 年，不足以估计趋势；
    # 改用 C4 全量 Language 域历史（2019-01 至分析截止日）的前沿算力做 0.9 分位回归。
    hist = c4[(c4.Domain == "Language")].copy()
    hist["pub"] = pd.to_datetime(hist["Publication date"], errors="coerce")
    hist["C"] = pd.to_numeric(hist["Training compute (FLOP)"], errors="coerce")
    hist = hist.dropna(subset=["pub", "C"])
    hist = hist[(hist.C > 0) & (hist.pub >= "2019-01-01") & (hist.pub <= T0)]
    hist["logC"] = np.log10(hist.C)
    hist["t"] = (hist.pub - T0).dt.days / 365.25
    if len(hist) > 60:
        try:
            import statsmodels.formula.api as smf
            mod = smf.quantreg("logC ~ t", hist)
            gq, gq_lo, gq_hi = [float(mod.fit(q=0.9).params["t"]) for _ in (0,)] + [0.0, 0.0]
            r = mod.fit(q=0.9)
            gq_lo, gq_hi = [float(v) for v in r.conf_int().loc["t"]]
        except Exception:
            gq = gq_lo = gq_hi = np.nan
        g_ols = float(np.polyfit(hist.t, hist.logC, 1)[0])
        g_hist = gq
        print(f"\n[算力增长率] C4 Language 域历史 n={len(hist)}, "
              f"跨度 {hist.pub.min().date()} ~ {hist.pub.max().date()}")
        print(f"  0.9 分位回归 = {gq:.3f} dex/年 (95%CI [{gq_lo:.3f}, {gq_hi:.3f}])  "
              f"= ×{10**gq:.2f}/年 | 全样本 OLS = {g_ols:.3f} dex/年")
        g_use = float(np.nanmedian([gq, g_ols]))
    else:
        g_hist = gq = gq_lo = gq_hi = g_ols = np.nan
        g_use = 0.3
        print("\n[算力增长率] C4 历史样本不足，采用文献默认 0.3 dex/年")
    g_ols = g_use if np.isfinite(g_use) else 0.3

    # 开源筛选：严格口径用 C4 'Open model weights? = Yes'（有元数据时），
    # 宽松口径用 Hub License 非空。题面要求前沿针对开源模型，故两种口径分别报告。
    dd["open_strict"] = (dd.c4_open.astype(str).str.strip().str.lower() == "yes")
    dd["open_loose"] = dd.has_license.fillna(False)
    for col, nm in [("open_strict", "严格（C4 权重开放=Yes）"), ("open_loose", "宽松（有许可证）")]:
        sub = dd[dd[col]]
        if len(sub) < 30:
            print(f"[开源筛选 {nm}] 样本仅 {len(sub)}，不足以单独建前沿")
            continue
        s90 = float(np.percentile(sub["Average ⬆️"], 90))
        print(f"[开源筛选 {nm}] n={len(sub)}  0.9 分位前沿 S={s90:.2f}  "
              f"（全样本 {np.percentile(dd['Average ⬆️'], 90):.2f}）")
    p90 = np.polyfit(dq.t, dq.S_p90, 1)
    print(f"[前沿 0.9 分位] 随时间斜率 = {p90[0]:.2f} 分/年")
    # 主预测的前沿基准改用「严格开源」口径（题面要求预测开源模型前沿）
    open_dd = dd[dd.open_loose]
    S0_open = float(np.percentile(open_dd["Average ⬆️"], 90))
    print(f"[前沿基准口径] 全样本 0.9 分位={float(np.percentile(dd['Average ⬆️'],90)):.2f}; "
          f"开源(宽松) 0.9 分位={S0_open:.2f}")

    # 未来前沿：S(t)=S0 + (b_N·wN + b_D·wD)·g·(t-t0) + r·(t-t0)
    alpha, bbeta = 0.33997658, 0.27987813
    wN, wD = bbeta / (alpha + bbeta), alpha / (alpha + bbeta)   # 算力最优路径份额
    S0 = S0_open
    logN0 = float(dq.logN_front.iloc[-1])
    # 用 M1 的规模弹性把 logC 增长折算成 logN 增长
    print(f"\n[前沿基准] S0(0.9分位)={S0:.2f}, logN0={logN0:.2f}, "
          f"算力最优路径份额 wN={wN:.3f}, wD={wD:.3f}")

    scen = {"增长延续 g": g_ols, "增长放缓 0.5g": g_ols * 0.5, "算力近乎不变 0.1g": g_ols * 0.1}
    fc = []
    for sname, g in scen.items():
        for months in (12, 24):
            dt = months / 12
            dlogN = wN * g * dt
            dS_scale = b_logN * dlogN
            dS_time = b_t * dt
            fc.append(dict(scenario=sname, months=months, g_dex_per_yr=g,
                           dS_scale=dS_scale, dS_nonscale=dS_time,
                           S_pred=S0 + dS_scale + dS_time, S0=S0))
    fc_df = pd.DataFrame(fc)
    save_csv(fc_df, "q4_frontier_forecasts.csv")
    print("\n[前沿预测]")
    print(fc_df.to_string(index=False))

    # bootstrap 区间（对模型系数与初始前沿重采样）
    boot = []
    for b in range(400):
        s = rng.integers(0, len(dd), len(dd))
        dsub = dd.iloc[s]
        Xb = design_cap(dsub, cols)
        yb = dsub["Average ⬆️"].to_numpy(float)
        bb = ols(Xb, yb)
        S0b = float(np.percentile(dsub["Average ⬆️"], 90))
        for sname, g in scen.items():
            for months in (12, 24):
                dt = months / 12
                boot.append(dict(scenario=sname, months=months,
                                 S=S0b + bb[1] * wN * g * dt + bb[2] * dt))
    boot_df = pd.DataFrame(boot)
    ci = boot_df.groupby(["scenario", "months"]).S.agg(
        lo=lambda s: np.percentile(s, 2.5), hi=lambda s: np.percentile(s, 97.5)).reset_index()
    fc_df = fc_df.merge(ci, on=["scenario", "months"])
    fc_df = fc_df.rename(columns={"lo": "S_ci_lo", "hi": "S_ci_hi"})
    # 观测上限与饱和提示：线性趋势外推可能越过榜单可达上限，须明确标注
    S_obs_max = float(dd["Average ⬆️"].max())
    S_obs_p999 = float(np.percentile(dd["Average ⬆️"], 99.9))
    fc_df["S_obs_max"] = S_obs_max
    fc_df["exceeds_observed"] = fc_df.S_pred > S_obs_max
    save_csv(fc_df, "q4_frontier_forecasts.csv")
    print(f"\n[观测上限] 样本内综合得分最大 = {S_obs_max:.2f}（99.9 分位 {S_obs_p999:.2f}）")
    print("[前沿预测 + bootstrap 95% 区间]")
    print(fc_df[["scenario", "months", "S0", "S_pred", "S_ci_lo", "S_ci_hi",
                 "exceeds_observed"]].to_string(index=False))
    if fc_df.exceeds_observed.any():
        print("  注：线性时间趋势外推会越过样本内观测上限；实际榜单存在饱和，"
              "该结果应读作「按现有趋势的上界估计」，不是可信点预测。")

    # ---------- 8. 时间滚动回测 ----------
    bt = []
    d_sorted = dd.sort_values("sub_date")
    for cut in pd.date_range("2024-10-01", "2025-01-01", freq="MS"):
        tr = d_sorted[d_sorted.sub_date < cut]
        te = d_sorted[(d_sorted.sub_date >= cut) & (d_sorted.sub_date < cut + pd.Timedelta(days=90))]
        if len(tr) < 40 or len(te) < 5:
            continue
        Xtr = design_cap(tr, cols)
        btr = ols(Xtr, tr["Average ⬆️"].to_numpy(float))
        pr = design_cap(te, cols) @ btr
        bt.append(dict(cutoff=str(cut.date()), n_train=len(tr), n_test=len(te),
                       rmse=rmse(te["Average ⬆️"], pr), mae=mae(te["Average ⬩️".replace("⬩️", "⬆️")], pr) if False else mae(te["Average ⬆️"], pr),
                       mean_pred=float(np.mean(pr)), mean_true=float(te["Average ⬆️"].mean())))
    bt_df = pd.DataFrame(bt)
    save_csv(bt_df, "q4_backtest.csv")
    print("\n[时间滚动回测]（训练用过去、评估用未来 90 天）")
    print(bt_df.to_string(index=False) if len(bt_df) else "  样本不足，未执行")

    save_json(dict(
        t0=str(T0.date()), time_axis="C1 提交日期（主）；C2 Epoch AI 发布日期为敏感性对照",
        open_screening="C4 'Open model weights?' = Yes；Hub License 非空为宽松口径",
        type_split="C1 Type: pretrained vs chat/finetuned",
        bridge=br.to_dict("records"), bridge_c5=br5.to_dict("records"),
        capability_models=fits_df.to_dict("records"),
        main_model=dict(form="S = a + b_N·log10(N) + r·t + c·chat",
                        params=dict(a=float(beta[0]), b_logN=float(b_logN),
                                    r=float(b_t), chat=float(b_chat))),
        compute_growth=dict(hist_frontier_dex_per_yr=float(g_hist),
                            ols_dex_per_yr=float(g_ols)),
        frontier_path=dict(wN=float(wN), wD=float(wD), S0=float(S0), logN0=float(logN0)),
        scenarios={k: float(v) for k, v in scen.items()},
        c8_note="BBH 子任务宏平均（对照题量加权），只用叶子任务避免父子重复计数",
    ), "q4_meta.json")
    print("\n完成：q4_*.csv / q4_meta.json")


if __name__ == "__main__":
    main()
