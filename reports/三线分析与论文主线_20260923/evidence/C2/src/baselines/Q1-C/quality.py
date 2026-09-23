# -*- coding: utf-8 -*-
"""Q1-C 第一部分：熵权质量分 + 共同冲突规则（与等权对照）。

对应任务卡 Q1-C.md 第 1、2 条与 01_问题分析与Baseline实施计划.md §1.2-1.5：
  1. 在固定的评分开发参考集上计算熵权，处理零值和常数列；熵权不代表因果重要性。
  2. 沿用共同冲突定义 c_i = max_g z_ig - min_g z_ig，阈值 τ 由开发集分位数确定，
     并给出多阈值敏感性；消解规则 Q = (1-λ)Q_base + λ min_g z_ig 只报告预设网格。

输入不重新读取 A1/A2/A3 原始 JSONL，复用上游冻结的标准化质量矩阵 Z ∈ [0,1]^(N×22)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (METRIC_ORDER, Timer, save_json, save_table, spearman)


# --------------------------------------------------------------- 参考集切分
def split_reference(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """把 A1 按域分层切成评分开发参考集(dev)与核验集(verify)。

    熵权与冲突阈值只在 dev 上确定，然后冻结；verify 只用于敏感性/稳健性核验。
    """
    seed = cfg["seed"]
    frac = float(cfg["quality"]["dev_fraction"])
    strat = cfg["quality"]["stratify_by"]
    idx = np.arange(len(df))
    dev_idx, ver_idx = train_test_split(
        idx, test_size=1.0 - frac, random_state=seed, stratify=df[strat].values)
    out = pd.Series("verify", index=df.index, dtype=object)
    out.iloc[dev_idx] = "dev"
    out.iloc[ver_idx] = "verify"
    return out


# --------------------------------------------------------------- 熵权
def entropy_weights(Z: np.ndarray, eps: float) -> pd.DataFrame:
    """标准熵权法。

    p_ij = z_ij / Σ_i z_ij（z 已落在 [0,1]，无需再平移）
    e_j  = -1/ln(n) · Σ_i p_ij ln p_ij ，按约定 0·ln0 := 0
    d_j  = 1 - e_j（冗余度/差异系数）
    w_j  = d_j / Σ_k d_k

    退化列处理（任务卡第 1 条「处理零值和常数列」）：
      - 全零列 Σ_i z_ij = 0：p_ij 无定义，该列 d_j := 0；
      - 常数列 std_j = 0：p 均匀，e_j = 1，d_j 自然为 0（此处亦显式置 0 以防浮点误差）。
    权重在剩余列上重新归一化；退化列权重为 0 并在输出中标记。
    """
    Z = np.asarray(Z, dtype=float)
    n, m = Z.shape
    colsum = Z.sum(axis=0)
    std = Z.std(axis=0, ddof=0)
    is_all_zero = colsum <= 0.0
    is_constant = std <= 1e-15

    P = np.zeros_like(Z)
    ok = ~(is_all_zero | is_constant)
    P[:, ok] = Z[:, ok] / colsum[ok]

    k = 1.0 / np.log(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plnp = np.where(P > 0.0, P * np.log(np.where(P > 0.0, P, 1.0)), 0.0)
    e = -k * plnp.sum(axis=0)
    e = np.clip(e, 0.0, 1.0)
    d = 1.0 - e
    d[~ok] = 0.0
    tot = d.sum()
    w = d / tot if tot > 0 else np.full(m, 1.0 / m)

    # 零值放大敏感性：z' = (z+eps)/(1+eps) 后重算
    Zs = (Z + eps) / (1.0 + eps)
    Ps = Zs / Zs.sum(axis=0, keepdims=True)
    es = -k * np.where(Ps > 0, Ps * np.log(np.where(Ps > 0, Ps, 1.0)), 0.0).sum(axis=0)
    ds = 1.0 - np.clip(es, 0.0, 1.0)
    ws = ds / ds.sum() if ds.sum() > 0 else np.full(m, 1.0 / m)

    return pd.DataFrame({
        "colsum": colsum, "std": std,
        "entropy": e, "divergence": d,
        "weight_entropy": w, "weight_entropy_eps": ws,
        "weight_equal": np.full(m, 1.0 / m),
        "is_all_zero": is_all_zero, "is_constant": is_constant,
        "degenerate": ~ok,
    }, index=[f"m{i}" for i in range(m)])


def decorrelated_weights(Z: np.ndarray, thr: float = 0.95) -> tuple[np.ndarray, list]:
    """相关性去重后的熵权敏感性：对 |r|>thr 的指标对只保留方差较大者。

    动机见 §1.3「相关指标重复计权」——dsir_* 三列与 rps_doc_frac_chars_top_* 高度共线，
    等权/熵权都会给同一信息重复计权。
    """
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    m = Z.shape[1]
    drop: set[int] = set()
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            if i in drop or j in drop:
                continue
            if abs(C[i, j]) > thr:
                keep, kill = (i, j) if Z[:, i].std() >= Z[:, j].std() else (j, i)
                drop.add(kill)
                pairs.append((i, j, float(C[i, j]), kill))
    keep_idx = [i for i in range(m) if i not in drop]
    Zk = Z[:, keep_idx]
    colsum = Zk.sum(axis=0)
    std = Zk.std(axis=0)
    ok = ~((colsum <= 0) | (std <= 1e-15))
    P = np.zeros_like(Zk)
    P[:, ok] = Zk[:, ok] / colsum[ok]
    k = 1.0 / np.log(Zk.shape[0])
    e = -k * np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0).sum(axis=0)
    d = 1.0 - np.clip(e, 0.0, 1.0)
    d[~ok] = 0.0
    w = np.zeros(m)
    w[keep_idx] = d / d.sum() if d.sum() > 0 else 1.0 / len(keep_idx)
    return w, pairs


# --------------------------------------------------------------- 质量组与冲突
def group_matrix(Z: np.ndarray, groups: dict) -> tuple[np.ndarray, list[str]]:
    """组内算术平均得到 z_ig（§1.3）。"""
    cols = {m: i for i, m in enumerate(METRIC_ORDER)}
    names = list(groups.keys())
    G = np.column_stack([Z[:, [cols[m] for m in groups[g]]].mean(axis=1) for g in names])
    return G, names


def conflict_labels(G: np.ndarray, names: list[str], tau: float) -> dict:
    """冲突强度 c_i = max_g z_ig - min_g z_ig（范围差）。"""
    c = G.max(axis=1) - G.min(axis=1)
    amax = G.argmax(axis=1)
    amin = G.argmin(axis=1)
    return {
        "c": c,
        "conflict": (c > tau).astype(int),
        "max_group": np.array(names, dtype=object)[amax],
        "min_group": np.array(names, dtype=object)[amin],
        "min_group_score": G.min(axis=1),
        "max_group_score": G.max(axis=1),
    }


def resolved_quality(q_base: np.ndarray, min_group: np.ndarray, lam: float) -> np.ndarray:
    """Q_resolved = (1-λ) Q_base + λ min_g z_ig（短板约束的凸组合）。"""
    return (1.0 - lam) * q_base + lam * min_group


# --------------------------------------------------------------- 域级聚合
def domain_aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """各域样本量、均值、分位数、冲突率与正态近似 95% 区间。"""
    rows = []
    for keys, g in df.groupby(by, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(by, keys))
        rec["n"] = len(g)
        for col in ("Q_equal", "Q_entropy", "Q_resolved", "c"):
            v = g[col].to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(np.mean(v))
            rec[f"{col}_p05"] = float(np.quantile(v, 0.05))
            rec[f"{col}_p50"] = float(np.median(v))
            rec[f"{col}_p95"] = float(np.quantile(v, 0.95))
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            rec[f"{col}_ci95_lo"] = float(np.mean(v) - 1.96 * se) if len(v) > 1 else np.nan
            rec[f"{col}_ci95_hi"] = float(np.mean(v) + 1.96 * se) if len(v) > 1 else np.nan
        rec["conflict_rate"] = float(g["conflict"].mean())
        rec["missing_metric_cells"] = int(g["n_missing"].sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 主流程
def run_quality(cfg: dict, log=print) -> dict:
    qcfg = cfg["quality"]
    metrics = qcfg["metrics"]
    assert list(metrics) == METRIC_ORDER, "配置的指标顺序与规范顺序不一致"
    groups = qcfg["quality_groups"]
    lam = float(qcfg["lambda_default"])
    eps = float(qcfg["entropy"]["epsilon"])

    # ---- 读取上游冻结产物
    from common import load_quality_frame
    frames = {}
    for name in ("A1", "A2", "A3"):
        df = load_quality_frame(cfg, name)
        df = df.copy()
        df["n_missing"] = df[metrics].isna().any(axis=1).astype(int)
        frames[name] = df
        log(f"[quality] 读入 {name}: {df.shape[0]} 行 × {len(metrics)} 指标, 域={sorted(df['domain'].unique())}")

    A1 = frames["A1"]
    A1["split"] = split_reference(A1, cfg).values
    for n in ("A2", "A3"):
        frames[n]["split"] = "extended"

    # ---- 1) 熵权（只在 dev 上确定）
    dev = A1[A1["split"] == "dev"]
    ver = A1[A1["split"] == "verify"]
    Zdev = dev[metrics].to_numpy(dtype=float)
    Zver = ver[metrics].to_numpy(dtype=float)

    with Timer() as t_ent:
        W = entropy_weights(Zdev, eps)
    W.index = metrics
    W.index.name = "metric"

    Wv = entropy_weights(Zver, eps)
    Wv.index = metrics
    Wdc, corr_pairs = decorrelated_weights(Zdev)
    W["weight_entropy_verify"] = Wv["weight_entropy"].to_numpy()
    W["weight_entropy_decorrelated"] = Wdc
    grp_of = {m: g for g, ms in groups.items() for m in ms}
    W.insert(0, "group", [grp_of[m] for m in metrics])

    C = np.corrcoef(Zdev, rowvar=False)
    np.fill_diagonal(C, 0.0)
    W["corr_max_abs"] = np.abs(np.nan_to_num(C)).max(axis=1)
    W["corr_partner"] = [metrics[int(np.nan_to_num(C).__abs__()[i].argmax())] for i in range(len(metrics))]

    w_ent = W["weight_entropy"].to_numpy(dtype=float)
    w_eq = W["weight_equal"].to_numpy(dtype=float)

    log(f"[quality] 熵权完成 {t_ent.elapsed:.2f}s；退化列="
        f"{W.index[W['degenerate'].to_numpy()].tolist() or '无'}；"
        f"权重 top3={W['weight_entropy'].nlargest(3).round(4).to_dict()}")
    # 等权向量为常数，与它的秩相关无定义；改用有效指标数 1/Σw² 刻画权重集中度
    eff_n_ent = float(1.0 / np.sum(w_ent ** 2))
    eff_n_eq = float(1.0 / np.sum(w_eq ** 2))
    log(f"[quality] 有效指标数 1/Σw²：熵权={eff_n_ent:.2f} / 等权={eff_n_eq:.2f}（共 {len(metrics)} 个指标）")

    # ---- 2) 组分数、冲突、综合质量
    def apply_all(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        Z = df[metrics].to_numpy(dtype=float)
        G, names = group_matrix(Z, groups)
        out = pd.DataFrame({
            "id": df["id"].values, "domain": df["domain"].values,
            "source_set": df["source_set"].values if "source_set" in df else tag,
            "split": df["split"].values, "n_missing": df["n_missing"].values,
            "Q_equal": Z @ w_eq, "Q_entropy": Z @ w_ent,
            "c": G.max(axis=1) - G.min(axis=1),
            "min_group_score": G.min(axis=1), "max_group_score": G.max(axis=1),
        })
        out["max_group"] = np.array(names, dtype=object)[G.argmax(axis=1)]
        out["min_group"] = np.array(names, dtype=object)[G.argmin(axis=1)]
        for j, nm in enumerate(names):
            out[f"g_{nm}"] = G[:, j]
        return out

    parts = [apply_all(frames[n], n) for n in ("A1", "A2", "A3")]
    Q = pd.concat(parts, ignore_index=True)

    # ---- 3) 冲突阈值 τ：dev 分位数 + 多阈值敏感性
    tau_grid = qcfg["conflict"]["quantile_grid"]
    dev_q = Q[Q["split"] == "dev"]
    taus = {f"q{int(q*100)}": float(np.quantile(dev_q["c"], q)) for q in tau_grid}
    tau_def = taus[f"q{int(qcfg['conflict']['default_quantile']*100)}"]

    audit = []
    for tag, tau in taus.items():
        flag = (Q["c"] > tau).astype(int)
        rec = {"threshold_tag": tag, "quantile": tag, "tau": tau,
               "n_total": len(Q), "conflict_rate_total": float(flag.mean())}
        for n in ("A1", "A2", "A3"):
            m = Q["source_set"] == n
            rec[f"conflict_rate_{n}"] = float(flag[m].mean())
        for d, g in Q.assign(_f=flag).groupby("domain", observed=True):
            rec[f"conflict_rate_domain_{d}"] = float(g["_f"].mean())
        audit.append(rec)
    audit_df = pd.DataFrame(audit)

    Q["conflict"] = (Q["c"] > tau_def).astype(int)
    Q["Q_resolved"] = resolved_quality(Q["Q_equal"].to_numpy(),
                                       Q["min_group_score"].to_numpy(), lam)
    Q["Q_resolved_entropy"] = resolved_quality(Q["Q_entropy"].to_numpy(),
                                               Q["min_group_score"].to_numpy(), lam)

    # 冲突类型（max/min 组组合）
    ctypes = (Q[Q["conflict"] == 1]
              .groupby(["max_group", "min_group"], observed=True)
              .size().reset_index(name="n")
              .sort_values("n", ascending=False))
    ctypes["share_of_conflicts"] = ctypes["n"] / ctypes["n"].sum()
    # 分域冲突类型
    ctypes_dom = (Q[Q["conflict"] == 1]
                  .groupby(["domain", "max_group", "min_group"], observed=True)
                  .size().reset_index(name="n")
                  .sort_values(["domain", "n"], ascending=[True, False]))

    # ---- 4) λ 网格（只报告，不宣称唯一最优）
    lam_rows = []
    for l in qcfg["lambda_grid"]:
        qr = resolved_quality(Q["Q_equal"].to_numpy(), Q["min_group_score"].to_numpy(), float(l))
        lam_rows.append({
            "lambda": float(l),
            "Q_resolved_mean": float(qr.mean()),
            "Q_resolved_std": float(qr.std(ddof=1)),
            "spearman_vs_Q_equal": spearman(qr, Q["Q_equal"].to_numpy()),
            "spearman_vs_min_group": spearman(qr, Q["min_group_score"].to_numpy()),
        })
    lam_df = pd.DataFrame(lam_rows)

    # ---- 5) 域级聚合：A1 / A2 / A3 / 去重合并 / 扩展非重叠
    merged, overlap_stats = merge_dedup(frames)
    # 合并集同样套用冻结的熵权与冲突阈值，得到可与 A1/A2/A3 直接对照的域级 Q
    from_col = merged["_from"].to_numpy()
    merged = apply_all(merged, "merged")
    merged["_from"] = from_col
    merged["conflict"] = (merged["c"] > tau_def).astype(int)
    merged["Q_resolved"] = resolved_quality(
        merged["Q_equal"].to_numpy(), merged["min_group_score"].to_numpy(), lam)
    merged["Q_resolved_entropy"] = resolved_quality(
        merged["Q_entropy"].to_numpy(), merged["min_group_score"].to_numpy(), lam)
    merged["split"] = "merged"
    nonoverlap = merged[merged["_from"] != "A1"].copy()
    nonoverlap["split"] = "extended_nonoverlap"

    dom_frames = []
    for tag, dd in (("A1", Q[Q.source_set == "A1"]),
                    ("A2", Q[Q.source_set == "A2"]),
                    ("A3", Q[Q.source_set == "A3"]),
                    ("merged", merged),
                    ("extended_nonoverlap", nonoverlap)):
        agg = domain_aggregate(dd, ["domain"])
        agg.insert(0, "set", tag)
        dom_frames.append(agg)
    dom_df = pd.concat(dom_frames, ignore_index=True)
    dev_dom = domain_aggregate(Q[Q.split == "dev"], ["domain"]).assign(set="A1_dev_reference")
    ver_dom = domain_aggregate(Q[Q.split == "verify"], ["domain"]).assign(set="A1_verify")
    dom_df = pd.concat([dom_df, dev_dom, ver_dom], ignore_index=True)

    # ---- 6) 等权 vs 熵权对照（域级）
    cmp_rows = []
    for tag, dd in (("A1", Q[Q.source_set == "A1"]), ("merged", merged)):
        g = dd.groupby("domain", observed=True)[["Q_equal", "Q_entropy", "Q_resolved"]].mean()
        cmp_rows.append({
            "set": tag, "n_domains": len(g),
            "spearman_Q_equal_vs_Q_entropy": spearman(g["Q_equal"], g["Q_entropy"]),
            "mean_abs_rank_shift": mean_abs_rank_shift(g["Q_equal"], g["Q_entropy"]),
            "topk1_agree": bool(g["Q_equal"].idxmax() == g["Q_entropy"].idxmax()),
            "ranked_by_equal": " > ".join(g["Q_equal"].sort_values(ascending=False).index),
            "ranked_by_entropy": " > ".join(g["Q_entropy"].sort_values(ascending=False).index),
        })
    cmp_df = pd.DataFrame(cmp_rows)

    # ---- 7) 域映射 A16 + 域质量（供配比模型的质量特征用）
    from common import load_domain_mapping
    mp = load_domain_mapping(cfg)
    # 质量域 -> 合并集上的域质量（等权/熵权）
    qdom = merged.groupby("domain", observed=True).agg(
        n=("id", "size"),
        q_equal=("Q_equal", "mean"),
        q_entropy=("Q_entropy", "mean"),
        q_resolved=("Q_resolved", "mean"),
        conflict_rate=("conflict", "mean"),
    ).reset_index().rename(columns={"domain": "quality_domain"})
    mp = mp.merge(qdom, left_on="quality_domain", right_on="quality_domain", how="left")
    mapped = mp["n"].notna()
    mp["mapped"] = mapped
    mp["mixture_col"] = ["train_the_pile_" + d for d in mp["mixture_domain"]]
    mp["quality_domain"] = mp["quality_domain"].replace({"(none)": None})

    log(f"[quality] 冲突阈值 τ(q90)={tau_def:.4f}；冲突率={Q['conflict'].mean():.4f}；"
        f"映射 direct/near_direct={int(mapped.sum())}/{len(mp)}")

    # ---- 落盘
    save_table(W.reset_index(), cfg, "quality_entropy_weights.csv")
    save_table(pd.DataFrame(corr_pairs, columns=["i", "j", "corr", "dropped_idx"]).assign(
        metric_i=lambda d: [metrics[int(i)] for i in d["i"]],
        metric_j=lambda d: [metrics[int(j)] for j in d["j"]],
        metric_dropped=lambda d: [metrics[int(k)] for k in d["dropped_idx"]]
    )[["metric_i", "metric_j", "corr", "metric_dropped"]], cfg, "quality_corr_pairs.csv")
    save_table(Q, cfg, "quality_scores.parquet")
    save_table(dom_df, cfg, "quality_domain_aggregate.csv")
    save_table(audit_df, cfg, "quality_conflict_threshold_audit.csv")
    save_table(ctypes, cfg, "quality_conflict_types.csv")
    save_table(ctypes_dom, cfg, "quality_conflict_types_by_domain.csv")
    save_table(lam_df, cfg, "quality_lambda_grid.csv")
    save_table(cmp_df, cfg, "quality_weight_ablation.csv")
    save_table(mp, cfg, "quality_domain_mapping.csv")
    save_table(pd.DataFrame([overlap_stats]), cfg, "quality_sample_flow.csv")

    summary = {
        "entropy_seconds": t_ent.elapsed,
        "degenerate_metrics": W.index[W["degenerate"].to_numpy()].tolist(),
        "theta_tau_q90": tau_def,
        "tau_grid": taus,
        "conflict_rate_total": float(Q["conflict"].mean()),
        "effective_n_metrics_entropy": eff_n_ent,
        "effective_n_metrics_equal": eff_n_eq,
        "note_weight_rank_corr": "熵权与等权的秩相关无定义（等权向量为常数），"
                                 "故改用有效指标数 1/Σw² 刻画权重集中度",
        "n_dev": int((A1["split"] == "dev").sum()),
        "n_verify": int((A1["split"] == "verify").sum()),
        "sample_flow": overlap_stats,
        "lambda_default": lam,
        "domain_quality_merged": qdom.to_dict(orient="records"),
        "mapped_quality_domains": int(mapped.sum()),
    }
    save_json(summary, cfg, "quality_summary.json")
    return {"Q": Q, "W": W, "merged": merged, "domains": dom_df, "mapping": mp,
            "lambda_grid": lam_df, "comparison": cmp_df, "summary": summary}


def mean_abs_rank_shift(a, b) -> float:
    ra = pd.Series(a).rank(ascending=False)
    rb = pd.Series(b).rank(ascending=False)
    return float((ra - rb).abs().mean())


def merge_dedup(frames: dict) -> tuple[pd.DataFrame, dict]:
    """按 (quality_domain, id) 去重合并 A1/A2/A3；A1 优先保留。

    §0.2：A1 与 A2/A3 在 arxiv、github 上重叠，合并必须按域和 id 去重。
    """
    from common import METRIC_ORDER
    recs = []
    for name in ("A1", "A2", "A3"):
        d = frames[name]
        recs.append(pd.DataFrame({
            "id": d["id"].values, "domain": d["domain"].values,
            "source_set": name,
            "split": d["split"].values if "split" in d else name,
            "n_missing": d["n_missing"].values if "n_missing" in d else 0,
            **{m: d[m].values for m in METRIC_ORDER},
        }))
    allq = pd.concat(recs, ignore_index=True)
    # A1 优先级最高，其次 A2，再次 A3
    allq["_prio"] = allq["source_set"].map({"A1": 0, "A2": 1, "A3": 2})
    allq = allq.sort_values(["domain", "id", "_prio"], kind="mergesort")
    before = len(allq)
    ded = allq.drop_duplicates(subset=["domain", "id"], keep="first").copy()
    ded["_from"] = ded["source_set"]

    # 逐对重叠统计
    def ids(n):
        return set(map(tuple, frames[n][["domain", "id"]].values.tolist()))
    a1, a2, a3 = ids("A1"), ids("A2"), ids("A3")
    stats = {
        "A1_rows": len(frames["A1"]), "A2_rows": len(frames["A2"]), "A3_rows": len(frames["A3"]),
        "concat_rows": before, "merged_rows": len(ded), "dropped_duplicates": before - len(ded),
        "overlap_A1_A2_arxiv": len(a1 & a2), "overlap_A1_A3_github": len(a1 & a3),
        "overlap_A2_A3": len(a2 & a3),
        "merged_by_domain": ded.groupby("domain", observed=True).size().to_dict(),
    }
    return ded, stats
