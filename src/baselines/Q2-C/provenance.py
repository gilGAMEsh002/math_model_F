# -*- coding: utf-8 -*-
"""附件 B 辅助索引的溯源核验（B11 模型仓元数据、B12 Pythia 检查点索引）。

B11/B12 **都不含 Loss 观测**，因此不能进入任何标度律拟合——这一点在报告里必须
讲清楚，不能装作它们"已被使用"。但它们能独立支撑三件真实的事：

    A. B1 内部一致性：D_tokens_B 是否等于 steps × batch_tokens_M；C_FLOPs_1e21 是否
       等于 6ND。这是 D 与 C 两列的口径核验（不是拟合）。
    B. B12 覆盖对账：B1 里出现的每个 (模型规模, step) 是否都在官方检查点索引中，
       以及索引里有而 B1 未用的部分——用于界定 B1 的样本覆盖边界。
    C. B11 参数量核验：由模型仓的 weight 文件字节数反推 bytes/param，独立核验
       Pythia（B1）与 Cerebras（B2）使用的参数量列。bytes/param 若显著偏离单份
       fp32 量级，说明该仓存在重复权重副本——这是溯源提示，不是数据错误。

全部为只读核验，不改动任何拟合结果。
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from common import scaling_dir

# 少量已知无歧义的规模后缀（仅供 B11 反推参数量；解析失败的行记为 NaN 并跳过）
_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)\s*([mMbB])\s*$")


def parse_params_from_name(name: str) -> float:
    """从模型仓名尾部的 70m / 1.3B 之类后缀解析参数量（单位：个）。"""
    m = _SUFFIX.search(str(name).strip())
    if not m:
        return float("nan")
    val = float(m.group(1))
    return val * 1e6 if m.group(2).lower() == "m" else val * 1e9


# ----------------------------------------------------- A. B1 内部一致性
def _col(df: pd.DataFrame, *names: str):
    """取第一个存在的列名（read_b 会把 N_params_B/D_tokens_B 重命名为 N/D）。"""
    for n in names:
        if n in df.columns:
            return df[n].astype(float)
    return None


def b1_internal_consistency(b1: pd.DataFrame, log=None) -> pd.DataFrame:
    """D 与 steps×batch 的口径核验，以及 C_FLOPs 与 6ND 的量级核验。"""
    rows = []
    N_ = _col(b1, "N_params_B", "N")
    D_ = _col(b1, "D_tokens_B", "D")

    if {"steps", "batch_tokens_M"}.issubset(b1.columns) and D_ is not None:
        pred = b1["steps"].astype(float) * b1["batch_tokens_M"].astype(float) / 1000.0
        rel = np.abs(pred - D_) / D_
        rows.append({
            "check": "D_tokens_B ?= steps * batch_tokens_M / 1000",
            "n": int(len(rel)), "rel_err_median": float(rel.median()),
            "rel_err_max": float(rel.max()), "abs_dev_max": None,
            "verdict": ("一致（偏差为 batch 规模的舍入）" if rel.max() < 0.05
                        else "存在系统性不一致，须查明单位"),
        })

    if "C_FLOPs_1e21" in b1.columns and N_ is not None and D_ is not None:
        # 6ND：N、D 以十亿计，故 6ND 的单位是 1e18 FLOPs；C 列单位是 1e21 FLOPs。
        six_nd = 6.0 * N_ * D_ / 1000.0
        C = b1["C_FLOPs_1e21"].astype(float)
        rel = np.abs(six_nd - C) / np.maximum(np.abs(C), 1e-12)
        # C 列本身只有 4 位小数，N·D 小时相对舍入误差天然很大 → 同时看绝对偏差
        abs_dev = np.abs(six_nd - C)
        rows.append({
            "check": "C_FLOPs_1e21 ?= 6*N*D (N,D 以十亿计)",
            "n": int(len(rel)),
            "rel_err_median": float(np.median(rel)), "rel_err_max": float(rel.max()),
            "abs_dev_max": float(abs_dev.max()),
            "verdict": (f"一致（最大绝对偏差 {abs_dev.max():.1e}，源于 C 列 4 位小数舍入；"
                        f"故 C 不作为独立回归量）" if abs_dev.max() <= 1e-4
                        else "量级不符，须查明 C 列口径"),
        })

    out = pd.DataFrame(rows)
    if log:
        for _, r in out.iterrows():
            log(f"  [溯源 {r['check']}] n={r['n']} 中位相对偏差={r['rel_err_median']:.2e} "
                f"最大={r['rel_err_max']:.2e} → {r['verdict']}")
    return out


# ----------------------------------------------------- B. B12 检查点覆盖
def _match_labels(sizes: np.ndarray, labels: list[str]) -> dict:
    """把 B1 的 N_params_B 数值匹配到 B12 的规模标签（按对数距离最近）。"""
    nom = {lab: parse_params_from_name("x" + lab.replace("b", "B").replace("m", "m"))
           for lab in labels}
    nom = {k: v for k, v in nom.items() if np.isfinite(v)}
    out = {}
    for s in sizes:
        best, bd = None, np.inf
        for lab, v in nom.items():
            d = abs(np.log(float(s) * 1e9 / v))
            if d < bd:
                best, bd = lab, d
        out[float(s)] = (best, float(bd))
    return out


def checkpoint_coverage(cfg: dict, b1: pd.DataFrame, log=None) -> pd.DataFrame:
    """B1 的 (规模, step) 与官方检查点索引 B12 的对账。"""
    f = os.path.join(scaling_dir(cfg), cfg["data"]["b12_ckpt_index"])
    if not os.path.isfile(f):
        if log:
            log(f"  [溯源] 未找到 B12 索引 {f}，跳过检查点覆盖对账")
        return pd.DataFrame()
    b12 = pd.read_csv(f)
    b1N = _col(b1, "N_params_B", "N")
    if b1N is None or "steps" not in b1.columns:
        if log:
            log("  [溯源] B1 缺少参数量或 steps 列，跳过检查点覆盖对账")
        return pd.DataFrame()
    sizes = np.sort(b1N.unique())
    lab_map = _match_labels(sizes, sorted(b12["model_size"].unique()))
    rows = []
    for s, (lab, dist) in lab_map.items():
        if dist > 0.15:      # 对数距离过大说明规模不对应，不强行匹配
            rows.append({"N_params_B": float(s), "matched_label": "",
                         "match_log_dist": dist, "n_in_index": 0, "n_in_b1": 0,
                         "b1_not_in_index": np.nan, "index_not_in_b1": np.nan,
                         "verdict": "无对应规模标签，跳过"})
            continue
        s12 = set(b12.loc[b12["model_size"] == lab, "step"].astype(int))
        sb1 = set(b1.loc[b1N == s, "steps"].astype(int))
        rows.append({"N_params_B": float(s), "matched_label": lab, "match_log_dist": dist,
                     "n_in_index": len(s12), "n_in_b1": len(sb1),
                     "b1_not_in_index": len(sb1 - s12), "index_not_in_b1": len(s12 - sb1),
                     "verdict": ("B1 检查点全部命中索引" if not (sb1 - s12)
                                 else f"有 {len(sb1 - s12)} 个检查点不在索引中")})
    out = pd.DataFrame(rows)
    # 索引中存在而 B1 完全没有的规模（覆盖边界）
    used = {lab for lab, _ in lab_map.values() if lab}
    missing_sizes = sorted(set(b12["model_size"].unique()) - used)
    if log:
        hit = int(out["b1_not_in_index"].fillna(0).sum())
        extra = int(out["index_not_in_b1"].fillna(0).sum())
        log(f"  [溯源 B12] 匹配规模 {len(used)} 个；B1 检查点未命中索引 {hit} 个；"
            f"索引中有而 B1 未用 {extra} 个；索引中含而 B1 完全未覆盖的规模: "
            f"{missing_sizes if missing_sizes else '无'}")
    out.attrs["missing_sizes"] = missing_sizes
    return out


# ----------------------------------------------------- C. B11 参数量核验
def param_audit(cfg: dict, log=None, other_models=None) -> pd.DataFrame:
    """由模型仓 weight 字节数反推 bytes/param，独立核验参数量列。

    other_models: 其他来源用到的模型名集合（B9/B10）。B11 只覆盖少数开源仓，
    与 B9/B10 的模型集若无交集，则**无法**佐证它们的参数量——这一范围限制必须
    显式记录下来，不能让读者以为 B11 核验了全部来源。
    """
    f = os.path.join(scaling_dir(cfg), cfg["data"]["b11_open_meta"])
    if not os.path.isfile(f):
        if log:
            log(f"  [溯源] 未找到 B11 模型仓元数据 {f}，跳过参数量核验")
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d.dropna(subset=["weight_total_bytes"]).copy()
    d["N_est"] = d["model_repo"].map(parse_params_from_name)
    d = d[np.isfinite(d["N_est"]) & (d["N_est"] > 0)].copy()
    # 规模后缀（70m / 1.3B），仅用于分组阅读
    d["lab"] = d["model_repo"].map(
        lambda s: (m.group(0).strip() if (m := _SUFFIX.search(str(s).strip())) else ""))
    d["bytes_per_param"] = d["weight_total_bytes"].astype(float) / d["N_est"]
    # 单份 fp32 权重约 4 字节/参数；显著高于 6 判为存在重复权重副本
    d["single_copy"] = (d["bytes_per_param"] > 3.0) & (d["bytes_per_param"] < 6.0)
    d["family_prefix"] = d["model_repo"].str.split("/").str[0]
    out = d[["model_repo", "family_prefix", "lab", "N_est", "weight_file_count",
             "weight_total_bytes", "bytes_per_param", "single_copy"]]
    if log:
        g = out.groupby("family_prefix")["bytes_per_param"].agg(["min", "max", "count"])
        for fam, r in g.iterrows():
            log(f"  [溯源 B11] {fam:12s} n={int(r['count']):2d} "
                f"bytes/param {r['min']:.2f}–{r['max']:.2f} "
                f"{'（单份副本，与参数量列自洽）' if r['max'] < 6 else '（存在重复权重副本）'}")
        if other_models:
            repo_names = {str(s).split("/")[-1].strip().lower()
                          for s in d["model_repo"].astype(str)}
            other = {str(s).strip().lower() for s in other_models}
            inter = repo_names & other
            if inter:
                log(f"  [溯源 B11] 与 B9/B10 有 {len(inter)} 个同名模型，可交叉佐证: "
                    f"{sorted(inter)}")
            else:
                log(f"  [溯源 B11] 与 B9/B10 的模型集**无交集**"
                    f"（B11 覆盖 {len(repo_names)} 个开源仓，B9/B10 为 {len(other)} 个"
                    f"其他模型）→ B11 只能佐证 B1(Pythia)/B2(Cerebras) 的参数量，"
                    f"**不能**佐证 B9/B10")
    return out


# ----------------------------------------------------- D. 收敛标志核验
def convergence_audit(cfg: dict, loaded: dict, log=None) -> pd.DataFrame:
    """B4/B10 的 is_converged 列：若非全 1，未收敛行应剔除后再算误差。"""
    rows = []
    for key in ("b4_cross_family", "b10_large_baseline"):
        d = loaded.get(key)
        if d is None or "is_converged" not in d.columns:
            continue
        v = pd.to_numeric(d["is_converged"], errors="coerce")
        rows.append({"key": key, "file": cfg["data"][key], "n": int(len(v)),
                     "n_converged": int((v == 1).sum()),
                     "n_not_converged": int((v != 1).sum()),
                     "verdict": ("全部收敛，无行需剔除" if int((v != 1).sum()) == 0
                                 else f"有 {int((v != 1).sum())} 行未收敛，"
                                      f"未剔除会污染跨源误差")})
    out = pd.DataFrame(rows)
    if log:
        for _, r in out.iterrows():
            log(f"  [溯源 {r['key']:20s}] n={r['n']} 收敛={r['n_converged']} "
                f"未收敛={r['n_not_converged']} → {r['verdict']}")
    return out


# ----------------------------------------------------- 汇总
def provenance_audit(cfg: dict, b1: pd.DataFrame, log=None, other_models=None,
                     loaded=None) -> dict:
    """跑完 A/B/C/D 四项，返回 {内部一致性, 检查点覆盖, 参数量核验, 收敛标志}。"""
    b12 = checkpoint_coverage(cfg, b1, log=log)
    return {"b1_internal_consistency": b1_internal_consistency(b1, log=log),
            "b12_checkpoint_coverage": b12,
            "b11_param_audit": param_audit(cfg, log=log, other_models=other_models),
            "convergence_audit": (convergence_audit(cfg, loaded, log=log)
                                  if loaded else pd.DataFrame())}
