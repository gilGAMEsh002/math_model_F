# -*- coding: utf-8 -*-
"""
Step 1 预处理的独立验证。

不信任 quality_preprocess.py 的输出，从 real_attachments 原始 jsonl.xz 重算一个样本，
并检查会改变结论的错误类别：
  V1  Z 全部落在 [0,1]、无 NaN
  V2  抽样重算 vs parquet 逐列一致（抓列序/偏移/公式接线错误）
  V3  缺失填补语义：被填补的记录应得到完全相同的 z（同一均值向量 -> 同一压缩结果）
  V4  方向正确性：负向指标的 raw 与 z 应呈负秩相关，正向应为正秩相关
  V5  非单调指标：极端取值应被打到 0 分
  V6  二分类/六分类压缩与手算一致

用法：python src/preprocess/verify_preprocess.py
"""

from __future__ import annotations

import glob
import json
import lzma
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SRC_DIR))
A_DIR = os.path.join(ROOT, "real_attachments", "A_data_value")
OUT_DIR = os.path.join(ROOT, "artifacts", "q1", "preprocessed")
CFG = os.path.join(ROOT, "configs", "quality_preprocess_params.json")

N_SAMPLE = 4000
fails: list[str] = []
checks: list[str] = []


def ok(cond, msg):
    (checks if cond else fails).append(("PASS  " if cond else "FAIL  ") + msg)
    print(("  PASS  " if cond else "  FAIL  ") + msg, flush=True)


SCALARS_G = ["dsir_books", "dsir_math", "dsir_wiki",
             "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
             "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
             "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
             "rps_doc_mean_word_length", "rps_lines_uppercase_letter_fraction",
             "rps_lines_numerical_chars_fraction",
             "rps_lines_ending_with_terminal_punctution_mark"]
LISTS_G = [("fineweb_edu", 1), ("fluency_en", 2), ("ad_en", 2), ("qurater", 4),
           ("modernbert_cleanliness", 6), ("modernbert_readability", 6),
           ("modernbert_reasoning", 6), ("modernbert_professionalism", 6)]


def scan_missing(path: str):
    """全量扫描，返回 [(id, 缺失字段集合), ...]，用于定位被填补的记录。"""
    out = []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            miss = set()
            for n in SCALARS_G:
                v = r.get(n)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                    miss.add(n)
            for n, k in LISTS_G:
                v = r.get(n)
                if not isinstance(v, list) or len(v) != k:
                    miss.add(n); continue
                for x in v:
                    if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
                        miss.add(n); break
            if miss:
                out.append((r.get("id"), miss))
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def main() -> int:
    with open(CFG, encoding="utf-8") as f:
        P = json.load(f)
    order22 = P["spec"]["order22"]
    neg = set(P["spec"]["negative_metrics"])
    nonmono = set(P["spec"]["non_monotonic_metrics"])
    rank_r = np.array(P["spec"]["modernbert_rank"])
    med = np.array(P["scalar_median"])
    q_lo = np.array(P["qurater_dim_min"])
    q_hi = np.array(P["qurater_dim_max"])

    SCALARS = ["dsir_books", "dsir_math", "dsir_wiki",
               "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
               "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
               "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
               "rps_doc_mean_word_length", "rps_lines_uppercase_letter_fraction",
               "rps_lines_numerical_chars_fraction",
               "rps_lines_ending_with_terminal_punctution_mark"]
    LISTS = [("fineweb_edu", 1), ("fluency_en", 2), ("ad_en", 2), ("qurater", 4),
             ("modernbert_cleanliness", 6), ("modernbert_readability", 6),
             ("modernbert_reasoning", 6), ("modernbert_professionalism", 6)]

    a1_path = os.path.join(A_DIR, "slimpajama_quality_signal_sample.jsonl.xz")
    df1 = pd.read_parquet(os.path.join(OUT_DIR, "A1_processed.parquet"))

    print("\n=== V1 取值域 ===")
    for tag in ("A1", "A2", "A3"):
        d = pd.read_parquet(os.path.join(OUT_DIR, f"{tag}_processed.parquet"))[order22]
        z = d.to_numpy()
        ok(np.isfinite(z).all(), f"{tag}: 无 NaN/inf")
        ok(z.min() >= 0.0 and z.max() <= 1.0, f"{tag}: 全部落在 [0,1]（实际 [{z.min():.6f}, {z.max():.6f}]）")

    print("\n=== V2 抽样重算 vs parquet ===")
    raw, ids, miss_pos = [], [], []
    with lzma.open(a1_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLE:
                break
            r = json.loads(line)
            rec = np.full(47, np.nan)
            for j, n in enumerate(SCALARS):
                v = r.get(n)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rec[j] = float(v)
            off = 14
            bad = False
            for n, k in LISTS:
                v = r.get(n)
                if isinstance(v, list) and len(v) == k:
                    for t, x in enumerate(v):
                        if isinstance(x, (int, float)) and not isinstance(x, bool):
                            rec[off + t] = float(x)
                        else:
                            bad = True
                else:
                    bad = True
                off += k
            raw.append(rec); ids.append(r["id"]); miss_pos.append(bad)
    raw = np.vstack(raw)
    bad_cells = ~np.isfinite(raw)
    if bad_cells.any():
        raw[bad_cells] = np.take(med, np.where(bad_cells)[1])

    M = np.empty((len(raw), 22))
    idx = {n: i for i, n in enumerate(order22)}
    off, els = 14, {}
    for n, k in LISTS:
        els[n] = raw[:, off:off + k]; off += k
    for n in SCALARS:
        M[:, idx[n]] = raw[:, SCALARS.index(n)]
    M[:, idx["fineweb_edu"]] = els["fineweb_edu"][:, 0]
    for n in ("fluency_en", "ad_en"):
        z = els[n] - els[n].max(axis=1, keepdims=True)
        e = np.exp(z); e /= e.sum(axis=1, keepdims=True)
        M[:, idx[n]] = e[:, 1]
    qn = np.clip((els["qurater"] - q_lo) / np.where(q_hi > q_lo, q_hi - q_lo, 1.0), 0, 1)
    M[:, idx["qurater"]] = qn.mean(axis=1)
    for n in ("modernbert_cleanliness", "modernbert_readability",
              "modernbert_reasoning", "modernbert_professionalism"):
        z = els[n] - els[n].max(axis=1, keepdims=True)
        e = np.exp(z); e /= e.sum(axis=1, keepdims=True)
        M[:, idx[n]] = e @ rank_r
    for n in ("dsir_books", "dsir_math", "dsir_wiki"):
        v = M[:, idx[n]]
        M[:, idx[n]] = np.sign(v) * np.log1p(np.abs(v))

    Z = np.empty_like(M)
    for n in order22:
        j = idx[n]
        if n in nonmono:
            t = P["trapezoid"][n]
            L, a, b, U = t["L"], t["a"], t["b"], t["U"]
            x = np.log1p(np.clip(M[:, j], 0, None))
            y = np.zeros_like(x)
            lo_m = (x > L) & (x < a); y[lo_m] = (x[lo_m] - L) / (a - L)
            y[(x >= a) & (x <= b)] = 1.0
            hi_m = (x > b) & (x < U); y[hi_m] = (U - x[hi_m]) / (U - b)
            Z[:, j] = y
        else:
            lo, hi = P["minmax_bounds"][n]
            v = np.clip((M[:, j] - lo) / (hi - lo), 0, 1)
            Z[:, j] = (1.0 - v) if n in neg else v

    got = df1.set_index("id").loc[ids, order22].to_numpy()
    diff = np.abs(got - Z)
    worst = np.unravel_index(np.argmax(diff), diff.shape)
    ok(diff.max() < 1e-9,
       f"前 {N_SAMPLE} 条重算与 parquet 一致（最大绝对差 {diff.max():.3e}，"
       f"位置 {order22[worst[1]]}）")

    print("\n=== V3 缺失填补语义（全量扫描定位缺失记录）===")
    df3 = pd.read_parquet(os.path.join(OUT_DIR, "A3_processed.parquet"))
    a3_path = glob.glob(os.path.join(A_DIR, "slimpajama_quality_extended", "github_*.jsonl.xz"))[0]
    miss1 = scan_missing(a1_path)
    miss3 = scan_missing(a3_path)
    ok(len(miss1) == 18, f"A1 含缺失记录数 = {len(miss1)}（期望 18）")
    ok(len(miss3) == 1, f"A3 含缺失记录数 = {len(miss3)}（期望 1）")

    d1 = df1.set_index("id")
    d3 = df3.set_index("id")
    for mn in ("modernbert_reasoning", "modernbert_professionalism"):
        sub = [i for i, m in miss1 if mn in m]
        if sub:
            nun = d1.loc[sub, mn].round(10).nunique()
            ok(nun == 1,
               f"A1 缺失 {mn} 的 {len(sub)} 条记录，z 取值唯一性 = {nun}（期望 1，"
               f"值 {d1.loc[sub[0], mn]:.10f}）")
        sub3 = [i for i, m in miss3 if mn in m]
        if sub3 and sub:
            same = abs(d3.loc[sub3[0], mn] - d1.loc[sub[0], mn]) < 1e-12
            ok(same,
               f"A3 缺失 {mn} 的记录 z={d3.loc[sub3[0], mn]:.10f} 与 A1 填补值一致"
               f"（证明 A3 复用了 A1 冻结的均值向量）")

    print("\n=== V4 方向正确性（raw 与 z 的秩相关符号）===")
    raw_lookup = {}
    off = 14
    for n, k in LISTS:
        if k == 1:
            raw_lookup[n] = raw[:, off]
        off += k
    for j, n in enumerate(SCALARS):
        raw_lookup[n] = raw[:, j]
    for n in order22:
        if n in nonmono or n not in raw_lookup:
            continue
        rho = spearman(raw_lookup[n], Z[:, idx[n]])
        want_neg = n in neg
        ok((rho < -0.99) if want_neg else (rho > 0.99),
           f"{n:<50} rho={rho:+.4f}  期望{'负' if want_neg else '正'}相关")

    print("\n=== V5 非单调指标：梯形隶属度形状 ===")
    for n in nonmono:
        j = idx[n]
        t = P["trapezoid"][n]
        L, a, b, U = t["L"], t["a"], t["b"], t["U"]
        x = np.log1p(np.clip(raw_lookup[n], 0, None))
        zz = got[:, j]
        outside = (x < L) | (x > U)
        bad = int((zz[outside] > 1e-12).sum())
        ok(bad == 0, f"{n:<28} 区外(x<L 或 x>U) {int(outside.sum()):>5} 条全为 0 分（违例 {bad}）")
        inplat = (x >= a) & (x <= b)
        bad2 = int((np.abs(zz[inplat] - 1.0) > 1e-12).sum())
        ok(bad2 == 0, f"{n:<28} 平台区[a,b] {int(inplat.sum()):>5} 条全为 1 分（违例 {bad2}）")
        o = np.argsort(x)
        xs, zs = x[o], zz[o]
        left, right = xs <= a, xs >= b
        ok(bool((np.diff(zs[left]) >= -1e-12).all()), f"{n:<28} 左侧随原始值单调不减")
        ok(bool((np.diff(zs[right]) <= 1e-12).all()), f"{n:<28} 右侧随原始值单调不增")
    for n in nonmono:
        zfull = df1[n]
        ok((zfull == 1.0).mean() > 0.2,
           f"{n:<28} A1 全域平台区(=1.0)占比 {(zfull == 1.0).mean():.2%}")

    print("\n=== V6 压缩公式手算对照（压缩 -> MinMax 标定 全链路）===")
    with lzma.open(a1_path, "rt", encoding="utf-8") as f:
        r0 = json.loads(f.readline())
    row0 = df1.iloc[0]
    ok(row0["id"] == r0["id"], f"首条记录对齐：parquet id={row0['id']} == 文件 id={r0['id']}")

    def calib(v, n):
        lo, hi = P["minmax_bounds"][n]
        return (v - lo) / (hi - lo)

    z = np.array(r0["fluency_en"]); e = np.exp(z - z.max())
    p = calib(float((e / e.sum())[1]), "fluency_en")
    ok(abs(p - row0["fluency_en"]) < 1e-12,
       f"fluency_en 二分类softmax手算 {p:.12f} == parquet {row0['fluency_en']:.12f}")

    z = np.array(r0["ad_en"]); e = np.exp(z - z.max())
    p = calib(float((e / e.sum())[1]), "ad_en")
    ok(abs(p - row0["ad_en"]) < 1e-12,
       f"ad_en 无广告类softmax手算   {p:.12f} == parquet {row0['ad_en']:.12f}")

    z = np.array(r0["modernbert_readability"]); e = np.exp(z - z.max())
    p = calib(float((e / e.sum()) @ rank_r), "modernbert_readability")
    ok(abs(p - row0["modernbert_readability"]) < 1e-12,
       f"modernbert 六分类期望等级手算 {p:.12f} == parquet {row0['modernbert_readability']:.12f}")

    q = np.array(r0["qurater"])
    qavg = float(((q - q_lo) / np.where(q_hi > q_lo, q_hi - q_lo, 1.0)).mean())
    p = calib(qavg, "qurater")
    ok(abs(p - row0["qurater"]) < 1e-12,
       f"qurater 逐维MinMax后平均手算  {p:.12f} == parquet {row0['qurater']:.12f}")

    v = float(r0["dsir_books"]); p = calib(float(np.sign(v) * np.log1p(abs(v))), "dsir_books")
    ok(abs(p - row0["dsir_books"]) < 1e-12,
       f"dsir_books 符号保持对数手算    {p:.12f} == parquet {row0['dsir_books']:.12f}")

    lo, hi = P["minmax_bounds"]["rps_doc_frac_no_alph_words"]
    v = 1.0 - (float(r0["rps_doc_frac_no_alph_words"]) - lo) / (hi - lo)
    ok(abs(v - row0["rps_doc_frac_no_alph_words"]) < 1e-12,
       f"负向指标补转换手算            {v:.12f} == parquet {row0['rps_doc_frac_no_alph_words']:.12f}")

    print(f"\n{'='*70}\n通过 {len(checks)} 项，失败 {len(fails)} 项")
    for m in fails:
        print("  " + m)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
