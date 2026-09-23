# -*- coding: utf-8 -*-
"""
Step 1 数据预处理：SlimPajama-Meta-rater 质量信号 (A1/A2/A3) -> 标准化质量矩阵 Z ∈ [0,1]^(N×22)

流程（对应用户给定的 Step 1 七节）：
  1. 字段筛选      仅保留 22 个质量指标，丢弃 id/content/sub_path/_source_domain/_source_path（保留 id/domain 作索引）
  2. 缺失值处理    标量 -> A1 中位数；列表型 -> A1 均值向量
  3. 列表型压缩    单元素直取 / 二分类 softmax / 四维逐维 MinMax 后平均 / 六分类 softmax 期望
  4. 异常值处理    DSIR 三列做符号保持对数变换 sign(x)·log1p(|x|)
  5. 方向统一      MinMax + 负向补转换；三个非单调指标改用 log 空间梯形隶属度
  6. 标准化输出    Z ∈ [0,1]^(N×22)
  7. 落盘          A1/A2/A3_processed.parquet

【参考集冻结约定】
所有可调参数（中位数、均值向量、MinMax 边界、qurater 逐维边界、梯形隶属度分位数）**一律在 A1 上估计**，
再原样应用到 A2/A3。这样三套数据的 Z 处于同一尺度，赛题要求的「扩展集域级 Q 与抽样集对照」才有意义。
代价是 A2/A3 会有少量取值超出 A1 的 [min,max]，统一截断到 [0,1] 并在报告中统计截断率。

用法：
    python src/preprocess/quality_preprocess.py
"""

from __future__ import annotations

import glob
import json
import lzma
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径 ----------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SRC_DIR))          # D:\26mathmodel\q1_preprocessing
A_DIR = os.path.join(ROOT, "real_attachments", "A_data_value")
OUT_DIR = os.path.join(ROOT, "artifacts", "q1", "preprocessed")
CFG_DIR = os.path.join(ROOT, "configs")
REP_DIR = os.path.join(ROOT, "reports")

# ------------------------------------------------------------ 指标定义 ----------
# 14 个标量指标，原始字段即最终指标
SCALARS = [
    "dsir_books", "dsir_math", "dsir_wiki",
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_doc_mean_word_length", "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
]

# 8 个列表型指标及其固定长度
LISTS = [
    ("fineweb_edu", 1),
    ("fluency_en", 2),
    ("ad_en", 2),
    ("qurater", 4),
    ("modernbert_cleanliness", 6),
    ("modernbert_readability", 6),
    ("modernbert_reasoning", 6),
    ("modernbert_professionalism", 6),
]

# 六分类 logits 的等级 -> 质量映射（下标即档次，与 argmax/5 口径一致）
MB_RANK = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 负向指标：越大越差，Step 5 需做补转换（用户确认：仅 3 个重复/噪声类）
NEGATIVE = {
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
}

# 非单调指标：过长过短都不好，改用梯形隶属度（用户确认）
NON_MONOTONIC = [
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_mean_word_length",
]

# DSIR 原始分无界且随文档长度漂移，需符号保持对数变换
DSIR = ["dsir_books", "dsir_math", "dsir_wiki"]

# 22 个输出指标的顺序（也是 Z 的列序）
ORDER22 = ([n for n, _ in LISTS] + SCALARS)

# 原始读取矩阵的列布局：14 标量 + 33 个列表元素 = 47
RAW_COLS = SCALARS + [f"{n}[{i}]" for n, k in LISTS for i in range(k)]
RAW_W = len(RAW_COLS)
assert RAW_W == 47, RAW_W
assert len(ORDER22) == 22, len(ORDER22)

# 梯形隶属度分位数：平台区 [P25, P75]，两侧线性衰减至 [P05, P95] 处为 0
TRAP_Q = (0.05, 0.25, 0.75, 0.95)


# ================================================================ 工具函数 ======
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_bad(v) -> bool:
    """缺失判定：None、非数值、bool、NaN/±inf 一律视为缺失。

    源数据中的缺失以 JSON 的 NaN 字面量出现（不是 None），所以必须显式做 isfinite 检查。
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return True
    return not np.isfinite(v)


def iter_jsonl_xz(path: str):
    """流式逐行读取 xz 压缩的 JSONL，避免整文件解压进内存。"""
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def softmax(z: np.ndarray) -> np.ndarray:
    """按最后一维做数值稳定的 softmax。"""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """线性映射到 [0,1]；lo==hi 时返回全 0.5（该指标在此参考集无区分度）。"""
    if hi <= lo:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def trapezoid(x: np.ndarray, params: dict) -> np.ndarray:
    """梯形隶属度：平台区 [a,b] 得 1，[L,a) 与 (b,U] 线性衰减，区外得 0。"""
    L, a, b, U = params["L"], params["a"], params["b"], params["U"]
    y = np.zeros_like(x, dtype=float)
    lo = (x > L) & (x < a)
    y[lo] = (x[lo] - L) / (a - L)
    y[(x >= a) & (x <= b)] = 1.0
    hi = (x > b) & (x < U)
    y[hi] = (U - x[hi]) / (U - b)
    return y


# ============================================================== 读取原始数据 ====
def load_raw(path: str, domain_getter) -> tuple:
    """读取一个 jsonl.xz，返回 (RAW: (N,47) float, ids, domains, 缺失计数)。"""
    rows, ids, doms = [], [], []
    n_missing_cell = 0          # 缺失/非数值单元格总数
    n_rec_missing = 0           # 至少含一个缺失的记录数
    n = 0
    off_list = len(SCALARS)
    for r in iter_jsonl_xz(path):
        row = np.full(RAW_W, np.nan)
        bad = 0
        for j, name in enumerate(SCALARS):
            v = r.get(name)
            if _is_bad(v):
                bad += 1
            else:
                row[j] = float(v)
        off = off_list
        for name, k in LISTS:
            v = r.get(name)
            if isinstance(v, list) and len(v) == k:
                for i, x in enumerate(v):
                    if _is_bad(x):
                        bad += 1
                    else:
                        row[off + i] = float(x)
            else:
                bad += k
            off += k
        rows.append(row)
        ids.append(r.get("id", ""))
        doms.append(domain_getter(r))
        n_missing_cell += bad
        n_rec_missing += 1 if bad else 0
        n += 1
        if n % 50000 == 0:
            log(f"    已读取 {n:,} 条")
    raw = np.vstack(rows) if rows else np.empty((0, RAW_W))
    info = dict(n=n, missing_cells=n_missing_cell, records_with_missing=n_rec_missing)
    log(f"    完成 {n:,} 条，缺失单元格 {n_missing_cell}，含缺失记录 {n_rec_missing}")
    return raw, ids, doms, info


# ==================================================== 阶段 2/3/4：原始 -> 22 驱动值 ===
def fit_raw22_params(raw1: np.ndarray) -> dict:
    """在 A1 上估计缺失填充统计量与 qurater 逐维 MinMax 边界。"""
    med = np.nanmedian(raw1, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    q_off = len(SCALARS) + 1 + 2 + 2          # qurater 在 RAW_COLS 中的起始列
    q_cols = list(range(q_off, q_off + 4))
    q = raw1[:, q_cols]
    q_lo = np.nanmin(q, axis=0)
    q_hi = np.nanmax(q, axis=0)

    return dict(
        scalar_median=med.tolist(),
        qurater_dim_min=q_lo.tolist(),
        qurater_dim_max=q_hi.tolist(),
    )


def build_raw22(raw: np.ndarray, p: dict) -> tuple:
    """缺失填充 -> 列表压缩 -> DSIR 变换，得到 22 列的驱动值。返回 (M, dsir_clip)。"""
    x = raw.copy()

    # --- Step 2 缺失值处理：标量取 A1 中位数，列表型取 A1 均值向量
    med = np.array(p["scalar_median"])
    nan_cols = np.isnan(x)
    if nan_cols.any():
        x[nan_cols] = np.take(med, np.where(nan_cols)[1])

    M = np.empty((x.shape[0], 22), dtype=float)

    # --- Step 3.1 单元素列表：fineweb_edu 直接取
    off = len(SCALARS)
    M[:, ORDER22.index("fineweb_edu")] = x[:, off]
    off += 1

    # --- Step 3.2 二分类 logits：softmax 后取高质量类概率（index 1 为高质类）
    for name in ("fluency_en", "ad_en"):
        pr = softmax(x[:, off:off + 2])[:, 1]
        M[:, ORDER22.index(name)] = pr
        off += 2

    # --- Step 3.3 四维质量：逐子维度 MinMax（边界取自 A1）后取平均
    q_lo = np.array(p["qurater_dim_min"])
    q_hi = np.array(p["qurater_dim_max"])
    qn = np.empty((x.shape[0], 4))
    for j in range(4):
        qn[:, j] = minmax(x[:, off + j], q_lo[j], q_hi[j])
    M[:, ORDER22.index("qurater")] = qn.mean(axis=1)
    off += 4

    # --- Step 3.4 六分类 logits：softmax 概率对等级 r 求期望
    for name in ("modernbert_cleanliness", "modernbert_readability",
                 "modernbert_reasoning", "modernbert_professionalism"):
        pr = softmax(x[:, off:off + 6])
        M[:, ORDER22.index(name)] = pr @ MB_RANK
        off += 6

    # --- 标量直接搬运
    for j, name in enumerate(SCALARS):
        M[:, ORDER22.index(name)] = x[:, j]

    # --- Step 4 异常值处理：DSIR 符号保持对数变换
    dsir_idx = [ORDER22.index(n) for n in DSIR]
    v = M[:, dsir_idx]
    M[:, dsir_idx] = np.sign(v) * np.log1p(np.abs(v))

    return M


# ==================================================== 阶段 5：方向统一 + 归一化 ===
def fit_z_params(M1: np.ndarray) -> dict:
    """在 A1 上估计 MinMax 边界与梯形隶属度分位数。"""
    bounds = {}
    for name in ORDER22:
        if name in NON_MONOTONIC:
            continue
        j = ORDER22.index(name)
        col = M1[:, j]
        lo, hi = float(np.min(col)), float(np.max(col))
        # 落盘时保留 8 位有效数字，保证复现
        bounds[name] = [round(lo, 8), round(hi, 8)]

    trap = {}
    for name in NON_MONOTONIC:
        j = ORDER22.index(name)
        lg = np.log1p(np.clip(M1[:, j], 0, None))
        qs = np.quantile(lg, TRAP_Q)
        if qs[1] <= qs[0]:
            qs[1] = qs[0] + 1e-9
        if qs[3] <= qs[2]:
            qs[3] = qs[2] + 1e-9
        trap[name] = {k: round(float(v), 8)
                      for k, v in zip(("L", "a", "b", "U"), qs)}
    return dict(minmax_bounds=bounds, trapezoid=trap)


def apply_z(M: np.ndarray, p: dict) -> tuple:
    """方向统一 -> Z ∈ [0,1]^(N,22)。返回 (Z, 截断统计)。"""
    Z = np.empty_like(M)
    clip = {}
    for name in ORDER22:
        j = ORDER22.index(name)
        if name in NON_MONOTONIC:
            lg = np.log1p(np.clip(M[:, j], 0, None))
            Z[:, j] = trapezoid(lg, p["trapezoid"][name])
            continue
        lo, hi = p["minmax_bounds"][name]
        raw = M[:, j]
        if name in NEGATIVE:
            v = minmax(raw, lo, hi)
            v = 1.0 - v                      # 负向补转换（Step 5）
        else:
            v = minmax(raw, lo, hi)
        out_of_range = int(np.sum((raw < lo) | (raw > hi)))
        clip[name] = out_of_range
        Z[:, j] = np.clip(v, 0.0, 1.0)
    return Z, clip


# ================================================================== 落盘 =======
def to_frame(Z, ids, doms, tag):
    df = pd.DataFrame(Z, columns=ORDER22)
    df.insert(0, "domain", doms)
    df.insert(0, "id", ids)
    df["source_set"] = tag
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["min", "p01", "p25", "p50", "p75", "p99", "max", "mean", "std"]
    d = df[ORDER22]
    q = d.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T
    out = pd.DataFrame(index=ORDER22)
    out["min"] = d.min()
    out["p01"] = q[0.01]
    out["p25"] = q[0.25]
    out["p50"] = q[0.50]
    out["p75"] = q[0.75]
    out["p99"] = q[0.99]
    out["max"] = d.max()
    out["mean"] = d.mean()
    out["std"] = d.std()
    out["at_zero"] = (d == 0).mean()
    out["at_one"] = (d == 1).mean()
    return out[cols + ["at_zero", "at_one"]]


# =================================================================== 主流程 ====
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)

    a1_path = os.path.join(A_DIR, "slimpajama_quality_signal_sample.jsonl.xz")
    a2_path = glob.glob(os.path.join(A_DIR, "slimpajama_quality_extended", "arxiv_*.jsonl.xz"))
    a3_path = glob.glob(os.path.join(A_DIR, "slimpajama_quality_extended", "github_*.jsonl.xz"))
    if not (os.path.exists(a1_path) and a2_path and a3_path):
        log("!! 输入文件缺失，请先完成 git lfs pull")
        return 1
    files = [("A1", a1_path, lambda r: r.get("_source_domain", "unknown")),
             ("A2", a2_path[0], lambda r: "arxiv"),
             ("A3", a3_path[0], lambda r: "github")]

    # ---- 读 A1（参考集）
    log("读取 A1（参考集）...")
    raw1, ids1, dom1, info1 = load_raw(files[0][1], files[0][2])
    log("  估计缺失填充统计量与 qurater 逐维边界 ...")
    p_raw = fit_raw22_params(raw1)

    log("  构造 A1 的 22 列驱动值 ...")
    M1 = build_raw22(raw1, p_raw)
    log("  估计 MinMax 边界与梯形隶属度分位数 ...")
    p_z = fit_z_params(M1)
    params = dict(
        spec=dict(
            order22=ORDER22,
            negative_metrics=sorted(NEGATIVE),
            non_monotonic_metrics=NON_MONOTONIC,
            dsir_metrics=DSIR,
            modernbert_rank=list(MB_RANK),
            trapezoid_quantiles=list(TRAP_Q),
            reference_set="A1",
        ),
        **p_raw, **p_z,
    )
    del raw1          # 原始 47 列矩阵用不到了，释放；ids/domain 保留

    # ---- 逐集合处理
    frames, clips, infos, sums = {}, {}, {}, {}
    for tag, path, getter in files:
        if tag == "A1":
            log(f"[{tag}] 复用已读取的 A1 ...")
            Z, clip = apply_z(M1, params)
            ids, doms, info = ids1, dom1, info1
        else:
            log(f"[{tag}] 读取 {os.path.basename(path)} ...")
            raw, ids, doms, info = load_raw(path, getter)
            M = build_raw22(raw, params)
            Z, clip = apply_z(M, params)
            del raw, M
        df = to_frame(Z, ids, doms, tag)
        frames[tag] = df
        clips[tag] = clip
        infos[tag] = info
        sums[tag] = summarize(df)
        out = os.path.join(OUT_DIR, f"{tag}_processed.parquet")
        df.to_parquet(out, index=False)
        log(f"[{tag}] 写出 {out}  shape={df.shape}")

    # ---- 冻结参数
    cfg_path = os.path.join(CFG_DIR, "quality_preprocess_params.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    log(f"写出参数 {cfg_path}")

    # ---- 报告
    write_report(params, infos, clips, sums, frames)
    log("完成")
    return 0


def write_report(params, infos, clips, sums, frames) -> None:
    lines = []
    A = lines.append
    A("# Step 1 数据预处理报告\n")
    A("对象：SlimPajama-Meta-rater 质量信号 A1 / A2 / A3 → 标准化质量矩阵 Z ∈ [0,1]^(N×22)\n")
    A("## 1. 样本与缺失\n")
    A("| 集合 | 记录数 | 域 | 缺失单元格 | 含缺失记录 |")
    A("|---|---:|---|---:|---:|")
    doms = {t: {k: int(v) for k, v in frames[t]["domain"].value_counts().items()} for t in frames}
    for t in ("A1", "A2", "A3"):
        i = infos[t]
        A(f"| {t} | {i['n']:,} | {doms[t]} | {i['missing_cells']} | {i['records_with_missing']} |")
    A("")
    A("> 缺失指字段为 None、非数值或列表长度不符；填充值一律取自 A1 参考集的中位数 / 均值向量。\n")

    A("## 2. 冻结参数（在 A1 上估计，原样应用到 A2/A3）\n")
    A("### 2.1 MinMax 边界与方向\n")
    A("| 指标 | 下界(A1) | 上界(A1) | 方向 |")
    A("|---|---:|---:|---|")
    for n in ORDER22:
        if n in NON_MONOTONIC:
            continue
        lo, hi = params["minmax_bounds"][n]
        d = "负向(补转换)" if n in NEGATIVE else "正向"
        A(f"| `{n}` | {lo:.6g} | {hi:.6g} | {d} |")
    A("")
    A("### 2.2 非单调指标的梯形隶属度（log1p 空间）\n")
    A("| 指标 | L(P05) | a(P25) | b(P75) | U(P95) |")
    A("|---|---:|---:|---:|---:|")
    for n in NON_MONOTONIC:
        t = params["trapezoid"][n]
        A(f"| `{n}` | {t['L']:.6g} | {t['a']:.6g} | {t['b']:.6g} | {t['U']:.6g} |")
    A("")
    A("> 低于 L 或高于 U 记 0 分，[a,b] 区间记满分，两侧线性过渡。语义：过短/过长的文档都不适合作训练语料。\n")

    A("## 3. 超出 A1 参考范围的截断（仅 A2/A3 可能非零）\n")
    A("| 指标 | " + " | ".join(f"{t} 截断数" for t in ("A1", "A2", "A3")) + " |")
    A("|---|---:|---:|---:|")
    for n in ORDER22:
        if n in NON_MONOTONIC:
            continue
        A(f"| `{n}` | " + " | ".join(f"{clips[t][n]:,}" for t in ("A1", "A2", "A3")) + " |")
    A("")

    for t in ("A1", "A2", "A3"):
        A(f"## 4.{('A1','A2','A3').index(t)+1} {t} 输出分布\n")
        s = sums[t]
        A("| 指标 | min | p01 | p25 | p50 | p75 | p99 | max | mean | std | =0 占比 | =1 占比 |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n, r in s.iterrows():
            A(f"| `{n}` | {r['min']:.4f} | {r['p01']:.4f} | {r['p25']:.4f} | {r['p50']:.4f} | "
              f"{r['p75']:.4f} | {r['p99']:.4f} | {r['max']:.4f} | {r['mean']:.4f} | "
              f"{r['std']:.4f} | {r['at_zero']:.2%} | {r['at_one']:.2%} |")
        A("")

    A("## 5. 复现\n")
    A("```bash")
    A("python src/preprocess/quality_preprocess.py")
    A("```")
    A("")
    A("输出：`artifacts/q1/preprocessed/{A1,A2,A3}_processed.parquet`，"
      "参数：`configs/quality_preprocess_params.json`。\n")

    path = os.path.join(REP_DIR, "q1_preprocess_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"写出报告 {path}")


if __name__ == "__main__":
    sys.exit(main())
