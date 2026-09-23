# -*- coding: utf-8 -*-
"""Q1-C 公共工具：配置加载、路径解析、输入哈希、表格与元数据落盘、指标计算。

任务卡：建模方案/baselines/Q1-C.md
本模块不包含模型逻辑，所有可调参数一律来自 configs/baselines/Q1-C.yaml。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

# 22 个质量指标的规范顺序（与上游 order22 一致）
METRIC_ORDER = [
    "fineweb_edu", "fluency_en", "ad_en", "qurater",
    "modernbert_cleanliness", "modernbert_readability", "modernbert_reasoning",
    "modernbert_professionalism", "dsir_books", "dsir_math", "dsir_wiki",
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_doc_mean_word_length", "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
]

CONFIG_REL = os.path.join("configs", "baselines", "Q1-C.yaml")


# --------------------------------------------------------------------- 路径
def repo_root() -> str:
    """仓库根目录：本文件位于 <root>/src/baselines/Q1-C/common.py。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_config(path: str | None = None) -> dict:
    cfg_path = path or os.environ.get("Q1C_CONFIG") or os.path.join(repo_root(), CONFIG_REL)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(cfg_path)
    cfg["_root"] = repo_root()
    return cfg


def resolve(cfg: dict, rel: str) -> str:
    """相对仓库根目录解析；绝对路径原样返回。"""
    if os.path.isabs(rel):
        return rel
    return os.path.abspath(os.path.join(cfg["_root"], rel.replace("/", os.sep)))


def artifacts_dir(cfg: dict) -> str:
    d = resolve(cfg, cfg["outputs"]["artifacts_dir"])
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------- 哈希
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def hash_if_exists(path: str) -> str | None:
    return sha256_file(path) if path and os.path.isfile(path) else None


def config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------- 数据读取
def preprocessed_dir(cfg: dict) -> str:
    env = os.environ.get("Q1C_PREPROCESSED_DIR")
    if env:
        return env
    return resolve(cfg, cfg["upstream"]["preprocessed_dir"])


def load_quality_frame(cfg: dict, name: str) -> pd.DataFrame:
    """读取上游冻结的标准化质量矩阵 Z ∈ [0,1]^(N×22)。"""
    path = os.path.join(preprocessed_dir(cfg), f"{name}_processed.parquet")
    df = pd.read_parquet(path)
    missing = [m for m in METRIC_ORDER if m not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少指标列: {missing}")
    return df


def load_pairs(cfg: dict) -> dict:
    """读取配比/Loss 成对表，按 index 一对一连接并断言对齐（§6 第 1 条）。"""
    d = resolve(cfg, cfg["paths"]["regmix_dir"])
    out: dict = {}
    groups = [("train", cfg["splits"]["train"])]
    groups += [("final_test", s) for s in cfg["splits"]["final_test"]]
    groups += [("extrapolation", s) for s in cfg["splits"]["extrapolation_diagnostic"]]
    for kind, spec in groups:
        name = spec["name"] if "name" in spec else "train_1m"
        m = pd.read_csv(os.path.join(d, spec["mixture"]))
        l = pd.read_csv(os.path.join(d, spec["loss"]))
        if not np.array_equal(m["index"].values, l["index"].values):
            raise AssertionError(f"{spec['mixture']} 与 {spec['loss']} 的 index 不是一对一")
        X = m.drop(columns=["index"])
        Y = l.drop(columns=["index"])
        if list(X.columns) != cfg["mixture"]["input_cols"]:
            raise AssertionError(f"{spec['mixture']} 输入列序与配置不一致")
        if list(Y.columns) != cfg["mixture"]["target_cols"]:
            raise AssertionError(f"{spec['loss']} 目标列序与配置不一致")
        out[name] = {"kind": kind, "index": m["index"].values, "X": X, "Y": Y,
                     "mixture_file": spec["mixture"], "loss_file": spec["loss"]}
    return out


def load_domain_mapping(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg, cfg["paths"]["domain_mapping"]))


# --------------------------------------------------------------------- 落盘
def save_table(df: pd.DataFrame, cfg: dict, name: str, index: bool = False) -> str:
    path = os.path.join(artifacts_dir(cfg), name)
    if name.endswith(".parquet"):
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_json(obj, cfg: dict, name: str) -> str:
    path = os.path.join(artifacts_dir(cfg), name)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"不可序列化: {type(o)}")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_default)
    return path


def write_text(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------- 指标
def rmse(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def spearman(a, b) -> float:
    """秩相关。用平均秩处理并列，避免对 scipy 版本的依赖。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if den == 0.0:
        return float("nan")
    return float((ra * rb).sum() / den)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def topk_hit(y_true: np.ndarray, y_pred: np.ndarray, k: int, lower_is_better: bool = True) -> bool:
    """预测最优是否落入真实前 k（Loss 越低越好）。"""
    n = len(y_true)
    k = min(k, n)
    pred_best = int(np.argmin(y_pred)) if lower_is_better else int(np.argmax(y_pred))
    order = np.argsort(y_true if lower_is_better else -y_true, kind="mergesort")
    return pred_best in set(order[:k].tolist())


def regret(y_true: np.ndarray, y_pred: np.ndarray, lower_is_better: bool = True) -> float:
    """后悔值 = 预测最优配方的真实值 - 候选集真实最优值（§1.6）。"""
    pred_best = int(np.argmin(y_pred)) if lower_is_better else int(np.argmax(y_pred))
    best = float(np.min(y_true)) if lower_is_better else float(np.max(y_true))
    return float(y_true[pred_best] - best)


# --------------------------------------------------------------------- 环境
def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "sklearn", "scipy", "pyarrow"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info


def git_rev(cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Timer:
    """记录拟合时间（任务卡要求记录拟合时间与模型复杂度）。"""

    def __init__(self, label: str = ""):
        self.label = label
        self.t0 = time.perf_counter()
        self.elapsed: float | None = None

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.t0
        return self.elapsed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
