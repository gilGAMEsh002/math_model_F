# -*- coding: utf-8 -*-
"""Q2-C 公共工具：配置加载、双布局路径解析、附件 B 读取、指标与落盘。

任务卡：建模方案/baselines/Q2-C.md
协议：  建模方案/01_问题分析与Baseline实施计划.md §2（第二问）

布局说明（用户要求本工作在 26mathmodel 下新建独立目录）：
    26mathmodel/q2_c/src/baselines/Q2-C/common.py   ← 本文件
    26mathmodel/math_model_F/real_attachments/...   ← 数据仓库（并列）

因此本模块把「项目根」（q2_c，存放 configs/artifacts/reports）与
「上游根」（math_model_F，存放 real_attachments 与第一问产物）分开解析。
`paths.upstream_roots` 按序探测，取第一个存在者；q2_c 内容并入仓库后
第二候选 `..` 生效，无需改代码。
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

CONFIG_REL = os.path.join("configs", "baselines", "Q2-C.yaml")


# --------------------------------------------------------------------- 路径
def project_root() -> str:
    """项目根：<proj>/src/baselines/Q2-C/common.py → <proj>。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_config(path: str | None = None) -> dict:
    cfg_path = path or os.environ.get("Q2C_CONFIG") or os.path.join(project_root(), CONFIG_REL)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(cfg_path)
    cfg["_root"] = project_root()
    return cfg


def candidate_roots(cfg: dict) -> list[str]:
    """[项目根] + 上游根候选（按序去重）。"""
    roots = [cfg["_root"]]
    for rel in cfg["paths"]["upstream_roots"]:
        roots.append(os.path.abspath(os.path.join(cfg["_root"], rel.replace("/", os.sep))))
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def find_path(cfg: dict, rel: str, kind: str = "any") -> str:
    """在项目根与上游根中按序探测 rel。找不到时抛错并列出探测过的路径。

    只做读取解析；写入一律走 project_root（见 artifacts_dir）。
    """
    if os.path.isabs(rel):
        return rel
    rel_n = rel.replace("/", os.sep)
    tried = []
    for root in candidate_roots(cfg):
        p = os.path.abspath(os.path.join(root, rel_n))
        tried.append(p)
        ok = os.path.isdir(p) if kind == "dir" else os.path.isfile(p) if kind == "file" else os.path.exists(p)
        if ok:
            return p
    raise FileNotFoundError(f"未找到 {rel}（kind={kind}），已探测:\n  " + "\n  ".join(tried))


def resolve(cfg: dict, rel: str) -> str:
    return find_path(cfg, rel)


def artifacts_dir(cfg: dict) -> str:
    d = os.path.abspath(os.path.join(cfg["_root"], cfg["outputs"]["artifacts_dir"].replace("/", os.sep)))
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


# ------------------------------------------------------------- 附件 B 读取
def scaling_dir(cfg: dict) -> str:
    env = os.environ.get("Q2C_SCALING_DIR")
    if env:
        return env
    return find_path(cfg, cfg["paths"]["scaling_dir"], kind="dir")


def read_b(cfg: dict, key: str) -> pd.DataFrame:
    """按配置中的逻辑名读取一个附件 B 表，并统一 N/D/L/Q 列名。"""
    fname = cfg["data"][key]
    df = pd.read_csv(os.path.join(scaling_dir(cfg), fname))
    col = cfg["columns"]
    ren = {}
    if col["N"] in df.columns:
        ren[col["N"]] = "N"
    if col["D"] in df.columns:
        ren[col["D"]] = "D"
    if col["L"] in df.columns:
        ren[col["L"]] = "L"
    if col["Q"] in df.columns:
        ren[col["Q"]] = "Q"
    if col["source"] in df.columns:
        ren[col["source"]] = "source"
    if col["family"] in df.columns:
        ren[col["family"]] = "family"
    df = df.rename(columns=ren)
    for c in ("N", "D", "L"):
        if c not in df.columns:
            raise ValueError(f"{fname} 缺少必需列 {c}（现有 {list(df.columns)}）")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Q" in df.columns:
        df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
    df["source_file"] = fname
    df["source_key"] = key
    return df


def read_b3_trajectories(cfg: dict) -> pd.DataFrame:
    """B3：插值轨迹。每文件一个模型规模，含 interpolated 标记（位于 scaling_dir 下）。"""
    d = os.path.join(scaling_dir(cfg), cfg["data"]["b3_trajectories_dir"])
    if not os.path.isdir(d):
        raise FileNotFoundError(f"轨迹目录不存在: {d}")
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"):
            continue
        t = pd.read_csv(os.path.join(d, fn))
        col = cfg["columns"]
        t = t.rename(columns={col["N"]: "N", col["D"]: "D", col["L"]: "L"})
        t["N"] = pd.to_numeric(t["N"], errors="coerce")
        t["D"] = pd.to_numeric(t["D"], errors="coerce")
        t["L"] = pd.to_numeric(t["L"], errors="coerce")
        t["traj_file"] = fn
        rows.append(t)
    if not rows:
        raise FileNotFoundError(f"{d} 下没有轨迹 csv")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------- 第一问产物（Q1→Q2）
def q1_path(cfg: dict, name: str) -> str:
    return find_path(cfg, os.path.join(cfg["paths"]["q1_artifacts"], name), kind="file")


def load_q1_domain_mapping(cfg: dict) -> pd.DataFrame:
    """第一问导出的 域→质量 映射与冻结质量分（Q1→Q2 接口）。"""
    return pd.read_csv(q1_path(cfg, "quality_domain_mapping.csv"))


def load_q1_predictor(cfg: dict) -> dict:
    """第一问配比预测器接口描述符（列序、固定 v、参考 p0、训练支持范围）。"""
    with open(q1_path(cfg, "mixture_predictor.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------- 落盘
def save_table(df: pd.DataFrame, cfg: dict, name: str, index: bool = False) -> str:
    path = os.path.join(artifacts_dir(cfg), name)
    if name.endswith(".parquet"):
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def _json_safe(o):
    """递归把 numpy 标量转成 Python 标量，把 NaN/Inf 转成 None。

    NaN/Infinity 不是合法 JSON；Python 的 json 默认会写成裸 `NaN`，别的解析器读不了。
    这里统一落成 null，再用 allow_nan=False 兜底，保证产物是严格合法 JSON。
    """
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


def save_json(obj, cfg: dict, name: str) -> str:
    path = os.path.join(artifacts_dir(cfg), name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def write_text(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------- 指标
def rmse(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def bias(y, p) -> float:
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def r2(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def smape(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    return float(np.mean(np.where(den > 0, 2 * np.abs(p - y) / den, 0.0)))


# --------------------------------------------------------------------- 环境
def env_info() -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "pyarrow"):
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
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float | None = None
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.t0
        return self.elapsed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
