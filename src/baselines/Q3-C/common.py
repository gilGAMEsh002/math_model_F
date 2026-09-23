# -*- coding: utf-8 -*-
"""Q3-C 公共工具：配置加载、双布局路径解析、上游接口读取、指标与落盘。

任务卡：建模方案/baselines/Q3-C.md
协议：  建模方案/01_问题分析与Baseline实施计划.md §3、§5、§6

布局说明（用户要求本工作在 26mathmodel 下新建独立目录）：
    26mathmodel/q3_c/src/baselines/Q3-C/common.py     ← 本文件
    26mathmodel/math_model_F/real_attachments/...     ← 数据仓库（并列）
    26mathmodel/q2_c/artifacts/baselines/Q2-C/        ← Q2；C 产物（临时桥接，只读）

因此本模块把「项目根」（q3_c，存放 configs/artifacts/reports）与「上游根」
（math_model_F，存放 real_attachments 与第一问产物）分开解析。候选根按
`paths.upstream_roots` 顺序探测，取第一个存在者；q3_c 内容并入仓库后 `_root`
自身即生效，无需改代码。

Q2-C 产物当前只在并列的 q2_c 里（另在分支 baseline-q2-c 中），故额外登记
`paths.q2_bridge_root` 作为只读备用根。这是**临时桥接**，注册在 run_metadata 里，
Q2-C 并入主仓库后可删除此项，删除不改变任何结果（因为 _root 优先命中）。
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

CONFIG_REL = os.path.join("configs", "baselines", "Q3-C.yaml")


def ensure_utf8_console() -> None:
    """Windows 控制台默认 GBK，日志中的 ≡ / ⇒ / ≤ 等符号会抛 UnicodeEncodeError。

    这里把标准输出/错误切到 UTF-8 并对无法编码的字符退化替换；切换失败也绝不让
    日志把整个流程带崩（日志文件始终以 UTF-8 写入，不受此影响）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ensure_utf8_console()


# --------------------------------------------------------------------- 路径
def project_root() -> str:
    """项目根：<proj>/src/baselines/Q3-C/common.py → <proj>。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_config(path: str | None = None) -> dict:
    cfg_path = path or os.environ.get("Q3C_CONFIG") or os.path.join(project_root(), CONFIG_REL)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(cfg_path)
    cfg["_root"] = project_root()
    return cfg


def candidate_roots(cfg: dict) -> list[str]:
    """[项目根] + 上游根候选 + Q2 桥接根（按序去重）。"""
    roots = [cfg["_root"]]
    for rel in cfg["paths"]["upstream_roots"]:
        roots.append(os.path.abspath(os.path.join(cfg["_root"], rel.replace("/", os.sep))))
    bridge = cfg["paths"].get("q2_bridge_root")
    if bridge:
        roots.append(os.path.abspath(os.path.join(cfg["_root"], bridge.replace("/", os.sep))))
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def find_path(cfg: dict, rel: str, kind: str = "any") -> str:
    """在项目根与各上游根中按序探测 rel。找不到时抛错并列出探测过的路径。

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


def which_root(cfg: dict, rel: str) -> str:
    """返回 rel 实际命中的根目录，用于在元数据里如实登记来源（避免误以为读了另一份）。"""
    full = find_path(cfg, rel)
    best = ""
    for root in candidate_roots(cfg):
        if os.path.abspath(full).startswith(os.path.abspath(root)) and len(root) > len(best):
            best = root
    return best


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


def source_hash(here: str | None = None) -> str:
    """本目录**全部** .py 源码的聚合哈希（按文件名排序后逐个喂入）。

    为什么需要：配置哈希只能锁定旋钮，锁不住代码。改了数值逻辑而配置未动时，
    配置哈希不变，产物却已不同——读者无法分辨手上这批产物出自哪版代码。
    `run_metadata.json` 因而同时记录源码哈希，使"可复现"这句话可被核验。
    收整个目录而非固定清单，是为了让新增模块也自动进入哈希范围。
    """
    d = here or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(n for n in os.listdir(d) if n.endswith(".py")):
        h.update(name.encode("utf-8"))
        with open(os.path.join(d, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


# ------------------------------------------------------------- 上游接口读取
def q1_path(cfg: dict, name: str) -> str:
    return find_path(cfg, os.path.join(cfg["paths"]["q1_artifacts"], name), kind="file")


def q2_path(cfg: dict, name: str) -> str:
    return find_path(cfg, os.path.join(cfg["paths"]["q2_artifacts"], name), kind="file")


def load_q1_domain_mapping(cfg: dict) -> pd.DataFrame:
    """第一问导出的 域→质量 映射与冻结质量分（Q1→Q2→Q3 接口）。"""
    return pd.read_csv(q1_path(cfg, "quality_domain_mapping.csv"))


def load_q1_predictor(cfg: dict) -> dict:
    """第一问配比预测器接口描述符（列序、固定 v、参考 p0、训练支持范围）。"""
    with open(q1_path(cfg, cfg["upstream"]["q1_predictor_file"]), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_q2_predictor(cfg: dict) -> dict:
    """Q2→Q3 损失预测器的描述符（参数字典、上界、可辨识性声明、支持范围）。"""
    with open(q2_path(cfg, cfg["upstream"]["q2_predictor_file"]), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_c7(cfg: dict) -> pd.DataFrame:
    """C7 架构元数据：用于给上下文长度情景提供元数据依据。"""
    p = find_path(cfg, os.path.join(cfg["paths"]["c7_dir"], cfg["context"]["c7_file"]), kind="file")
    return pd.read_csv(p)


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


class TeeLog:
    """同时写 stdout 与日志文件；CLI 里也能看到进度。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = ""):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# --------------------------------------------------------------------- 指标
def rmse(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def bias(y, p) -> float:
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def rel_err(a, b) -> float:
    """相对误差 |a-b|/max(|b|,eps)。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# --------------------------------------------------------------------- 环境
def env_info() -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml"):
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
