"""Common utilities for Baseline A: hashing, IO, config, run metadata."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import lzma

REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj) -> str:
    """Deterministic hash of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return sha256_bytes(payload)


def iter_jsonl_xz(path: str | Path):
    """Stream records from an .jsonl.xz file without loading it into memory."""
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sanitize(obj):
    """Recursively convert numpy scalars and non-finite floats to strict-JSON values."""
    import math

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            return obj
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                 encoding="utf-8")
    return p


def load_config(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        import yaml

        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg["_config_path"] = str(p)
    cfg["_config_hash"] = sha256_bytes(text.encode("utf-8"))
    return cfg


def run_metadata(seed: int, config: dict | None = None, extra: dict | None = None) -> dict:
    meta = {
        "baseline_id": "Baseline-A",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": __import__("numpy").__version__,
    }
    try:
        import scipy

        meta["scipy"] = scipy.__version__
    except Exception:
        meta["scipy"] = None
    if config is not None:
        meta["config_path"] = config.get("_config_path")
        meta["config_hash"] = config.get("_config_hash")
    if extra:
        meta.update(extra)
    return meta
