"""Research environment common utilities: upstream paths, hashing, run records.

Design goals (see ../README.md):
- every run gets a unique run_id and an immutable runs/<run_id>/ directory;
- metadata records upstream commits, research code revision (incl. dirty diff),
  config, seed, input hashes, command, log, elapsed, metrics and failures;
- nothing here mutates upstream baseline trees.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]          # research/
LOCK_PATH = RESEARCH / "upstream.lock.json"
RUNS = RESEARCH / "runs"


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def line_worktree(lock: dict, line: str) -> Path:
    key = line.upper().lstrip("LINE-")
    return Path(lock["lines"][key]["worktree"])


def line_commit(lock: dict, line: str) -> str:
    return lock["lines"][line.upper().lstrip("LINE-")]["commit"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def research_revision() -> dict:
    """Current research-branch commit plus a hash of any uncommitted diff."""
    def git(*args):
        return subprocess.run(["git", "-C", str(RESEARCH.parent), *args],
                              capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    diff = subprocess.run(["git", "-C", str(RESEARCH.parent), "diff"],
                          capture_output=True, text=True).stdout
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": head,
        "dirty": bool(dirty),
        "dirty_files": [l for l in dirty.splitlines() if l][:50],
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else "",
    }


def env_fingerprint() -> dict:
    fp = {"python": sys.version.split()[0], "platform": platform.platform(),
          "executable": sys.executable}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml", "matplotlib"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = None
    return fp


def new_run_id(prefix: str = "run") -> str:
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{prefix}_{os.getpid()}"


class Run:
    """Context manager that writes an immutable run record.

    Usage:
        with Run("acceptance", cfg=..., seed=0) as run:
            run.log("checking ...")
            run.set_metrics({...})
            run.add_inputs({"path": "/abs/file"})
    A failed run is recorded with status="failed" and the traceback, and is never
    marked successful.
    """

    def __init__(self, prefix: str, cfg=None, seed: int | None = None,
                 command: str | None = None, inputs: dict | None = None):
        self.lock = load_lock()
        self.run_id = new_run_id(prefix)
        self.dir = RUNS / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)   # never overwrite
        self.t0 = time.perf_counter()
        self.meta = {
            "run_id": self.run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "running",
            "command": command or " ".join(sys.argv),
            "cwd": os.getcwd(),
            "seed": seed,
            "config": cfg,
            "upstream": {k: {"branch": v["branch"], "commit": v["commit"]}
                         for k, v in self.lock["lines"].items()},
            "c_upstream_hashes": self.lock.get("key_input_sha256", {}),
            "research_revision": research_revision(),
            "environment": env_fingerprint(),
        }
        self.metrics: dict = {}
        self.failures: list = []
        self._log = (self.dir / "log.txt").open("w", encoding="utf-8")
        if inputs:
            self.add_inputs(inputs)

    def add_inputs(self, inputs: dict):
        """Record sha256 for each given path (kept small: hash names, not copies)."""
        self.meta.setdefault("inputs", {})
        for name, path in inputs.items():
            p = Path(path)
            self.meta["inputs"][name] = {
                "path": str(p), "exists": p.exists(),
                "sha256": sha256_file(p) if p.is_file() else None,
                "bytes": p.stat().st_size if p.is_file() else None,
            }
        return self

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def set_metrics(self, metrics: dict):
        self.metrics.update(metrics)

    def _write(self, status: str):
        self.meta["status"] = status
        self.meta["elapsed_s"] = round(time.perf_counter() - self.t0, 3)
        self.meta["metrics"] = self.metrics
        self.meta["failures"] = self.failures
        (self.dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.dir / "metrics.json").write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _snapshot_source(self):
        """Persist the actual research source/config of this run (recoverable), plus git patch."""
        import tarfile
        snap = self.dir / "source_snapshot.tar.gz"
        with tarfile.open(snap, "w:gz") as tf:
            for rel in ("src", "configs", "protocol.yaml", "requirements.txt"):
                p = RESEARCH / rel
                if p.exists():
                    tf.add(p, arcname=f"research/{rel}")
        # actual patch since the branch base, so the run's code can be restored
        patch = subprocess.run(["git", "-C", str(RESEARCH.parent), "diff"],
                               capture_output=True, text=True).stdout
        (self.dir / "research.patch").write_text(patch, encoding="utf-8")
        self.meta["source_snapshot"] = str(snap.relative_to(self.dir))
        self.meta["patch_bytes"] = len(patch)

    def __enter__(self):
        self._snapshot_source()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            import traceback
            self.failures.append({"type": exc_type.__name__, "message": str(exc),
                                  "traceback": "".join(traceback.format_exception(exc_type, exc, tb))})
            self._log.write(self.failures[-1]["traceback"])
            self.meta["error"] = str(exc)
            self._write("failed")
        else:
            self._write("success")
        self._log.close()
        return False
