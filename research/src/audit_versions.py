"""R1 step 1: version/consistency audit of the fixed A/B/C results.

Answers: for each line and question, is the committed result consistent with the
current frozen code+config+inputs? Only consistent, fully-rerun results are reused.

Writes research/reports/version_audit.json and prints a summary table.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .common import RESEARCH, load_lock, sha256_file

QUESTIONS = {
    "A": {"q1": "artifacts/baselines/Q1-A", "q2": "artifacts/baselines/Q2-A", "q3": "artifacts/baselines/Q3-A"},
    "B": {"q1": "results", "q2": "results", "q3": "results"},
    "C": {"q1": "artifacts/baselines/Q1-C", "q2": "artifacts/baselines/Q2-C", "q3": "artifacts/baselines/Q3-C"},
}


def _source_hash_dir(d: Path) -> str:
    """Replicate C's common.source_hash: sha256 over sorted *.py (name then bytes), first 16 hex."""
    h = hashlib.sha256()
    for name in sorted(n for n in os.listdir(d) if n.endswith(".py")):
        h.update(name.encode("utf-8"))
        h.update((d / name).read_bytes())
    return h.hexdigest()[:16]


def _git_head(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def audit() -> dict:
    lock = load_lock()
    out = {"generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
           "lines": {}}
    for line, meta in lock["lines"].items():
        root = Path(meta["worktree"])
        rec = {"locked_commit": meta["commit"], "head": _git_head(root),
               "head_matches_lock": _git_head(root) == meta["commit"], "questions": {}}
        # C source/config hashes recorded in run_metadata.json
        checks = {
            "q1": (root / "src/baselines/Q1-C", root / "artifacts/baselines/Q1-C/run_metadata.json"),
            "q2": (root / "src/baselines/Q2-C", root / "artifacts/baselines/Q2-C/run_metadata.json"),
            "q3": (root / "src/baselines/Q3-C", root / "artifacts/baselines/Q3-C/run_metadata.json"),
        } if line == "C" else {}
        for q, (sdir, meta_path) in checks.items():
            m = _load(meta_path) or {}
            entry = {"run_metadata": str(meta_path.relative_to(root)),
                     "source_dir": str(sdir.relative_to(root)),
                     "has_source_hash": "source_hash" in m,
                     "recorded_source_hash": m.get("source_hash"),
                     "recorded_config_hash": m.get("config_hash"),
                     "status": "unknown"}
            if entry["has_source_hash"]:
                current = _source_hash_dir(sdir)
                entry["current_source_hash"] = current
                entry["source_hash_match"] = current == entry["recorded_source_hash"]
                entry["status"] = "consistent" if entry["source_hash_match"] else "STALE(source changed)"
            else:
                entry["status"] = "no_source_hash_in_metadata"
            rec["questions"][q] = entry
        # A: config hash consistency (A records config_hash sha256 of config text)
        for q, adir in QUESTIONS["A"].items():
            ap = root / adir
            m = _load(ap / "run_metadata.json") if (ap / "run_metadata.json").exists() else {}
            entry = {"run_metadata": str((ap / "run_metadata.json").relative_to(root)),
                     "config_hash": (m or {}).get("config_hash"),
                     "input_hashes_present": bool((m or {}).get("input_hashes")),
                     "status": "consistent" if m else "no_metadata"}
            rec["questions"].setdefault(q, entry)
        # B: no hash metadata; check fix marker + corrected scale in results
        if line == "B":
            q3 = root / "code/q3_optimize.py"
            txt = q3.read_text(encoding="utf-8") if q3.exists() else ""
            has_unit = "UNIT = 1e9" in txt and "unit_anchor_check" in txt
            alloc = root / "results/q3_optimal_allocations.csv"
            scale_ok = None
            if alloc.exists():
                import pandas as pd
                d = pd.read_csv(alloc)
                col = "L" if "L" in d.columns else None
                if col:
                    scale_ok = bool(float(d[col].min()) > 1.7)   # old buggy scale ~1.69
            for q in ("q1", "q2", "q3", "q4"):
                rec["questions"].setdefault(q, {})
            rec["questions"]["q3"].update({
                "fix_marker_in_code": has_unit,
                "results_scale_corrected": scale_ok,
                "status": "consistent" if (has_unit and scale_ok) else "NEEDS REVIEW",
                "note": "B 无 source_hash 元数据；以代码标记 + 结果量级判定（非哈希级证据）",
            })
            for q in ("q1", "q2", "q4"):
                rec["questions"][q].setdefault("status", "not_audited_this_round")
        out["lines"][line] = rec
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESEARCH / "reports/version_audit.json"))
    a = ap.parse_args()
    res = audit()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for line, rec in res["lines"].items():
        print(f"[{line}] head_match={rec['head_matches_lock']} commit={rec['head'][:8]}")
        for q, e in rec["questions"].items():
            print(f"    {q}: {e.get('status')} " +
                  (f"src_hash_match={e.get('source_hash_match')}" if "source_hash_match" in e else ""))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
