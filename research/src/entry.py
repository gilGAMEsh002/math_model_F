"""Research entry point. Simple commands only (no scheduler).

    python -m src.entry check                          # inputs + environment
    python -m src.entry experiment --name acceptance   # run lightweight checks
    python -m src.entry line --name C --question q3    # resolve/record command (dry)
    python -m src.entry line --name C --question q3 --execute
    python -m src.entry aggregate                      # summarize runs

Every command writes an immutable runs/<run_id>/ record.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .common import (RESEARCH, Run, env_fingerprint, line_commit, line_worktree,
                     load_lock, sha256_file)

LINE_COMMANDS = {
    "A": {"q1": ["python", "-m", "src.baselines.q1a.run"],
          "q2": ["python", "-m", "src.baselines.q2a.run"],
          "q3": ["python", "-m", "src.baselines.q3a.run"]},
    "B": {"q1": ["python", "code/q1_quality.py"],
          "q2": ["python", "code/q2_scaling.py"],
          "q3": ["python", "code/q3_optimize.py"],
          "q4": ["python", "code/q4_frontier.py"]},
    "C": {"q1": ["python", "run_q1.py", "--only", "q1c"],
          "q2": ["python", "src/baselines/Q2-C/run_all.py"],
          "q3": ["python", "src/baselines/Q3-C/run_all.py"]},
}


def cmd_check(args):
    lock = load_lock()
    with Run("check", command=" ".join(sys.argv)) as run:
        run.log("environment: " + json.dumps(env_fingerprint(), ensure_ascii=False))
        ok = True
        for line, meta in lock["lines"].items():
            wt = Path(meta["worktree"])
            exists = wt.is_dir()
            head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            match = head == meta["commit"]
            run.log(f"line {line}: worktree={wt} exists={exists} head={head[:8]} "
                    f"locked={meta['commit'][:8]} match={match}")
            ok = ok and exists and match
        # key input hashes
        cand_map = {
            "C_q1_predictor": "line-c/artifacts/baselines/Q1-C/mixture_predictor.json",
            "C_q2_predictor": "line-c/artifacts/baselines/Q2-C/loss_predictor.json",
            "C_q3_alloc": "line-c/artifacts/baselines/Q3-C/optimal_allocations.csv",
            "C_unit_anchor": "line-c/artifacts/baselines/Q3-C/verify_unit_anchor.csv",
            "C_kappa_ablation": "line-c/artifacts/baselines/Q3-C/kappa_ablation.csv",
            "A_mixture_predictor": "line-a/artifacts/baselines/Q1-A/mixture_predictor.json",
            "A_loss_predictor": "line-a/artifacts/baselines/Q2-A/loss_predictor.json",
            "B_q2_params": "line-b/results/q2_scaling_parameters.json",
            "B_q3_alloc": "line-b/results/q3_optimal_allocations.csv",
        }
        mism = []
        for name, expected in lock.get("key_input_sha256", {}).items():
            rel = cand_map.get(name)
            if rel is None:
                continue
            p = Path(lock["upstream_root"]) / rel
            run.add_inputs({name: p})
            actual = sha256_file(p) if p.exists() else None
            if actual != expected:
                mism.append(name)
                run.log(f"  HASH MISMATCH {name}: {actual} != {expected}")
        run.set_metrics({"lines_ok": ok, "hash_mismatches": mism})
        run.log(f"check: lines_ok={ok}; hash_mismatches={mism}")
        if not ok or mism:
            raise SystemExit(2)


def cmd_experiment(args):
    # Path A: run the built-in lightweight acceptance suite.
    name = args.name
    # Path C: reproducible three-line comparison (R1).
    if name == "r1":
        from .r1_compare import run_r1
        with Run("r1", command=" ".join(sys.argv)) as run:
            res = run_r1(run.dir)
            run.set_metrics(res["summary"])
            # also drop the summary into reports/ for the paper
            (RESEARCH / "reports" / "r1_summary.json").write_text(
                json.dumps(res["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
            for f in ("r1_q1_metrics.csv", "r1_q1_perdomain.csv", "r1_q1_set_identity.csv",
                      "r1_q3_degenerate.csv", "r1_q1_comparison.png"):
                src = run.dir / f
                if src.exists():
                    (RESEARCH / "reports" / f).write_bytes(src.read_bytes())
            run.log("R1 complete: " + json.dumps(res["summary"], ensure_ascii=False))
        return

    # Path D: R2a stability experiment.
    if name == "r2a":
        from .r2a_stability import run_r2a
        with Run("r2a", command=" ".join(sys.argv)) as run:
            res = run_r2a(outdir=run.dir)
            run.set_metrics(res["summary"])
            for f in ("r2a_stability.csv", "r2a_prereview.json", "r2a_summary.json", "r2a_regret.png"):
                s = run.dir / f
                if s.exists():
                    (RESEARCH / "reports" / f).write_bytes(s.read_bytes())
            run.log("R2a complete: " + json.dumps(res["summary"]["lambda_inside_A4A5"], ensure_ascii=False))
        return

    # Path E: R1-Q2 / R2b delegated modules — run under a Run record and copy outputs.
    if name in ("r1q2", "r2b", "r3a", "r3b"):
        import subprocess
        import sys as _sys
        mod = {"r1q2": "src.r1_q2_compare", "r2b": "src.r2b_source_quality",
               "r3a": "src.r3a_score_alignment", "r3b": "src.r3b_strategy"}[name]
        pats = {"r1q2": ("r1_q2_",), "r2b": ("r2b_",),
                "r3a": ("r3a_",), "r3b": ("r3b_",)}[name]
        with Run(name, command=" ".join(sys.argv)) as run:
            proc = subprocess.run([_sys.executable, "-m", mod], capture_output=True, text=True,
                                  cwd=str(RESEARCH))
            (run.dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (run.dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
            copied = []
            for f in sorted((RESEARCH / "reports").iterdir()):
                if f.name.startswith(pats) and f.is_file():
                    (RESEARCH / "reports" / f.name)  # keep in reports
                    (run.dir / f.name).write_bytes(f.read_bytes())
                    copied.append(f.name)
            run.set_metrics({"returncode": proc.returncode, "reports": copied})
            run.log(f"{mod} rc={proc.returncode}; copied {len(copied)} reports")
            if proc.returncode != 0:
                raise RuntimeError(f"{mod} failed rc={proc.returncode}: {proc.stderr[-400:]}")
        return

    if name == "acceptance":
        from .checks import run_acceptance, summarize
        with Run("acceptance", command=" ".join(sys.argv)) as run:
            checks = run_acceptance()
            summary = summarize(checks)
            run.set_metrics({"summary": summary, "checks": checks})
            lines = ["# 轻量验收结果（E0-A）", "",
                     f"- run_id: `{run.run_id}`", f"- 汇总: {summary}", "",
                     "| id | check | status | expected | actual | 说明 |",
                     "|---|---|---|---|---|---|"]
            for c in checks:
                lines.append(f"| {c['id']} | {c['name']} | **{c['status']}** | "
                             f"`{json.dumps(c['expected'], ensure_ascii=False)}` | "
                             f"`{json.dumps(c['actual'], ensure_ascii=False)}` | {c['notes']} |")
            lines += ["", "> 状态口径：pass 通过；fail 执行了但不符预期（实现问题）；"
                          "blocked 无法执行（缺输入/未持久化），**不计为通过**；n/a 该线不适用。"]
            (RESEARCH / "reports" / f"acceptance_{run.run_id}.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            run.log(f"acceptance summary: {summary}")
        return

    # Path B: a declared experiment config (R1/R2/R3). This round only records the
    # plan unless the config explicitly sets run: true; a plan is NOT a result.
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        with Run(f"plan_{cfg.get('id', 'exp')}", cfg=cfg, command=" ".join(sys.argv)) as run:
            run.set_metrics({"declared": True, "executed": bool(cfg.get("run", False)),
                             "config": cfg})
            run.log(f"config={args.config} status={cfg.get('status')} run={cfg.get('run', False)}")
            if cfg.get("run", False):
                raise SystemExit("config requests execution, but no runner is implemented for it yet")
            run.log("declared only: recorded as planned, NOT executed")
        return

    raise SystemExit(f"unknown experiment (name={args.name}, config={args.config})")


def cmd_line(args):
    lock = load_lock()
    line = args.name.upper()
    q = args.question.lower()
    cmd = LINE_COMMANDS[line][q]
    wt = line_worktree(lock, line)
    full = " ".join(cmd)
    with Run(f"line{line}_{q}", command=f"cd {wt} && {full}") as run:
        run.set_metrics({"line": line, "question": q, "mode": "execute" if args.execute else "dry"})
        run.log(f"line={line} question={q} upstream_commit={line_commit(lock, line)[:8]}")
        run.log(f"cwd={wt}")
        run.log(f"command={full}")
        if not args.execute:
            run.log("dry run: not executed (status recorded as planned, NOT a result)")
            run.set_metrics({"executed": False})
            return
        # Requirement: never write into the reference worktree. Use an independent
        # execution copy under research/work/ (git-ignored), with the canonical
        # read-only real_attachments symlinked in; capture outputs into the run dir.
        import os
        import shutil
        exec_root = RESEARCH / "work" / f"exec-{line}-{run.run_id}"
        if exec_root.exists():
            shutil.rmtree(exec_root)
        shutil.copytree(wt, exec_root,
                        ignore=shutil.ignore_patterns("real_attachments", ".git", "*.jsonl.xz",
                                                      "__pycache__", ".venv", "work"))
        ra = Path(lock["canonical_data"]["real_attachments"])
        link = exec_root / "real_attachments"
        try:
            if ra.exists() and not link.exists():
                os.symlink(ra, link)
                run.log(f"linked canonical data (read-only): {ra} -> {link}")
        except OSError as exc:
            run.log(f"[warn] could not symlink real_attachments: {exc}")
        run.log(f"execution copy (reference worktree untouched): {exec_root}")
        run.set_metrics({"exec_root": str(exec_root)})
        proc = subprocess.run(cmd, cwd=str(exec_root), capture_output=True, text=True)
        (run.dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (run.dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        # copy produced outputs back into the run record
        outdir = run.dir / "exec_out"
        for sub in ("artifacts", "results", "reports", "figures"):
            s = exec_root / sub
            if s.exists():
                shutil.copytree(s, outdir / sub, dirs_exist_ok=True)
        run.set_metrics({"executed": True, "returncode": proc.returncode})
        run.log(f"returncode={proc.returncode}; outputs copied to {outdir}")
        if proc.returncode != 0:
            raise RuntimeError(f"line command failed rc={proc.returncode}")


def cmd_aggregate(args):
    runs = sorted((RESEARCH / "runs").glob("*/meta.json"))
    rows = []
    for m in runs:
        d = json.loads(m.read_text(encoding="utf-8"))
        rows.append({"run_id": d.get("run_id"), "status": d.get("status"),
                     "command": d.get("command"), "elapsed_s": d.get("elapsed_s"),
                     "seed": d.get("seed"), "upstream": json.dumps(
                         {k: v["commit"][:8] for k, v in d.get("upstream", {}).items()},
                         ensure_ascii=False)})
    out = RESEARCH / "reports" / f"aggregate_{args.tag or 'latest'}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "status", "command", "elapsed_s", "seed", "upstream"])
        w.writeheader()
        w.writerows(rows)
    print(f"aggregate -> {out} ({len(rows)} runs)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="research")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check").set_defaults(func=cmd_check)

    pe = sub.add_parser("experiment")
    pe.add_argument("--name", default=None)
    pe.add_argument("--config", default=None)
    pe.set_defaults(func=lambda a: cmd_experiment(a))

    pl = sub.add_parser("line")
    pl.add_argument("--name", required=True, choices=["A", "B", "C"])
    pl.add_argument("--question", required=True, choices=["q1", "q2", "q3", "q4"])
    pl.add_argument("--execute", action="store_true")
    pl.set_defaults(func=cmd_line)

    pa = sub.add_parser("aggregate")
    pa.add_argument("--tag", default=None)
    pa.set_defaults(func=cmd_aggregate)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
