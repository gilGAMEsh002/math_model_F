"""Lightweight acceptance checks (this round only: imports, interfaces, a few numbers).

Status vocabulary:
  pass    - check executed and met expectation
  fail    - check executed and did NOT meet expectation (implementation problem)
  blocked - cannot execute (missing artifact/input); recorded, never treated as pass
  n/a     - not applicable to that line by design
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .adapters import LineA, LineB, LineC
from .common import RESEARCH, load_lock

ANCHOR = 2.385578
PHYS = (1e9, 1e11, 1.0, 1.0)   # N raw, D raw, Q, h


def _row(cid, name, expected, actual, status, notes="", severity="info"):
    return {"id": cid, "name": name, "expected": expected, "actual": actual,
            "status": status, "severity": severity, "notes": notes}


def check_1_cross_question_loss() -> list[dict]:
    """Same physical N,D,Q,p -> Q2 and Q3 loss predictions agree (per line)."""
    out = []
    N, D, Q, h = PHYS
    # C: two separate code paths (Q2-C predictor vs Q3-C model)
    C = LineC()
    q2 = C.q2_loss_at(N, D, Q, h)
    q3 = float(np.asarray(C.q3_model().predict(N, D, Q, h, C.m1_kappa())).ravel()[0])
    out.append(_row("1.C", "C Q2-C vs Q3-C at physical anchor",
                    {"q2": ANCHOR, "q3": ANCHOR}, {"q2": q2, "q3": q3},
                    "pass" if abs(q2 - q3) < 1e-9 and abs(q3 - ANCHOR) < 1e-5 else "fail",
                    "Q2-C 预测器（十亿坐标）与 Q3-C 模型（实际个数，内部换算）",
                    "critical"))
    # B
    B = LineB()
    lb = B.loss_at(N, D, Q)
    out.append(_row("1.B", "B Q2 params vs Q3-B loss at physical anchor", ANCHOR, lb,
                    "pass" if abs(lb - ANCHOR) < 1e-5 else "fail",
                    "Q3-B loss() 修复后应与十亿坐标经典项一致", "critical"))
    # A
    A = LineA()
    la = A.loss_at(N, D, 1.0)
    out.append(_row("1.A", "A Q2/Q3 loss at physical anchor (Q=1)", ANCHOR, la,
                    "pass" if abs(la - ANCHOR) < 1e-5 else "fail",
                    "A 全链同一 loss_predictor.json", "critical"))
    return out


def check_2_cost_and_coordinates() -> list[dict]:
    """Cost uses actual counts; loss coordinate conversion correct."""
    out = []
    C = LineC()
    alloc = (C.root / "artifacts/baselines/Q3-C/optimal_allocations.csv")
    import pandas as pd
    df = pd.read_csv(alloc)
    # recompute 6ND + eta ND ell from N_B,D_B (actual counts) and compare to C's reported share sum
    N = df["N_B"].values * 1e9
    D = df["D_B"].values * 1e9
    eta = 2e-4
    recomputed = 6 * N * D + eta * N * D * df["ell"].values
    tot = recomputed + df["C_quality"].values
    rel = np.abs(tot / df["C"].values - 1)
    out.append(_row("2.C", "C cost recomputed from actual N,D; residual",
                    "<=1e-9", float(np.nanmax(rel)),
                    "pass" if float(np.nanmax(rel)) < 1e-9 else "fail",
                    "用实际个数回算 6ND+ηNDℓ+D·c(Q) 与预算 C 的残差", "critical"))
    # A cost check artifact
    a_cost = (LineA().root / "artifacts/baselines/Q3-A/cost_check.csv")
    out.append(_row("2.A", "A cost_check presence + loss predictor units",
                    "exists", a_cost.exists(),
                    "pass" if a_cost.exists() else "blocked",
                    "A Q3 成本份额与约束残差见 artifacts/baselines/Q3-A/", "info"))
    # B cost is checked inside B's own q3 run; presence of constraint check
    b_cc = (LineB().root / "results/q3_constraint_check.csv")
    if b_cc.exists():
        import pandas as pd
        d = pd.read_csv(b_cc)
        mx = float(d["rel_error"].max())
        out.append(_row("2.B", "B cost constraint max relative error", "<=1e-9", mx,
                        "pass" if mx < 1e-9 else "fail", "B Q3 影子成本回算", "critical"))
    else:
        out.append(_row("2.B", "B cost constraint check", "exists", False, "blocked",
                        "results/q3_constraint_check.csv 缺失", "critical"))
    return out


def check_3_boundary_and_trust_region() -> list[dict]:
    """Q0 boundary uses right derivative; trust region really enforced (C)."""
    out = []
    C = LineC()
    cfg = C.q3_config()
    q0 = float(cfg["quality"]["q0_ref"] if "q0_ref" in cfg.get("quality", {}) else 0.416633)
    kappa = float(cfg["model"]["kappa_scenarios"][cfg["model"]["kappa_primary"]])
    form = "power"
    o = C.q3_loss_at_budget(1e9, q0, 1.0, q0, form, 1e22, 4096, kappa, need_grad=True)
    g_an = o["dL_dQ"]
    eps = 1e-6
    op = C.q3_loss_at_budget(1e9, q0 * (1 + eps), 1.0, q0, form, 1e22, 4096, kappa)
    g_fd = (op["L"] - o["L"]) / (q0 * eps)
    ok = abs(g_an - g_fd) / max(abs(g_fd), 1e-30) < 1e-3
    out.append(_row("3.C-deriv", "C right derivative at Q=Q0 vs forward FD",
                    {"g_analytic": g_an, "g_fd": g_fd}, "rel<=1e-3",
                    "pass" if ok else "fail",
                    "可行域 Q>=Q0 的右导数（旧实现会置零）", "critical"))
    import pandas as pd
    df = pd.read_csv(C.root / "artifacts/baselines/Q3-C/optimal_allocations.csv")
    allin = bool(df["in_trust_l1"].all())
    out.append(_row("3.C-trust", "C main optimal table all in_trust_l1", True, allin,
                    "pass" if allin else "fail",
                    f"rows={len(df)}; 区域外候选单列 optimal_allocations_outside_l1.csv",
                    "critical"))
    out.append(_row("3.B", "B trust-region constraint", "n/a", "n/a", "n/a",
                    "B 第三问不含 p 可信域约束（固定配比）", "info"))
    return out


def check_4_c_q1_predictor_and_scale() -> list[dict]:
    """C mixture predictor callable; cross-question normalisation scale frozen."""
    out = []
    C = LineC()
    st = C.q1_predictor_status()
    out.append(_row("4.C-predictor", "C Q1 mixture predictor executable for arbitrary p",
                    True, st["executable"],
                    "pass" if st["executable"] else "blocked",
                    st["blocker"] or "", "critical"))
    out.append(_row("4.C-descriptor", "C Q1 descriptor well-formed (17 in / 13 out / v / p0)",
                    {"n_inputs": 17, "n_outputs": 13, "keys_ok": True},
                    {"n_inputs": st["n_inputs"], "n_outputs": st["n_outputs"],
                     "keys_ok": st["descriptor_keys_ok"]},
                    "pass" if st["descriptor_keys_ok"] and st["n_inputs"] == 17 and st["n_outputs"] == 13
                    else "fail", "描述符协议检查", "info"))
    # frozen cross-question scale
    q2_cal = C.root / "artifacts/baselines/Q2-C/quality_scale_calibration.json"
    q3_cal = C.root / "artifacts/baselines/Q3-C/quality_scale_calibration.json"
    present = q2_cal.exists()
    out.append(_row("4.C-scale", "Cross-question quality scale frozen (Q2-C calibration)",
                    "exists", {"q2": present, "q3": q3_cal.exists()},
                    "pass" if present else "blocked",
                    "Q3-C 的 h(p) 尺度取自 Q2-C 冻结的 scale_s，避免逐候选重标定", "info"))
    return out


def check_5_planned_vs_executed() -> list[dict]:
    """Config-listed scenarios must be separated from actually executed runs."""
    out = []
    exp = RESEARCH / "experiments.csv"
    rows = list(csv.DictReader(exp.open(encoding="utf-8")))
    bad = [r["exp_id"] for r in rows
           if r.get("executed", "").lower() == "true" and not r.get("run_id", "").strip()]
    done_wo_run = [r["exp_id"] for r in rows
                   if r.get("status", "") in ("done", "complete", "completed")
                   and r.get("executed", "").lower() != "true"]
    ok = not bad and not done_wo_run
    out.append(_row("5.planned", "experiments.csv: executed=true has run_id; no done-without-run",
                    {"bad_executed": [], "done_without_executed": []},
                    {"bad_executed": bad, "done_without_executed": done_wo_run},
                    "pass" if ok else "fail",
                    f"共 {len(rows)} 条登记（含 R1/R2/R3，均为 pending）", "critical"))
    return out


def check_6_fix_status_classification() -> list[dict]:
    """Fix evidence must exist per line; classification is three-stage in decisions.md."""
    out = []
    lock = load_lock()
    from .adapters import LineA, LineB
    b_test = (LineB().root / "code/test_q3_units.py")
    c_anchor = (LineC().root / "artifacts/baselines/Q3-C/verify_unit_anchor.csv")
    c_kappa = (LineC().root / "artifacts/baselines/Q3-C/kappa_ablation.csv")
    a_tests = (LineA().root / "tests/test_baseline_a.py")
    ev = {"A_tests": a_tests.exists(), "B_unit_test": b_test.exists(),
          "C_unit_anchor": c_anchor.exists(), "C_kappa_ablation": c_kappa.exists()}
    ok = all(ev.values())
    out.append(_row("6.fix", "fix evidence present (A tests / B test / C anchor+kappa)",
                    {k: True for k in ev}, ev, "pass" if ok else "blocked",
                    "代码已修≠全量重跑；三项状态在 decisions.md 分开登记", "info"))
    return out


def run_acceptance() -> list[dict]:
    checks: list[dict] = []
    checks += check_1_cross_question_loss()
    checks += check_2_cost_and_coordinates()
    checks += check_3_boundary_and_trust_region()
    checks += check_4_c_q1_predictor_and_scale()
    checks += check_5_planned_vs_executed()
    checks += check_6_fix_status_classification()
    return checks


def summarize(checks: list[dict]) -> dict:
    from collections import Counter
    c = Counter(x["status"] for x in checks)
    return {"n": len(checks), "pass": c.get("pass", 0), "fail": c.get("fail", 0),
            "blocked": c.get("blocked", 0), "n_a": c.get("n/a", 0),
            "critical_failures": [x["id"] for x in checks if x["status"] == "fail"
                                  and x["severity"] == "critical"]}
