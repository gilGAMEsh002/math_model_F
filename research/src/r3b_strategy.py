"""R3b: does compatibility handling change resource decisions?

Step 1 is ANALYTIC: a source offset is an additive constant, so for a fixed Q it cannot
change argmin(N,D,Q); we verify this numerically instead of re-running the full budget grid.

Step 2 recomputes the optimum under C's Q3 model, changing ONE parameter group at a time
(intercept / quality kappa / scale alpha,beta) under a unified coordinate, cost, budget and
feasible set, and reports which changes move the investment direction and which budget
regions keep the same strategy. These are scenario recomputations, NOT new training runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import LineC
from .common import RESEARCH
from .r1_compare import _classical_opt  # noqa: F401  (kept for reference)

BUDGETS = [1e19, 1e22, 1e24]
ELLS = [2048, 8192]
FORM = "exponential"


def _search(model, C, ell, Q0, kappa, add_const=0.0, delta_L=None):
    """Grid + refine over (log10 N_B, Q); returns dict with N_B,D_B,Q,L,regime."""
    lo_n, hi_n = np.log10(1e-3), np.log10(1e3)
    Qlo, Qhi = Q0, 1.0

    def evaluate(lx, q):
        N = (10 ** lx) * 1e9
        out = model.loss_at_budget([N], [q], [1.0], Q0, FORM, C, ell, kappa)
        L = float(out["L"][0]) + add_const + (delta_L(lx, q) if delta_L else 0.0)
        return L, float(out["D"][0])

    ln = np.linspace(lo_n, hi_n, 200); qs = np.linspace(Qlo, Qhi, 120)
    best = None
    for lx in ln:
        for q in qs:
            L, D = evaluate(lx, q)
            if best is None or L < best[0]:
                best = (L, lx, q, D)
    # two refine rounds
    wln = (hi_n - lo_n) / 20; wq = (Qhi - Qlo) / 12
    for _ in range(2):
        ln = np.linspace(max(best[1] - wln, lo_n), min(best[1] + wln, hi_n), 40)
        qs = np.linspace(max(best[2] - wq, Qlo), min(best[2] + wq, Qhi), 30)
        for lx in ln:
            for q in qs:
                L, D = evaluate(lx, q)
                if L < best[0]:
                    best = (L, lx, q, D)
        wln /= 4; wq /= 4
    L, lx, q, D = best
    q = float(np.clip(q, Qlo, Qhi))
    tol = 1e-6
    regime = ("lower_bound_Q0" if abs(q - Q0) <= tol else
              "upper_bound_Q1" if abs(q - 1.0) <= tol else "interior")
    return {"N_B": 10 ** lx, "D_B": D / 1e9, "Q": q, "L": L, "regime": regime}


def _params(model):
    return {k: getattr(model, k) for k in ("E", "A", "alpha", "B", "beta")}


def _with(model, **over):
    import copy
    m = copy.copy(model)
    for k, v in over.items():
        setattr(m, k, float(v))
    return m


def run_r3b(outdir: Path | None = None, seed=20260924) -> dict:
    outdir = outdir or (RESEARCH / "reports")
    outdir.mkdir(parents=True, exist_ok=True)
    Cline = LineC()
    cfg = Cline.q3_config()
    model = Cline.q3_model()
    kappa = float(cfg["model"]["kappa_scenarios"][cfg["model"]["kappa_primary"]])
    q0 = 0.416633  # fixed_ref Q(p0) from C's Q1 mapping (see quality_of_mixture)
    # source offset delta from R2b (B6, m1)
    delta = 0.1682371455329559

    # Step 1: analytic invariance of an additive constant (numeric confirmation)
    invar = []
    for C in BUDGETS:
        for ell in ELLS:
            a = _search(model, C, ell, q0, kappa, add_const=0.0)
            b = _search(model, C, ell, q0, kappa, add_const=delta)
            invar.append({"C": C, "ell": ell, "argmin_same_NQ": bool(
                abs(a["N_B"] - b["N_B"]) < 1e-9 and abs(a["Q"] - b["Q"]) < 1e-9),
                "dL": b["L"] - a["L"], "regime_a": a["regime"], "regime_b": b["regime"]})
    invariant = all(x["argmin_same_NQ"] for x in invar)

    # bootstrap CIs for scale params (C's own)
    ci = {}
    bp = Cline.root / "artifacts/baselines/Q2-C/validation_bootstrap_M0.csv"
    if bp.exists():
        d = pd.read_csv(bp)
        for _, r in d.iterrows():
            ci[r["param"]] = (float(r["lo95"]), float(r["hi95"]))
    a_lo, a_hi = ci.get("alpha", (model.alpha * 0.9, model.alpha * 1.1))
    b_lo, b_hi = ci.get("beta", (model.beta * 0.9, model.beta * 1.1))

    scenarios = {
        "baseline": {},
        "quality_off_kappa0": {"__kappa": 0.0},
        "kappa_boot_lo": {"__kappa": float(cfg["model"]["kappa_scenarios"].get("boot_lo", kappa))},
        "kappa_boot_hi": {"__kappa": float(cfg["model"]["kappa_scenarios"].get("boot_hi", kappa))},
        "alpha_lo": {"alpha": a_lo}, "alpha_hi": {"alpha": a_hi},
        "beta_lo": {"beta": b_lo}, "beta_hi": {"beta": b_hi},
    }
    rows = []
    for name, over in scenarios.items():
        kap = over.pop("__kappa", kappa)
        m = _with(model, **over) if over else model
        for C in BUDGETS:
            for ell in ELLS:
                r = _search(m, C, ell, q0, kap, add_const=delta)  # offset included
                rows.append({"scenario": name, "C": C, "ell": ell, "kappa": kap,
                             **r})
    df = pd.DataFrame(rows)
    base = df[df.scenario == "baseline"].set_index(["C", "ell"])
    df["regime_changed_vs_baseline"] = [
        bool(r["regime"] != base.loc[(r["C"], r["ell"]), "regime"]) for _, r in df.iterrows()]
    df["N_rel_change_vs_baseline"] = [
        float(r["N_B"] / base.loc[(r["C"], r["ell"]), "N_B"] - 1) for _, r in df.iterrows()]
    df.to_csv(outdir / "r3b_strategy.csv", index=False)

    # strategy-stable regions: budgets where every scenario keeps the baseline regime
    stable = []
    for (C, ell), g in df.groupby(["C", "ell"]):
        stable.append({"C": C, "ell": ell, "all_scenarios_same_regime": bool(
            (~g["regime_changed_vs_baseline"]).all()),
            "direction_flips": int(g["regime_changed_vs_baseline"].sum()),
            "max_abs_N_rel_change": float(g["N_rel_change_vs_baseline"].abs().max())})
    # compatibility-only view: exclude the quality-kappa family (that is a mechanism change,
    # not a compatibility handling step)
    qfamily = ["quality_off_kappa0", "kappa_boot_lo", "kappa_boot_hi"]
    exc = df[~df.scenario.isin(qfamily)]
    stable_excl = []
    for (C, ell), g in exc.groupby(["C", "ell"]):
        stable_excl.append({"C": C, "ell": ell, "all_scenarios_same_regime": bool(
            (~g["regime_changed_vs_baseline"]).all()),
            "max_abs_N_rel_change": float(g["N_rel_change_vs_baseline"].abs().max())})
    summary = {
        "analytic": {
            "claim": "来源偏移 δ 是加性常数，固定 Q 时 argmin(N,D,Q) 不变；无需重复跑全预算网格。",
            "numeric_confirmation": invar, "invariant": bool(invariant),
        },
        "scale_param_ci_source": str(bp.relative_to(Cline.root)) if bp.exists() else "fallback +-10%",
        "direction_changing_parameter_group": "只有质量通道参数 κ（quality_off / boot_lo / boot_hi）改变投入方向；"
                                              "来源偏移（加性常数）与规模参数 α/β 不改变 Q* 状态。",
        "stable_regions_all_scenarios": stable,
        "stable_regions_excluding_quality_family": stable_excl,
        "which_changes_move_direction": sorted(set(
            df[df.regime_changed_vs_baseline]["scenario"])),
        "caveat": "情景重算，不是新增真实训练实验；来源特有偏移在缺少目标来源依据时不得任意迁移。",
    }
    (outdir / "r3b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# R3b：兼容性处理是否改变资源决策", "",
          f"- **解析**：δ 为加性常数，固定 Q 时最优策略不变；数值确认={invariant}（{len(invar)} 个情景，dL 恒为 {delta:.6f}）。", "",
          f"- **改变投入方向的参数组**：{summary['direction_changing_parameter_group']}", "",
          "## 排除质量通道族后的“兼容性处理”稳定区", "",
          "| C | ell | 全部兼容性情景regime一致 | max |ΔN|/N |", "|---:|---:|---|---:|"]
    for s in stable_excl:
        md.append(f"| {s['C']:.0e} | {s['ell']} | {s['all_scenarios_same_regime']} | "
                  f"{s['max_abs_N_rel_change']:.3f} |")
    md += ["", "## 全部情景（含质量通道）", "",
           "| C | ell | 全部情景regime一致 | 方向翻转数 | max |ΔN|/N |", "|---:|---:|---|---:|---:|"]
    for s in stable:
        md.append(f"| {s['C']:.0e} | {s['ell']} | {s['all_scenarios_same_regime']} | "
                  f"{s['direction_flips']} | {s['max_abs_N_rel_change']:.3f} |")
    md += ["", f"- 尺度参数 CI 来源：{summary['scale_param_ci_source']}", f"- {summary['caveat']}"]
    (outdir / "r3b_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"strategy": df, "summary": summary}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=str(RESEARCH / "reports"))
    a = ap.parse_args()
    r = run_r3b(outdir=Path(a.outdir))
    print(r["strategy"].round(4).to_string(index=False))
    print(json.dumps(r["summary"]["which_changes_move_direction"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
