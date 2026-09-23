"""Q3-A: one-dimensional resource search at fixed quality and mixture.

Fixes Q = Q0 and p = p0, then for each compute budget C and context length ell solves

    min_{N,D}  L0(N,D) + gamma (1 - Q0)
    s.t.       D * [(6 + eta*ell) * N + max(g(Q0)-g(Q0), 0)] <= C,   eta = 2e-4,

so the budget binds and D = C / ((6+eta*ell) N). The optimum is found by a dense
log-N search and cross-checked against the closed-form N* when the additive power
law applies. N,D are in billions; budgets are converted with C_B = C/1e18.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import REPO, load_config, read_json, run_metadata, sha256_file, write_json
from src.common.scaling import predict_l0

ETA_DEFAULT = 2.0e-4


def g_cost(form: dict, q):
    t = form["type"]
    gam, lam = float(form["gamma"]), float(form["lambda"])
    q = float(q)
    if t == "exponential":
        return gam * np.exp(lam * q)
    if t == "power":
        return gam * q ** lam
    if t == "log":
        return gam * np.log(1 + lam * q)
    raise ValueError(t)


def loss_fixed(N_B, D_B, theta, gamma, Q0):
    return float(predict_l0(theta, N_B, D_B) + gamma * (1.0 - Q0))


def solve_one(C_flops, ell, theta, gamma, Q0, cfg):
    C_flops = float(C_flops)
    a = 6.0 + float(cfg["eta"]) * float(ell)
    C_B = float(C_flops) / 1e18
    n_grid = int(cfg["n_grid"])
    n_lo, n_hi = float(cfg["n_search_range_B"][0]), float(cfg["n_search_range_B"][1])
    Ns = np.logspace(np.log10(n_lo), np.log10(n_hi), n_grid)
    Ds = C_B / (a * Ns)
    Ls = predict_l0(theta, Ns, Ds) + gamma * (1.0 - Q0)

    i = int(np.argmin(Ls))
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda lx: loss_fixed(10 ** lx, C_B / (a * 10 ** lx), theta, gamma, Q0),
                          bounds=(np.log10(n_lo), np.log10(n_hi)), method="bounded",
                          options={"xatol": 1e-10})
    N_num = float(10 ** res.x)
    D_num = C_B / (a * N_num)
    L_num = float(res.fun)

    # analytic reference (additive power law, unconstrained)
    E, logA, alpha, logB, beta = theta
    A, B = np.exp(logA), np.exp(logB)
    N_ana = ((alpha * A) / (beta * B) * (C_B / a) ** beta) ** (1.0 / (alpha + beta))
    D_ana = C_B / (a * N_ana)
    L_ana = loss_fixed(N_ana, D_ana, theta, gamma, Q0)

    # support-restricted optimum
    sup = cfg["data_support"]
    mask = (Ns >= sup["N_B"][0]) & (Ns <= sup["N_B"][1]) & (Ds >= sup["D_B"][0]) & (Ds <= sup["D_B"][1])
    if mask.any():
        k = int(np.argmin(Ls[mask]))
        j = int(np.where(mask)[0][k])
        N_sup, D_sup, L_sup = float(Ns[j]), float(Ds[j]), float(Ls[j])
        sup_boundary = bool(k in (0, int(mask.sum()) - 1))
        sup_feasible = True
    else:
        N_sup = D_sup = L_sup = float("nan")
        sup_boundary = False
        sup_feasible = False

    def flag(N, D):
        return {"N_in_support": bool(sup["N_B"][0] <= N <= sup["N_B"][1]),
                "D_in_support": bool(sup["D_B"][0] <= D <= sup["D_B"][1])}

    out = {
        "C_flops": float(C_flops), "ell": int(ell), "a": a,
        "N_num_B": float(N_num), "D_num_B": float(D_num), "L_num": L_num,
        "N_analytic_B": float(N_ana), "D_analytic_B": float(D_ana), "L_analytic": L_ana,
        "rel_diff_L": abs(L_num - L_ana) / (abs(L_ana) + 1e-30),
        "rel_diff_N": abs(N_num - N_ana) / (abs(N_ana) + 1e-30),
        "N_support_B": N_sup, "D_support_B": D_sup, "L_support": L_sup,
        "support_optimum_at_boundary": sup_boundary,
        "support_optimum_feasible": sup_feasible,
        "N_near_search_boundary": bool(i in (0, n_grid - 1)),
    }
    n_actual, d_actual = N_num * 1e9, D_num * 1e9
    total = a * n_actual * d_actual
    out.update({
        "train_cost_flops": 6.0 * n_actual * d_actual,
        "attention_cost_flops": cfg["eta"] * ell * n_actual * d_actual,
        "quality_cost_flops": 0.0,
        "total_cost_flops": total,
        "cost_constraint_residual": (total - C_flops) / C_flops,
        "train_share": 6.0 / a, "attention_share": cfg["eta"] * ell / a,
        "quality_share": 0.0, "unused_share": float(max(0.0, (C_flops - total) / C_flops)),
        "extrapolation": not (flag(N_num, D_num)["N_in_support"] and flag(N_num, D_num)["D_in_support"]),
        **flag(N_num, D_num),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/baselines/Q3-A.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    outdir = REPO / cfg["paths"]["out"]
    outdir.mkdir(parents=True, exist_ok=True)
    q2dir = REPO / cfg["paths"]["q2_out"]

    lp = read_json(q2dir / "loss_predictor.json")
    theta = np.array([lp["E"], np.log(lp["A"]), lp["alpha"], np.log(lp["B"]), lp["beta"]])
    gamma = lp["gamma"]
    p0 = read_json(REPO / cfg["paths"]["q1_out"] / "p0.json")
    Q0 = p0["Q0"]
    eta = cfg["optimization"]["eta"]
    assert abs(eta - lp["cost_model"]["eta"]) < 1e-15, "eta mismatch between Q2 and Q3 configs"

    rows = []
    for C in cfg["optimization"]["budgets_flops"]:
        for ell in cfg["optimization"]["context_lengths"]:
            r = solve_one(C, ell, theta, gamma, Q0, {**cfg["optimization"], "eta": eta})
            r["budget"] = C
            rows.append(r)
    scen = pd.DataFrame(rows)
    scen.to_csv(outdir / "optimal_allocations.csv", index=False)
    scen.to_csv(outdir / "optimization_scenarios.csv", index=False)

    # budget paths (continuous in log C)
    grid = np.logspace(np.log10(cfg["optimization"]["budget_grid_range"][0]),
                       np.log10(cfg["optimization"]["budget_grid_range"][1]),
                       int(cfg["optimization"]["budget_grid_n"]))
    paths = []
    for ell in cfg["optimization"]["context_lengths"]:
        for C in grid:
            r = solve_one(C, ell, theta, gamma, Q0, {**cfg["optimization"], "eta": eta})
            r["budget"] = C
            paths.append(r)
    pd.DataFrame(paths).to_csv(outdir / "budget_paths.csv", index=False)

    # cost-form check (inactive at fixed Q; record g(Q0) for transparency)
    cost_rows = []
    for name, form in cfg["optimization"]["cost_forms"].items():
        cost_rows.append({"form": name, "g_Q0": float(g_cost(form, Q0)),
                          "extra_cost_at_Q0": 0.0,
                          "note": "Q3-A 固定 Q=Q0，[g(Q)-g(Q0)]_+ = 0，成本形式不改变本问结果。"})
    pd.DataFrame(cost_rows).to_csv(outdir / "cost_check.csv", index=False)

    # transition candidates: none expected at fixed Q and p
    trans = [{
        "candidate": "none_detected",
        "reason": "Q3-A 固定 Q=Q0、p=p0，加性质量项为常数；训练/注意力成本份额之比在固定 ell 下不随 C 变化，不产生结构转移。",
        "check": "budget_paths 中 N*/D*/份额随 log C 连续变化",
        "max_share_variation": float(pd.DataFrame(paths).groupby("ell")["train_share"].std().max()),
    }]
    pd.DataFrame(trans).to_csv(outdir / "transition_candidates.csv", index=False)

    summary = {
        "budgets": cfg["optimization"]["budgets_flops"],
        "context_lengths": cfg["optimization"]["context_lengths"],
        "eta": eta,
        "Q0": Q0,
        "gamma": gamma,
        "theta_units": lp["units"],
        "n_scenarios": len(rows),
        "max_rel_diff_L_numeric_vs_analytic": float(scen["rel_diff_L"].max()),
        "max_rel_diff_N_numeric_vs_analytic": float(scen["rel_diff_N"].max()),
        "n_extrapolated": int(scen["extrapolation"].sum()),
        "n_boundary": int(scen["N_near_search_boundary"].sum()),
        "cost_forms_inactive": True,
    }
    write_json(outdir / "optimization_summary.json", summary)
    write_json(outdir / "run_metadata.json", run_metadata(seed, cfg, {
        "q2_loss_predictor_hash": sha256_file(q2dir / "loss_predictor.json"),
        "q1_p0_hash": sha256_file(REPO / cfg["paths"]["q1_out"] / "p0.json"),
        "output_dir": str(outdir.relative_to(REPO)),
    }))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
