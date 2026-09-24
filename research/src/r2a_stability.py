"""R2a: does prediction improvement form STABLE selection improvement?

Part 0 (pre-review): how does each channel move the *ranking* of candidate mixtures?
For a separable model the mixture channel is a positive monotone transform of Q(p), so
argmin_p L = argmax_p Q(p), independent of (C, ell, kappa, omega). We reuse C's own
`verify_p_channel.csv` numeric evidence and record the analytic conclusion instead of
re-running a necessarily identical selection.

Part 1 (experiment): on the SAME candidate pools (A6-A11) compare
  point selection  = argmin predicted equal-weight eval loss,
  stable selection = argmin(ensemble_mean + lambda * ensemble_std), lambda chosen
                     INSIDE A4/A5 only,
  random-pick expectation,
reporting TRUE top-1/top-5 regret (from the actual mixture experiments), and selection
stability. A6-A11 are exploratory external checks; A12-A15 are diagnostics only.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import LineA, LineB, LineC
from .common import RESEARCH

V = np.full(13, 1.0 / 13)
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0]
POOLS = [("test_mixture_1m.csv", "test_pile_loss_1m.csv", "1m"),
         ("test_mixture_60m.csv", "test_pile_loss_60m.csv", "60m"),
         ("test_mixture_1B.csv", "test_pile_loss_1B.csv", "1B")]


def train_matrix(rt: Path):
    M = pd.read_csv(rt / "train_mixture_1m.csv"); L = pd.read_csv(rt / "train_pile_loss_1m.csv")
    P = M.drop(columns=["index"]).to_numpy(float); P = P / P.sum(1, keepdims=True)
    return P, L.drop(columns=["index"]).to_numpy(float), list(M.columns[1:])


def set_matrix(rt: Path, mf: str, lf: str):
    M = pd.read_csv(rt / mf); L = pd.read_csv(rt / lf)
    P = M.drop(columns=["index"]).to_numpy(float); P = P / P.sum(1, keepdims=True)
    return M["index"].to_numpy(), P, L.drop(columns=["index"]).to_numpy(float)


# ------------------------------------------------------------------ A ridge bootstrap
def a_bootstrap(P, Y, fidx, lam, B, seed):
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(B):
        idx = rng.integers(0, len(P), len(P))
        X = P[np.ix_(idx, fidx)]
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        Xs = (X - mu) / sd; ybar = Y[idx].mean(0)
        W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Y[idx] - ybar))
        fits.append((mu, sd, W, ybar))
    return fits


def a_predict(fits, P, fidx):
    X = P[:, fidx]
    return np.stack([((X - mu) / sd) @ W + ybar for (mu, sd, W, ybar) in fits])


# ------------------------------------------------------------------ B quadratic bootstrap
def _b_design(P):
    n = P.shape[1]
    inter = np.column_stack([P[:, i] * P[:, j] for i, j in combinations(range(n), 2)]) * n
    return np.hstack([P, inter])


def b_bootstrap(P, Y, alpha, B, seed):
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(B):
        idx = rng.integers(0, len(P), len(P))
        X = _b_design(P[idx])
        fits.append(np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y[idx]))
    return np.array(fits)


def b_predict(fits, P):
    X = _b_design(P)
    return np.stack([X @ W for W in fits])


# ------------------------------------------------------------------ C per-tree ensemble
def c_trees(P):
    import joblib
    model = joblib.load(RESEARCH / "artifacts/c_q1_predictor/model.joblib")
    X = P / P.sum(1, keepdims=True)
    return np.stack([t.predict(X) for t in model.estimators_])


# ------------------------------------------------------------------ selection
def _eval(pred):                     # pred (n,13) -> (n,)
    return pred @ V


def regret(y, pred_eval, k=1):
    sel = np.argsort(pred_eval)[:k]
    return float(np.mean(y[sel] - y.min()))


def choose_lambda(mean, std, y, lams=LAMS):
    best_l, best_r = 0.0, None
    for l in lams:
        r = regret(y, _eval(mean + l * std))
        if best_r is None or r < best_r:
            best_l, best_r = l, r
    return best_l, best_r


def prereview() -> dict:
    C = LineC()
    o = {"separable_mixture_channel": {
        "claim": "h(p) 是 Q(p) 的正单调变换且与 (C,ell,kappa,omega) 无关 ⇒ argmin_p L = argmax_p Q(p)；"
                 "配比通道不改变候选排序，无需重复运行必然同选方的实验。",
        "form": "L = E + A n^-a + B (d Q^kappa h(p))^-b；A/B 第三问固定 p=p0（通道关闭）。",
    }}
    vp = C.root / "artifacts/baselines/Q3-C/verify_p_channel.csv"
    if vp.exists():
        d = pd.read_csv(vp)
        o["separable_mixture_channel"]["existing_numeric_evidence"] = {
            "file": str(vp.relative_to(C.root)),
            "cells": int(len(d)),
            "argmin_L_is_argmax_h_true": int(d["argmin_L_is_argmax_h"].sum()),
        }
    return o


def run_r2a(seed=20260923, B=100, outdir: Path | None = None) -> dict:
    outdir = outdir or (RESEARCH / "reports")
    outdir.mkdir(parents=True, exist_ok=True)
    A, Bline, C = LineA(), LineB(), LineC()
    rt = A.root / "real_attachments/A_data_value/regmix_tables"
    Ptr, Ytr, cols = train_matrix(rt)
    ref = A.mix["model"]["reference_domain"]
    fidx = [i for i, c in enumerate(cols) if c != ref]
    a_fits = a_bootstrap(Ptr, Ytr, fidx, float(A.mix["model"]["ridge_lambda"]), B, seed)
    b_fits = b_bootstrap(Ptr, Ytr, 0.021544346900318822, B, seed)
    c_fits = c_trees(Ptr)                              # (T, n, 13), no refit

    # lambda chosen inside A4/A5 from each line's own ensemble
    ytr = Ytr @ V
    lam, reg = {}, {}
    for name, batch in (("A", a_predict(a_fits, Ptr, fidx)),
                        ("B", b_predict(b_fits, Ptr)),
                        ("C", c_fits)):
        lam[name], reg[name] = choose_lambda(batch.mean(0), batch.std(0), ytr)
        # lambda must be chosen on the *eval-loss* ensemble spread
    rows = []
    for mf, lf, key in POOLS:
        idx, P, Y = set_matrix(rt, mf, lf)
        y = Y @ V
        batches = {"A": a_predict(a_fits, P, fidx), "B": b_predict(b_fits, P), "C": c_trees(P)}
        mean = {k: _eval(v.mean(0)) for k, v in batches.items()}
        std = {k: v.std(0) @ V for k, v in batches.items()}
        rand = float(y.mean() - y.min())
        for name in ("A", "B", "C"):
            pt = regret(y, mean[name])
            st = regret(y, mean[name] + lam[name] * std[name])
            rows.append({"set": key, "line": name, "n": len(P), "lambda": lam[name],
                         "point_top1_regret": pt, "stable_top1_regret": st,
                         "point_top5_regret": regret(y, mean[name], 5),
                         "stable_top5_regret": regret(y, mean[name] + lam[name] * std[name], 5),
                         "random_expected_regret": rand,
                         "stable_beats_point": bool(st < pt - 1e-12),
                         "point_beats_random": bool(pt < rand)})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "r2a_stability.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sets = ["1m", "60m", "1B"]
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        w = 0.25
        for i, line in enumerate(["A", "B", "C"]):
            pt = [df[(df.line == line) & (df.set == s)]["point_top1_regret"].mean() for s in sets]
            ax.bar(np.arange(3) + (i - 1) * w, pt, width=w, label=f"{line} point")
        rnd = [df[df.set == s]["random_expected_regret"].mean() for s in sets]
        ax.plot(np.arange(3), rnd, "k--o", label="random expectation")
        ax.set_xticks(range(3)); ax.set_xticklabels(sets)
        ax.set_ylabel("true top-1 regret"); ax.set_title(
            "R2a: selection regret vs random (A6-A11, exploratory)", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(outdir / "r2a_regret.png", dpi=160); plt.close(fig)
    except Exception as exc:
        print("figure failed:", exc)
    pre = prereview()
    (outdir / "r2a_prereview.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"B": B, "seed": seed, "lambda_inside_A4A5": lam, "train_regret_at_lambda": reg,
               "exploratory": "A6-A11 已被查看；本结果为探索性外部复核，非首次盲测；A12-A15 仅诊断",
               "prereview": pre, "per_set": df.to_dict("records")}
    (outdir / "r2a_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stability": df, "summary": summary}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=str(RESEARCH / "reports"))
    a = ap.parse_args()
    res = run_r2a(outdir=Path(a.outdir))
    print(res["stability"].round(4).to_string(index=False))
    print("lambda(inside A4/A5):", json.dumps(res["summary"]["lambda_inside_A4A5"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
