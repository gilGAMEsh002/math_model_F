"""Read-only audit recomputations for Q2 delta/gamma decomposition and B8 cells.

Run from the research checkout:
  research/.venv/bin/python research/reviews/20260924_audit/q2_recompute.py
Outputs q2_recompute.csv beside this script. No source files are modified.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
LOCK = json.loads((RESEARCH / "upstream.lock.json").read_text(encoding="utf-8"))
A_ROOT = Path(LOCK["lines"]["A"]["worktree"])
PRED = json.loads((A_ROOT / "artifacts/baselines/Q2-A/loss_predictor.json").read_text())
E, A, ALPHA, B, BETA = [float(PRED[k]) for k in ("E", "A", "alpha", "B", "beta")]
DATA = A_ROOT / "real_attachments/B_scaling_laws"
FILES = {
    "B6": "supplementary_NQ_experiment.csv",
    "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv",
}


def analyze_source(tag: str, df: pd.DataFrame, data_type: str) -> dict:
    ncoord = df["N_params_B"].to_numpy(float)
    dcoord = df["D_tokens_B"].to_numpy(float)
    x = 1.0 - df["Q_score"].to_numpy(float)
    r = df["val_loss"].to_numpy(float) - (
        E + A * ncoord ** (-ALPHA) + B * dcoord ** (-BETA)
    )
    sse0 = float(r @ r)
    delta = float(r.mean())
    sse_delta = float(((r - delta) @ (r - delta)))
    denom = float(x @ x)
    gamma0 = max(0.0, float((x @ r) / denom)) if denom > 0 else 0.0
    sse_gamma = float(((r - gamma0 * x) @ (r - gamma0 * x)))
    xc, rc = x - x.mean(), r - r.mean()
    gamma_joint = max(0.0, float((xc @ rc) / (xc @ xc))) if (xc @ xc) > 0 else 0.0
    delta_joint = float(r.mean() - gamma_joint * x.mean())
    resid = r - (delta_joint + gamma_joint * x)
    sse_joint = float(resid @ resid)
    out = {
        "source": tag, "data_type": data_type, "n": len(df),
        "sse_M0": sse0,
        "delta_only": delta, "sse_delta_only": sse_delta,
        "delta_only_gain_frac_M0": (sse0 - sse_delta) / sse0,
        "gamma_only_nonnegative": gamma0, "sse_gamma_only": sse_gamma,
        "gamma_only_gain_frac_M0": (sse0 - sse_gamma) / sse0,
        "delta_joint": delta_joint, "gamma_joint_nonnegative": gamma_joint,
        "sse_joint": sse_joint, "joint_gain_frac_M0": (sse0 - sse_joint) / sse0,
        "quality_gain_after_delta_frac_delta_sse": (sse_delta - sse_joint) / sse_delta,
    }
    if tag == "B8":
        keys = ["N_params_B", "D_tokens_B"]
        cells = df.groupby(keys).agg(rows=("Q_score", "size"), n_unique_q=("Q_score", "nunique"))
        g = df.copy()
        g["x"] = x
        g["r"] = r
        g["x_cell_centered"] = g["x"] - g.groupby(keys)["x"].transform("mean")
        g["r_cell_centered"] = g["r"] - g.groupby(keys)["r"].transform("mean")
        xx = g["x_cell_centered"].to_numpy()
        rr = g["r_cell_centered"].to_numpy()
        slope = float(xx @ rr / (xx @ xx))
        corr = float(np.corrcoef(xx, rr)[0, 1])
        counts = cells["n_unique_q"].value_counts().to_dict()
        row_counts = cells["rows"].value_counts().to_dict()
        out.update({
            "n_nd_cells": int(len(cells)), "q_count_distribution": json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True),
            "rows_per_cell_distribution": json.dumps({str(k): int(v) for k, v in row_counts.items()}, sort_keys=True),
            "duplicate_ndq_rows": int(df.duplicated(keys + ["Q_score"]).sum()),
            "within_cell_slope_resid_on_1mQ": slope,
            "within_cell_corr_resid_1mQ": corr,
        })
    return out


def main() -> None:
    results = []
    for tag, fname in FILES.items():
        df = pd.read_csv(DATA / fname)
        if tag == "B8":
            for kind, part in df.groupby("data_type", sort=True):
                results.append(analyze_source(tag, part.reset_index(drop=True), str(kind)))
        else:
            results.append(analyze_source(tag, df, "all"))
    out = pd.DataFrame(results)
    out.to_csv(HERE / "q2_recompute.csv", index=False)
    cols = ["source", "data_type", "n", "delta_only_gain_frac_M0", "gamma_only_gain_frac_M0", "joint_gain_frac_M0", "delta_joint", "gamma_joint_nonnegative", "n_nd_cells", "q_count_distribution", "within_cell_slope_resid_on_1mQ", "within_cell_corr_resid_1mQ"]
    print(out.reindex(columns=cols).to_string(index=False))
    print(f"saved: {HERE / 'q2_recompute.csv'}")


if __name__ == "__main__":
    main()
