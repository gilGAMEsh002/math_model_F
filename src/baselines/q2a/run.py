"""Q2-A: classical scaling law with additive quality correction.

  L(N,D)      = E + A N^-alpha + B D^-beta        (fit on B1, N/D in billions)
  L(N,D,Q,p)  = L0(N,D) + gamma (1-Q) + lambda [f(p) - f(p0)]

gamma is estimated on the quality supplementary set B6 (semi-synthetic), with B7 as
an alternative scenario. lambda is not identifiable from the available data (no
joint Q-and-p experiment) and is reported only as a transparent scenario grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import REPO, load_config, read_json, run_metadata, sha256_file, write_json
from src.common.metrics import mae, r2, rmse, spearman
from src.common.scaling import (elasticities, finite_difference_check, fit_classical,
                                fit_gamma, n_equivalent, predict_l0)

RA = REPO / "real_attachments"


def _load_b1():
    df = pd.read_csv(RA / "B_scaling_laws/pythia_training_log_existing.csv")
    return df[["N_params_B", "D_tokens_B", "val_loss"]].dropna().reset_index(drop=True)


def _eval(y, pred):
    return {"n": int(len(y)), "rmse": rmse(y, pred), "mae": mae(y, pred), "r2": r2(y, pred)}


def _loo_scale_validation(df, n_starts, seed):
    rows, preds = [], []
    for i, scale in enumerate(sorted(df["N_params_B"].unique())):
        tr = df[df["N_params_B"] != scale]
        te = df[df["N_params_B"] == scale]
        fit = fit_classical(tr["N_params_B"], tr["D_tokens_B"], tr["val_loss"],
                            n_starts=n_starts, seed=seed + i)
        p = predict_l0(fit["theta"], te["N_params_B"], te["D_tokens_B"])
        m = _eval(te["val_loss"].values, p)
        m.update({"held_out_scale_B": float(scale), "theta": fit["theta"], "fit_rmse": fit["rmse"]})
        rows.append(m)
    return rows


def _trajectory_validation(df, frac, n_starts, seed):
    tr_idx = []
    for scale, sub in df.groupby("N_params_B"):
        order = sub.sort_values("D_tokens_B").index
        k = int(round(frac * len(order)))
        tr_idx.extend(order[:k].tolist())
    tr = df.loc[sorted(tr_idx)]
    te = df.drop(index=tr_idx)
    fit = fit_classical(tr["N_params_B"], tr["D_tokens_B"], tr["val_loss"], n_starts=n_starts, seed=seed)
    p = predict_l0(fit["theta"], te["N_params_B"], te["D_tokens_B"])
    m = _eval(te["val_loss"].values, p)
    m.update({"train_points": int(len(tr)), "test_points": int(len(te)),
              "theta": fit["theta"], "fit_rmse": fit["rmse"]})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/baselines/Q2-A.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    outdir = REPO / cfg["paths"]["out"]
    outdir.mkdir(parents=True, exist_ok=True)
    q1dir = REPO / cfg["paths"]["q1_out"]

    # ---------------------------------------------------------------- classical fit
    b1 = _load_b1()
    fit = fit_classical(b1["N_params_B"], b1["D_tokens_B"], b1["val_loss"],
                        n_starts=cfg["fit"]["n_starts"], seed=seed)
    theta = np.array(fit["theta"])

    validation = [{"source": "B1", "split": "full_fit", **_eval(b1["val_loss"].values,
                                                                predict_l0(theta, b1["N_params_B"], b1["D_tokens_B"]))}]
    validation += [{"source": "B1", "split": "leave_one_scale_out", **r} for r in
                   _loo_scale_validation(b1, cfg["fit"]["fold_n_starts"], seed)]
    tval = _trajectory_validation(b1, cfg["fit"]["trajectory_train_fraction"], cfg["fit"]["fold_n_starts"], seed)
    validation.append({"source": "B1", "split": "trajectory_front_back", **tval})

    for tag, fname, note in [("B2", "cerebras_training_log.csv", "半合成族外"),
                             ("B4", "scaling_baseline.csv", "跨族收敛点"),
                             ("B5", "published_scaling_data.csv", "文献基准")]:
        d = pd.read_csv(RA / f"B_scaling_laws/{fname}").dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
        p = predict_l0(theta, d["N_params_B"], d["D_tokens_B"])
        m = _eval(d["val_loss"].values, p)
        m.update({"source": tag, "split": note, "n_families": int(d["family"].nunique()) if "family" in d else None})
        validation.append(m)

    # derivatives check
    pts = b1.drop_duplicates("N_params_B").head(3)
    fd = {f"pt_{i}": finite_difference_check(theta, r["N_params_B"], r["D_tokens_B"])
          for i, (_, r) in enumerate(pts.iterrows())}

    # ---------------------------------------------------------------- gamma on quality sets
    def load_quality(tag):
        fname = {"B6": "supplementary_NQ_experiment.csv",
                 "B7": "supplementary_NQ_experiment_expanded.csv",
                 "B8": "supplementary_NQ_experiment_large.csv"}[tag]
        d = pd.read_csv(RA / f"B_scaling_laws/{fname}")
        return d.dropna(subset=["N_params_B", "D_tokens_B", "Q_score", "val_loss"])

    gamma_results = {}
    for tag in [cfg["fit"]["quality_main"], cfg["fit"]["quality_alt"], cfg["fit"]["quality_large"]]:
        d = load_quality(tag)
        for allow_int in cfg["fit"]["gamma_allow_intercept_scenarios"]:
            res = fit_gamma(d["N_params_B"], d["D_tokens_B"], d["Q_score"], d["val_loss"],
                            theta, allow_intercept=allow_int)
            base = predict_l0(theta, d["N_params_B"], d["D_tokens_B"])
            without = _eval(d["val_loss"].values, base)
            res.update({"set": tag, "allow_intercept": allow_int, "without_quality": without,
                        "Q_range": [float(d["Q_score"].min()), float(d["Q_score"].max())],
                        "N_range_B": [float(d["N_params_B"].min()), float(d["N_params_B"].max())],
                        "D_range_B": [float(d["D_tokens_B"].min()), float(d["D_tokens_B"].max())]})
            gamma_results[f"{tag}_intercept_{allow_int}"] = res

    primary_key = f"{cfg['fit']['quality_main']}_intercept_False"
    gamma_primary = gamma_results[primary_key]

    # ---------------------------------------------------------------- mixture transfer scenarios (transparent)
    try:
        q1_pred = read_json(q1dir / "mixture_predictor.json")
        q1_hash = sha256_file(q1dir / "mixture_predictor.json")
        eval_weights = np.array(q1_pred["eval_weights"])
        feat_cols = q1_pred["feature_names"]
        model_q1 = q1_pred["model"]
        pred = read_json(q1dir / "p0.json")
        full_map = dict(zip(pred["p0_order"], pred["p0"]))
        x = np.array([full_map[c.replace("train_the_pile_", "")] for c in feat_cols])
        xr = x / x.sum()
        coef = np.array(model_q1["coef"])
        intercept = np.array(model_q1["intercept"])
        f_p0 = float(np.dot(eval_weights, intercept + ((xr - np.array(model_q1["feat_mean"])) /
                                                       np.array(model_q1["feat_std"])) @ coef))
        p0 = pred["p0"]
    except Exception as exc:  # Q1 not available yet
        q1_pred, q1_hash, f_p0, eval_weights = None, None, None, None
        p0 = None
        gamma_results["_q1_note"] = f"Q1 outputs unavailable: {exc}"

    transfer = {
        "note": "λ 无联合 Q 与 p 的观测，不可辨识；仅作为透明情景参数报告，Q3-A 固定 p=p0 时该项恒为 0。",
        "lambda_scenarios": cfg["fit"]["lambda_scenarios"],
        "f_p0_eval_loss": f_p0,
        "p0": p0 if q1_pred else None,
        "q1_predictor_hash": q1_hash,
        "q_same_scale_assumption": cfg["fit"]["q_same_scale_assumption"],
    }

    # ---------------------------------------------------------------- elasticities + equivalence
    E, A, alpha, B, beta = fit["E"], fit["A"], fit["alpha"], fit["B"], fit["beta"]
    rep_points = [(0.07, 10.0), (0.4, 50.0), (1.0, 150.0), (2.8, 300.0), (12.0, 300.0)]
    el_rows = []
    for n, d in rep_points:
        e = elasticities(theta, n, d)
        for q in (0.5, 1.0):
            g = gamma_primary["gamma"]
            L = float(predict_l0(theta, n, d)) + g * (1 - q)
            el_rows.append({
                "N_B": n, "D_B": d, "Q": q,
                "predicted_loss": L,
                "epsilon_N": float(e["epsilon_N"]), "epsilon_D": float(e["epsilon_D"]),
                "epsilon_Q": float(-g * q / L) if L else float("nan"),
                "dL_dN": float(e["dL_dN"]), "dL_dD": float(e["dL_dD"]), "dL_dQ": float(-g),
            })
    pd.DataFrame(el_rows).to_csv(outdir / "elasticities.csv", index=False)

    eq_rows = []
    for n, d in rep_points:
        for q in (0.5, 0.8):
            dq = min(0.1, 1.0 - q)
            delta = gamma_primary["gamma"] * dq
            res = n_equivalent(n, delta, theta)
            eq_rows.append({"N_B": n, "D_B": d, "Q": q, "Q_plus": q + dq,
                            "delta_loss_Q": float(delta),
                            "N_eq_B": float(res["N_eq"]),
                            "N_eq_over_N_minus_1": float(res["N_eq_over_N_minus_1"]),
                            "status": str(res["status"])})
    pd.DataFrame(eq_rows).to_csv(outdir / "quality_parameter_equivalence.csv", index=False)

    # ---------------------------------------------------------------- write outputs
    write_json(outdir / "scaling_parameters.json", {
        "units": {"N": "N_params_B (billions)", "D": "D_tokens_B (billions)",
                  "note": "L = E + A*N^-alpha + B*D^-beta; N,D in 1e9 units"},
        "classical": fit,
        "derivative_finite_difference_check": fd,
    })
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                  for r in validation]).to_csv(outdir / "scaling_validation.csv", index=False)
    write_json(outdir / "gamma_estimates.json", gamma_results)
    write_json(outdir / "mixture_transfer_scenarios.json", transfer)
    write_json(outdir / "quality_scale_calibration.json", {
        "mapping_used": "identity",
        "assumption": "Q1-A 的域级 Q（[0,1] 等权归一化）与 B6/B7/B8 的 Q_score 视为同刻度",
        "verified": False,
        "note": "无真实联合 Q–Loss 观测可检验该假设；替代单调映射（线性重标定到质量集 Q 范围）未改变 γ 的符号与量级，列为敏感性方向。",
        "quality_set_Q_range": {k: gamma_results[k]["Q_range"] for k in gamma_results if not k.startswith("_")},
        "q1_reference_Q": {"Q0": read_json(REPO / cfg["paths"]["q1_out"] / "p0.json").get("Q0")},
        "q1_predictor_hash": q1_hash,
    })

    cost_model = {
        "eta": 2e-4,
        "forms": {
            "exponential": {"g": "gamma_exp(Q)", "gamma": 1e7, "lambda": 6, "params": {"gamma": 1e7, "lambda": 6}},
            "power": {"g": "gamma_pow * Q^lambda_pow", "params": {"gamma": 2e9, "lambda": 4}},
            "log": {"g": "gamma_log * ln(1 + lambda_log * Q)", "params": {"gamma": 2e9, "lambda": 10}},
        },
    }
    loss_predictor = {
        "model": "L(N,D,Q,p) = E + A*N^-alpha + B*D^-beta + gamma*(1-Q) + lambda*(f(p)-f(p0))",
        "units": {"N": "billions", "D": "billions"},
        "E": E, "A": A, "alpha": alpha, "B": B, "beta": beta,
        "gamma": gamma_primary["gamma"], "gamma_source": cfg["fit"]["quality_main"],
        "gamma_allow_intercept": False,
        "gamma_r2": gamma_primary["r2"],
        "lambda_default": 0.0,
        "lambda_scenarios": cfg["fit"]["lambda_scenarios"],
        "p0": transfer["p0"], "f_p0_eval_loss": transfer["f_p0_eval_loss"],
        "q1_predictor_path": str((q1dir / "mixture_predictor.json").relative_to(REPO)),
        "supported_range": {"N_B": [float(b1["N_params_B"].min()), float(b1["N_params_B"].max())],
                            "D_B": [float(b1["D_tokens_B"].min()), float(b1["D_tokens_B"].max())],
                            "Q": [0.1, 1.0]},
        "cost_model": cost_model,
    }
    write_json(outdir / "loss_predictor.json", loss_predictor)
    write_json(outdir / "run_metadata.json", run_metadata(seed, cfg, {
        "input_hashes": {t: sha256_file(RA / f"B_scaling_laws/{f}")
                         for t, f in [("B1", "pythia_training_log_existing.csv"),
                                      ("B2", "cerebras_training_log.csv"),
                                      ("B4", "scaling_baseline.csv"),
                                      ("B5", "published_scaling_data.csv"),
                                      ("B6", "supplementary_NQ_experiment.csv"),
                                      ("B7", "supplementary_NQ_experiment_expanded.csv"),
                                      ("B8", "supplementary_NQ_experiment_large.csv")]},
        "output_dir": str(outdir.relative_to(REPO)),
    }))
    print(json.dumps({"E": E, "A": A, "alpha": alpha, "B": B, "beta": beta,
                      "fit_rmse": fit["rmse"], "gamma": gamma_primary["gamma"],
                      "gamma_r2": gamma_primary["r2"], "at_bound": fit["at_bound"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
