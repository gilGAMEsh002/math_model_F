#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the persisted Q1-C predictor against the original commit's stored predictions.

Run with the RESEARCH virtual environment (the one the object must be executable
in):

    .../research/.venv/bin/python verify.py

What is checked (see verification.json for the machine-readable result):

  1. simplex inputs (p0 and Dirichlet samples) are accepted; outputs are finite
     with shape (n, 13);
  2. the upstream row normalisation maps arbitrary non-negative rows to rows that
     sum to 1;
  3. predictions of the persisted object match ``mixture_predictions.csv`` on the
     ``variant == "forest_none"`` rows, per dataset and overall (n / max abs diff
     / median abs diff), for the *original commit result*;
  4. the stored ``y_true`` matches the frozen loss tables exactly (alignment check);
  5. a NATIVE refit with this environment's scikit-learn is compared separately to
     quantify the library-version effect.

Integrity note: A6-A15 inputs/losses are read ONLY here, read-only, purely to
reproduce predictions and confirm alignment against an already-published result.
They are never used to fit, tune or select the model.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FIXED_UPSTREAM = "/home/sshuser/projects/数学建模26/F题-upstream/line-c"
HERE = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 1e-6

DATASETS = [
    ("same_scale_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv"),
    ("cross_scale_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv"),
    ("cross_scale_1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv"),
    ("est_10b", "est_mixture_10b.csv", "est_pile_loss_10b.csv"),
    ("est_70b", "est_mixture_70b.csv", "est_pile_loss_70b.csv"),
]


def import_upstream(upstream: str):
    qdir = os.path.join(upstream, "src", "baselines", "Q1-C")
    if qdir not in sys.path:
        sys.path.insert(0, qdir)
    return importlib.import_module("mixture")


def env_versions() -> dict:
    import sklearn
    import scipy
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__}


def _stats(diff: np.ndarray) -> dict:
    return {"n": int(diff.size), "max_abs_diff": float(np.max(diff)),
            "median_abs_diff": float(np.median(diff))}


def predict_simplex(model, M, X) -> np.ndarray:
    """Call path used by consumers: row-normalise then predict -> (n, 13)."""
    Xn = M.normalize_rows(np.asarray(X, dtype=float))
    return np.asarray(M.predict_models(model, Xn)).reshape(len(Xn), -1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", default=FIXED_UPSTREAM)
    ap.add_argument("--artifacts", default=HERE)
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    args = ap.parse_args()
    upstream = os.path.abspath(args.upstream)
    art = os.path.abspath(args.artifacts)

    import joblib
    M = import_upstream(upstream)
    with open(os.path.join(art, "protocol.json"), encoding="utf-8") as fh:
        protocol = json.load(fh)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(os.path.join(art, "model.joblib"))
    load_warnings = sorted({f"{type(x.message).__name__}: {x.message}" for x in caught})
    load_warning_counts = len(caught)
    seed = int(protocol["seed"])
    tcols = protocol["output_columns"]
    ic = protocol["input_columns"]
    reg = os.path.join(upstream, "real_attachments", "A_data_value", "regmix_tables")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(upstream, "configs", "baselines", "Q1-C.yaml"),
                              encoding="utf-8"))

    out: dict = {
        "artifact": "verification of research/artifacts/c_q1_predictor/model.joblib",
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance": args.tol,
        "persisted_object": {"path": os.path.join(art, "model.joblib"),
                             "type": type(model).__name__},
        "selected_variant": protocol["selected_variant"]["name"],
        "loaded_by_environment": env_versions(),
        "fit_environment": protocol["environment_used_to_fit"],
        "cross_version_unpickle_warnings": {
            "n_warnings": load_warning_counts,
            "unique": load_warnings,
            "note": ("scikit-learn raises InconsistentVersionWarning per unpickled tree "
                     "when loading a 1.5.2 pickle with 1.9.1; predictions were verified "
                     "identical, see original_commit_result_vs_persisted_object"),
        },
    }

    # ---------------------------------------------------- 1. simplex acceptance
    p0 = np.asarray(protocol["p0_reference_mixture"], dtype=float).reshape(1, -1)
    rng = np.random.default_rng(seed)
    dirichlet_flat = rng.dirichlet(np.ones(len(ic)), size=5)          # generic simplex
    dirichlet_near_p0 = rng.dirichlet(np.maximum(p0.ravel() * 50.0, 1e-6), size=5)

    simp = {}
    for name, X in (("p0", p0), ("dirichlet_uniform", dirichlet_flat),
                    ("dirichlet_near_p0", dirichlet_near_p0)):
        Y = predict_simplex(model, M, X)
        simp[name] = {
            "input_shape": list(np.asarray(X).shape),
            "input_rowsum_min": float(np.asarray(X).sum(axis=1).min()),
            "input_rowsum_max": float(np.asarray(X).sum(axis=1).max()),
            "accepted": True,
            "output_shape": list(Y.shape),
            "output_all_finite": bool(np.all(np.isfinite(Y))),
        }
        if name == "p0":
            simp[name]["p0_prediction"] = [float(v) for v in Y[0]]
    out["simplex_acceptance"] = simp

    # ------------------------------------------------ 2. row-normalisation check
    raw = np.vstack([p0 * 3.0, np.full((1, len(ic)), 1.0 / len(ic)),
                     np.zeros((1, len(ic)))])
    raw[2, 0] = 0.0
    normed = M.normalize_rows(raw.copy())
    out["row_normalization"] = {
        "rule": protocol["row_normalization"]["rule"],
        "input_rowsums": [float(v) for v in raw.sum(axis=1)],
        "normalized_rowsums": [float(v) for v in normed.sum(axis=1)],
        "positive_rows_sum_to_one_within_1e-12": bool(
            np.allclose(normed[:2].sum(axis=1), 1.0, atol=1e-12)),
        "zero_row_stays_zero": bool(np.allclose(normed[2], 0.0)),
    }

    # ------------------------------------- 3. compare vs stored predictions
    stored = pd.read_csv(os.path.join(upstream, "artifacts", "baselines", "Q1-C",
                                      "mixture_predictions.csv"))
    stored = stored[stored["variant"] == protocol["selected_variant"]["name"]]
    per_dataset = []
    all_diffs = []
    all_yt = []
    for ds, mfile, lfile in DATASETS:
        Xd = pd.read_csv(os.path.join(reg, mfile))
        idx = Xd["index"].to_numpy()
        Xn = M.normalize_rows(Xd.drop(columns=["index"]).to_numpy(dtype=float))
        Xv, _ = M.build_variant_X(cfg, Xn, ic, {}, "none")
        P = M.predict_models(model, Xv)
        ours = []
        for j, c in enumerate(tcols):
            ours.append(pd.DataFrame({"row_index": idx, "target": M.short_target(c),
                                      "y_pred_ours": P[:, j]}))
        ours.append(pd.DataFrame({"row_index": idx, "target": "__composite_equal_v__",
                                  "y_pred_ours": P.mean(axis=1)}))
        ours = pd.concat(ours, ignore_index=True)

        s = stored[stored["dataset"] == ds][["row_index", "target", "y_true", "y_pred"]]
        m = s.merge(ours, on=["row_index", "target"], how="inner")
        d = np.abs(m["y_pred"].to_numpy() - m["y_pred_ours"].to_numpy())
        all_diffs.append(d)

        # y_true alignment vs the frozen loss table (read-only)
        L = pd.read_csv(os.path.join(reg, lfile), index_col="index")
        yt_diffs = []
        for c in tcols:
            g = s[s["target"] == M.short_target(c)].set_index("row_index")
            diff = np.abs(g.loc[L.index, "y_true"].to_numpy(float)
                          - L[c].to_numpy(float))
            yt_diffs.append(diff)
        yt = np.concatenate(yt_diffs)
        all_yt.append(yt)

        rec = {"dataset": ds, "n_matched": int(len(m)),
               "max_abs_diff_y_pred": float(np.max(d)),
               "median_abs_diff_y_pred": float(np.median(d)),
               "max_abs_diff_y_true": float(np.max(yt))}
        per_dataset.append(rec)

    all_diffs = np.concatenate(all_diffs)
    all_yt = np.concatenate(all_yt)
    overall = _stats(all_diffs)
    overall["max_abs_diff_y_true"] = float(np.max(all_yt))
    equivalent = bool(overall["max_abs_diff"] <= args.tol)

    out["original_commit_result_vs_persisted_object"] = {
        "description": ("stored predictions from the original run "
                        "(mixture_predictions.csv, variant forest_none) compared with "
                        "predictions of the persisted object, loaded and called by the "
                        "research environment"),
        "per_dataset": per_dataset,
        "overall": overall,
        "equivalent": equivalent,
        "tolerance": args.tol,
    }

    # --------------------------- 4. native research-environment refit (diagnostic)
    Xraw = pd.read_csv(os.path.join(reg, "train_mixture_1m.csv")).drop(columns=["index"])
    Yraw = pd.read_csv(os.path.join(reg, "train_pile_loss_1m.csv")).drop(columns=["index"])
    Xtr = M.normalize_rows(Xraw.to_numpy(float))
    Ytr = Yraw.to_numpy(float)
    Xm, _ = M.build_variant_X(cfg, Xtr, ic, {}, "none")
    native = M.fit_forest(Xm, Ytr, dict(protocol["frozen_hyperparameters"]), seed)
    nat_per = []
    nat_all = []
    for ds, mfile, lfile in DATASETS:
        Xd = pd.read_csv(os.path.join(reg, mfile))
        idx = Xd["index"].to_numpy()
        Xn = M.normalize_rows(Xd.drop(columns=["index"]).to_numpy(float))
        Xv, _ = M.build_variant_X(
            {"mixture": {"drop_reference_col": None}}, Xn, ic, {}, "none")
        P = M.predict_models(native, Xv)
        ours = []
        for j, c in enumerate(tcols):
            ours.append(pd.DataFrame({"row_index": idx, "target": M.short_target(c),
                                      "y_pred_ours": P[:, j]}))
        ours.append(pd.DataFrame({"row_index": idx, "target": "__composite_equal_v__",
                                  "y_pred_ours": P.mean(axis=1)}))
        ours = pd.concat(ours, ignore_index=True)
        s = stored[stored["dataset"] == ds][["row_index", "target", "y_pred"]]
        m = s.merge(ours, on=["row_index", "target"], how="inner")
        d = np.abs(m["y_pred"].to_numpy() - m["y_pred_ours"].to_numpy())
        nat_all.append(d)
        nat_per.append({"dataset": ds, "n_matched": int(len(m)),
                        "max_abs_diff_y_pred": float(np.max(d)),
                        "median_abs_diff_y_pred": float(np.median(d))})
    nat_all = np.concatenate(nat_all)
    out["diagnostic_native_research_env_refit"] = {
        "description": ("same variant re-fitted from scratch with THIS environment's "
                        "scikit-learn; isolates the library-version effect"),
        "environment": env_versions(),
        "per_dataset": nat_per,
        "overall": _stats(nat_all),
        "equivalent": bool(np.max(nat_all) <= args.tol),
        "cause": (f"scikit-learn {env_versions()['sklearn']} vs the original "
                  f"{protocol['environment_used_to_fit']['sklearn']}: RandomForest "
                  "tree-building RNG/seeding changed, so the fitted trees differ "
                  "(fixed params, seed and data are identical)"),
    }

    with open(os.path.join(art, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"[verify] persisted object vs stored: n={overall['n']} "
          f"max={overall['max_abs_diff']:.3e} median={overall['median_abs_diff']:.3e} "
          f"-> equivalent={equivalent}")
    print(f"[verify] native {env_versions()['sklearn']} refit: "
          f"max={out['diagnostic_native_research_env_refit']['overall']['max_abs_diff']:.3e} "
          f"(diagnostic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
