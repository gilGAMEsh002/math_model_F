#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild Baseline C's selected Q1-C mixture predictor as a persisted object.

This script refits EXACTLY ONE variant -- the one that the original run's
``selected_model.overall_best`` corresponds to -- using the unmodified upstream
implementation.  Nothing is vendored or rewritten and no cross-validation or
hyper-parameter search is performed.

Variant determination (evidence; full details are written into protocol.json):

  * ``run_all.build_predictor`` takes ``selected_model.overall_best`` from
    ``mixture.run_mixture``'s ``best`` row, which comes from
    ``cv_select(cfg, Xtr, Ytr, cands)``.  ``cv_select`` is called with
    ``Xtr = normalize_rows(A4)`` -- the 17 row-normalised mixture columns, with
    NO quality feature appended.  Therefore the winning candidate
    ``rf_n600_d12_l1_f1.0`` was selected in quality-feature mode ``none``.
  * The winning CV row equals the first row of ``mixture_cv_grid.csv``:
    ``cv_composite_rmse = 0.15538620982757276``.
  * The only fitted variant carrying those frozen forest params in mode ``none``
    is ``forest_none`` (see the ``variants`` dict in ``run_mixture``).
  * Empirical confirmation: re-fitting ``forest_none`` reproduces the stored
    ``mixture_predictions.csv`` rows with ``variant == "forest_none"`` to
    floating-point tolerance; no other variant is close (verify.py reports the
    full comparison).

Only A4/A5 (``train_mixture_1m.csv`` / ``train_pile_loss_1m.csv``) are read here.
A6-A15 are never touched by this fitting script.

Environment note
----------------
Bit-exact reproduction of the original forest requires the library versions of
the original run (Python 3.11 / numpy 2.4.6 / pandas 3.0.3 / scipy 1.17.1 /
scikit-learn 1.5.2).  Under a newer scikit-learn the RandomForest is re-fitted
with a different tree-building RNG path and predictions drift by up to ~0.013
(median ~1e-3).  The pinned environment was used to produce ``model.joblib``;
``verify.py`` re-loads and calls it with the research virtual environment and
also quantifies the newer-environment refit difference.

Usage
-----
    python rebuild.py [--upstream DIR] [--out DIR]

``rebuild.py`` writes ``model.joblib`` and ``protocol.json``.  Verification
(``verification.json``) is produced by ``verify.py`` because it needs the
research virtual environment to load the persisted object.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

FIXED_UPSTREAM = "/home/sshuser/projects/数学建模26/F题-upstream/line-c"
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------- helpers
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_rev(cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def import_upstream(upstream: str):
    """Import the upstream Q1-C modules by path (no vendoring)."""
    qdir = os.path.join(upstream, "src", "baselines", "Q1-C")
    if qdir not in sys.path:
        sys.path.insert(0, qdir)
    return importlib.import_module("mixture")


def env_versions() -> dict:
    import sklearn
    import scipy
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def load_pairs_train(cfg: dict, upstream: str):
    """Read A4/A5 only, enforcing the exact config column order and row alignment."""
    reg = os.path.join(upstream, "real_attachments", "A_data_value", "regmix_tables")
    mpath = os.path.join(reg, "train_mixture_1m.csv")
    lpath = os.path.join(reg, "train_pile_loss_1m.csv")
    m = pd.read_csv(mpath)
    l = pd.read_csv(lpath)
    if not np.array_equal(m["index"].values, l["index"].values):
        raise AssertionError("A4/A5 index is not one-to-one")
    X = m.drop(columns=["index"])
    Y = l.drop(columns=["index"])
    if list(X.columns) != cfg["mixture"]["input_cols"]:
        raise AssertionError("A4 input column order != config input_cols")
    if list(Y.columns) != cfg["mixture"]["target_cols"]:
        raise AssertionError("A5 target column order != config target_cols")
    return mpath, lpath, X, Y


# --------------------------------------------------------------------- main
def build(upstream: str, out: str) -> dict:
    cfg_path = os.path.join(upstream, "configs", "baselines", "Q1-C.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    desc_path = os.path.join(upstream, "artifacts", "baselines", "Q1-C",
                             "mixture_predictor.json")
    with open(desc_path, encoding="utf-8") as fh:
        descriptor = json.load(fh)
    meta_path = os.path.join(upstream, "artifacts", "baselines", "Q1-C",
                             "run_metadata.json")
    with open(meta_path, encoding="utf-8") as fh:
        run_meta = json.load(fh)

    M = import_upstream(upstream)

    # ---- frozen selected candidate (read from the descriptor; assert label<->params)
    best = descriptor["selected_model"]["overall_best"]
    params = dict(descriptor["selected_model"]["forest"])
    label = best["label"]
    expected_label = (f"rf_n{params['n_estimators']}_d{params['max_depth']}"
                      f"_l{params['min_samples_leaf']}_f{params['max_features']}")
    if label != expected_label:
        raise AssertionError(f"descriptor label {label!r} != params label {expected_label!r}")
    if best["kind"] != "forest":
        raise AssertionError("overall_best is not a forest candidate")
    seed = int(cfg["seed"])

    # ---- data (A4/A5 only)
    mpath, lpath, Xraw, Yraw = load_pairs_train(cfg, upstream)
    Xtr = M.normalize_rows(Xraw.to_numpy(dtype=float))
    Ytr = Yraw.to_numpy(dtype=float)
    cols = list(Xraw.columns)
    tcols = list(cfg["mixture"]["target_cols"])

    # ---- selected variant: mode "none" (CV selection space), no drop_ref, no Q
    #      qmap is only used for equal/entropy modes; "none" returns before touching it.
    Xm, names = M.build_variant_X(cfg, Xtr, cols, {}, "none")
    if names != cols:
        raise AssertionError("mode 'none' unexpectedly changed feature columns")

    # ---- refit (single model; NO cv_select, NO hyper-parameter search)
    model = M.fit_forest(Xm, Ytr, params, seed)

    # ---- sanity: descriptor p0 == mean of normalised A4
    p0 = [float(x) for x in Xtr.mean(axis=0)]
    p0_desc = [float(x) for x in descriptor["p0_reference_mixture_mean"]]
    if not np.allclose(p0, p0_desc, rtol=0.0, atol=1e-12):
        raise AssertionError("recomputed p0 != descriptor p0")

    os.makedirs(out, exist_ok=True)
    model_path = os.path.join(out, "model.joblib")
    import joblib
    joblib.dump(model, model_path)

    n_targets = len(tcols)
    protocol = {
        "artifact": "Baseline C Q1-C selected mixture predictor",
        "baseline_id": cfg["baseline_id"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "fixed_upstream_worktree": os.path.abspath(upstream),
            "source_commit": git_rev(upstream),
            "config_path": os.path.abspath(cfg_path),
            "config_sha256": sha256_file(cfg_path),
            "config_hash_upstream": _config_hash(cfg),
            "a4_mixture": {"path": os.path.relpath(mpath, upstream),
                           "sha256": sha256_file(mpath)},
            "a5_loss": {"path": os.path.relpath(lpath, upstream),
                        "sha256": sha256_file(lpath)},
            "descriptor": {"path": os.path.relpath(desc_path, upstream),
                           "sha256": sha256_file(desc_path)},
            "original_run": {
                "run_command": run_meta.get("run_command"),
                "run_utc": run_meta.get("run_utc"),
                "git_rev": run_meta.get("git_rev"),
                "config_hash": run_meta.get("config_hash"),
                "config_yaml_sha256": run_meta.get("input_hashes_sha256", {}).get("config_yaml"),
                "environment": run_meta.get("environment"),
                "note": ("the original run's config (sha256 ...bdb41dde) differs from the "
                         "fixed tree's config (sha256 ...0f18d026) only in the non-model "
                         "upstream.* path strings; all mixture-relevant knobs are identical"),
            },
        },
        "selected_variant": {
            "name": "forest_none",
            "kind": "forest",
            "quality_feature_mode": "none",
            "unmapped_domain_policy": {
                "applies": False,
                "reason": ("mode 'none' uses only the 17 row-normalised mixture columns; "
                           "no Q(p) feature is built"),
                "config_primary_unmapped_policy": cfg["mixture"]["quality_feature"]["primary_unmapped_policy"],
                "config_proxy_unmapped_policy": cfg["mixture"]["quality_feature"]["proxy_unmapped_policy"],
            },
            "drop_reference_col": None,
            "determination_evidence": [
                "run_mixture calls cv_select(cfg, Xtr, Ytr, cands) with Xtr = normalize_rows(A4) (17 cols, no Q); selection space is mode 'none'",
                f"overall_best label {label} with cv_composite_rmse={best['cv_composite_rmse']!r} equals mixture_cv_grid.csv row 0",
                "the only fitted variant with these params in mode none is forest_none",
                "verify.py: refit reproduces stored mixture_predictions.csv variant 'forest_none' (no other variant is close)",
            ],
        },
        "frozen_hyperparameters": params,
        "seed": seed,
        "target_weights_v": descriptor["target_weights_v"],
        "target_weights_note": descriptor["target_weights_note"],
        "input_columns": cols,
        "n_inputs": len(cols),
        "output_columns": tcols,
        "output_short_names": [M.short_target(c) for c in tcols],
        "n_outputs": n_targets,
        "p0_reference_mixture": p0,
        "p0_note": descriptor["row_normalization"],
        "row_normalization": {
            "rule": "p_i := p_i / sum_j p_j  (rows with sum<=0 become all-zero -> 1.0 divisor)",
            "applies_to": "both fitting inputs and call-time inputs",
            "A4_raw_rowsum_range": [float(Xraw.to_numpy(float).sum(1).min()),
                                    float(Xraw.to_numpy(float).sum(1).max())],
            "sums_to_one": True,
        },
        "input_units": "row-normalised mixture proportions (non-negative, rows sum to 1)",
        "output_units": descriptor["output_units"],
        "training": {
            "n_rows": int(len(Xtr)),
            "n_features": int(Xm.shape[1]),
            "data_used": ["A4 train_mixture_1m.csv", "A5 train_pile_loss_1m.csv"],
            "held_out_sets_not_used_for_fitting": ["A6-A11", "A12-A15"],
            "method": "upstream mixture.fit_forest (RandomForestRegressor, multioutput)",
        },
        "environment_used_to_fit": env_versions(),
        "how_to_call": (
            "import sys, joblib, numpy as np\n"
            "sys.path.insert(0, '<upstream>/src/baselines/Q1-C')\n"
            "from mixture import normalize_rows, predict_models\n"
            "model = joblib.load('research/artifacts/c_q1_predictor/model.joblib')\n"
            "p = np.asarray(p, dtype=float)                 # (n, 17) non-negative proportions\n"
            "y = predict_models(model, normalize_rows(p))   # (n, 13) val losses, config target order\n"
            "# e.g. the research environment is enough to load and call this object;\n"
            "# scikit-learn emits an InconsistentVersionWarning (1.5.2 -> 1.9.1) but predictions match.\n"
        ),
        "reproduction": {
            "exact_requires": {
                "python": "3.11",
                "numpy": "2.4.6", "pandas": "3.0.3",
                "scipy": "1.17.1", "scikit-learn": "1.5.2",
            },
            "bootstrap": [
                "python -m venv .venv && .venv/bin/pip install uv",
                ".venv/bin/uv python install 3.11",
                ".venv/bin/uv venv --python 3.11 /tmp/opencode/venv311",
                ".venv/bin/uv pip install --python /tmp/opencode/venv311/bin/python "
                "numpy==2.4.6 pandas==3.0.3 scipy==1.17.1 scikit-learn==1.5.2 pyyaml joblib",
            ],
            "note": ("model.joblib was produced with the pinned versions above; it also "
                     "loads and predicts in the research venv (see verification.json)"),
        },
    }
    proto_path = os.path.join(out, "protocol.json")
    with open(proto_path, "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, ensure_ascii=False, indent=2)

    # ---- in-sample reproducibility fingerprint (no held-out data touched)
    train_pred = M.predict_models(model, Xm)
    fingerprint = {
        "train_composite_r2": float(_r2(Ytr.mean(axis=1), train_pred.mean(axis=1))),
        "train_pred_sum": float(train_pred.sum()),
    }
    print(f"[rebuild] fitted {label} on {Xm.shape} -> {Ytr.shape}; saved {model_path}")
    print(f"[rebuild] env: {protocol['environment_used_to_fit']}")
    print(f"[rebuild] fingerprint: {fingerprint}")
    return protocol


def _config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _r2(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", default=FIXED_UPSTREAM)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    build(os.path.abspath(args.upstream), os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
