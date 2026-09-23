"""Minimal meaningful tests for Baseline A (plan section 6).

Focus is on checks that would change conclusions if wrong: mixture/Loss join and
normalisation, quality transforms and dedup, N/D units and the analytic optimum,
derivative/optimisation consistency, and the frozen Q1 predictor protocol.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.common.io_utils import REPO
from src.common.quality import aggregate, fit_calibration, score_record
from src.common.scaling import (elasticities, finite_difference_check, fit_classical,
                                n_equivalent, predict_l0)
from src.common.metrics import softmax

RA = REPO / "real_attachments"
Q1 = REPO / "artifacts/baselines/Q1-A"
Q2 = REPO / "artifacts/baselines/Q2-A"
Q3 = REPO / "artifacts/baselines/Q3-A"
METRICS = json.loads((REPO / "configs/shared/quality_metrics.json").read_text(encoding="utf-8"))["metrics"]
GROUPS = json.loads((REPO / "configs/shared/quality_metrics.json").read_text(encoding="utf-8"))["groups"]
DOMAINS = json.loads((REPO / "configs/shared/domains.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ mixtures
def test_mixture_loss_join_and_columns():
    m = pd.read_csv(RA / "A_data_value/regmix_tables/train_mixture_1m.csv")
    l = pd.read_csv(RA / "A_data_value/regmix_tables/train_pile_loss_1m.csv")
    assert (m["index"].values == l["index"].values).all()
    pcols = [c for c in m.columns if c.startswith("train_the_pile_")]
    lcols = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in l.columns
             if c.startswith("metric/")]
    assert len(pcols) == 17 and lcols == DOMAINS["loss_targets"]


def test_mixture_renormalisation():
    m = pd.read_csv(RA / "A_data_value/regmix_tables/train_mixture_1m.csv")
    pcols = [c for c in m.columns if c.startswith("train_the_pile_")]
    raw = m[pcols].values
    assert raw.min() >= 0
    norm = raw / raw.sum(axis=1, keepdims=True)
    assert np.allclose(norm.sum(axis=1), 1.0)
    # documented rounding: raw row sums within 0.996-1.003
    assert 0.99 < raw.sum(axis=1).min() and raw.sum(axis=1).max() < 1.01


# ------------------------------------------------------------------ quality
def _toy_record():
    rec = {}
    for m in METRICS:
        rec[m["name"]] = 1.0
    rec["fineweb_edu"] = [2.0]
    rec["fluency_en"] = [0.0, 5.0]          # clearly fluent (index 1)
    rec["ad_en"] = [0.0, 5.0]               # clearly no-ad (index 1)
    for mm in ("modernbert_cleanliness", "modernbert_readability", "modernbert_reasoning",
               "modernbert_professionalism"):
        rec[mm] = [0.0, 1.0, 2.0, 3.0, 4.0, 9.0]  # argmax level 5
    rec["qurater"] = [1.0, 1.0, 1.0, 1.0]
    return rec


def test_quality_transforms_and_range():
    calib = fit_calibration([_toy_record() for _ in range(5)], METRICS)
    dims = score_record(_toy_record(), METRICS, calib)
    assert set(dims) == {m["name"] for m in METRICS}
    assert all(0.0 <= v <= 1.0 for v in dims.values())
    # ordinal: argmax over 6 logits -> level 5 -> 1.0
    assert dims["modernbert_cleanliness"] == pytest.approx(1.0)
    # softmax index 1 for fluency/ad
    assert dims["fluency_en"] > 0.9
    assert dims["ad_en"] > 0.9
    agg = aggregate(dims, METRICS, GROUPS)
    assert abs(agg["conflict"] - (max(agg[f"z_{g}"] for g in GROUPS) -
                                 min(agg[f"z_{g}"] for g in GROUPS))) < 1e-12


def test_softmax_index_direction():
    assert softmax([0.0, 5.0])[1] > softmax([5.0, 0.0])[1]


def test_dedup_and_reference_calibration():
    audit = json.loads((REPO / "artifacts/audit/quality_audit.json").read_text(encoding="utf-8"))
    assert audit["overlaps"] == {"A1_and_A2": 1419, "A1_and_A3": 10000, "A2_and_A3": 0}
    assert audit["merged_unique_records"] == 261086
    prep = json.loads((Q1 / "quality_preprocessing.json").read_text(encoding="utf-8"))
    assert prep["reference_set"] == "A1"
    assert prep["n_reference"] == 51230
    assert "fill_value" in prep["calibration"]["metrics"]["fineweb_edu"]


# ------------------------------------------------------------------ scaling
def test_fit_derivatives_and_equivalence():
    b1 = pd.read_csv(RA / "B_scaling_laws/pythia_training_log_existing.csv")
    fit = fit_classical(b1["N_params_B"], b1["D_tokens_B"], b1["val_loss"], n_starts=5, seed=1)
    theta = np.array(fit["theta"])
    assert fit["rmse"] < 0.05
    fd = finite_difference_check(theta, 1.0, 20.0)
    assert fd["rel_error_N"] < 1e-6 and fd["rel_error_D"] < 1e-6
    res = n_equivalent(1.0, 0.01, theta)
    assert res["status"] == "finite" and res["N_eq"] > 1.0
    unreachable = n_equivalent(1.0, 1e9, theta)
    assert unreachable["status"] in {"not_reachable", "infinite"}


def test_analytic_optimum_matches_grid():
    lp = json.loads((Q2 / "loss_predictor.json").read_text(encoding="utf-8"))
    theta = np.array([lp["E"], np.log(lp["A"]), lp["alpha"], np.log(lp["B"]), lp["beta"]])
    C_B = 10.0
    a = 6.0 + 2e-4 * 2048
    E, logA, alpha, logB, beta = theta
    A, B = np.exp(logA), np.exp(logB)
    Nstar = ((alpha * A) / (beta * B) * (C_B / a) ** beta) ** (1 / (alpha + beta))
    Dstar = C_B / (a * Nstar)
    # derivative of L(N) wrt N vanishes at N*
    L = lambda n: predict_l0(theta, n, C_B / (a * n))
    eps = 1e-5
    d = (L(Nstar * (1 + eps)) - L(Nstar * (1 - eps))) / (2 * Nstar * eps)
    assert abs(d) < 1e-6
    assert Dstar * a * Nstar == pytest.approx(C_B, rel=1e-12)


# ------------------------------------------------------------------ Q1 predictor protocol
def test_q1_predictor_protocol_reproducible():
    mp = json.loads((Q1 / "mixture_predictor.json").read_text(encoding="utf-8"))
    p0 = json.loads((Q1 / "p0.json").read_text(encoding="utf-8"))
    model = mp["model"]
    full = dict(zip(p0["p0_order"], p0["p0"]))
    # documented protocol: drop the reference domain, do NOT renormalise the 16-vector
    x = np.array([full[c.replace("train_the_pile_", "")] for c in model["feature_columns"]])
    assert np.allclose(x, np.array(model["feat_mean"]), atol=1e-12)  # p0 is the training mean
    z = (x - np.array(model["feat_mean"])) / np.array(model["feat_std"])
    assert np.allclose(z, 0.0)
    pred = z @ np.array(model["coef"]) + np.array(model["intercept"])
    ev = float(np.array(mp["eval_weights"]) @ pred)
    assert ev == pytest.approx(float(np.mean(np.array(model["intercept"]))), abs=1e-12)
    lp = json.loads((Q2 / "loss_predictor.json").read_text(encoding="utf-8"))
    assert ev == pytest.approx(lp["f_p0_eval_loss"], abs=1e-9)
    assert len(mp["eval_weights"]) == 13


def test_q1_evaluation_uses_held_out_sets():
    m = json.loads((Q1 / "q1_metrics.json").read_text(encoding="utf-8"))
    assert m["sets"]["test_1m"]["nature"] == "observed"
    assert m["sets"]["est_10b"]["nature"] == "extrapolated"
    # honest reporting: cross-scale R2 is not silently dropped
    assert "r2" in m["sets"]["test_1B"]["eval_loss"]


# ------------------------------------------------------------------ Q3 optimisation
def test_gamma_sign_and_b8_in_support_diagnostic():
    g = json.loads((Q2 / "gamma_estimates.json").read_text(encoding="utf-8"))
    assert g["B6_intercept_False"]["gamma"] > 0
    assert g["B7_intercept_False"]["gamma"] > 0
    # B8's Q-loss relation is inverted even inside B1 support: keep it as a negative
    # cross-source diagnostic, never pool it into gamma.
    assert g["B8_intercept_False"]["gamma"] < 0
    assert g["B8_intercept_False"]["n_in_B1_support"] > 0
    assert g["B8_intercept_False"]["residual_corr_1_minus_Q_in_B1_support"] < 0
    assert g["B6_intercept_False"]["residual_corr_1_minus_Q_in_B1_support"] > 0


def test_q3_cost_constraint_and_analytic():
    scen = pd.read_csv(Q3 / "optimal_allocations.csv")
    assert abs(scen["cost_constraint_residual"]).max() < 1e-9
    assert np.allclose(scen["train_share"] + scen["attention_share"] + scen["quality_share"], 1.0)
    assert np.allclose(scen["train_share"], 6.0 / (6.0 + 2e-4 * scen["ell"]))
    assert scen["rel_diff_L"].max() < 1e-8
    assert (scen["unused_share"] >= 0).all()


def test_q3_extrapolation_flags_consistent():
    scen = pd.read_csv(Q3 / "optimal_allocations.csv")
    for _, r in scen.iterrows():
        expected = not (0.070542 <= r["N_num_B"] <= 11.965825 and 0.134 <= r["D_num_B"] <= 299.893)
        assert bool(r["extrapolation"]) == expected
