"""Q1-A: equal-weight quality scoring, conflict resolution, domain mixture ridge regression.

Runs the full Baseline-A first question:
  1. stream A1-A3, scalarise 22 metrics, fit frozen normalisation on A1;
  2. score every record, deduplicate by (quality_domain, id), aggregate per domain;
  3. report conflict statistics and a lambda grid for the weakest-group rule;
  4. fit a 16-dim + intercept ridge regression per validation domain on A4/A5,
     select the penalty by internal CV, and evaluate on A6-A11 (1M/60M/1B) and
     A12-A15 (10B/70B, extrapolated) with the equal-weight evaluation loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import REPO, iter_jsonl_xz, load_config, read_json, sha256_file, stable_hash, write_json, run_metadata
from src.common.quality import aggregate, fit_calibration, resolve, score_record
from src.common.metrics import bootstrap_mean_ci, mae, r2, rmse, spearman, topk_regret

RA = REPO / "real_attachments"
QUALITY_FILES = [
    ("A1", "A_data_value/slimpajama_quality_signal_sample.jsonl.xz", None),
    ("A2", "A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz", "arxiv"),
    ("A3", "A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz", "github"),
]


# --------------------------------------------------------------------------- quality
def _strip_content(rec):
    return {k: v for k, v in rec.items() if k != "content"}


def fit_calibration_on_A1(metrics):
    """Stream A1 once, keeping only the 22 metric fields, and fit the frozen params."""
    tag, rel, hint = QUALITY_FILES[0]
    records = [_strip_content(rec) for rec in iter_jsonl_xz(RA / rel)]
    return fit_calibration(records, metrics), len(records)


def _score_all(metrics, groups, calib):
    """Stream A1>A2>A3 in priority order, dedup by (domain,id), score on the fly."""
    seen = set()
    acc = {k: [] for k in ("domain", "set", "id", "Q_base", "min_group", "conflict", "nmiss")}
    acc.update({f"z_{g}": [] for g in groups})
    for tag, rel, hint in QUALITY_FILES:
        for rec in iter_jsonl_xz(RA / rel):
            dom = rec.get("_source_domain") or hint
            key = (dom, rec.get("id"))
            if key in seen:
                continue
            seen.add(key)
            dims = score_record(_strip_content(rec), metrics, calib)
            agg = aggregate(dims, metrics, groups)
            acc["domain"].append(dom)
            acc["set"].append(tag)
            acc["id"].append(rec.get("id"))
            acc["Q_base"].append(agg["Q_base"])
            acc["min_group"].append(agg["min_group"])
            acc["conflict"].append(agg["conflict"])
            acc["nmiss"].append(sum(1 for m in metrics if rec.get(m["name"]) is None))
            for g in groups:
                acc[f"z_{g}"].append(agg[f"z_{g}"])
    arr = {
        "domain": np.array(acc["domain"], dtype=object), "set": np.array(acc["set"], dtype=object),
        "id": np.array(acc["id"], dtype=object), "Q_base": np.array(acc["Q_base"]),
        "min_group": np.array(acc["min_group"]), "conflict": np.array(acc["conflict"]),
        "nmiss": np.array(acc["nmiss"]),
    }
    for g in groups:
        arr[f"z_{g}"] = np.array(acc[f"z_{g}"])
    return arr


def _domain_table(arr, groups, lambdas, tau, seed, bootstrap_n, lam_selected):
    df = pd.DataFrame({k: arr[k] for k in ("domain", "set", "Q_base", "min_group", "conflict", "nmiss")})
    for g in groups:
        df[f"z_{g}"] = arr[f"z_{g}"]
    rows = []
    for dom, sub in df.groupby("domain"):
        for setname, s in [("A1", sub[sub["set"] == "A1"]),
                           ("A2", sub[sub["set"] == "A2"]),
                           ("A3", sub[sub["set"] == "A3"]),
                           (None, sub)]:
            if setname is not None and len(s) == 0:
                continue
            label = setname or "merged"
            row = {
                "quality_domain": dom, "set": label, "n": int(len(s)),
                "Q_base_mean": float(s["Q_base"].mean()),
                "Q_base_median": float(s["Q_base"].median()),
                "Q_base_std": float(s["Q_base"].std()),
                "conflict_mean": float(s["conflict"].mean()),
                "conflict_rate": float((s["conflict"] > tau).mean()),
                "min_group_mean": float(s["min_group"].mean()),
                "missing_metric_rate": float(s["nmiss"].mean() / 22.0),
            }
            for g in groups:
                row[f"z_{g}_mean"] = float(s[f"z_{g}"].mean())
            qb = s["Q_base"].values
            mg = s["min_group"].values
            for lam in lambdas:
                row[f"Q_resolved_lambda_{lam}"] = float(np.mean((1 - lam) * qb + lam * mg))
            qres = (1 - lam_selected) * qb + lam_selected * mg
            row["Q_resolved_selected"] = float(qres.mean())
            row["lambda_selected"] = lam_selected
            mean, lo, hi = bootstrap_mean_ci(qres, n_boot=bootstrap_n, seed=seed)
            row["Q_resolved_ci_low"] = lo
            row["Q_resolved_ci_high"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def _select_lambda(arr, lambdas, reserved_fraction, seed, pool_mask):
    """Reserved-subset lambda selection, restricted to the given pool (A1 only)."""
    domain = arr["domain"][pool_mask]
    qbase_all = arr["Q_base"][pool_mask]
    ming_all = arr["min_group"][pool_mask]
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(domain), dtype=bool)
    for d in np.unique(domain):
        idx = np.where(domain == d)[0]
        n_res = max(1, int(round(reserved_fraction * len(idx))))
        mask[rng.choice(idx, size=n_res, replace=False)] = True
    qb, mg = qbase_all[mask], ming_all[mask]
    dom = domain[mask]
    base_by_dom = pd.Series(qb).groupby(dom).mean()
    grid = []
    for lam in lambdas:
        q_lam = (1 - lam) * qb + lam * mg
        res_by_dom = pd.Series(q_lam).groupby(dom).mean()
        rho = spearman(base_by_dom.reindex(res_by_dom.index).values, res_by_dom.values)
        grid.append({"lambda": lam, "domain_rank_spearman_vs_base": rho,
                     "mean_q_shrinkage": float(np.mean(qb - q_lam)),
                     "reserved_n": int(mask.sum())})
    selected = 0.0
    for row in grid:
        if row["lambda"] > 0 and row["domain_rank_spearman_vs_base"] >= 0.99:
            selected = row["lambda"]
            break
    return grid, selected, int(mask.sum())


# --------------------------------------------------------------------------- mixtures
def _load_mixture_pair(mixture_rel, loss_rel):
    m = pd.read_csv(RA / mixture_rel)
    l = pd.read_csv(RA / loss_rel)
    assert (m["index"].values == l["index"].values).all(), "index mismatch"
    pcols = [c for c in m.columns if c.startswith("train_the_pile_")]
    lcols = [c for c in l.columns if c.startswith("metric/")]
    return m["index"].values, m[pcols], l[lcols]


def _ridge_fit(Xs, yc, lam):
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ yc)
    return w


def _ridge_cv(X, Y, penalty_grid, folds, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[idx] = np.arange(n) % folds
    norm = Y.std(axis=0)
    norm[norm == 0] = 1.0
    scores = {lam: [] for lam in penalty_grid}
    for f in range(folds):
        val = idx[fold_ids[idx] == f]
        tr = idx[fold_ids[idx] != f]
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X[tr] - mu) / sd
        ybar = Y[tr].mean(axis=0)
        for lam in penalty_grid:
            w = _ridge_fit(Xs, Y[tr] - ybar, lam)
            pred = ((X[val] - mu) / sd) @ w + ybar
            scores[lam].append(np.mean(np.sqrt(np.mean((Y[val] - pred) ** 2, axis=0)) / norm))
    return {lam: float(np.mean(v)) for lam, v in scores.items()}


def _predict(model, X):
    return ((X - model["feat_mean"]) / model["feat_std"]) @ model["coef"] + model["intercept"]


def _eval_block(y_true, y_pred, targets, weights):
    out = {}
    for i, t in enumerate(targets):
        out[t] = {"rmse": rmse(y_true[:, i], y_pred[:, i]),
                  "mae": mae(y_true[:, i], y_pred[:, i]),
                  "r2": r2(y_true[:, i], y_pred[:, i]),
                  "spearman": spearman(y_true[:, i], y_pred[:, i])}
    ev_t = y_true @ weights
    ev_p = y_pred @ weights
    out["eval_loss"] = {"rmse": rmse(ev_t, ev_p), "mae": mae(ev_t, ev_p),
                        "r2": r2(ev_t, ev_p), "spearman": spearman(ev_t, ev_p)}
    for k in (1, 3, 5):
        out["eval_loss"][f"top{k}_regret"] = topk_regret(ev_t, ev_p, k)["mean_regret"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/baselines/Q1-A.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    outdir = REPO / cfg["paths"]["out"]
    outdir.mkdir(parents=True, exist_ok=True)

    qcfg = read_json(REPO / cfg["paths"]["shared_quality"])
    dcfg = read_json(REPO / cfg["paths"]["shared_domains"])
    metrics = qcfg["metrics"]
    groups = qcfg["groups"]
    lambdas = list(cfg["quality"]["lambda_grid"])

    calib, n_ref = fit_calibration_on_A1(metrics)
    write_json(outdir / "quality_preprocessing.json", {
        "version": qcfg["version"], "reference_set": "A1", "n_reference": n_ref,
        "calibration": calib,
        "reserved_fraction": cfg["quality"]["reserved_fraction"],
        "paths": {tag: rel for tag, rel, _ in QUALITY_FILES},
        "input_hashes": {tag: sha256_file(RA / rel) for tag, rel, _ in QUALITY_FILES},
    })

    arr = _score_all(metrics, groups, calib)
    a1_mask = arr["set"] == "A1"
    conflicts = arr["conflict"]
    # tau and the lambda reserved subset are frozen on the A1 reference/development
    # sample only, so A2/A3 remain an independent migration check.
    tau = float(np.quantile(conflicts[a1_mask], cfg["quality"]["conflict_quantile"]))
    grid, lam_selected, n_reserved = _select_lambda(
        arr, lambdas, cfg["quality"]["reserved_fraction"], seed, a1_mask)

    domain_df = _domain_table(arr, groups, lambdas, tau, seed, cfg["quality"]["bootstrap_n"], lam_selected)
    domain_df.to_csv(outdir / "domain_quality.csv", index=False)

    # conflict audit: threshold sensitivity + specific conflict types (A1 pool)
    conflict_audit = {"tau": tau, "quantile": cfg["quality"]["conflict_quantile"],
                      "pool": "A1", "lambda_grid": grid, "lambda_selected": lam_selected,
                      "reserved_n": n_reserved, "tau_sensitivity": {}}
    for q in cfg["quality"]["conflict_quantile_sensitivity"]:
        t = float(np.quantile(conflicts[a1_mask], q))
        conflict_audit["tau_sensitivity"][str(q)] = {
            "tau": t,
            "conflict_rate_A1": float((conflicts[a1_mask] > t).mean()),
            "conflict_rate_all_sets": float((conflicts > t).mean()),
            "by_domain_A1": {d: float((conflicts[a1_mask & (arr["domain"] == d)] > t).mean())
                             for d in sorted(set(arr["domain"][a1_mask]))},
        }
    types = []
    edu = arr["z_educational"][a1_mask]
    ad = arr["z_alignment"][a1_mask]
    fmt = arr["z_length_format"][a1_mask]
    rel = arr["z_relevance"][a1_mask]
    for name, m in [("edu_high_ad_low", (edu > np.quantile(edu, 0.75)) & (ad < np.quantile(ad, 0.25))),
                    ("format_low", fmt <= np.quantile(fmt, 0.10)),
                    ("relevance_low", rel <= np.quantile(rel, 0.10))]:
        types.append({"type": name, "count": int(m.sum()), "rate": float(m.mean())})
    conflict_audit["conflict_types"] = types
    conflict_audit["n_dedup"] = int(len(arr["domain"]))
    conflict_audit["n_A1"] = int(a1_mask.sum())
    write_json(outdir / "conflict_audit.json", conflict_audit)
    pd.DataFrame([{"tau_quantile": k, **v} for k, v in conflict_audit["tau_sensitivity"].items()],
                 ).to_csv(outdir / "conflict_audit.csv", index=False)

    # domain mapping + Q0 (proxy for inferred domains computed first)
    merged = domain_df[domain_df["set"] == "merged"]
    proxy = float(np.average(merged["Q_resolved_selected"], weights=merged["n"]))
    mapping_rows = []
    for dom in dcfg["mixture_domains"]:
        mp = dcfg["mapping"][dom]
        qd = mp["quality_domain"]
        if qd is not None:
            row = domain_df[(domain_df["quality_domain"] == qd) & (domain_df["set"] == "merged")]
            q_val = float(row["Q_resolved_selected"].iloc[0]) if len(row) else float("nan")
            n_val = int(row["n"].iloc[0]) if len(row) else 0
        else:
            # inferred: no direct quality domain; use documented corpus-wide proxy
            q_val, n_val = proxy, 0
        mapping_rows.append({"mixture_domain": dom, "quality_domain": qd or "",
                             "mapping_type": mp["mapping_type"], "Q_resolved": q_val,
                             "n_quality": n_val, "is_proxy": qd is None})
    write_json(outdir / "domain_mapping.json", {
        "reference_domain": dcfg["reference_domain"],
        "proxy_rule_inferred": dcfg["proxy_rule_inferred"],
        "proxy_Q": proxy,
        "lambda_selected": lam_selected,
        "rows": mapping_rows,
    })
    pd.DataFrame(mapping_rows).to_csv(outdir / "domain_mapping.csv", index=False)

    # ------------------------------------------------------------------ ridge
    splits = read_json(REPO / cfg["paths"]["shared_splits"])
    ref_dom = dcfg["reference_domain"]
    idx_tr, P_tr, L_tr = _load_mixture_pair(splits["mixture"]["train"]["mixture"],
                                            splits["mixture"]["train"]["loss"])
    pcols_all = list(P_tr.columns)
    feat_cols = [c for c in pcols_all if c != f"train_the_pile_{ref_dom}"]
    targets = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in L_tr.columns]
    # row-normalise proportions
    Pn_tr = P_tr.values / P_tr.values.sum(axis=1, keepdims=True)
    Xtr, Ytr = Pn_tr[:, [pcols_all.index(c) for c in feat_cols]], L_tr.values
    cv_scores = _ridge_cv(Xtr, Ytr, cfg["ridge"]["penalty_grid"], cfg["ridge"]["cv_folds"], seed)
    lam_ridge = min(cv_scores, key=cv_scores.get)
    feat_mean = Xtr.mean(axis=0)
    feat_std = Xtr.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    Xs = (Xtr - feat_mean) / feat_std
    intercept = Ytr.mean(axis=0)
    coef = _ridge_fit(Xs, Ytr - intercept, lam_ridge)

    model = {"feature_columns": feat_cols, "reference_domain": ref_dom, "targets": targets,
             "coef": coef.tolist(), "intercept": intercept.tolist(),
             "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
             "ridge_lambda": lam_ridge, "cv_scores": cv_scores}
    # full-row-sum normalisation constants for external calls: features are renormalised
    p0 = Pn_tr.mean(axis=0)
    weights = np.full(len(targets), 1.0 / len(targets))

    eval_rows = []
    metrics_out = {"ridge_lambda": lam_ridge, "cv_scores": cv_scores,
                   "reference_domain": ref_dom, "eval_weights": "equal", "sets": {}}
    for tag, s in splits["mixture"].items():
        idx_t, P_t, L_t = _load_mixture_pair(s["mixture"], s["loss"])
        Pn_t = P_t.values / P_t.values.sum(axis=1, keepdims=True)
        Xt = Pn_t[:, [pcols_all.index(c) for c in feat_cols]]
        pred = _predict(model, Xt)
        metrics_out["sets"][tag] = _eval_block(L_t.values, pred, targets, weights)
        metrics_out["sets"][tag]["train_mean_reference"] = _eval_block(
            L_t.values, np.tile(intercept, (len(L_t), 1)), targets, weights)
        metrics_out["sets"][tag]["nature"] = s["nature"]
        block = pd.DataFrame({"set": tag, "index": idx_t, "predicted_eval_loss": pred @ weights,
                              "actual_eval_loss": L_t.values @ weights})
        for i, t in enumerate(targets):
            block[f"pred_{t}"] = pred[:, i]
            block[f"actual_{t}"] = L_t.values[:, i]
        eval_rows.append(block)

    pd.concat(eval_rows, ignore_index=True).to_csv(outdir / "mixture_predictions.csv", index=False)
    write_json(outdir / "q1_metrics.json", metrics_out)

    # p0 / Q0
    q_mapping = {r["mixture_domain"]: r["Q_resolved"] for r in mapping_rows}
    dom_order = dcfg["mixture_domains"]
    p0_map = dict(zip([c.replace("train_the_pile_", "") for c in pcols_all], p0))
    inferred_used = [d for d in dom_order if dcfg["mapping"][d]["quality_domain"] is None]
    q0_terms = []
    for dom in dom_order:
        qv = q_mapping.get(dom, float("nan"))
        if not np.isfinite(qv):
            qv = proxy
        q0_terms.append(p0_map[dom] * qv)
    Q0 = float(np.sum(q0_terms))
    dn = [d for d in dom_order if not dcfg["mapping"][d]["quality_domain"] is None]
    w_dn = np.array([p0_map[d] for d in dn])
    Q0_dn = float(np.sum([p0_map[d] * q_mapping[d] for d in dn]) / w_dn.sum()) if w_dn.sum() > 0 else float("nan")
    write_json(outdir / "p0.json", {
        "p0_order": dom_order, "p0": [float(p0_map[d]) for d in dom_order],
        "p0_source": "A4 训练配比行归一化后的均值（各行列和≈1，均值列和≈1）",
        "Q0": Q0,
        "Q0_definition": "sum_i p0_i * Q_domain(i)，inferred 域用全质量域样本量加权均值代理",
        "inferred_domains_using_proxy": inferred_used,
        "proxy_Q": proxy,
        "Q0_direct_near_renormalised": Q0_dn,
        "domain_Q_source": "artifacts/baselines/Q1-A/domain_quality.csv (set=merged, Q_resolved_selected)",
    })

    # quality metric dictionary + per-record quality scores (deliverables)
    pd.DataFrame(metrics).to_csv(outdir / "quality_metric_dictionary.csv", index=False)
    qs = pd.DataFrame({
        "quality_domain": arr["domain"], "set": arr["set"], "id": arr["id"],
        "Q_base": arr["Q_base"], "min_group": arr["min_group"], "conflict": arr["conflict"],
        "Q_resolved_selected": (1 - lam_selected) * arr["Q_base"] + lam_selected * arr["min_group"],
        "n_missing_metrics": arr["nmiss"],
    })
    for g in groups:
        qs[f"z_{g}"] = arr[f"z_{g}"]
    qs.to_parquet(outdir / "quality_scores.parquet", index=False)

    input_protocol = {
        "description": "给定 17 维配比向量 p（顺序见 p0_order），按下述步骤得到 13 个 Loss 预测。",
        "steps": [
            "x_full = p / sum(p)  # 行归一化到和为 1",
            "x = [x_full[i] for i, name in enumerate(p0_order) if name != reference_domain]  # 按 feature_columns 顺序删除参考域",
            "z = (x - feat_mean) / feat_std",
            "pred_targets = z @ coef + intercept  # 长度 13，顺序见 targets",
            "pred_eval_loss = pred_targets @ eval_weights  # 等权 1/13",
        ],
        "warning": "不要在删除参考域后再次对 16 维向量归一化。",
    }
    full = {"model": model, "feature_names": feat_cols, "targets": targets,
            "eval_weights": weights.tolist(), "lambda_grid": lambdas, "lambda_selected": lam_selected,
            "tau": tau, "Q0": Q0, "p0": [float(p0_map[d]) for d in dom_order],
            "p0_order": dom_order, "mapping": mapping_rows,
            "reference_domain": ref_dom, "input_protocol": input_protocol,
            "supported_p_range": {c: [float(v.min()), float(v.max())] for c, v in
                                  zip(feat_cols, Xtr.T)}}
    write_json(outdir / "mixture_predictor.json", full)

    # metadata
    write_json(outdir / "run_metadata.json", run_metadata(seed, cfg, {
        "n_merged_records": int(len(arr["domain"])),
        "ridge_lambda": lam_ridge, "lambda_selected": lam_selected, "tau": tau,
        "input_hashes": {tag: sha256_file(RA / rel) for tag, rel, _ in QUALITY_FILES},
        "output_dir": str(outdir.relative_to(REPO)),
    }))
    print(json.dumps({"lambda_selected": lam_selected, "ridge_lambda": lam_ridge,
                      "Q0": Q0, "n_merged": int(len(arr["domain"])),
                      "cv_scores": cv_scores}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
