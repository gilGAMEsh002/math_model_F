"""Q1-A quality scoring: scalarisation, frozen normalisation, conflict resolution.

All 22 metrics are mapped to [0,1] with higher = better. Normalisation is estimated
on the reference set A1 only, then frozen and applied to A2/A3. See
configs/shared/quality_metrics.json for the frozen dictionary.
"""
from __future__ import annotations

import numpy as np

from .metrics import softmax

_ALREADY_UNIT = {"ordinal_level", "softmax_index"}
_PERCENTILE_KINDS = {"scalar", "element0", "zmean"}
SUITABLE_KNOTS = (5, 25, 75, 95)


def _raw_scalarize(value, spec, qurater_stats):
    """Scalarise one record's raw value before percentile normalisation.

    Returns np.nan for missing values, or a value already in [0,1] for
    ordinal_level / softmax_index kinds.
    """
    kind = spec["kind"]
    if value is None:
        return np.nan
    try:
        if kind == "scalar":
            return float(value)
        if kind == "element0":
            arr = np.asarray(value, dtype=float).ravel()
            return float(arr[0]) if arr.size else np.nan
        if kind == "softmax_index":
            if len(value) <= spec["positive_index"]:
                return np.nan
            return float(softmax(value)[spec["positive_index"]])
        if kind == "ordinal_level":
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 0:
                return np.nan
            return float(np.argmax(arr)) / (arr.size - 1)
        if kind == "zmean":
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 0 or qurater_stats is None:
                return np.nan
            mean, std = qurater_stats
            z = (arr - mean) / std
            return float(np.mean(z))
    except (TypeError, ValueError):
        return np.nan
    raise ValueError(f"unknown metric kind: {kind}")


def _percentile_knots(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"p05": 0.0, "p25": 0.0, "p50": 0.5, "p75": 1.0, "p95": 1.0}
    return {k: float(np.percentile(v, q)) for k, q in
            (("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))}


def fit_calibration(records, metrics) -> dict:
    """Estimate frozen normalisation parameters on the reference set."""
    # qurater sub-dimension statistics (each standardised then averaged).
    qurater = next((m for m in metrics if m["kind"] == "zmean"), None)
    qurater_stats = None
    if qurater is not None:
        cols = [[] for _ in range(qurater.get("subdims", 4))]
        for r in records:
            v = r.get(qurater["name"])
            if v is None:
                continue
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size != len(cols):
                continue
            for i, x in enumerate(arr):
                cols[i].append(x)
        means = np.array([np.mean(c) if c else 0.0 for c in cols], dtype=float)
        stds = np.array([np.std(c) if c else 1.0 for c in cols], dtype=float)
        stds[stds == 0] = 1.0
        qurater_stats = (means, stds)

    collected = {m["name"]: [] for m in metrics}
    missing = {m["name"]: 0 for m in metrics}
    for r in records:
        for m in metrics:
            name = m["name"]
            if name not in r or r[name] is None:
                missing[name] += 1
                continue
            x = _raw_scalarize(r.get(name), m, qurater_stats)
            if np.isfinite(x):
                collected[name].append(x)
            else:
                missing[name] += 1

    calib = {
        "qurater_stats": None if qurater_stats is None else {
            "mean": qurater_stats[0].tolist(), "std": qurater_stats[1].tolist()
        },
        "metrics": {},
    }
    n = len(records)
    for m in metrics:
        name = m["name"]
        vals = np.asarray(collected[name], dtype=float)
        entry = {
            "kind": m["kind"],
            "direction": m["direction"],
            "already_unit": m["kind"] in _ALREADY_UNIT,
            "n_observed": int(vals.size),
            "missing_rate": float(missing[name] / n) if n else float("nan"),
        }
        entry.update(_percentile_knots(vals))
        # Expected utility under the reference distribution, used to impute missing
        # values consistently across metric kinds (never rewarded or penalised).
        if entry["already_unit"]:
            normed = np.clip(vals, 0.0, 1.0)
        else:
            normed = np.array([_normalise(v, entry["direction"], entry) for v in vals], dtype=float)
        entry["fill_value"] = float(normed.mean()) if normed.size else 0.5
        calib["metrics"][name] = entry
    return calib


def _normalise(x: float, direction: str, knots: dict) -> float:
    p05, p25, p50, p75, p95 = (knots["p05"], knots["p25"], knots["p50"], knots["p75"], knots["p95"])
    if not np.isfinite(x):
        return 0.5
    if direction == "increasing":
        if p95 <= p05:
            return 0.5
        return float(np.clip((x - p05) / (p95 - p05), 0.0, 1.0))
    if direction == "decreasing":
        if p95 <= p05:
            return 0.5
        return float(1.0 - np.clip((x - p05) / (p95 - p05), 0.0, 1.0))
    if direction == "suitable_interval":
        if p05 == p25 == p75 == p95:
            return 0.5
        return float(np.interp(x, [p05, p25, p75, p95], [0.0, 1.0, 1.0, 0.0]))
    raise ValueError(f"unknown direction: {direction}")


def score_record(record, metrics, calib) -> dict:
    """Return the 22 normalised dimensions plus group scores for one record."""
    qs = calib.get("qurater_stats")
    qurater_stats = None
    if qs is not None:
        qurater_stats = (np.asarray(qs["mean"], dtype=float), np.asarray(qs["std"], dtype=float))

    dims = {}
    for m in metrics:
        name = m["name"]
        entry = calib["metrics"][name]
        x = _raw_scalarize(record.get(name), m, qurater_stats)
        if not np.isfinite(x):
            dims[name] = float(entry.get("fill_value", 0.5))
        elif entry["already_unit"]:
            dims[name] = float(np.clip(x, 0.0, 1.0))
        else:
            dims[name] = _normalise(x, entry["direction"], entry)
    return dims


def aggregate(dims: dict, metrics, groups) -> dict:
    """Aggregate 22 dimensions into Q_base, group scores, conflict and min-group."""
    values = np.array([dims[m["name"]] for m in metrics], dtype=float)
    q_base = float(values.mean())
    group_scores = {}
    for g in groups:
        members = [m["name"] for m in metrics if m["group"] == g]
        group_scores[g] = float(np.mean([dims[n] for n in members])) if members else float("nan")
    gvals = np.array([group_scores[g] for g in groups], dtype=float)
    conflict = float(np.nanmax(gvals) - np.nanmin(gvals))
    out = {"Q_base": q_base, "conflict": conflict, "min_group": float(np.nanmin(gvals))}
    out.update({f"z_{g}": group_scores[g] for g in groups})
    return out


def resolve(q_base: float, min_group: float, lam: float) -> float:
    return float((1.0 - lam) * q_base + lam * min_group)
