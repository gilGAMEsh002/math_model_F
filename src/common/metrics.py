"""Evaluation metrics used across Baseline A."""
from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size < 3:
        return float("nan")
    rt, rp = _rankdata(yt), _rankdata(yp)
    rt = rt - rt.mean()
    rp = rp - rp.mean()
    denom = np.sqrt(np.sum(rt ** 2) * np.sum(rp ** 2))
    return float(np.sum(rt * rp) / denom) if denom > 0 else float("nan")


def topk_regret(y_true, y_pred, k: int = 3) -> dict:
    """Regret of the predicted-best candidates: true loss of predicted top-k minus true best.

    Lower is better. Reported as the average over the top-k selected candidates.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return {"k": k, "mean_regret": float("nan"), "best_regret": float("nan"), "best_is_true_best": False}
    k = min(k, yt.size)
    sel = np.argsort(yp)[:k]
    best = float(np.min(yt))
    regrets = yt[sel] - best
    return {
        "k": int(k),
        "mean_regret": float(np.mean(regrets)),
        "best_regret": float(np.min(regrets)),
        "best_is_true_best": bool(np.isclose(np.min(regrets), 0.0)),
    }


def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                      chunk: int = 100) -> tuple:
    """Percentile bootstrap CI for the mean, computed in chunks to bound memory."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, v.size, size=(m, v.size))
        means[done:done + m] = v[idx].mean(axis=1)
        done += m
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()
