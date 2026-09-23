"""Classical and quality-augmented scaling-law utilities for Baseline A (Q2-A).

N and D are used in billions (N_params_B, D_tokens_B) for numerical conditioning;
the unit metadata is stored with every fitted parameter set. Cost calculation in
Q3 converts these coefficients back to actual parameter/token counts.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# parameter vector: [E, logA, alpha, logB, beta]
BOUNDS_LOWER = np.array([0.0, -40.0, 1e-3, -40.0, 1e-3])
BOUNDS_UPPER = np.array([60.0, 40.0, 3.0, 40.0, 3.0])


def predict_l0(theta, n, d):
    E, logA, alpha, logB, beta = theta
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    return E + np.exp(logA) * n ** (-alpha) + np.exp(logB) * d ** (-beta)


def _residuals(theta, n, d, y, w=None):
    r = predict_l0(theta, n, d) - y
    if w is not None:
        r = r * w
    return r


def fit_classical(n, d, y, n_starts=40, seed=0, max_nfev=20000):
    """Multi-start bounded nonlinear least squares for L = E + A n^-a + B d^-b."""
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    starts = [np.array([min(y), -2.0, 0.1, -2.0, 0.1])]
    for _ in range(n_starts):
        starts.append(np.array([
            rng.uniform(0.0, max(0.5, float(y.min()) * 1.2)),
            rng.uniform(-6, 6), rng.uniform(0.02, 1.0),
            rng.uniform(-6, 6), rng.uniform(0.02, 1.0)]))
    for x0 in starts:
        x0 = np.clip(x0, BOUNDS_LOWER + 1e-9, BOUNDS_UPPER - 1e-9)
        try:
            res = least_squares(_residuals, x0, args=(n, d, y), bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
                                method="trf", max_nfev=max_nfev)
        except Exception:
            continue
        cost = float(np.sum(res.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, res.x, res)
    if best is None:
        raise RuntimeError("classical scaling fit failed for all starts")
    cost, theta, res = best
    pred = predict_l0(theta, n, d)
    r = y - pred
    rmse = float(np.sqrt(np.mean(r ** 2)))
    # boundary-hit flags
    at_bound = [bool(np.isclose(theta[i], BOUNDS_LOWER[i], rtol=1e-6) or
                     np.isclose(theta[i], BOUNDS_UPPER[i], rtol=1e-6))
                for i in range(len(theta))]
    return {
        "theta": theta.tolist(),
        "E": float(theta[0]), "A": float(np.exp(theta[1])), "alpha": float(theta[2]),
        "B": float(np.exp(theta[3])), "beta": float(theta[4]),
        "cost": cost, "rmse": rmse, "n_points": int(len(y)),
        "n_starts_used": len(starts), "at_bound": at_bound,
    }


def fit_gamma(n, d, q, y, theta, w=None, allow_intercept=False):
    """Estimate additive quality term: L = L0(N,D) + gamma (1-Q) [+ c]."""
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    base = predict_l0(theta, n, d)
    resid = y - base
    x = 1.0 - q
    if allow_intercept:
        X = np.column_stack([x, np.ones_like(x)])
    else:
        X = x[:, None]
    if w is not None:
        Xw = X * np.asarray(w)[:, None]
    else:
        Xw = X
    coef, *_ = np.linalg.lstsq(Xw, resid * (np.asarray(w) if w is not None else 1.0), rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((resid - pred) ** 2))
    ss_tot = float(np.sum((resid - resid.mean()) ** 2))
    return {
        "gamma": float(coef[0]),
        "intercept": float(coef[1]) if allow_intercept else 0.0,
        "rmse": float(np.sqrt(np.mean((resid - pred) ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "residual_std": float(np.std(resid)),
        "n_points": int(len(y)),
    }


def elasticities(theta, n, d, y_true=None):
    """Elasticity of L wrt N and D for the classical part; Q handled separately."""
    E, logA, alpha, logB, beta = theta
    A, B = np.exp(logA), np.exp(logB)
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    L = predict_l0(theta, n, d)
    dL_dN = -alpha * A * n ** (-alpha - 1)
    dL_dD = -beta * B * d ** (-beta - 1)
    return {
        "epsilon_N": dL_dN * n / L,
        "epsilon_D": dL_dD * d / L,
        "dL_dN": dL_dN,
        "dL_dD": dL_dD,
        "predicted_loss": L,
    }


def finite_difference_check(theta, n, d, rel=1e-6):
    """Relative error of analytic vs central finite-difference derivatives."""
    out = {}
    for name, idx in (("N", 0), ("D", 1)):
        x = n if name == "N" else d
        h = x * rel
        f_plus = predict_l0(theta, n + h if name == "N" else n, d + h if name == "D" else d)
        f_minus = predict_l0(theta, n - h if name == "N" else n, d - h if name == "D" else d)
        fd = (f_plus - f_minus) / (2 * h)
        an = elasticities(theta, n, d)["dL_dN" if name == "N" else "dL_dD"]
        out[f"dL_d{name}_analytic"] = float(an)
        out[f"dL_d{name}_finite_diff"] = float(fd)
        out[f"rel_error_{name}"] = float(abs(an - fd) / (abs(fd) + 1e-30))
    return out


def n_equivalent(n, delta_loss, theta):
    """Parameters needed to match a quality gain dL at fixed D: N_eq = (N^-a - dL/A)^(-1/a)."""
    E, logA, alpha, logB, beta = theta
    A = np.exp(logA)
    scalar = np.ndim(n) == 0 and np.ndim(delta_loss) == 0
    n = np.asarray(n, dtype=float)
    delta_loss = np.asarray(delta_loss, dtype=float)
    inside = n ** (-alpha) - delta_loss / A
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eq = np.where(inside > 0, inside ** (-1.0 / alpha), np.nan)
    status = np.where(inside > 0, "finite", np.where(np.isclose(inside, 0), "infinite", "not_reachable"))
    if scalar:
        n_eq = n_eq.item()
        status = status.item()
        n = n.item()
    return {"N_eq": n_eq, "N_eq_over_N_minus_1": np.asarray(n_eq) / np.asarray(n) - 1.0, "status": status}
