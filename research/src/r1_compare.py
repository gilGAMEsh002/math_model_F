"""R1: verifiable comparison of the three repaired lines.

Layer A (end-to-end): Q1 mixture prediction compared on the SAME candidate sets
(A6-A11 / est) with the SAME 13-domain equal weights; per-domain errors, Spearman,
top-1 regret and random-pick expected regret.

Layer B (minimal controlled): Q3 solver/model consistency under a degenerate setting
(quality and mixture channels OFF, same budget/cost) -- the classical model must give
the same optimum across lines before any quality/mixture channel is switched on.

Provenance and caliber caveats are written next to every number. Nothing here picks a
paper main line.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import LineA, LineB, LineC
from .common import RESEARCH, load_lock

SETS = {"test_1m": "1m", "test_60m": "60m", "test_1B": "1B",
        "est_10b": "est_10b", "est_70b": "est_70b"}
C_SETS = {"same_scale_1m": "1m", "cross_scale_60m": "60m", "cross_scale_1B": "1B",
          "est_10b": "est_10b", "est_70b": "est_70b"}
B_SPLITS = set(SETS.keys())


# ------------------------------------------------------------------ metrics
def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy().copy(); rb = pd.Series(b).rank().to_numpy().copy()
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _metrics(Y, Yhat, v):
    y = Y @ v; yh = Yhat @ v
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    mae = float(np.mean(np.abs(y - yh)))
    ss = float(np.sum((y - yh) ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss / st) if st > 0 else float("nan")
    k = int(np.argmin(yh))
    top1 = float(y[k] - y.min())
    rnd = float(y.mean() - y.min())
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": _spearman(y, yh),
            "top1_regret": top1, "random_expected_regret": rnd,
            "top1_is_oracle": bool(abs(top1) < 1e-12)}


# ------------------------------------------------------------------ ingestion
def load_a() -> dict:
    A = LineA()
    df = pd.read_csv(A.root / "artifacts/baselines/Q1-A/mixture_predictions.csv")
    out = {}
    for setcol, key in SETS.items():
        d = df[df["set"] == setcol]
        if d.empty:
            continue
        doms = [c[len("pred_"):] for c in df.columns if c.startswith("pred_")]
        out[key] = {"index": d["index"].to_numpy(),
                    "domains": doms,
                    "Y": d[[f"actual_{t}" for t in doms]].to_numpy(float),
                    "Yhat": d[[f"pred_{t}" for t in doms]].to_numpy(float),
                    "provenance": f"A baseline-a {load_lock()['lines']['A']['commit'][:8]} ridge, calibre=A own"}
    return out


def load_c(variant: str = "best_entropy_proxyQ") -> dict:
    C = LineC()
    df = pd.read_csv(C.root / "artifacts/baselines/Q1-C/mixture_predictions.csv")
    df = df[(df["variant"] == variant) & (df["target"] != "__composite_equal_v__")]
    out = {}
    for dscol, key in C_SETS.items():
        d = df[df["dataset"] == dscol]
        if d.empty:
            continue
        piv = d.pivot_table(index="row_index", columns="target", values=["y_true", "y_pred"],
                            aggfunc="first")
        doms = list(piv["y_true"].columns)
        out[key] = {"index": piv.index.to_numpy(),
                    "domains": doms,
                    "Y": piv["y_true"].to_numpy(float),
                    "Yhat": piv["y_pred"].to_numpy(float),
                    "provenance": f"C baseline-c {load_lock()['lines']['C']['commit'][:8]} forest({variant})"}
    return out


def load_b() -> dict:
    """Reconstruct B's predictions from its persisted ridge coefficients (fit_intercept=False)."""
    B = LineB()
    npz = np.load(B.root / "results/q1_mixture_model.npz", allow_pickle=True)
    coef_quad = npz["coef_quad"]                     # (13, 153)
    from itertools import combinations
    rt = B.root / "real_attachments/A_data_value/regmix_tables"
    groups = {"test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
              "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
              "test_1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
              "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
              "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv")}
    out = {}
    for split, (mf, lf) in groups.items():
        M = pd.read_csv(rt / mf); L = pd.read_csv(rt / lf)
        assert list(M["index"]) == list(L["index"]), split
        P = M.drop(columns=["index"]).to_numpy(float)
        P = P / P.sum(axis=1, keepdims=True)
        inter = np.column_stack([P[:, i] * P[:, j] for i, j in combinations(range(P.shape[1]), 2)])
        inter = inter * P.shape[1]
        X = np.hstack([P, inter])
        Yhat = X @ coef_quad.T
        doms = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in L.columns[1:]]
        out[SETS[split]] = {"index": M["index"].to_numpy(), "domains": doms,
                            "Y": L.drop(columns=["index"]).to_numpy(float),
                            "Yhat": Yhat,
                            "provenance": f"B baseline-b {load_lock()['lines']['B']['commit'][:8]} "
                                          f"quadratic ridge (reconstructed from q1_mixture_model.npz)"}
    return out


# ------------------------------------------------------------------ layer A
def layer_a_q1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    A, B, C = load_a(), load_b(), load_c()
    v = np.full(13, 1.0 / 13)
    rows, perdom, ident = [], [], []
    for key in SETS.values():
        if key not in A or key not in B or key not in C:
            continue
        # align on shared candidate ids and require identical targets
        common = np.intersect1d(np.intersect1d(A[key]["index"], B[key]["index"]), C[key]["index"])
        ia = {x: i for i, x in enumerate(A[key]["index"])}
        ib = {x: i for i, x in enumerate(B[key]["index"])}
        ic = {x: i for i, x in enumerate(C[key]["index"])}
        doms = A[key]["domains"]
        # domain axis must match by name across lines
        same_domains = (B[key]["domains"] == doms == C[key]["domains"]) or \
                       (set(B[key]["domains"]) == set(doms) == set(C[key]["domains"]))
        if not same_domains:
            ident.append({"set": key, "common_n": int(common.size), "y_identical": False,
                          "note": f"domain axis differs: A={doms[:3]} B={B[key]['domains'][:3]} C={C[key]['domains'][:3]}"})
            continue
        # reorder each line's columns to A's domain order
        def reord(obj):
            cols = [obj["domains"].index(t) for t in doms]
            return obj["Y"][:, cols], obj["Yhat"][:, cols]
        Ya, Yha = reord(A[key]); Yb, Yhb = reord(B[key]); Yc, Yhc = reord(C[key])
        idxA = [ia[x] for x in common]; idxB = [ib[x] for x in common]; idxC = [ic[x] for x in common]
        yA, yB, yC = Ya[idxA], Yb[idxB], Yc[idxC]
        ymax = float(np.max(np.abs(yA - yB))) if yA.shape == yB.shape else float("inf")
        ymax = max(ymax, float(np.max(np.abs(yA - yC))))
        ident.append({"set": key, "common_n": int(common.size), "y_true_max_abs_diff": ymax,
                      "same_candidates": bool(ymax < 1e-6),
                      "note": "y_true 一致即同一候选集与同一真实 Loss"})
        if ymax >= 1e-6:
            continue
        for name, Y, Yh in (("A", yA, Yha[idxA]), ("B", yB, Yhb[idxB]), ("C", yC, Yhc[idxC])):
            m = _metrics(Y, Yh, v)
            m.update({"line": name, "set": key, "n": int(common.size),
                      "provenance": {"A": A, "B": B, "C": C}[name][key]["provenance"]})
            rows.append(m)
            for j, t in enumerate(doms):
                e = Y[:, j] - Yh[:, j]
                perdom.append({"line": name, "set": key, "target": t,
                               "rmse": float(np.sqrt(np.mean(e ** 2))),
                               "bias": float(np.mean(e))})
    return pd.DataFrame(rows), pd.DataFrame(perdom), pd.DataFrame(ident)


# ------------------------------------------------------------------ layer B
def _classical_opt(E, A_, alpha, B_, beta, C_B, a, n_lo=1e-4, n_hi=1e4):
    """Grid+local min of E + A n^-a + B (C_B/(a n))^-b over n; returns n*, d*, L*."""
    ns = np.logspace(np.log10(n_lo), np.log10(n_hi), 20000)
    L = E + A_ * ns ** (-alpha) + B_ * (C_B / (a * ns)) ** (-beta)
    i = int(np.argmin(L))
    # local golden refine on log10 n
    lo, hi = np.log10(ns[max(i - 1, 0)]), np.log10(ns[min(i + 1, len(ns) - 1)])
    f = lambda lx: E + A_ * (10 ** lx) ** (-alpha) + B_ * (C_B / (a * 10 ** lx)) ** (-beta)
    for _ in range(200):
        m1 = lo + (hi - lo) * 0.382; m2 = lo + (hi - lo) * 0.618
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    n = 10 ** (0.5 * (lo + hi))
    return float(n), float(C_B / (a * n)), float(f(np.log10(n)))


def layer_b_q3_degenerate(C=1e22, ell=4096) -> pd.DataFrame:
    A, B, C_ = LineA(), LineB(), LineC()
    a = 6 + 2e-4 * ell
    C_B = C / 1e18
    rows = []

    # A: analytic (and grid cross-check)
    n_an = ((A.alpha * A.A) / (A.beta * A.B) * (C_B / a) ** A.beta) ** (1 / (A.alpha + A.beta))
    n_g, d_g, L_g = _classical_opt(A.E, A.A, A.alpha, A.B, A.beta, C_B, a)
    rows.append({"line": "A", "method": "analytic+grid",
                 "N_B": n_an, "D_B": C_B / (a * n_an),
                 "L": A.E + A.A * n_an ** (-A.alpha) + A.B * (C_B / (a * n_an)) ** (-A.beta),
                 "provenance": "A loss_predictor.json (classical)"})
    # B: from its own loss function (billions internally)
    n_b, d_b, L_b = _classical_opt(*(_b_params(B)), C_B, a)
    rows.append({"line": "B", "method": "grid on B loss()",
                 "N_B": n_b, "D_B": d_b, "L": L_b,
                 "provenance": "B q2_scaling_parameters.json + code/q3_optimize.loss"})
    # C: M1 with kappa=0 (quality channel off) via its own model
    st = C_.q2_stage("stage2_M1_staged") or {}
    p = C_.q3_config()["model"]["fixed_params"]
    E, Aa, al, Bb, be = (float(st.get(k, p[k])) for k in ("E", "A", "alpha", "B", "beta"))
    n_c, d_c, L_c = _classical_opt(E, Aa, al, Bb, be, C_B, a)
    # also through C's model.predict at the grid optimum and at anchor
    Cmodel = C_.q3_model()
    L_c_model = float(np.asarray(Cmodel.predict(n_c * 1e9, d_c * 1e9, 1.0, 1.0, 0.0)).ravel()[0])
    rows.append({"line": "C", "method": "grid on C model (kappa=0)",
                 "N_B": n_c, "D_B": d_c, "L": L_c_model,
                 "provenance": "C Q2-C M1 params + Q3-C model.predict(kappa=0)"})
    df = pd.DataFrame(rows)
    ref = float(df.loc[df.line == "A", "N_B"].iloc[0])
    df["N_rel_to_A"] = df["N_B"] / ref - 1.0
    return df


def _b_params(B: LineB):
    d = json.loads((B.root / "results/q2_scaling_parameters.json").read_text("utf-8"))
    # locate classical params anywhere in the json
    def find(o):
        if isinstance(o, dict):
            if {"E", "A", "alpha", "B", "beta"} <= set(o):
                return o
            for v in o.values():
                r = find(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r:
                    return r
        return None
    p = find(d) or {}
    return (float(p["E"]), float(p["A"]), float(p["alpha"]), float(p["B"]), float(p["beta"]))


def make_figure(q1: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = ["1m", "60m", "1B", "est_10b", "est_70b"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, metric, title in ((axes[0], "rmse", "Composte RMSE (13 domains, equal v)"),
                              (axes[1], "spearman", "Spearman (ranking)"),
                              (axes[2], "top1_regret", "Top-1 regret")):
        w = 0.25
        for i, line in enumerate(["A", "B", "C"]):
            vals = [q1[(q1.line == line) & (q1.set == s)][metric].mean()
                    if not q1[(q1.line == line) & (q1.set == s)].empty else np.nan for s in sets]
            ax.bar(np.arange(len(sets)) + (i - 1) * w, vals, width=w, label=line)
        ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets, rotation=20)
        ax.set_title(title, fontsize=9); ax.axvspan(2.5, 4.5, color="grey", alpha=0.12)
    axes[0].legend(fontsize=8)
    fig.suptitle("R1 Layer A: Q1 mixture prediction on the SAME candidate sets "
                 "(grey = estimated/extrapolated)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_r1(outdir: Path) -> dict:
    """Execute R1 comparison, write tables+figure into outdir, return a summary dict."""
    outdir.mkdir(parents=True, exist_ok=True)
    q1, perdom, ident = layer_a_q1()
    q3 = layer_b_q3_degenerate()
    q1.to_csv(outdir / "r1_q1_metrics.csv", index=False)
    perdom.to_csv(outdir / "r1_q1_perdomain.csv", index=False)
    ident.to_csv(outdir / "r1_q1_set_identity.csv", index=False)
    q3.to_csv(outdir / "r1_q3_degenerate.csv", index=False)
    fig = outdir / "r1_q1_comparison.png"
    try:
        make_figure(q1, fig)
    except Exception as exc:                      # figure is not allowed to break R1
        fig = None
        print("figure failed:", exc)
    # provenance of each captured upstream input
    lock = load_lock()
    summary = {
        "q1_rows": len(q1), "q1_all_same_candidates": bool(ident["same_candidates"].all())
        if len(ident) else False,
        "q1_sets": ident[["set", "common_n", "same_candidates"]].to_dict("records") if len(ident) else [],
        "q3_max_N_rel_spread": float(q3["N_rel_to_A"].abs().max()) if len(q3) else None,
        "upstream": {k: v["commit"][:8] for k, v in lock["lines"].items()},
        "figure": str(fig.name) if fig else None,
    }
    return {"q1": q1, "perdomain": perdom, "identity": ident, "q3": q3, "summary": summary}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="best_entropy_proxyQ",
                    help="C Q1 stored variant to use in Layer A (default matches selected_model)")
    ap.add_argument("--outdir", default=str(RESEARCH / "reports"))
    args = ap.parse_args()
    res = run_r1(Path(args.outdir))
    q1, q3, ident = res["q1"], res["q3"], res["identity"]
    if len(q1):
        print("== Layer A Q1 ==")
        print(q1[["line", "set", "n", "rmse", "r2", "spearman",
                  "top1_regret", "random_expected_regret"]].round(4).to_string(index=False))
    print("== set identity =="); print(ident.to_string(index=False))
    print("== Layer B Q3 degenerate =="); print(q3.round(6).to_string(index=False))
    print("== summary =="); print(json.dumps(res["summary"], ensure_ascii=False))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
