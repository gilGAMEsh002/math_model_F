"""Read frozen Git objects, verify saved predictions, write review-only evidence.

Run from the repository root: python3 reports/三线分析与论文主线_20260923/collect_evidence.py
No model training; no original input or baseline output is modified.
"""
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
COMMITS = {
    "A": "eed45774a89192ce914b15479cc193139097d308",
    "B": "03921456b512d2c8dbd5ed052ef0f22d8de91412",
    "C": "e35bf06f506cf3e7a5fc93b06e8a77fa39a8c1f3",
    "C2": "6f101de192768f449b27ca4d8bf84c0fcb0a6ebc",
    "C3": "70c0accac07a9be4fe67b81d38c7a4d0af420403",
}
manifest = {"collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "commits": COMMITS, "files": [],
            "scope": "saved-artifact verification; no training rerun"}


def get(line, path, snapshot=True):
    data = subprocess.check_output(["git", "show", f"{COMMITS[line]}:{path}"], cwd=REPO)
    manifest["files"].append({"line": line, "path": path,
                              "sha256": hashlib.sha256(data).hexdigest()})
    if snapshot:
        dest = OUT / "evidence" / line / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def frame(line, path, snapshot=True):
    return pd.read_csv(io.BytesIO(get(line, path, snapshot)), encoding="utf-8-sig")


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((pred-y)**2))),
            "r2": float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),
            "spearman": float(spearmanr(y, pred).statistic),
            "top1_regret": float(y[np.argmin(pred)]-y.min()),
            "random_expected_regret": float(y.mean()-y.min())}


inventory = {}
for line, sha in COMMITS.items():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", sha], cwd=REPO, text=True).splitlines()
    inventory[line] = [p for p in paths if p.startswith(("reports/", "src/", "artifacts/", "configs/", "tests/", "scripts/", "results/", "code/", "sections/"))]
    for p in inventory[line]:
        if p.endswith((".md", ".py", ".tex")) or p == "results/q1_mixture_model.npz" or (p.endswith((".json", ".csv", ".yaml")) and "predictions" not in p and p.startswith(("artifacts/baselines/", "configs/baselines/", "results/"))):
            get(line, p)

a = frame("A", "artifacts/baselines/Q1-A/mixture_predictions.csv", False)
c = frame("C", "artifacts/baselines/Q1-C/mixture_predictions.csv", False)
a_recorded = json.loads(get("A", "artifacts/baselines/Q1-A/q1_metrics.json"))
c_recorded = frame("C", "artifacts/baselines/Q1-C/mixture_ranking_cross_scale.csv")
c_est = frame("C", "artifacts/baselines/Q1-C/mixture_extrapolation_consistency.csv")
c_rmse = frame("C", "artifacts/baselines/Q1-C/mixture_metrics_same_scale_1m.csv")
c_recorded = pd.concat([c_recorded, c_est], ignore_index=True)
names = {"test_1m": "same_scale_1m", "test_60m": "cross_scale_60m", "test_1B": "cross_scale_1B", "est_10b": "est_10b", "est_70b": "est_70b"}
rows, checks = [], []
for aset, dataset in names.items():
    ad = a[a["set"] == aset].sort_values("index")
    am = metrics(ad.actual_eval_loss, ad.predicted_eval_loss)
    for k in ["rmse", "r2", "spearman", "top1_regret"]:
        assert np.isclose(am[k], a_recorded["sets"][aset]["eval_loss"][k], atol=1e-10)
    rows.append({"line": "A", "dataset": dataset, "variant": "ridge_A", **am})
    for variant in ["best_entropy_proxyQ", "forest_none", "forest_pertarget_none", "ridge_none"]:
        cd = c[(c.dataset == dataset) & (c.variant == variant) & (c.target == "__composite_equal_v__")].sort_values("row_index")
        assert len(cd) == len(ad)
        assert np.array_equal(ad["index"].to_numpy(), cd.row_index.to_numpy())
        assert np.allclose(ad.actual_eval_loss, cd.y_true, atol=1e-10)
        cm = metrics(cd.y_true, cd.y_pred)
        saved = c_recorded[(c_recorded.dataset == dataset) & (c_recorded.variant == variant) & (c_recorded.target == "__composite_equal_v__")].iloc[0]
        assert np.isclose(cm["spearman"], saved.spearman, atol=1e-10)
        assert np.isclose(cm["top1_regret"], saved.regret, atol=1e-10)
        if dataset == "same_scale_1m":
            saved_m = c_rmse[(c_rmse.variant == variant) & (c_rmse.target == "__composite_equal_v__")].iloc[0]
            assert np.isclose(cm["rmse"], saved_m.rmse, atol=1e-10)
            assert np.isclose(cm["r2"], saved_m.r2, atol=1e-10)
        rows.append({"line": "C", "dataset": dataset, "variant": variant, **cm})
    checks.append({"dataset": dataset, "n": len(ad), "same_row_ids_and_target": True, "saved_metrics_recomputed": True})

q3 = frame("A", "artifacts/baselines/Q3-A/optimal_allocations.csv")
cost = (6+2e-4*q3.ell)*q3.N_num_B*q3.D_num_B*1e18
assert np.allclose(cost, q3.total_cost_flops)
assert np.allclose(q3.quality_cost_flops, 0)
assert np.all(cost <= q3.C_flops*(1+1e-12))

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "q1_comparison_verified.csv", index=False)
for name, data in [("branch_inventory.json", inventory), ("manifest.json", manifest),
                   ("verification.json", {"q1": checks, "q3_cost_formula_and_feasibility": True,
                    "q3_scenarios": len(q3), "q3_extrapolated": int(q3.extrapolation.sum()),
                    "q3_attention_training_equal_context": 6/2e-4,
                    "limits": "Verifies stored predictions and algebra, not training reproducibility or unseen-data performance."})]:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(f"Verified {len(rows)} prediction summaries, {len(q3)} cost scenarios; evidence saved to {OUT}")
