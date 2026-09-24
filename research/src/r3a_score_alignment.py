"""R3a: score-coordinate alignment and transfer-assumption audit.

Apply A's and C's scoring rules to the SAME raw quality records (A1), decomposing the
difference into preprocessing / weighting / conflict rule; validate any monotone mapping
on a held-out split. A's scoring runs in a subprocess (its package is also named `src`),
so the result is written to a side file and read back.

Explicit boundary: aligning TEXT scores does NOT anchor them to B6-B8 `Q_score` (no shared
observations), so the cross-attachment mapping remains a hypothetical scenario.
B's rule is not applied to A1 (same-source rebuild A1=50,412; LFS unavailable).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import LineA, LineC
from .common import RESEARCH, sha256_file

METRIC_CFG = json.loads((RESEARCH.parent / "configs/shared/quality_metrics.json").read_text("utf-8"))
A_METRICS = [m["name"] for m in METRIC_CFG["metrics"]]
A_SUBPROC = r'''
import json, lzma, sys
import pandas as pd
sys.path.insert(0, {root!r})
from src.common.quality import fit_calibration, score_record, aggregate
metrics = json.load(open({cfg!r}, encoding="utf-8"))["metrics"]
groups = json.load(open({cfg!r}, encoding="utf-8"))["groups"]
recs = []
with lzma.open({a1!r}, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            d = json.loads(line); recs.append({{k: v for k, v in d.items() if k != "content"}})
calib, _ = fit_calibration(recs, metrics), None
rows = []
for r in recs:
    dims = score_record(r, metrics, calib)
    agg = aggregate(dims, metrics, groups)
    rows.append({{"id": r.get("id"), **{{m["name"]: dims[m["name"]] for m in metrics}},
                  "S_A_equal": agg["Q_base"], "S_A_min_group": agg["min_group"], "conflict_A": agg["conflict"]}})
pd.DataFrame(rows).to_parquet({out!r}, index=False)
print("Z_A rows", len(rows))
'''


def _compute_A_Z(A: LineA, out: Path):
    cfg = str(RESEARCH.parent / "configs/shared/quality_metrics.json")
    from .common import load_lock
    canon = Path(load_lock()["canonical_data"]["real_attachments"])
    a1 = str(canon / "A_data_value/slimpajama_quality_signal_sample.jsonl.xz")
    if Path(a1).stat().st_size < 10_000:
        raise RuntimeError(f"A1 at canonical root looks like an LFS pointer ({a1}); "
                           "need the real attachments")
    code = "import sys; sys.path.insert(0, {!r})\n".format(str(A.root)) + \
        A_SUBPROC.format(root=str(A.root), cfg=cfg, a1=a1, out=str(out))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(A.root))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-800:])
    return proc.stdout.strip()


def run_r3a(seed=20260924, outdir: Path | None = None) -> dict:
    outdir = outdir or (RESEARCH / "reports")
    outdir.mkdir(parents=True, exist_ok=True)
    A, C = LineA(), LineC()
    za_path = outdir / "_r3a_Z_A.parquet"
    msg = _compute_A_Z(A, za_path)
    ZA = pd.read_parquet(za_path)
    ZC = pd.read_parquet(C.root / "artifacts/q1/preprocessed/A1_processed.parquet")
    w = pd.read_csv(C.root / "artifacts/baselines/Q1-C/quality_entropy_weights.csv")
    wmap = dict(zip(w["metric"], w["weight_entropy"]))
    w_ent = np.array([wmap[m] for m in A_METRICS], float); w_ent = w_ent / w_ent.sum()
    w_eq = np.full(len(A_METRICS), 1.0 / len(A_METRICS))
    Zc = ZC.set_index("id")[A_METRICS]
    qs_all = pd.read_parquet(C.root / "artifacts/baselines/Q1-C/quality_scores.parquet")
    if "source_set" in qs_all.columns:
        qs_all = qs_all[qs_all["source_set"] == "A1"]
    qs = qs_all[["id", "Q_entropy", "Q_resolved_entropy"]].drop_duplicates("id")
    S_CC = Zc.to_numpy(float) @ w_ent
    S_CA = Zc.to_numpy(float) @ w_eq
    c_index = {k: i for i, k in enumerate(Zc.index)}
    m = ZA.merge(qs, on="id", how="inner")
    ZAm = m[A_METRICS].to_numpy(float)
    S_A = m["S_A_equal"].to_numpy(float)
    S_AC = ZAm @ w_ent
    S_CCa = S_CC[[c_index[k] for k in m["id"]]]
    S_CAa = S_CA[[c_index[k] for k in m["id"]]]
    S_C = m["Q_entropy"].to_numpy(float)
    S_Cres = m["Q_resolved_entropy"].to_numpy(float)
    from scipy.stats import spearmanr
    scr = pd.DataFrame({"id": m["id"], "S_A_equal": S_A, "S_Aprep_Cweight": S_AC,
                        "S_Cprep_equal": S_CAa, "S_C_entropy": S_C, "S_C_resolved": S_Cres,
                        "S_C_entropy_recomputed": S_CCa})
    scr.to_csv(outdir / "r3a_score_alignment.csv", index=False)
    corr = {"n_matched": int(len(m))}
    for k in ["S_Aprep_Cweight", "S_Cprep_equal", "S_C_entropy", "S_C_resolved", "S_C_entropy_recomputed"]:
        corr[f"spearman_S_A_equal_vs_{k}"] = float(spearmanr(S_A, scr[k])[0])
    corr["mean_abs_S_A_equal_minus_S_C_entropy"] = float(np.mean(np.abs(S_A - S_C)))
    rng = np.random.default_rng(seed); perm = rng.permutation(len(S_A)); half = len(S_A) // 2
    fit, val = perm[:half], perm[half:]
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(S_A[fit], S_C[fit])
        pred = iso.predict(S_A[val])
        mapping = {"method": "isotonic", "valid_spearman": float(spearmanr(S_C[val], pred)[0]),
                   "valid_rmse": float(np.sqrt(np.mean((S_C[val] - pred) ** 2)))}
    except Exception as exc:
        mapping = {"method": "unavailable", "error": str(exc)}
    verdict = {
        "text_score_alignment": {"A_vs_C_spearman": corr["spearman_S_A_equal_vs_S_C_entropy"],
                                 "monotone_mapping": mapping},
        "decomposition": {"preprocessing": "S_Cprep_equal vs S_A_equal",
                          "weighting": "S_Aprep_Cweight vs S_A_equal",
                          "conflict_rule": "S_C_resolved vs S_C_entropy"},
        "b_rule": {"status": "blocked/hypothetical",
                   "reason": "B 同源重建样本(A1=50412≠51230)且重建数据 LFS 未 pull；未在 A1 上应用 B 规则"},
        "critical": "文本评分对齐 ≠ 与 B6–B8 的 Q_score 建立真实训练收益锚点；两类 Q 缺共同观测，跨附件映射只能作假设情景。",
    }
    (outdir / "r3a_summary.json").write_text(json.dumps(
        {"corr": corr, "verdict": verdict, "A_subprocess": msg,
         "inputs": {"A1_sha256": sha256_file(A.root / "real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz"),
                    "C_preprocessed_sha256": sha256_file(C.root / "artifacts/q1/preprocessed/A1_processed.parquet")}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# R3a：评分坐标对齐与迁移假设审计", "",
          f"- 匹配记录数：{corr['n_matched']}；A 侧子进程：{msg}", "",
          "| 对齐 | Spearman |", "|---|---:|"]
    for k in ["S_Aprep_Cweight", "S_Cprep_equal", "S_C_entropy", "S_C_resolved", "S_C_entropy_recomputed"]:
        md.append(f"| S_A_equal vs {k} | {corr['spearman_S_A_equal_vs_'+k]:.4f} |")
    md += ["", f"- A↔C 平均绝对差：{corr['mean_abs_S_A_equal_minus_S_C_entropy']:.4f}",
           f"- 单调映射（isotonic，fit/验证各半）：{json.dumps(mapping, ensure_ascii=False)}", "",
           "- 分解：预处理差异 = S_Cprep_equal−S_A_equal；赋权差异 = S_Aprep_Cweight−S_A_equal；冲突规则 = S_C_resolved−S_C_entropy。",
           "- **B**：未在 A1 上应用（同源重建样本 + LFS 未 pull），记为 blocked/假设分支。", "",
           "> **关键边界**：文本评分对齐 ≠ 与 B6–B8 的 `Q_score` 的真实训练收益锚点；缺共同观测时跨附件映射只作假设情景。"]
    (outdir / "r3a_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"corr": corr, "verdict": verdict}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=str(RESEARCH / "reports"))
    a = ap.parse_args()
    r = run_r3a(outdir=Path(a.outdir))
    print(json.dumps(r["corr"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
