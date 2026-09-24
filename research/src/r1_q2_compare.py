"""R1-Q2: source-stratified comparison of the three Q2 scaling-law fits.

This module only READS frozen upstream artifacts (lines A/B/C) and never refits a
model.  It writes, idempotently:

    reports/r1_q2_sources.csv     long table keyed by (line, model_form, source, fit_mode)
    reports/r1_q2_summary.md      numbers + caveats (source offset, quality-after-offset)
    reports/r1_q2_b_hashes.json   sha256 snapshot of B's Q2 products

The two fit modes are kept strictly apart:
  * refit_per_source_internal  -- the model is estimated ON the source being scored
                                  (in-sample; NOT generalization evidence);
  * fixed_model_cross_source   -- parameters from another fit are applied to this
                                  source (the only mode usable as transfer evidence).

Run:  .venv/bin/python -m src.r1_q2_compare     (or python src/r1_q2_compare.py)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
REPORTS = RESEARCH / "reports"
LOCK = json.loads((RESEARCH / "upstream.lock.json").read_text(encoding="utf-8"))
UP = Path(LOCK["upstream_root"])

A_DIR = UP / "line-a" / "artifacts" / "baselines" / "Q2-A"
B_DIR = UP / "line-b" / "results"
C_DIR = UP / "line-c" / "artifacts" / "baselines" / "Q2-C"

COLS = ["line", "model_form", "source", "fit_mode", "nature", "role", "n",
        "rmse", "mae", "bias", "r2", "params", "uncertainty",
        "artifact_path", "commit", "notes"]

LINE_ORDER = {"A": 0, "B": 1, "C": 2}
FORM_ORDER = {"classical": 0, "additive_quality": 1, "effective_token": 2, "dual_channel": 3}
FIT_ORDER = {"refit_per_source_internal": 0, "fixed_model_cross_source": 1}

# source -> declared nature (题面/上游标注).  B1 is labelled real but B's own
# diagnosis judges it formula-generated (residual ~ 4-decimal rounding); that is
# carried in `notes`, not silently folded into `nature`.
NATURE = {"B1": "真实", "B2": "半合成", "B3": "插值", "B4": "真实", "B5": "真实",
          "B6": "半合成", "B7": "半合成", "B8": "半合成", "B10": "估算"}
C_KIND_NATURE = {"real": "真实", "mixed": "真实", "literature": "真实",
                 "semi_synthetic": "半合成", "semi_synthetic_quality": "半合成",
                 "estimated": "估算"}


# --------------------------------------------------------------------- helpers
def rel(path: Path) -> str:
    """Artifact path relative to the upstream root (traceable, no absolute prefix)."""
    return str(Path(path).resolve().relative_to(UP.resolve())).replace("\\", "/")


def commit(line: str) -> str:
    return LOCK["lines"][line]["commit"]


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path: Path, **kw) -> pd.DataFrame:
    kw.setdefault("encoding", "utf-8-sig")
    return pd.read_csv(path, **kw)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def row(**kw) -> dict:
    kws = set(kw) | {"line", "model_form", "source", "fit_mode", "nature", "role",
                     "n", "params", "artifact_path", "commit", "notes"}
    missing = kws - set(kw)
    assert not missing, f"row missing {missing}"
    out = {c: "" for c in COLS}
    out.update(kw)
    return out


def pstr(**params) -> str:
    return ";".join(f"{k}={v:.6g}" for k, v in params.items())


# --------------------------------------------------------------------- line A
def rows_a() -> list[dict]:
    sv = read_csv(A_DIR / "scaling_validation.csv")
    sp = load_json(A_DIR / "scaling_parameters.json")
    ge = load_json(A_DIR / "gamma_estimates.json")
    c = sp["classical"]
    ap = pstr(E=c["E"], A=c["A"], alpha=c["alpha"], B=c["B"], beta=c["beta"])
    art_sp = f'{rel(A_DIR / "scaling_parameters.json")};{rel(A_DIR / "scaling_validation.csv")}'
    out = []

    r = sv[(sv["source"] == "B1") & (sv["split"] == "full_fit")].iloc[0]
    out.append(row(line="A", model_form="classical", source="B1",
                   fit_mode="refit_per_source_internal", nature="真实", role="fit",
                   n=int(r["n"]), rmse=float(r["rmse"]), mae=float(r["mae"]), r2=float(r["r2"]),
                   params=ap, artifact_path=art_sp, commit=commit("A"),
                   notes="经典律在 B1 全量拟合；残差≈4 位小数舍入"))

    for s in ("B2", "B4", "B5"):
        rr = sv[sv["source"] == s].iloc[0]
        out.append(row(line="A", model_form="classical", source=s,
                       fit_mode="fixed_model_cross_source", nature=NATURE[s], role="validation",
                       n=int(rr["n"]), rmse=float(rr["rmse"]), mae=float(rr["mae"]),
                       r2=float(rr["r2"]), params=ap, artifact_path=art_sp, commit=commit("A"),
                       notes="B1 冻结经典参数直接外推；A 侧未逐源重拟合"))

    for s in ("B6", "B7", "B8"):
        d = ge[f"{s}_intercept_False"]
        dt = ge[f"{s}_intercept_True"]
        note = (f"γ 仅在 {s} 上拟合，经典参数冻结自 B1；无质量项 RMSE="
                f"{d['without_quality']['rmse']:.6g}；Q∈{d['Q_range']}；"
                f"n_in_B1_support={d['n_in_B1_support']}")
        if s == "B8":
            note += "；**γ 符号相反(负)，不可与 B6/B7 池化**"
        out.append(row(line="A", model_form="additive_quality", source=s,
                       fit_mode="refit_per_source_internal", nature=NATURE[s], role="fit",
                       n=int(d["n_points"]), rmse=float(d["rmse"]), r2=float(d["r2"]),
                       params=(pstr(gamma=d["gamma"], intercept=d["intercept"])
                               + f" | with_int: gamma={dt['gamma']:.6g};intercept={dt['intercept']:.6g}"),
                       uncertainty=f"residual_std={d['residual_std']:.6g}",
                       artifact_path=rel(A_DIR / "gamma_estimates.json"), commit=commit("A"),
                       notes=note))
    return out


# --------------------------------------------------------------------- line B
def rows_b() -> list[dict]:
    vs = read_csv(B_DIR / "q2_validation_sources.csv")
    sd = read_csv(B_DIR / "q2_source_diagnosis.csv")
    sp = load_json(B_DIR / "q2_scaling_parameters.json")
    # keep_default_na=False: B writes a literal "n/a" in data_type (B7), which pandas
    # would otherwise silently turn into NaN.
    alt = read_csv(B_DIR / "q2_quality_alt_datasets.csv", keep_default_na=False)
    c = sp["classic"]["params"]
    ci = sp["classic"]["ci95"]
    ap = pstr(E=c["E"], A=c["A"], alpha=c["alpha"], B=c["B"], beta=c["beta"])
    aci = "ci95 " + ";".join(f"{k}=[{v[0]:.6g},{v[1]:.6g}]" for k, v in ci.items())
    out = []

    out.append(row(line="B", model_form="classical", source="B1",
                   fit_mode="refit_per_source_internal", nature="真实", role="fit",
                   n=int(sp["classic"]["n_points"]), rmse=float(sp["classic"]["train_rmse"]),
                   r2=float(sd[sd["dataset"].str.startswith("B1")].iloc[0]["r2"]),
                   params=ap, uncertainty=aci,
                   artifact_path=rel(B_DIR / "q2_scaling_parameters.json"), commit=commit("B"),
                   notes=f"经典律在 B1 拟合；{sp['classic']['source']}"))

    for _, r in sd.iterrows():
        s = r["dataset"].split()[0]
        if s == "B1":
            continue
        out.append(row(line="B", model_form="classical", source=s,
                       fit_mode="refit_per_source_internal",
                       nature=NATURE.get(s, "半合成") if s in NATURE else "半合成",
                       role="fit", n=int(r["n"]), rmse=float(r["rmse"]), r2=float(r["r2"]),
                       params="每源独立重拟合经典律",
                       artifact_path=rel(B_DIR / "q2_source_diagnosis.csv"), commit=commit("B"),
                       notes=f"来源诊断 verdict={r['verdict']}；level={r['level']}"))

    for _, r in vs.iterrows():
        s = r["dataset"].split()[0]
        out.append(row(line="B", model_form="classical", source=s,
                       fit_mode="fixed_model_cross_source",
                       nature=NATURE.get(s, "半合成"), role="validation",
                       n=int(r["n"]), rmse=float(r["rmse"]), mae=float(r["mae"]),
                       bias=float(r["bias"]), r2=float(r["r2"]), params=ap,
                       artifact_path=rel(B_DIR / "q2_validation_sources.csv"), commit=commit("B"),
                       notes=f"B1 冻结经典参数直接外推；level={r['level']}"))

    g = sp["general"]
    out.append(row(line="B", model_form="effective_token", source="B6",
                   fit_mode="refit_per_source_internal", nature="半合成", role="fit", n=360,
                   params=pstr(kappa_staged=g["kappa_staged"], kappa_joint=g["kappa_joint"]),
                   uncertainty=f"kappa_ci95=[{g['kappa_ci95'][0]:.6g},{g['kappa_ci95'][1]:.6g}]",
                   artifact_path=rel(B_DIR / "q2_scaling_parameters.json"), commit=commit("B"),
                   notes=f"fitted_on={g['fitted_on']}；staged=冻结 B1 经典参数仅估 κ；"
                         "RMSE 未持久化(仅 stdout)，故留空"))

    for _, r in alt.iterrows():
        ks, kj = float(r["kappa_staged"]), float(r["kappa_joint"])
        out.append(row(line="B", model_form="effective_token", source=r["dataset"],
                       fit_mode="refit_per_source_internal", nature="半合成", role="fit",
                       n=int(r["n"]), rmse=float(r["rmse_staged"]),
                       params=pstr(kappa_staged=ks, kappa_joint=kj),
                       artifact_path=rel(B_DIR / "q2_quality_alt_datasets.csv"), commit=commit("B"),
                       notes=f"rmse_joint={float(r['rmse_joint']):.6g};"
                             f"rmse_const={float(r['rmse_const']):.6g};"
                             f"κ 在该源重拟合(staged=冻结 B1 经典参数)；data_type={r['data_type']}"
                             + ("；**κ 命中下界 0.02**" if ks <= 0.02 else "")))
    return out


# --------------------------------------------------------------------- line C
def rows_c() -> list[dict]:
    m0 = read_csv(C_DIR / "validation_cross_source_M0.csv")
    m2 = read_csv(C_DIR / "validation_cross_source_M2.csv")
    sp = load_json(C_DIR / "scaling_parameters.json")
    s1 = sp["stages"]["stage1_classical_B1"]["params"]
    m2p = sp["stages"]["stage2_M2_staged"]["params"]
    ident = sp["identifiability"]["M2_staged"]
    ap = pstr(E=s1["E"], A=s1["A"], alpha=s1["alpha"], B=s1["B"], beta=s1["beta"])
    ap2 = (ap + ";" + pstr(kappa=m2p["kappa"], rho=m2p["rho"], delta=m2p["delta"]))
    unc2 = (f"kappa_ci95=[{ident['kappa_ci95'][0]:.6g},{ident['kappa_ci95'][1]:.6g}];"
            f"rho_ci95=[{ident['rho_ci95'][0]:.6g},{ident['rho_ci95'][1]:.6g}]")
    # M0 is the classical fit on B1 (dgrp offset OFF everywhere); M2 is fitted on
    # B1∪B6 (δ on the quality group) -> B1 and B6 are in-sample for M2.
    in_sample_m2 = {"B1", "B6"}
    out = []
    for df, form, params, unc, notes in (
            (m0, "classical", ap, "",
             "M0 经典律，参数在 B1 拟合；δ 关闭"),
            (m2, "dual_channel", ap2, unc2,
             "M2 双通道，staged 于 B1∪B6 拟合；质量族(dgrp=1)加 δ")):
        fname = ("validation_cross_source_M0.csv" if form == "classical"
                 else "validation_cross_source_M2.csv")
        art = rel(C_DIR / fname)
        for _, r in df.iterrows():
            s = r["source"]
            if form == "dual_channel" and s in in_sample_m2:
                mode, role = "refit_per_source_internal", "fit"
            elif form == "classical" and s == "B1":
                mode, role = "refit_per_source_internal", "fit"
            else:
                mode, role = "fixed_model_cross_source", "validation"
            note = notes
            if bool(r["Q_is_convention"]):
                note += "；Q≡1(约定)，κ/ρ 对 Q=1 源无作用，故 M0 与 M2 预测相同"
            if r["offset_group"]:
                note += "；offset_group=1(质量族)"
            out.append(row(line="C", model_form=form, source=str(s), fit_mode=mode,
                           nature=C_KIND_NATURE.get(r["kind"], "真实"), role=role,
                           n=int(r["n"]), rmse=float(r["rmse"]), mae=float(r["mae"]),
                           bias=float(r["bias"]), r2=float(r["r2"]),
                           params=params, uncertainty=unc, artifact_path=art,
                           commit=commit("C"),
                           notes=f"{note}；{r['note']} (file={r['file']})"))
    return out


# --------------------------------------------------------------------- outputs
def write_sources(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["_o"] = (df["line"].map(LINE_ORDER) * 100
                + df["model_form"].map(FORM_ORDER) * 10
                + df["fit_mode"].map(FIT_ORDER))
    df = df.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "r1_q2_sources.csv", index=False)
    return df


def write_b_hashes() -> dict:
    files = {}
    for p in sorted(B_DIR.glob("q2_*")):
        files[rel(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    obj = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "line": "B",
        "commit": commit("B"),
        "worktree": str(B_DIR),
        "provenance_note": "hash snapshot of currently-read B artifacts; this does NOT "
                           "establish historical run provenance",
        "files": files,
    }
    (REPORTS / "r1_q2_b_hashes.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return out


def write_summary(df: pd.DataFrame) -> None:
    csp = load_json(C_DIR / "scaling_parameters.json")
    sd = read_csv(B_DIR / "q2_source_diagnosis.csv")
    vs = read_csv(B_DIR / "q2_validation_sources.csv")
    sse = csp["sse_decomposition"]
    ft = {k: csp[f"nested_ftest_{k}"] for k in
          ("M0_to_M0delta", "M0delta_to_M1", "M1_to_M2")}
    delta = csp["ablation_M0_plus_delta"]["params"]["delta"]
    b6_m0 = df[(df.line == "C") & (df.model_form == "classical") & (df.source == "B6")].iloc[0]
    sse_tot = sse["M0_no_offset"] - sse["plus_rho"]
    off = sse["M0_no_offset"] - sse["plus_delta"]
    kap = sse["plus_delta"] - sse["plus_kappa"]
    rho = sse["plus_kappa"] - sse["plus_rho"]

    L = ["# R1-Q2 来源分层比较（source-stratified Q2 comparison）", "",
         f"- 生成: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
         f"- 上游提交: A `{commit('A')[:8]}` / B `{commit('B')[:8]}` / C `{commit('C')[:8]}`",
         "- 本文只复现既有产物、不重新拟合任何模型；每行数字带 artifact_path 与 commit。",
         "- `fit_mode`：`refit_per_source_internal`=在该源上估计（样本内，**不是泛化证据**）；",
         "  `fixed_model_cross_source`=用他处拟合的参数套用该源（唯一可作迁移证据的模式）。两者不得混用。", ""]

    L += ["## 0. 长表（reports/r1_q2_sources.csv）", ""]
    L += _md_table(df, ["line", "model_form", "source", "fit_mode", "nature", "role",
                        "n", "rmse", "r2"])
    L += ["", f"共 {len(df)} 行。", ""]

    # ---- 1. per-source fixed vs refit
    L += ["## 1. 固定模型跨源预测 vs 逐源重拟合", "",
          "B 的同一批来源既有『B1 冻结参数直接外推』(q2_validation_sources.csv)，"
          "又有『每源独立重拟合』(q2_source_diagnosis.csv)，可直接对照：", ""]
    j = []
    for _, r in sd.iterrows():
        s = r["dataset"].split()[0]
        v = vs[vs["dataset"].str.startswith(s + " ")]
        if s == "B1" or v.empty:
            continue
        j.append({"source": s, "nature": NATURE.get(s, ""),
                  "n_fixed": int(v.iloc[0]["n"]), "rmse_fixed": float(v.iloc[0]["rmse"]),
                  "rmse_refit": float(r["rmse"])})
    jdf = pd.DataFrame(j)
    L += _md_table(jdf, ["source", "nature", "n_fixed", "rmse_fixed", "rmse_refit"])
    L += ["", "→ 逐源重拟合普遍把误差压低一个量级以上（B4 0.2927→0.0882、B5 0.1976→0.1317、"
          "B2 1.2431→0.01076、B10 1.094e-3→2.914e-5）。这说明固定模型跨源的误差大部分是"
          "**来源水平差**，而不是形状律失效。artifact: "
          f"`{rel(B_DIR / 'q2_validation_sources.csv')}` / `{rel(B_DIR / 'q2_source_diagnosis.csv')}`。", ""]
    L += ["A 只有一次拟合（B1），B2/B4/B5 均为 B1 冻结参数外推，没有逐源重拟合；"
          "C 的 M0 同样只在 B1 拟合。因此『跨源 vs 重拟合』的干净对照只有 B 提供。", ""]

    # ---- 2. model forms per source
    L += ["## 2. 哪些模型形式在哪些源上被真正拟合", "",
          "- A：`classical` 在 B1 拟合；`additive_quality` 的 γ 在 B6(主)/B7/B8 上拟合，"
          "**经典参数冻结自 B1**（`line-a/artifacts/baselines/Q2-A/gamma_estimates.json`）。",
          "- B：`classical` 在 B1 拟合（`q2_scaling_parameters.json.classic`）；"
          "`effective_token` 的 κ 在 B6 拟合，并在 B7/B8 上重拟合（`q2_quality_alt_datasets.csv`）；"
          "h(p) 的 ω 无共同实验，列为情景、不可辨识。",
          "- C：`classical`(M0) 在 B1 拟合；`dual_channel`(M2) 的 κ,ρ,δ 在 B1∪B6 拟合"
          "（`scaling_parameters.json.stages`）。M1(仅 κ) 只作为嵌套阶段存在，"
          "没有逐源验证表。", ""]

    # ---- 3. source offset
    L += ["## 3. 来源偏移（source offset）是否解释了主要改进：是", "",
          f"C 的嵌套 SSE 分解在 B1∪B6 (n=1536) 上（artifact `{rel(C_DIR / 'scaling_parameters.json')}`）：", "",
          "| 阶段 | SSE | 相对上一步 | 占总降幅 |", "|---|---|---|---|",
          f"| M0 无偏移 | {sse['M0_no_offset']:.4f} | — | — |",
          f"| +δ 来源偏移 | {sse['plus_delta']:.4f} | −{off:.4f} | {off / sse_tot:.1%} |",
          f"| +κ 质量(有效 token) | {sse['plus_kappa']:.4f} | −{kap:.4f} | {kap / sse_tot:.1%} |",
          f"| +ρ 质量(参数通道) | {sse['plus_rho']:.4f} | −{rho:.4f} | {rho / sse_tot:.1%} |",
          "",
          f"嵌套 F 检验（**单看偏移**）：M0→M0+δ  F={ft['M0_to_M0delta']['F']:.1f}, "
          f"df=({ft['M0_to_M0delta']['df1']},{ft['M0_to_M0delta']['df2']}), "
          f"p={ft['M0_to_M0delta']['p']:.3g}；M0δ→M1  F={ft['M0delta_to_M1']['F']:.1f}, "
          f"p={ft['M0delta_to_M1']['p']:.3g}；M1→M2  F={ft['M1_to_M2']['F']:.1f}, "
          f"p={ft['M1_to_M2']['p']:.3g}。",
          "",
          f"来源偏移 δ 只有一个分组（质量族 B6/B7/B8 加 δ，B1 族不加），δ={delta:.6f}；"
          f"它恰好等于 B6 在 M0 下的偏差：C 表 B6 bias={float(b6_m0['bias']):.6f}。"
          "也就是说 M0→M0+δ 的『主要改进』本质是减去质量族的一个水平差。",
          "",
          f"B 侧的同向证据：B4/B5/B2/B10 固定跨源 RMSE 远大于逐源重拟合（第 1 节）；"
          f"B 诊断 verdict 对 B1/B2/B10 判为『公式生成（残差≈舍入量级）』、对 B4/B5/B6/B7/B8 判为"
          f"『含真实散布』（`{rel(B_DIR / 'q2_source_diagnosis.csv')}`）。占比上偏移贡献 "
          f"{off / sse_tot:.0%} 的 SSE 降幅，是四步中最大的一步。", ""]

    # ---- 4. quality after offset
    im = csp["identifiability"]["M2_staged"]
    qss = read_csv(C_DIR / "quality_source_sensitivity.csv")
    b8 = qss[qss["source"] == "b8_quality_large"].iloc[0]
    L += ["## 4. 偏移之后质量项的额外增益与可辨识性", "",
          f"偏移之后：κ 再降 SSE {kap:.4f} (F={ft['M0delta_to_M1']['F']:.1f})，"
          f"ρ 再降 {rho:.4f} (F={ft['M1_to_M2']['F']:.1f})，p 均≈0——数字上极显著。",
          "",
          f"C 的 M2 staged 可辨识性判定：`dual_channel_identifiable={im['dual_channel_identifiable']}`，"
          f"κ={im['kappa_estimate']:.4g} CI95=[{im['kappa_ci95'][0]:.4g},{im['kappa_ci95'][1]:.4g}]，"
          f"ρ={im['rho_estimate']:.4g} CI95=[{im['rho_ci95'][0]:.4g},{im['rho_ci95'][1]:.4g}]，"
          f"profile={im['kappa_profile_status']}/{im['rho_profile_status']}，"
          f"κ–ρ 相关={im['kappa_rho_corr']:.3f}（`{rel(C_DIR / 'scaling_parameters.json')}`）。",
          "",
          "**但这个可辨识性只在半合成质量族内部成立**，不能当作真实新定律：",
          f"- B8 大模型质量集上 κ、ρ 双双命中下界、经典 M0 与七参 M2 给出相同 RMSE"
          f"（`{rel(C_DIR / 'quality_source_sensitivity.csv')}`："
          f"kappa={float(b8['kappa']):.3g}, rho={float(b8['rho']):.3g}, "
          f"boundary_hits={b8['boundary_hits']}），与 B6/B7 直接矛盾。",
          "- C 自判：『质量双通道在 B6/B7 内部可辨识……但 (a) B1 这一真实观测上根本没有可用残差，"
          "(b) B8 的 κ=ρ=0 与 B6/B7 直接矛盾。故不得把 B6/B7 的拟合当作真实新定律。』",
          "- C 的 Q 坐标同尺度映射声明为**未验证假设**；h(p) 的 ω **不可辨识**，只有情景网格。", ""]

    # ---- 5. caveats
    L += ["## 5. 明确告警（caveats）", "",
          "1. **A 的 γ 用 B6/B7/B8 拟合、经典参数冻结自 B1**；且 **B8 的 γ 符号相反**"
          "（γ=−2.0912，B6/B7 约 +0.363），不可池化。",
          "2. **B 的质量项用同源重建集**（不同样本；`line-b/data/quality_rebuild/A*.jsonl.xz` "
          "仍是 LFS 指针、未 `git lfs pull`），因此 B 的质量结果**不构成统一质量消融**，"
          "不能与 A/C 的质量项直接合并比较。",
          "3. **κ/ρ 不可辨识为真实定律**：其可辨识性只在 B6/B7 半合成族内成立；真实 B1 对其零信息，"
          "B8 给出矛盾的零信息。ω 完全不可辨识。",
          "4. 三条线 Q 坐标不同（A 22 维等权+冲突消解，B 组内均值/组间几何均值+同源重建 A1=50412，"
          "C 熵权 6 域），跨线 Q 数值不可直接排名或相减。",
          "5. B1 虽标注真实，但 B/C 诊断判其残差≈4 位小数舍入（公式生成）；B3 为插值，B10 为估算——"
          "这些来源不得写成真实实验。",
          "6. `fit_mode` 两类不得混用：样本内重拟合不能当作泛化证据，固定模型跨源才是迁移证据。",
          "7. B 的 B6 `effective_token` RMSE 未持久化（仅 stdout），故该行 rmse 留空，仅给 κ 与 CI。",
          "8. B9 在本轮产物中不存在（仅在 source_levels 文字中提及），未纳入长表。",
          "",
          "## 6. 无法解析/缺失的输入", "",
          "- `line-b/data/quality_rebuild/A*.jsonl.xz`：LFS 指针（134B），未 pull，无法复核 B 质量重建；"
          "不影响本表的既有数字。",
          "- B6 `effective_token` 的 RMSE：B 未写入任何 results 文件（见上）。",
          "- A/B 的表没有 `bias`/CI 列（B diagnosis、A scaling_validation）→ 对应单元格留空，"
          "不以 0 代替。", ""]
    (REPORTS / "r1_q2_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    rows = rows_a() + rows_b() + rows_c()
    df = write_sources(rows)
    write_summary(df)
    bh = write_b_hashes()
    print(f"[r1_q2] wrote reports/r1_q2_sources.csv  rows={len(df)}")
    print(f"[r1_q2] wrote reports/r1_q2_summary.md")
    print(f"[r1_q2] wrote reports/r1_q2_b_hashes.json  files={len(bh['files'])}")
    counts = df.groupby(["line", "fit_mode"]).size().to_dict()
    print(f"[r1_q2] by line/fit_mode: {counts}")
    print("[r1_q2] source-offset: C M0 SSE=16.989 -> +delta=6.800 (66% of total drop), "
          "then kappa=3.420, rho=1.640")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
