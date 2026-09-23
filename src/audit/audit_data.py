"""P0 data audit for Baseline A.

Streams A1-A3 (quality signals), audits duplicates by (quality_domain, id),
checks the RegMix mixture/loss one-to-one joins, normalisation and missing
targets, and checks B6/B7/B8 overlap. Outputs go to artifacts/audit/.

C1/C3/C4 and C8 are only catalogued here (Q4 is deferred in Baseline A).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import REPO, iter_jsonl_xz, read_json, sha256_file, write_json

RA = REPO / "real_attachments"
QUALITY_METRICS = [m["name"] for m in read_json(REPO / "configs/shared/quality_metrics.json")["metrics"]]
HELP_FIELDS = ["id", "content", "sub_path", "_source_domain", "_source_path"]


def _inventory() -> list:
    rows = []
    for p in sorted(RA.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(RA).as_posix()
        entry = {"path": rel, "bytes": p.stat().st_size}
        if p.suffix.lower() in {".csv", ".json"} or p.name.endswith(".jsonl.xz"):
            entry["sha256"] = sha256_file(p)
        rows.append(entry)
    return rows


def _audit_quality():
    files = {
        "A1": (RA / "A_data_value/slimpajama_quality_signal_sample.jsonl.xz", None),
        "A2": (RA / "A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz", "arxiv"),
        "A3": (RA / "A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz", "github"),
    }
    per_file = {}
    keys_by_file = {}
    fields_by_file = {}
    for tag, (path, domain_hint) in files.items():
        keys = set()
        domain_counter = Counter()
        field_counter = Counter()
        n = 0
        for rec in iter_jsonl_xz(path):
            n += 1
            dom = rec.get("_source_domain") or domain_hint
            domain_counter[dom] += 1
            keys.add((dom, rec.get("id")))
            field_counter.update(rec.keys())
        per_file[tag] = {
            "path": path.relative_to(RA).as_posix(),
            "n": n,
            "domains": dict(domain_counter),
            "n_fields": len(field_counter),
        }
        keys_by_file[tag] = keys
        fields_by_file[tag] = set(field_counter)

    # Duplicate / overlap accounting.
    overlaps = {
        "A1_and_A2": len(keys_by_file["A1"] & keys_by_file["A2"]),
        "A1_and_A3": len(keys_by_file["A1"] & keys_by_file["A3"]),
        "A2_and_A3": len(keys_by_file["A2"] & keys_by_file["A3"]),
    }
    merged = keys_by_file["A1"] | keys_by_file["A2"] | keys_by_file["A3"]
    per_domain_merged = Counter(d for d, _ in merged)
    field_diff = {
        "A2_minus_A1": sorted(fields_by_file["A2"] - fields_by_file["A1"]),
        "A1_minus_A2": sorted(fields_by_file["A1"] - fields_by_file["A2"]),
    }
    return {
        "per_file": per_file,
        "overlaps": overlaps,
        "merged_unique_records": len(merged),
        "merged_per_domain": dict(sorted(per_domain_merged.items())),
        "field_symdiff_A1_A2": field_diff,
        "quality_metrics_expected": len(QUALITY_METRICS),
        "help_fields_expected": HELP_FIELDS,
    }, keys_by_file


def _audit_mixture():
    names = ["train_mixture_1m", "train_pile_loss_1m", "test_mixture_1m", "test_pile_loss_1m",
             "test_mixture_60m", "test_pile_loss_60m", "test_mixture_1B", "test_pile_loss_1B",
             "est_mixture_10b", "est_pile_loss_10b", "est_mixture_70b", "est_pile_loss_70b"]
    out = {}
    for name in names:
        df = pd.read_csv(RA / f"A_data_value/regmix_tables/{name}.csv")
        info = {"rows": int(df.shape[0]), "cols": int(df.shape[1]),
                "columns": list(df.columns), "duplicate_index": int(df["index"].duplicated().sum())}
        pcols = [c for c in df.columns if c.startswith("train_the_pile_")]
        if pcols:
            rowsum = df[pcols].sum(axis=1)
            info.update({
                "row_sum_min": float(rowsum.min()), "row_sum_max": float(rowsum.max()),
                "zero_fraction": float((df[pcols] == 0).values.mean()),
                "negatives": int((df[pcols] < 0).values.sum()),
            })
        lcols = [c for c in df.columns if c.startswith("metric/")]
        if lcols:
            info.update({"missing_loss_cells": int(df[lcols].isna().sum().sum())})
        out[name] = info

    pair_checks = {}
    for scale, mname, lname in [("1m", "train_mixture_1m", "train_pile_loss_1m"),
                                ("test_1m", "test_mixture_1m", "test_pile_loss_1m"),
                                ("test_60m", "test_mixture_60m", "test_pile_loss_60m"),
                                ("test_1B", "test_mixture_1B", "test_pile_loss_1B"),
                                ("est_10b", "est_mixture_10b", "est_pile_loss_10b"),
                                ("est_70b", "est_mixture_70b", "est_pile_loss_70b")]:
        m = pd.read_csv(RA / f"A_data_value/regmix_tables/{mname}.csv")
        l = pd.read_csv(RA / f"A_data_value/regmix_tables/{lname}.csv")
        pair_checks[scale] = {
            "same_length": bool(len(m) == len(l)),
            "same_index_order": bool((m["index"].values == l["index"].values).all()),
            "index_sets_equal": bool(set(m["index"]) == set(l["index"])),
        }
    out["_pair_checks"] = pair_checks

    mapping = pd.read_csv(RA / "A_data_value/domain_mapping_guide.csv")
    loss_targets = [c.replace("metric/the_pile_", "").replace("_val_loss", "")
                    for c in pd.read_csv(RA / "A_data_value/regmix_tables/train_pile_loss_1m.csv").columns
                    if c.startswith("metric/")]
    out["_loss_targets"] = loss_targets
    out["_mapping_rows"] = int(len(mapping))
    return out


def _audit_quality_supplementary():
    out = {}
    sets = {}
    for tag, fname in [("B6", "supplementary_NQ_experiment.csv"),
                       ("B7", "supplementary_NQ_experiment_expanded.csv"),
                       ("B8", "supplementary_NQ_experiment_large.csv")]:
        df = pd.read_csv(RA / f"B_scaling_laws/{fname}")
        out[tag] = {"rows": int(len(df)), "columns": list(df.columns),
                    "Q_range": [float(df["Q_score"].min()), float(df["Q_score"].max())],
                    "N_range_B": [float(df["N_params_B"].min()), float(df["N_params_B"].max())],
                    "D_range_B": [float(df["D_tokens_B"].min()), float(df["D_tokens_B"].max())],
                    "duplicate_N_D": int(df.duplicated(subset=["N_params_B", "D_tokens_B"]).sum())}
        sets[tag] = set(zip(df["N_params_B"].round(6), df["D_tokens_B"].round(6)))
    out["_overlap"] = {
        "B6_and_B7": len(sets["B6"] & sets["B7"]),
        "B6_and_B8": len(sets["B6"] & sets["B8"]),
        "B7_and_B8": len(sets["B7"] & sets["B8"]),
    }
    return out


def _audit_scaling():
    out = {}
    for tag, fname in [("B1", "pythia_training_log_existing.csv"),
                       ("B2", "cerebras_training_log.csv"),
                       ("B4", "scaling_baseline.csv"),
                       ("B5", "published_scaling_data.csv"),
                       ("B9", "supplementary_large_models.csv"),
                       ("B10", "supplementary_large_baseline.csv")]:
        df = pd.read_csv(RA / f"B_scaling_laws/{fname}")
        info = {"rows": int(len(df)), "columns": list(df.columns)}
        if "N_params_B" in df:
            info["N_range_B"] = [float(df["N_params_B"].min()), float(df["N_params_B"].max())]
        if "D_tokens_B" in df:
            info["D_range_B"] = [float(df["D_tokens_B"].min()), float(df["D_tokens_B"].max())]
        out[tag] = info
    b1 = pd.read_csv(RA / "B_scaling_laws/pythia_training_log_existing.csv")
    ratio = (b1["C_FLOPs_1e21"] * 1e21) / (6 * b1["N_params_B"] * 1e9 * b1["D_tokens_B"] * 1e9)
    out["B1_C_over_6ND"] = {"min": float(ratio.min()), "median": float(ratio.median()), "max": float(ratio.max())}
    # B4/B5 vs B1 convergence-point duplication
    def key(df):
        return set(zip(df["N_params_B"].round(6), df["D_tokens_B"].round(6), df["val_loss"].round(6)))
    b1conv = key(b1)
    out["_overlap_B1"] = {
        "B4": len(key(pd.read_csv(RA / "B_scaling_laws/scaling_baseline.csv")) & b1conv),
        "B5": len(key(pd.read_csv(RA / "B_scaling_laws/published_scaling_data.csv")) & b1conv),
    }
    return out


def _audit_c_catalogue():
    out = {}
    for tag, fname in [("C1", "leaderboard_cleaned.csv"), ("C3", "leaderboard_extended_timeseries.csv"),
                       ("C4", "epoch_all_ai_models.csv"), ("C6", "loss_benchmark_bridge_expanded.csv"),
                       ("C7", "model_architecture_metadata.csv")]:
        df = pd.read_csv(RA / f"C_efficiency_evolution/{fname}")
        out[tag] = {"rows": int(len(df)), "columns": list(df.columns)}
    dr = RA / "C_efficiency_evolution/detailed_results"
    dirs = [d for d in dr.iterdir() if d.is_dir()]
    jsons = list(dr.rglob("*.json"))
    corrupt = 0
    for j in jsons:
        try:
            json.loads(j.read_text(encoding="utf-8")[:200] + "}")
        except Exception:
            pass
    out["C8_catalogue"] = {"model_dirs": len(dirs), "json_files": len(jsons),
                           "note": "Q4 deferred: only catalogued, not parsed."}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "artifacts/audit"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = _inventory()
    pd.DataFrame(inventory).to_csv(outdir / "file_inventory.csv", index=False)

    quality, keys_by_file = _audit_quality()
    write_json(outdir / "quality_audit.json", quality)

    # write deduplicated key list count only (keys themselves are large)
    merged = keys_by_file["A1"] | keys_by_file["A2"] | keys_by_file["A3"]
    write_json(outdir / "quality_dedup_keys.json",
               {"n_merged": len(merged), "n_A1": len(keys_by_file["A1"]),
                "n_A2": len(keys_by_file["A2"]), "n_A3": len(keys_by_file["A3"])})

    write_json(outdir / "mixture_audit.json", _audit_mixture())
    write_json(outdir / "quality_supplementary_audit.json", _audit_quality_supplementary())
    write_json(outdir / "scaling_audit.json", _audit_scaling())
    write_json(outdir / "c_catalogue.json", _audit_c_catalogue())

    summary = {
        "inventory_files": len(inventory),
        "total_bytes": int(sum(e["bytes"] for e in inventory)),
        "quality": {k: quality[k] for k in ("overlaps", "merged_unique_records", "merged_per_domain")},
        "notes": [
            "A1-A3 全量读取；合并按 (quality_domain, id) 去重。",
            "配比表与 Loss 表仅按同一数据组的 index 一对一连接。",
            "C 附件仅编目（Q4 在 Baseline A 中暂缓）。",
        ],
    }
    write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
