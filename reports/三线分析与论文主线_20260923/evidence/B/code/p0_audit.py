# -*- coding: utf-8 -*-
"""
P0：数据审计与口径冻结。
生成文件清单、行数、字段、缺失率、来源类别与内容哈希。
"""
from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, save_csv, save_json, is_lfs_pointer, sha256_file, SOURCE_LEVEL

# 文件 -> (编号, 性质, 绑定问题)
FILE_META = {
    "A_data_value/slimpajama_quality_signal_sample.jsonl.xz": ("A1", "observed", "问题一"),
    "A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz": ("A2", "observed", "问题一"),
    "A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz": ("A3", "observed", "问题一"),
    "A_data_value/regmix_tables/train_mixture_1m.csv": ("A4", "observed", "问题一"),
    "A_data_value/regmix_tables/train_pile_loss_1m.csv": ("A5", "observed", "问题一"),
    "A_data_value/regmix_tables/test_mixture_1m.csv": ("A6", "observed", "问题一"),
    "A_data_value/regmix_tables/test_pile_loss_1m.csv": ("A7", "observed", "问题一"),
    "A_data_value/regmix_tables/test_mixture_60m.csv": ("A8", "observed", "问题一"),
    "A_data_value/regmix_tables/test_pile_loss_60m.csv": ("A9", "observed", "问题一"),
    "A_data_value/regmix_tables/test_mixture_1B.csv": ("A10", "observed", "问题一"),
    "A_data_value/regmix_tables/test_pile_loss_1B.csv": ("A11", "observed", "问题一"),
    "A_data_value/regmix_tables/est_mixture_10b.csv": ("A12", "subset", "问题一"),
    "A_data_value/regmix_tables/est_pile_loss_10b.csv": ("A13", "estimated", "问题一"),
    "A_data_value/regmix_tables/est_mixture_70b.csv": ("A14", "subset", "问题一"),
    "A_data_value/regmix_tables/est_pile_loss_70b.csv": ("A15", "estimated", "问题一"),
    "A_data_value/domain_mapping_guide.csv": ("A16", "reference", "问题一"),
    "A_data_value/regmix_domain_summary.csv": ("A17", "observed", "问题一"),
    "A_data_value/regmix_domain_sample.jsonl.xz": ("A18", "observed", "问题一"),
    "B_scaling_laws/pythia_training_log_existing.csv": ("B1", "observed", "问题二"),
    "B_scaling_laws/cerebras_training_log.csv": ("B2", "semi_synthetic", "问题二"),
    "B_scaling_laws/scaling_baseline.csv": ("B4", "observed", "问题二"),
    "B_scaling_laws/published_scaling_data.csv": ("B5", "observed", "问题二"),
    "B_scaling_laws/supplementary_NQ_experiment.csv": ("B6", "semi_synthetic", "问题二"),
    "B_scaling_laws/supplementary_NQ_experiment_expanded.csv": ("B7", "semi_synthetic", "问题二"),
    "B_scaling_laws/supplementary_NQ_experiment_large.csv": ("B8", "semi_synthetic", "问题二"),
    "B_scaling_laws/supplementary_large_models.csv": ("B9", "observed", "问题二"),
    "B_scaling_laws/supplementary_large_baseline.csv": ("B10", "estimated", "问题二"),
    "B_scaling_laws/open_model_family_metadata.csv": ("B11", "observed", "问题二"),
    "B_scaling_laws/pythia_checkpoint_index.csv": ("B12", "observed", "问题二"),
    "C_efficiency_evolution/leaderboard_cleaned.csv": ("C1", "observed", "问题四"),
    "C_efficiency_evolution/leaderboard_enhanced.csv": ("C2", "observed", "问题四"),
    "C_efficiency_evolution/leaderboard_extended_timeseries.csv": ("C3", "mixed", "问题四"),
    "C_efficiency_evolution/epoch_all_ai_models.csv": ("C4", "observed", "问题四"),
    "C_efficiency_evolution/loss_benchmark_bridge.csv": ("C5", "mixed", "问题四"),
    "C_efficiency_evolution/loss_benchmark_bridge_expanded.csv": ("C6", "mixed", "问题四"),
    "C_efficiency_evolution/model_architecture_metadata.csv": ("C7", "observed", "问题三/四"),
    "C_efficiency_evolution/data/train-00000-of-00001.parquet": ("C9", "observed", "问题四"),
}
TRAJ_DIR = "B_scaling_laws/training_trajectories"


def main():
    rows = []
    # 普通文件
    for rel, (aid, level, prob) in FILE_META.items():
        p = DATA / rel
        if not p.exists():
            rows.append(dict(id=aid, path=rel, exists=False, bytes=0, rows=None,
                             cols=None, missing_rate=None, source_level=level,
                             problem=prob, sha256="", note="文件不存在"))
            continue
        rec = dict(id=aid, path=rel, exists=True, bytes=p.stat().st_size, rows=None,
                   cols=None, missing_rate=None, source_level=SOURCE_LEVEL[level],
                   problem=prob, sha256="", note="")
        if is_lfs_pointer(p):
            rec.update(exists=True, note="Git LFS 指针（实体未拉取）", sha256="")
            rows.append(rec)
            continue
        if p.suffix == ".csv":
            try:
                df = pd.read_csv(p)
                rec["rows"], rec["cols"] = df.shape
                rec["missing_rate"] = round(float(df.isna().mean().mean()), 5)
            except Exception as e:
                rec["note"] = f"读取失败: {e}"
        elif p.suffix == ".parquet":
            try:
                df = pd.read_parquet(p)
                rec["rows"], rec["cols"] = df.shape
                rec["missing_rate"] = round(float(df.isna().mean().mean()), 5)
            except Exception as e:
                rec["note"] = f"读取失败: {e}"
        rec["sha256"] = sha256_file(p)[:16]
        rows.append(rec)

    # 轨迹目录 B3
    tdir = DATA / TRAJ_DIR
    if tdir.is_dir():
        files = sorted(tdir.glob("*.csv"))
        tot = 0
        for f in files:
            try:
                tot += len(pd.read_csv(f))
            except Exception:
                pass
        rows.append(dict(id="B3", path=f"{TRAJ_DIR}/*.csv ({len(files)} 个)", exists=True,
                         bytes=sum(f.stat().st_size for f in files), rows=tot, cols=4,
                         missing_rate=None, source_level=SOURCE_LEVEL["interpolated"],
                         problem="问题二", sha256="", note="Pythia 插值轨迹"))

    # C8 逐任务目录
    c8 = DATA / "C_efficiency_evolution/detailed_results"
    if c8.is_dir():
        dirs = [d for d in c8.iterdir() if d.is_dir()]
        njson = sum(1 for d in dirs for _ in d.glob("*.json"))
        rows.append(dict(id="C8", path="C_efficiency_evolution/detailed_results/", exists=True,
                         bytes=sum(f.stat().st_size for d in dirs for f in d.glob("*.json")),
                         rows=njson, cols=None, missing_rate=None,
                         source_level=SOURCE_LEVEL["observed"], problem="问题四",
                         sha256="", note=f"{len(dirs)} 个模型子目录 / {njson} 个 JSON"))

    df = pd.DataFrame(rows)
    save_csv(df, "audit_files.csv")

    summary = {
        "data_root": str(DATA),
        "n_files": int(len(df)),
        "n_missing_entity": int((df["note"].astype(str).str.contains("LFS 指针")).sum()),
        "total_bytes": int(df["bytes"].sum()),
        "by_level": df["source_level"].value_counts().to_dict(),
    }
    save_json(summary, "audit_summary.json")
    print(json_dumps(summary))
    print(df[["id", "exists", "rows", "cols", "bytes", "source_level", "note"]].to_string(index=False))


def json_dumps(o):
    import json
    return json.dumps(o, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
