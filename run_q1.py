# -*- coding: utf-8 -*-
"""问题一（Q1）完整流程一键入口：预处理 -> 独立验证 -> Q1-C 建模。

本目录把原分支中分处两个仓库的内容合并到同一根目录：

    D:\\26mathmodel\\q1-c\\
    ├── src\\preprocess\\               ← 原 D:\\26mathmodel\\q1_preprocessing\\src\\preprocess\\
    │   ├── quality_preprocess.py       Step 1：原始 JSONL -> Z ∈ [0,1]^(N×22)
    │   └── verify_preprocess.py        独立重算验证（46 项）
    ├── src\\baselines\\Q1-C\\          ← 原 math_model_F 仓库 baseline-q1-c 分支
    │   ├── quality.py                  entropy weight quality score + conflict
    │   ├── mixture.py                  17 domain mixture -> 13 validation loss
    │   ├── diagnostics.py              post-hoc diagnostics
    │   ├── run_all.py                  端到端入口
    │   └── common.py                   IO / metrics / hash
    ├── configs\\baselines\\Q1-C.yaml   Q1-C 全部可调旋钮（上游路径已改为本目录内相对路径）
    ├── configs\\quality_preprocess_params.json
    ├── artifacts\\q1\\preprocessed\\   Step 1 产物（A1/A2/A3_processed.parquet）
    ├── artifacts\\baselines\\Q1-C\\    Q1-C 产物（25 个文件）
    ├── reports\\                       两份报告
    └── real_attachments\\              原始数据附件（530 MB，2012 文件）

两个上游脚本都用「相对 __file__」解出根目录（SRC_DIR 上两级），因此搬迁后
无需改任何代码；Q1-C 的 config 里也只有上游三条路径被改过。

用法（在任意目录下均可执行）：

    python run_q1.py                     # 全流程：预处理 -> 验证 -> Q1-C
    python run_q1.py --skip-verify       # 跳过独立验证
    python run_q1.py --skip-preprocess   # 复用已有 parquet，只跑 Q1-C
    python run_q1.py --only q1c          # 只跑指定步骤
    python run_q1.py --list              # 列出步骤

耗时参考：预处理约 20 s，验证约 5 s，Q1-C 约 93 s。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# 上游 Q1-C 报告的运行命令是 `PYTHONIOENCODING=utf-8 python ...`，即子进程按 UTF-8 输出。
# 这里把本脚本自身的输出也固定为 UTF-8，避免父进程走控制台默认 GBK、子进程走 UTF-8
# 导致同一屏里两种编码混排。若控制台中文显示为乱码，执行 `chcp 65001` 切到 UTF-8 代码页。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# (步骤 id, 说明, 脚本相对路径)
STEPS: list[tuple[str, str, str]] = [
    ("preprocess", "Step 1 数据预处理：原始 JSONL -> Z ∈ [0,1]^(N×22)",
     os.path.join("src", "preprocess", "quality_preprocess.py")),
    ("verify", "Step 1 独立验证：不信任上一步产物，从原始 JSONL 重算核对",
     os.path.join("src", "preprocess", "verify_preprocess.py")),
    ("q1c", "Q1-C 建模：熵权质量分 + 冲突消解 + 随机森林配比回归",
     os.path.join("src", "baselines", "Q1-C", "run_all.py")),
]

LINE = "=" * 72


def run_step(step_id: str, desc: str, script_rel: str) -> tuple[bool, float]:
    """跑一个步骤，返回 (是否成功, 耗时秒)。"""
    script = os.path.join(ROOT, script_rel)
    if not os.path.isfile(script):
        print(f"[缺失] {script}")
        return False, 0.0

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print(f"\n{LINE}\n[{step_id}] {desc}\n命令: {sys.executable} {script_rel}\n{LINE}", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, script], cwd=ROOT, env=env)
    elapsed = time.perf_counter() - t0

    status = "成功" if proc.returncode == 0 else f"失败 (exit={proc.returncode})"
    print(f"\n[{step_id}] {status}，耗时 {elapsed:.1f}s", flush=True)
    return proc.returncode == 0, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Q1 完整流程：预处理 -> 独立验证 -> Q1-C 建模",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], choices=[s[0] for s in STEPS],
                        help="只跑指定步骤（可重复）")
    parser.add_argument("--skip", action="append", default=[], choices=[s[0] for s in STEPS],
                        help="跳过指定步骤（可重复）")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="等价于 --skip preprocess（复用已有 parquet）")
    parser.add_argument("--skip-verify", action="store_true", help="等价于 --skip verify")
    parser.add_argument("--keep-going", action="store_true",
                        help="某步失败后继续跑后续步骤（默认失败即停）")
    parser.add_argument("--list", action="store_true", help="列出步骤后退出")
    args = parser.parse_args()

    if args.list:
        for sid, desc, rel in STEPS:
            print(f"  {sid:11s} {desc}\n              {rel}")
        return 0

    skip = set(args.skip)
    if args.skip_preprocess:
        skip.add("preprocess")
    if args.skip_verify:
        skip.add("verify")

    todo = [s for s in STEPS if (not args.only or s[0] in args.only) and s[0] not in skip]

    print(f"Q1 完整流程  根目录 {ROOT}")
    print(f"计划执行 {len(todo)} 个步骤: {', '.join(s[0] for s in todo) or '(无)'}")
    if skip:
        print(f"已跳过: {', '.join(sorted(skip))}")

    t_all = time.perf_counter()
    results: list[tuple[str, bool, float]] = []
    for sid, desc, rel in todo:
        ok, elapsed = run_step(sid, desc, rel)
        results.append((sid, ok, elapsed))
        if not ok and not args.keep_going:
            print(f"\n[{sid}] 失败，已中止。修好后可只重跑该步："
                  f"python run_q1.py --only {sid}", flush=True)
            break

    # ---- 汇总
    total = time.perf_counter() - t_all
    print(f"\n{LINE}\n汇总\n{LINE}")
    # 注：不用 ✓/✗ —— 这两个字符不在 GBK 码表内，Windows 控制台(代码页 936)会抛
    # UnicodeEncodeError。改用纯 ASCII 标记，避免依赖控制台编码。
    for sid, ok, elapsed in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {sid:11s} {elapsed:7.1f}s")
    skipped = [s[0] for s in STEPS if s not in todo]
    if skipped:
        print(f"  [SKIP] {', '.join(skipped)}")
    print(f"  总耗时 {total:.1f}s")

    failed = [sid for sid, ok, _ in results if not ok]
    if failed:
        print(f"\n失败步骤: {', '.join(failed)}")
        return 1

    print(f"\n产物：")
    print(f"  预处理  artifacts/q1/preprocessed/{{A1,A2,A3}}_processed.parquet")
    print(f"  Q1-C    artifacts/baselines/Q1-C/（25 个文件）")
    print(f"  报告    reports/q1_preprocess_report.md、reports/baselines/Q1-C.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
