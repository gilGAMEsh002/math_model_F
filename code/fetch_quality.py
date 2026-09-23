# -*- coding: utf-8 -*-
"""
从公开原始来源 SlimPajama-Meta-rater 同源重建 A1/A2/A3。

背景：仓库内 A1/A2/A3 的 .jsonl.xz 为 Git LFS 指针，GitHub 上 LFS 对象不存在（404），
无法获得实体。数据说明给出的来源为 HuggingFace `opendatalab/SlimPajama-Meta-rater`，
且 A2/A3 分片名与该数据集文件一一对应、字段结构一致，故按同源重建。

重建规则（写入论文，标注为「同源重建」）：
  A2 = arxiv/part-6777d8857c6e-000486.jsonl   全量，剔除 content，保留 24 字段
  A3 = github/part-6777d8857c6e-000275.jsonl  全量，剔除 content，保留 24 字段
  A1 = 7 域按 SlimPajama 自然配比分层抽样，保留 27 字段（含 content），
       每域在给定字节预算内做蓄水池抽样

实现：按字节区间并行分块拉取（HTTP Range 206 已验证），单流慢（~0.17 MB/s），
并行 8 路显著提速；分块边界丢弃半行，保证 JSONL 行完整性。
"""
from __future__ import annotations

import json
import lzma
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/resolve/main/"
OUT = Path(__file__).resolve().parent.parent / "data" / "quality_rebuild"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
SEED = 20260101
CHUNK = 12 << 20          # 每个 Range 请求 12 MB
WORKERS = 8

METRICS = [
    "dsir_books", "fluency_en", "rps_lines_ending_with_terminal_punctution_mark",
    "modernbert_cleanliness", "qurater", "rps_doc_num_sentences", "rps_doc_word_count",
    "ad_en", "rps_doc_frac_no_alph_words", "modernbert_reasoning",
    "rps_doc_frac_chars_top_2gram", "rps_lines_uppercase_letter_fraction",
    "rps_doc_frac_unique_words", "rps_lines_numerical_chars_fraction", "fineweb_edu",
    "dsir_math", "rps_doc_mean_word_length", "dsir_wiki",
    "rps_doc_frac_chars_top_3gram", "rps_doc_unigram_entropy",
    "modernbert_professionalism", "modernbert_readability",
]
FIELDS_EXT = ["id", "sub_path"] + METRICS
FIELDS_SAMPLE = ["id", "content", "sub_path", "_source_domain", "_source_path"] + METRICS

# A1 分层抽样：域 -> (分片, 目标条数, 字节预算MB)
A1_PLAN = {
    "commoncrawl":  ("commoncrawl/part-6777d8857c6e-000049.jsonl", 26700, 1500),
    "c4":           ("c4/part-6777d8857c6e-000103.jsonl",          13700,  800),
    "book":         ("book/part-6777d8857c6e-003454.jsonl",         2150,  700),
    "arxiv":        ("arxiv/part-6777d8857c6e-000486.jsonl",        2360,  250),
    "github":       ("github/part-6777d8857c6e-000275.jsonl",       2660,  200),
    "wikipedia":    ("wikipedia/part-6777d8857c6e-000341.jsonl",    1940,  150),
    "stackexchange":("stackexchange/part-6777d8857c6e-002279.jsonl", 1690,  120),
}
EXT_JOBS = {
    "A2_arxiv": ("arxiv/part-6777d8857c6e-000486.jsonl", "A2_arxiv.jsonl.xz"),
    "A3_github": ("github/part-6777d8857c6e-000275.jsonl", "A3_github.jsonl.xz"),
}


def log(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def file_size(url, retry=6):
    for i in range(retry):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception as e:
            if i == retry - 1:
                raise
            time.sleep(2 + 2 * i)


def fetch_range(url, start, end, retry=8):
    """拉取 [start, end] 闭区间字节，失败重试。"""
    for i in range(retry):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "curl/8", "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception:
            if i == retry - 1:
                return None
            time.sleep(1.5 * (i + 1))
    return None


def _rows_from_chunk(data, is_first, is_last):
    """把字节块切成完整 JSONL 行；非首块丢弃头部半行，非末块丢弃尾部半行。"""
    if not is_first:
        i = data.find(b"\n")
        if i < 0:
            return []
        data = data[i + 1:]
    if is_last:
        lines = data.split(b"\n")
        if data.endswith(b"\n"):
            lines = lines[:-1]
    else:
        i = data.rfind(b"\n")
        if i < 0:
            return []
        lines = data[:i].split(b"\n")
    return [ln for ln in lines if ln.strip()]


def parallel_rows(url, total, budget=None, workers=WORKERS, on_progress=None):
    """按字节区间并行产出完整 JSONL 行（顺序不保证）。"""
    limit = total if budget is None else min(total, budget)
    ranges = [(s, min(s + CHUNK, limit) - 1) for s in range(0, limit, CHUNK)]
    n_done = [0]

    def work(rg):
        i, (s, e) = rg
        d = fetch_range(url, s, e)
        if d is None:
            log(f"    !! 分块失败 {s}-{e}")
            return []
        rows = _rows_from_chunk(d, s == 0, e >= limit - 1)
        with LOCK:
            n_done[0] += 1
            if on_progress and n_done[0] % 20 == 0:
                on_progress(n_done[0], len(ranges))
        return rows

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, (i, rg)) for i, rg in enumerate(ranges)]
        for f in as_completed(futs):
            yield from f.result()


def rebuild_ext(name, rel, outname):
    out = OUT / outname
    if out.exists():
        log(f"[skip] {name} 已存在 {out.stat().st_size/1e6:.1f} MB")
        return
    url, t0, n = BASE + rel, time.time(), 0
    total = file_size(url)
    log(f"[get ] {name}: {total/1e6:.0f} MB, 并行 {WORKERS} 路")
    tmp = out.with_suffix(".part")
    with lzma.open(tmp, "wt", encoding="utf-8", preset=6) as f:
        for raw in parallel_rows(url, total):
            try:
                o = json.loads(raw)
            except Exception:
                continue
            f.write(json.dumps({k: o.get(k) for k in FIELDS_EXT},
                               ensure_ascii=False) + "\n")
            n += 1
            if n % 50000 == 0:
                log(f"  {name}: {n} 行  {(time.time()-t0)/60:.1f} 分钟")
    tmp.replace(out)
    log(f"[done] {name}: {n} 行 -> {out.name} {out.stat().st_size/1e6:.1f} MB "
        f"耗时 {(time.time()-t0)/60:.1f} 分钟")


def reservoir_stream(rel, target, budget_mb, domain, url_total):
    """在字节预算内对某域做蓄水池抽样（并行分块，无偏于所读前缀）。"""
    url = BASE + rel
    rng = random.Random(SEED + (abs(hash(domain)) % 10000))
    res, n = [], 0
    t0 = time.time()
    for raw in parallel_rows(url, url_total, budget=budget_mb * 1024 * 1024):
        try:
            o = json.loads(raw)
        except Exception:
            continue
        n += 1
        o["_source_domain"] = domain
        o["_source_path"] = rel
        if len(res) < target:
            res.append(o)
        else:
            j = rng.randint(0, n - 1)
            if j < target:
                res[j] = o
    log(f"  域 {domain}: 读取 {n} 条, 抽样 {len(res)} 条, {(time.time()-t0)/60:.1f} 分钟")
    return res


def rebuild_a1():
    out = OUT / "A1_sample.jsonl.xz"
    if out.exists():
        log(f"[skip] A1 已存在 {out.stat().st_size/1e6:.1f} MB")
        return
    names = list(A1_PLAN)
    sizes = {}
    for dom in names:
        rel = A1_PLAN[dom][0]
        sizes[dom] = file_size(BASE + rel)
    log("分片大小: " + ", ".join(f"{d}={s/1e6:.0f}MB" for d, s in sizes.items()))
    parts = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(reservoir_stream, rel, tgt, mb, dom, sizes[dom]): dom
                for dom, (rel, tgt, mb) in A1_PLAN.items()}
        for f in as_completed(futs):
            parts.extend(f.result())
    rng = random.Random(SEED)
    rng.shuffle(parts)
    tmp = out.with_suffix(".part")
    with lzma.open(tmp, "wt", encoding="utf-8", preset=6) as f:
        for o in parts:
            f.write(json.dumps({k: o.get(k) for k in FIELDS_SAMPLE},
                               ensure_ascii=False) + "\n")
    tmp.replace(out)
    log(f"[done] A1: {len(parts)} 行 -> {out.name} {out.stat().st_size/1e6:.1f} MB")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"输出目录 {OUT}")
    if which in ("all", "a1"):
        rebuild_a1()
    if which in ("all", "ext"):
        for k, (r, o) in EXT_JOBS.items():
            rebuild_ext(k, r, o)
    log("全部完成")


if __name__ == "__main__":
    main()
