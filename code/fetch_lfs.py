# -*- coding: utf-8 -*-
"""通过 GitHub LFS Batch API 直接拉取缺失的大文件（A1/A2/A3/A18）。"""
import json
import sys
import urllib.request
from pathlib import Path

REPO = "https://github.com/gilGAMEsh002/math_model_F.git"
SRC = Path(r"C:\Users\Administrator\Desktop\math_model_F-main\real_attachments")
DST = Path(__file__).resolve().parent.parent / "data" / "lfs"
DST.mkdir(parents=True, exist_ok=True)

FILES = [
    "A_data_value/slimpajama_quality_signal_sample.jsonl.xz",
    "A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz",
    "A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz",
    "A_data_value/regmix_domain_sample.jsonl.xz",
]


def parse_pointer(p):
    d = {}
    for line in p.read_text().splitlines():
        if line.startswith("version"):
            continue
        k, _, v = line.partition(" ")
        d[k.strip()] = v.strip()
    return d["oid"], int(d["size"])


def batch(objects):
    req = urllib.request.Request(
        REPO.replace(".git", ".git") + "/info/lfs/objects/batch",
        data=json.dumps({"operation": "download", "transfers": ["basic"],
                         "objects": objects}).encode(),
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-Type": "application/vnd.git-lfs+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(url, out, size):
    tmp = out.with_suffix(out.suffix + ".part")
    got = 0
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            print(f"\r  {out.name}: {got/1e6:.1f}/{size/1e6:.1f} MB", end="", flush=True)
    print()
    if got != size:
        raise RuntimeError(f"size mismatch {got} != {size}")
    tmp.replace(out)


def main():
    todo = []
    for rel in FILES:
        p = SRC / rel
        oid, size = parse_pointer(p)
        out = DST / Path(rel).name
        if out.exists() and out.stat().st_size == size:
            print(f"[skip] {out.name} 已存在")
            continue
        todo.append((rel, oid, size, out))
    if not todo:
        print("全部就绪")
        return
    resp = batch([{"oid": o, "size": s} for _, o, s, _ in todo])
    by_oid = {o["oid"]: o for o in resp.get("objects", [])}
    for rel, oid, size, out in todo:
        o = by_oid.get(oid, {})
        if "actions" not in o:
            print(f"[FAIL] {rel}: {o.get('error')}")
            continue
        print(f"[get ] {rel}  {size/1e6:.1f} MB")
        download(o["actions"]["download"]["href"], out, size)
        print(f"[ ok ] {out}")


if __name__ == "__main__":
    sys.exit(main())
