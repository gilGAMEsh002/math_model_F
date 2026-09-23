# -*- coding: utf-8 -*-
"""检查图注长度：图注应 <=20 字（表注不受此限）。"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pat = re.compile(r"\\caption\{([^}]*)\}")
strip = re.compile(r"\\[a-zA-Z]+|\$[^$]*\$|[{}]")
figpat = re.compile(r"\\label\{(fig:[^}]*)\}")

bad = []
for f in sorted((ROOT / "sections").glob("*.tex")) + [ROOT / "main.tex"]:
    txt = f.read_text(encoding="utf-8")
    for m in pat.finditer(txt):
        cap = m.group(1)
        tail = txt[m.end():m.end() + 200]
        is_fig = "fig:" in tail
        n = len(strip.sub("", cap))
        if is_fig:
            mark = "  <-- 超过 20 字" if n > 20 else ""
            print(f"[图注] {n:3d} 字  {cap}{mark}")
            if n > 20:
                bad.append((f.name, cap, n))
print(f"\n共 {len(bad)} 个图注超过 20 字")
sys.exit(1 if bad else 0)
