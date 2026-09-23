#!/bin/bash
MIRROR="http://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/win32/miktex/tm/packages"
WORK=auto
declare -A ALIAS=( [kvdefinekeys]=kvoptions [kvsetkeys]=kvsetkeys [kvoptions]=kvoptions
  [suffix]=bigfoot [bigstrut]=multirow [bigdelim]=multirow [indentfirst]=indentfirst
  [calc]=latex-tools [titletoc]=titlesec [subfig]=caption [ms]=psnfss [bm]=latex-tools )
for round in 1 2 3 4 5 6 7 8; do
  xelatex -interaction=nonstopmode main.tex > /tmp/loop.log 2>&1
  miss=$(grep -oE "File \`[^']+' not found" /tmp/loop.log | sed "s/File \`//;s/' not found//" | sort -u)
  [ -z "$miss" ] && { echo "ROUND $round: 无缺失文件"; break; }
  echo "ROUND $round 缺失: $(echo $miss | tr '\n' ' ')"
  got=0
  for f in $miss; do
    base=$(basename "$f" | sed 's/\.[^.]*$//')
    for cand in "${ALIAS[$base]:-}" "$base"; do
      [ -z "$cand" ] && continue
      out="$WORK/$cand.tar.lzma"
      [ -s "$out" ] && continue
      curl -sSL -o "$out" "$MIRROR/$cand.tar.lzma" 2>/dev/null
      sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
      if [ "$sz" -gt 1000 ]; then echo "   + $cand ($sz)"; got=1; else rm -f "$out"; fi
    done
  done
  [ "$got" = "0" ] && { echo "ROUND $round: 无法自动获取，停止"; break; }
  python - <<'PY'
import lzma,tarfile,io
from pathlib import Path
D=Path(r"C:\Users\Administrator\AppData\Roaming\MiKTeX")
for f in Path('auto').glob("*.tar.lzma"):
    try:
        tf=tarfile.open(fileobj=io.BytesIO(lzma.decompress(f.read_bytes())))
        for m in tf.getmembers():
            if m.name.startswith("texmf/"):
                m.name=m.name[6:]
                if m.name: tf.extract(m,D,set_attrs=False)
    except Exception: pass
PY
  initexmf --update-fndb > /dev/null 2>&1
done
