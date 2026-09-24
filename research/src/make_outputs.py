"""Produce the two remaining figures (data already exists) and render the manuscript PDF.

No new experiments: figures use existing reports/*.csv; PDF renders manuscript/正文.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import RESEARCH
from .adapters import LineB

FIG = RESEARCH / "reports/figures"
MAN = RESEARCH / "manuscript"


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # F2: per-domain RMSE by line (1M)
    p = RESEARCH / "reports/r1_q1_perdomain.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["set"] == "1m"]
        piv = d.pivot_table(index="target", columns="line", values="rmse")
        piv = piv.sort_values("A")
        ax = piv.plot(kind="bar", figsize=(11, 3.6), width=0.8)
        ax.set_ylabel("RMSE (1M, per domain)")
        ax.set_title("Per-domain RMSE on the same candidate set (A6/A7, exploratory)", fontsize=9)
        ax.legend(fontsize=7, ncol=4); plt.tight_layout()
        plt.savefig(FIG / "F2_perdomain_rmse_1m.png", dpi=160); plt.close()

    # F8: A vs C score alignment scatter with isotonic mapping
    p = RESEARCH / "reports/r3a_score_alignment.csv"
    if p.exists():
        d = pd.read_csv(p)
        sub = d.sample(min(5000, len(d)), random_state=0)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(sub["S_A_equal"], sub["S_C_entropy"], s=3, alpha=0.25, label="records")
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip").fit(d["S_A_equal"], d["S_C_entropy"])
            xs = np.linspace(d["S_A_equal"].min(), d["S_A_equal"].max(), 200)
            ax.plot(xs, iso.predict(xs), "r-", lw=1.5, label="isotonic A->C")
        except Exception:
            pass
        ax.set_xlabel("A equal-weight score"); ax.set_ylabel("C entropy score")
        ax.set_title("A/C text-score alignment (A1, n=51230)", fontsize=9)
        ax.legend(fontsize=7); plt.tight_layout()
        plt.savefig(FIG / "F8_score_alignment.png", dpi=160); plt.close()


def _cjk_font():
    for name in ("SimHei.ttf", "KaiTi.ttf", "simsun.ttf"):
        p = Path(LineB().root) / name
        if p.exists():
            return str(p)
    return None


def render_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer

    font = _cjk_font()
    base = "Helvetica"
    if font:
        pdfmetrics.registerFont(TTFont("CJK", font))
        base = "CJK"
    h1 = ParagraphStyle("h1", fontName=base, fontSize=15, leading=20, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName=base, fontSize=12.5, leading=17, spaceAfter=4)
    body = ParagraphStyle("body", fontName=base, fontSize=9.5, leading=14, spaceAfter=4)
    pre = ParagraphStyle("pre", fontName=base, fontSize=7.2, leading=9)

    md = (MAN / "正文.md").read_text(encoding="utf-8").splitlines()
    story = []
    buf_pre = []

    def flush():
        if buf_pre:
            story.append(Preformatted("\n".join(buf_pre), pre))
            story.append(Spacer(1, 4)); buf_pre.clear()

    figdir = RESEARCH / "reports"
    for ln in md:
        if ln.startswith("[[FIG:"):
            flush()
            spec = ln.strip()[6:-2]
            name, _, cap = spec.partition("|")
            fp = figdir / name if not name.startswith("figures/") else figdir / name
            if fp.exists():
                from PIL import Image as PILImage  # noqa: F401
                try:
                    from reportlab.lib.utils import ImageReader
                    ir = ImageReader(str(fp)); iw, ih = ir.getSize()
                    w = 170 * mm
                    story.append(Image(str(fp), width=w, height=w * ih / iw))
                    story.append(Paragraph(cap, pre)); story.append(Spacer(1, 4))
                except Exception as exc:
                    story.append(Paragraph("(图嵌入失败: %s)" % exc, pre))
            else:
                story.append(Paragraph("(缺图: %s)" % name, pre))
            continue
        if ln.startswith("|") or ln.startswith("$$") or ln.strip().startswith("\\["):
            buf_pre.append(ln); continue
        flush()
        if ln.startswith("# "):
            story.append(Paragraph(ln[2:], h1))
        elif ln.startswith("## "):
            story.append(Paragraph(ln[3:], h2))
        elif ln.startswith("### "):
            story.append(Paragraph(ln[4:], h2))
        elif ln.strip() == "" or ln.strip() == "---":
            story.append(Spacer(1, 2))
        else:
            import re as _re
            ln = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ln)
            esc = (ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            esc = esc.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            story.append(Paragraph(esc, body))
    flush()
    out = MAN / "正文.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm,
                      leftMargin=18 * mm, rightMargin=18 * mm,
                      title="算力约束下提升大语言模型能力的资源配置建模").build(story)
    return out, base


def main():
    figures()
    out, base = render_pdf()
    print("figures ->", FIG)
    print("pdf ->", out, "| font:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
