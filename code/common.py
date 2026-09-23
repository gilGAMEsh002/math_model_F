# -*- coding: utf-8 -*-
"""
公共模块：数据路径、来源分级、全局绘图样式、通用工具。
所有问题脚本共用，保证口径一致（见《Baseline 公共约定与接口》）。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT_CANDIDATES = [
    ROOT / "data" / "lfs_src" / "real_attachments",          # LFS 完整克隆（含 A1-A3）
    Path(r"C:\Users\Administrator\Desktop\math_model_F-main\real_attachments"),
]
DATA = next((p for p in DATA_ROOT_CANDIDATES if p.is_dir()), DATA_ROOT_CANDIDATES[-1])

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 20260101
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# 来源分级标签（题面/数据说明要求区分）
# ----------------------------------------------------------------------------
SOURCE_LEVEL = {
    "observed": "真实观测",
    "semi_synthetic": "半合成（真实标度律校准 + 噪声）",
    "interpolated": "插值生成",
    "estimated": "模型估算",
    "subset": "子集（非独立实验）",
    "reference": "参考/说明",
    "mixed": "混合来源",
}

# 17 个 The Pile 训练域（顺序与 A4/A5 列顺序一致）
TRAIN_DOMAINS = [
    "arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
    "dm_mathematics", "github", "philpapers", "stackexchange", "enron_emails",
    "gutenberg_pg_19", "pile_cc", "ubuntu_irc", "europarl", "hackernews",
    "pubmed_abstracts", "uspto_backgrounds",
]
# A5/A7/... Loss 表中实际存在的 13 个验证域（缺 nih_exporter/enron_emails/europarl/philpapers）
LOSS_DOMAINS = [
    "arxiv", "freelaw", "pubmed_central", "wikipedia_en", "dm_mathematics",
    "github", "stackexchange", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
    "hackernews", "pubmed_abstracts", "uspto_backgrounds",
]
# 4 个只有训练配比、无对应交叉熵损失的域
NO_LOSS_DOMAINS = ["nih_exporter", "enron_emails", "europarl", "philpapers"]

# 7 个质量信号域
QUALITY_DOMAINS = ["arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"]

# 22 个质量指标：14 个标量 + 8 个多维列表型
SCALAR_METRICS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "dsir_books", "dsir_wiki",
    "dsir_math", "qurater", "ad_en", "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
]
LIST_METRICS = [
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction", "rps_lines_ending_with_terminal_punctuation_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
ALL_METRICS = SCALAR_METRICS + LIST_METRICS

# 质量语义分组（依据 SlimPajama-Meta-rater 数据卡：教育价值 / 可读性 / 专业性 / 噪声）
QUALITY_GROUPS = {
    "edu_value":  ["fineweb_edu", "dsir_books", "dsir_wiki", "dsir_math", "qurater",
                   "modernbert_reasoning"],
    "readability": ["modernbert_readability", "modernbert_cleanliness", "fluency_en",
                    "rps_doc_unigram_entropy"],
    "professionalism": ["modernbert_professionalism", "rps_doc_mean_word_length",
                        "rps_doc_frac_unique_words"],
    "noise_free": ["ad_en", "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
                   "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
                   "rps_lines_ending_with_terminal_punctuation_mark",
                   "rps_lines_numerical_chars_fraction", "rps_doc_word_count",
                   "rps_doc_num_sentences"],
}
# 反向指标：原始值越高越差，需做补转换
NEGATIVE_METRICS = ["ad_en", "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
                    "rps_lines_uppercase_letter_fraction", "rps_doc_frac_no_alph_words",
                    "rps_lines_numerical_chars_fraction"]

# ----------------------------------------------------------------------------
# 全局绘图样式（统一配色 + SimSun 中文字体）
# ----------------------------------------------------------------------------
PALETTE = ["#2E5C8A", "#C8553D", "#4E8A6B", "#E0A33E", "#7B5EA7",
           "#3FA7B5", "#B5527E", "#7A8B3F", "#8C6D4F", "#5C6B7A"]
GROUP_COLORS = {"edu_value": "#2E5C8A", "readability": "#4E8A6B",
                "professionalism": "#E0A33E", "noise_free": "#C8553D"}
DOMAIN_COLORS = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(TRAIN_DOMAINS)}
QUALITY_COST_COLORS = {"exp": "#C8553D", "power": "#2E5C8A", "log": "#4E8A6B"}
CTX_COLORS = {2048: "#A8C6DF", 4096: "#7BA7CC", 8192: "#4E8A6B",
              32768: "#E0A33E", 131072: "#C8553D"}


def _register_fonts():
    """注册工作目录内的中文字体，确保 SimSun 可用、不出方块。"""
    from matplotlib import font_manager
    font_files = ["simsun.ttf", "SimHei.ttf", "KaiTi.ttf", "LiSu.ttf"]
    for ff in font_files:
        p = ROOT / ff
        if p.exists():
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:
                pass
    for fam in ["SimSun", "SimHei", "KaiTi", "LiSu"]:
        try:
            font_manager.findfont(fam, fallback_to_default=False)
        except Exception:
            pass


_register_fonts()
plt.rcParams.update({
    "font.sans-serif": ["SimSun", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 330,
    "savefig.bbox": "tight",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "axes.edgecolor": "#4A4A4A",
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

try:
    import seaborn as sns
    sns.set_theme(style="ticks", palette=PALETTE, rc=plt.rcParams)
    sns.set_context("notebook")
    plt.rcParams.update({
        "font.sans-serif": ["SimSun", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "savefig.dpi": 330,
    })
except Exception:
    sns = None


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def savefig(fig, name, tight=True):
    """统一保存：PNG(330dpi) + PDF 矢量，返回 PNG 路径。"""
    png = FIGURES / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {png}")
    return png


def panel_label(ax, text, dx=-0.09, dy=1.04):
    """子图 (a)(b)(c) 标注。"""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="bottom", ha="left")


def sha256_file(path, blocksize=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(blocksize), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERR:{e}"


def is_lfs_pointer(path):
    """检测 Git LFS 指针文件（未拉取实体）。"""
    try:
        if os.path.getsize(path) > 4096:
            return False
        with open(path, "rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs")
    except Exception:
        return False


def save_json(obj, name):
    p = RESULTS / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"[json] {p}")
    return p


def save_csv(df, name, float_format="%.6g"):
    p = RESULTS / name
    df.to_csv(p, index=False, float_format=float_format)
    print(f"[csv ] {p}  {df.shape}")
    return p


def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman(y, yhat):
    from scipy.stats import spearmanr
    return float(spearmanr(y, yhat).statistic)
