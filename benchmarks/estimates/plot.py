#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""生成正文里的两张定量图（1-5 带宽、1-6 显存）。

为什么单独用脚本画：这两张图的每个数字都要能追溯到 CSV，
改了估算参数重跑一次即可，不靠手改 SVG。

用法（在仓库根目录）：
    uv run benchmarks/estimates/plot.py            # 输出到 docs/public/figures/
    uv run benchmarks/estimates/plot.py --preview  # 额外导出 PNG 到 figures/out/preview/

数据口径：vram-8k.csv 是 1.6 节的估算账本（f16 KV，8K 上下文）；
bandwidth.csv 的带宽取自各厂商规格页，tok/s 由 带宽 ÷ 权重字节 推出。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "public" / "figures"
PREVIEW = ROOT / "figures" / "out" / "preview"

INK = "#111827"
MUTED = "#6b7280"
RULE = "#e5e7eb"
BLUE = "#3b82f6"
BLUE_LIGHT = "#bfdbfe"
AMBER = "#f59e0b"
GRAY = "#d1d5db"

# 图里要出现中文，显式指定一个 cairo 与浏览器都能解析的字体。
for family in ("Hiragino Sans GB", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"):
    if any(family == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break
plt.rcParams.update(
    {
        "svg.fonttype": "path",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": RULE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 11,
        "axes.unicode_minus": False,
    }
)


def read_csv(name: str) -> list[dict[str, str]]:
    with open(HERE / name, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_vram(preview: bool) -> None:
    rows = read_csv("vram-8k.csv")
    models = [r["model"] for r in rows]
    weights = [float(r["weights_gb"]) for r in rows]
    kv = [float(r["kv_cache_gb"]) for r in rows]
    runtime = [float(r["runtime_gb"]) for r in rows]
    margin = [
        (w + k + rt) * float(row["margin_ratio"])
        for w, k, rt, row in zip(weights, kv, runtime, rows)
    ]

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=100)
    y = list(range(len(models)))
    left = [0.0] * len(models)

    for values, color, label, hatch in (
        (weights, BLUE, "权重（GGUF 文件）", None),
        (kv, BLUE_LIGHT, "KV Cache（8K 上下文，f16）", None),
        (runtime, GRAY, "计算缓冲 + 运行时开销", None),
        (margin, "none", "20% 安全余量", "///"),
    ):
        ax.barh(y, values, left=left, height=0.5, color=color, edgecolor=INK if hatch else "white",
                linewidth=0.8 if hatch else 0.6, hatch=hatch, label=label, zorder=3)
        left = [a + b for a, b in zip(left, values)]

    for index, total in enumerate(left):
        ax.text(total + 0.35, index, f"{total:.1f} GB", va="center", ha="left", fontsize=11, color=INK)

    ax.axvline(18.5, color=AMBER, linestyle=(0, (4, 3)), linewidth=1.6, zorder=2)
    ax.text(19.2, 0.18, "24GB Mac 可用额度\n约 18–19 GB（估算）",
            ha="left", va="center", fontsize=10, color=AMBER, linespacing=1.5)

    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 26.5)
    ax.set_xlabel("GB（10 亿字节，与文件大小同口径）")
    ax.set_title("三个模型在 24GB Mac 上的显存账本", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=10, ncol=4)
    fig.tight_layout()
    save(fig, "1-6-vram-budget", preview)


def plot_bandwidth(preview: bool) -> None:
    rows = read_csv("bandwidth.csv")
    bandwidths = [float(r["bandwidth_gb_s"]) for r in rows]
    weights = float(rows[0]["weights_gb"])
    ceiling = [b / weights for b in bandwidths]

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=100)
    xs = [40, 4200]
    ax.plot(xs, [x / weights for x in xs], color=BLUE, linewidth=2, zorder=2,
            label=f"理论上限 = 带宽 ÷ {weights} GB（权重）")

    offsets = {"Apple M4": (-10, 14), "RTX 4090": (-10, 16), "H100 SXM": (-10, 16)}
    for row, b, c in zip(rows, bandwidths, ceiling):
        ax.scatter([b], [c], s=64, color=BLUE, zorder=4)
        dx, dy = offsets.get(row["device"], (10, 12))
        ax.annotate(f"{row['device']}\n{c:.0f} token/s", (b, c), textcoords="offset points",
                    xytext=(dx, dy), fontsize=10, color=INK, ha="right" if dx < 0 else "left")

    # 实测点：M4 上 tg128 = 17.2 tok/s（benchmarks/prefill-decode/depth.csv）
    ax.scatter([120], [17.2], s=64, facecolor="white", edgecolor=AMBER, linewidth=2, zorder=5)
    ax.annotate("M4 实测 17.2 token/s\n（达到上限的 82%）", (120, 17.2), textcoords="offset points",
                xytext=(14, -34), fontsize=10, color=AMBER)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(60, 5000)
    ax.set_ylim(5, 1200)
    ax.set_xticks([100, 500, 1000, 3000], ["100", "500", "1000", "3000"])
    ax.set_yticks([10, 30, 100, 300, 600], ["10", "30", "100", "300", "600"])
    ax.set_xlabel("内存带宽（GB/s）")
    ax.set_ylabel("decode 速度（token/s）")
    ax.set_title("decode 速度随带宽线性增长，与算力无关", loc="left", fontsize=14, color=INK, pad=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "1-5-bandwidth-scaling", preview)


def read_quant(name: str) -> list[dict[str, str]]:
    """量化实验的 CSV 在 benchmarks/quantization/，与本目录的估算账本分开。"""
    with open(ROOT / "benchmarks" / "quantization" / name, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_group_size(preview: bool) -> None:
    rows = read_quant("group-size.csv")
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=100)
    groups = [16, 32, 64, 128, 256, 512, 1024]

    for bits, color, label in ((4, BLUE, "4 位"), (8, "#9ca3af", "8 位")):
        xs, ys = [], []
        for row in rows:
            if int(row["bits"]) != bits or row["scheme"] != "symmetric" or row["group"] == "per-tensor":
                continue
            xs.append(int(row["group"]))
            ys.append(float(row["rmse_norm"]))
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, zorder=3)
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color=color)
        per_tensor = next(
            float(r["rmse_norm"]) for r in rows
            if int(r["bits"]) == bits and r["scheme"] == "symmetric" and r["group"] == "per-tensor"
        )
        ax.axhline(per_tensor, color=color, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"逐张量 {per_tensor:.2f}", xy=(16, per_tensor), xytext=(2, 7), textcoords="offset points",
                    va="bottom", ha="left", fontsize=9, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(groups, [str(g) for g in groups])
    ax.set_xlim(14, 2600)
    ax.set_xlabel("分组大小（每个 scale 管多少个权重）")
    ax.set_ylabel("归一化 RMSE（越小越好）")
    ax.set_title("分组越小误差越低，但每组的 scale 也是要存的", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "4-2-group-size", preview)


def plot_ppl_domain(preview: bool) -> None:
    rows = read_quant("ppl-domain.csv")
    labels = [r["domain"] for r in rows]
    values = [float(r["ppl"]) for r in rows]
    chars = [int(r["chars"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=100)
    y = list(range(len(labels)))
    ax.barh(y, values, height=0.55, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.2, zorder=3)
    for index, (value, n) in enumerate(zip(values, chars)):
        ax.text(value + 0.12, index, f"{value:.2f}", va="center", ha="left", fontsize=11, color=INK)
        ax.text(value + 0.75, index, f"样本 {n} 字符", va="center", ha="left", fontsize=9, color=MUTED)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 11.5)
    ax.set_xlabel("困惑度（同一模型、同一参数）")
    ax.set_title("换个语料域，困惑度差 3 倍多", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "4-6-ppl-domain", preview)


def save(fig: plt.Figure, name: str, preview: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", format="svg", facecolor="white")
    if preview:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{name}.png", format="png", dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote docs/public/figures/{name}.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="额外导出 PNG 审阅图")
    args = parser.parse_args()
    plot_vram(args.preview)
    plot_bandwidth(args.preview)
    plot_group_size(args.preview)
    plot_ppl_domain(args.preview)
