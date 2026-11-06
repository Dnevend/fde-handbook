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
from matplotlib.ticker import NullFormatter

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


def read_rag(name: str) -> list[dict[str, str]]:
    """第五部分的实验数据在 benchmarks/rag/。"""
    with open(ROOT / "benchmarks" / "rag" / name, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_long_context(preview: bool) -> None:
    rows = [row for row in read_rag("long-context.csv") if row["mode"] == "long_context"]
    rag = next(row for row in read_rag("long-context.csv") if row["mode"] == "rag_top5")

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    positions = {"0.0": "开头", "0.5": "中间", "1.0": "结尾"}
    colors = {"0.0": BLUE, "0.5": AMBER, "1.0": "#10b981"}
    offsets = {"0.0": (8, 10), "0.5": (-34, 12), "1.0": (8, -18)}
    for key, label in positions.items():
        subset = [row for row in rows if row["position"] == key]
        xs = [int(row["context_tokens"]) for row in subset]
        ys = [float(row["ttft_s"]) for row in subset]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.6, color=colors[key], label=f"答案在{label}")
        dx, dy = offsets[key]
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=9, color=colors[key])

    rag_tokens = int(rag["context_tokens"])
    rag_ttft = float(rag["ttft_s"])
    ax.scatter([rag_tokens], [rag_ttft], marker="*", s=180, color=INK, zorder=5, label=f"RAG top-5（{rag_tokens} token）")
    ax.annotate(f"RAG：{rag_ttft:.1f}s\n上下文只有 {rag_tokens / 1000:.1f}k token",
                (rag_tokens, rag_ttft), textcoords="offset points", xytext=(16, 22), fontsize=9, color=INK, ha="left")

    ax.set_xscale("log")
    ax.set_xticks([2000, 8000, 20000], ["2k", "8k", "20k"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("上下文 token 数")
    ax.set_ylabel("首字延迟（秒）")
    ax.set_ylim(0, 130)
    ax.set_title("答案都在，代价是首字延迟随上下文线性涨", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "5-1-long-context", preview)


def plot_chunking(preview: bool) -> None:
    rows = read_rag("chunking.csv")
    labels = {"structure": "按结构", "fixed": "按长度", "semantic": "按语义"}
    names = [labels[row["strategy"]] for row in rows]
    recall = [float(row["recall@5"]) for row in rows]
    mrr = [float(row["mrr@10"]) for row in rows]
    chunks = [int(row["chunks"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    x = list(range(len(names)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], recall, width, color=BLUE, label="Recall@5", zorder=3)
    ax.bar([i + width / 2 for i in x], mrr, width, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1, label="MRR@10", zorder=3)
    for i, (r, m, c) in enumerate(zip(recall, mrr, chunks)):
        ax.text(i, -0.07, f"{c} 块", ha="center", fontsize=9, color=MUTED)
        ax.text(i - width / 2, r + 0.01, f"{r:.2f}", ha="center", fontsize=10, color=INK)
        ax.text(i + width / 2, m + 0.01, f"{m:.2f}", ha="center", fontsize=10, color=INK)

    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylabel("指标（越高越好）")
    ax.set_title("同一套向量，只换切法：按结构最稳，语义切分反而更碎", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "5-3-chunking", preview)


def plot_ann(preview: bool) -> None:
    rows = read_rag("ann.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)

    for dataset, label, color in (("real", "真实语料 794 块", BLUE), ("random-20k", "随机向量 2 万条", AMBER)):
        subset = [row for row in rows if row["dataset"] == dataset and row["method"] == "hnsw"]
        xs = [int(row["ef_search"]) for row in subset]
        recall = [float(row["recall@10"]) for row in subset]
        latency = [float(row["p50_ms"]) for row in subset]
        left.plot(xs, recall, marker="o", linewidth=2, color=color, label=label)
        right.plot(xs, latency, marker="o", linewidth=2, color=color, label=label)

    left.set_xscale("log", base=2)
    left.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
    left.set_ylim(0, 1.05)
    left.set_xlabel("efSearch")
    left.set_ylabel("Recall@10（对暴力检索）")
    left.set_title("召回：真实数据很快饱和", loc="left", fontsize=13, pad=10)
    left.legend(frameon=False, fontsize=9, loc="lower right")

    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
    right.set_xlabel("efSearch")
    right.set_ylabel("单次查询 p50（毫秒，对数轴）")
    right.set_title("延迟：随机向量上并不便宜", loc="left", fontsize=13, pad=10)

    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("HNSW 的收益取决于数据分布，不能拿随机向量外推", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "5-5-ann", preview)


def plot_retrieval(preview: bool) -> None:
    rows = [row for row in read_rag("retrieval.csv") if row["category"] in {"direct", "condition", "cross", "identifier"}]
    categories = [("direct", "直接摘取"), ("condition", "条件判断"), ("cross", "跨段组合"), ("identifier", "标识符")]
    strategies = [("bm25", "BM25", AMBER), ("dense", "向量", BLUE), ("hybrid", "混合 RRF", "#10b981")]

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    width = 0.25
    x = list(range(len(categories)))
    for index, (strategy, label, color) in enumerate(strategies):
        values = []
        for key, _ in categories:
            row = next(item for item in rows if item["strategy"] == strategy and item["category"] == key)
            values.append(float(row["mrr@10"]))
        offset = (index - 1) * width
        ax.bar([i + offset for i in x], values, width, color=color, label=label, zorder=3)
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=INK)

    ax.set_xticks(x, [name for _, name in categories])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("MRR@10")
    ax.set_title("没有一路通吃：语义题向量领先，标识符题混合领先", loc="left", fontsize=13, color=INK, pad=14)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "5-6-retrieval", preview)


def plot_rerank(preview: bool) -> None:
    rows = {row["stage"]: row for row in read_rag("rerank.csv")}
    metrics = [("recall@5", "Recall@5"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]

    fig, ax = plt.subplots(figsize=(9.6, 4.0), dpi=100)
    x = list(range(len(metrics)))
    width = 0.34
    for offset, stage, label, color in ((-width / 2, "before", "重排前", GRAY), (width / 2, "after", "重排后", BLUE)):
        values = [float(rows[stage][key]) for key, _ in metrics]
        ax.bar([i + offset for i in x], values, width, color=color, label=label, zorder=3)
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.02, f"{value:.3f}", ha="center", fontsize=9, color=INK)

    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标")
    ax.set_title("重排把相关段落推进前 5，但第一名不一定更稳", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "5-7-rerank", preview)



def read_evaluation(name: str) -> list[dict[str, str]]:
    """第六部分的实验数据在 benchmarks/evaluation/。"""
    with open(ROOT / "benchmarks" / "evaluation" / name, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_recall_curve(preview: bool) -> None:
    rows = read_evaluation("retrieval-k-curve.csv")
    labels = {"bm25": "BM25", "dense": "稠密检索", "rrf": "RRF 混合"}
    colors = {"bm25": MUTED, "dense": BLUE, "rrf": AMBER}

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    for strategy, label in labels.items():
        subset = [row for row in rows if row["strategy"] == strategy]
        xs = [int(row["k"]) for row in subset]
        ys = [float(row["recall"]) for row in subset]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, color=colors[strategy], label=label)
        at_five = next(row for row in subset if row["k"] == "5")
        offsets = {"bm25": (-30, -18), "dense": (10, 10), "rrf": (12, -14)}
        ax.annotate(
            f"{float(at_five['recall']):.2f}",
            (5, float(at_five["recall"])),
            textcoords="offset points",
            xytext=offsets[strategy],
            fontsize=9,
            color=colors[strategy],
        )

    ax.axvline(5, color=RULE, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.6, 1.02)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("截断位置 k")
    ax.set_ylabel("Recall@k")
    ax.set_title("召回很快贴顶：k=5 之后，三条线都挤在 0.94 以上", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "6-2-recall-curve", preview)


def plot_metric_shape(preview: bool) -> None:
    rows = [
        row
        for row in read_evaluation("retrieval-metrics.csv")
        if row["relevance"] == "answer" and row["category"] == "all"
    ]
    order = ["bm25", "dense", "rrf"]
    labels = {"bm25": "BM25", "dense": "稠密", "rrf": "RRF"}
    metrics = [("recall@5", "Recall@5"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]
    by_strategy = {row["strategy"]: row for row in rows}

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    width = 0.24
    palette = [BLUE, BLUE_LIGHT, "#93c5fd"]
    for index, (key, label) in enumerate(metrics):
        values = [float(by_strategy[strategy][key]) for strategy in order]
        positions = [position + (index - 1) * width for position in range(len(order))]
        bars = ax.bar(positions, values, width, color=palette[index], label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", fontsize=9, color=INK)

    ax.set_xticks(range(len(order)), [labels[strategy] for strategy in order])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("指标值（越高越好）")
    ax.set_title("召回都接近 1，差距在排序：MRR 与 nDCG 把 RRF 拉开", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    save(fig, "6-2-metric-shape", preview)



def plot_judge_confusion(preview: bool) -> None:
    rows = read_evaluation("judge-confusion.csv")
    labels = ["correct", "partial", "wrong"]
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        matrix[labels.index(row["human"])][labels.index(row["judge"])] = int(row["count"])

    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=100)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max(row) for row in matrix))
    for i, human in enumerate(labels):
        for j, judge in enumerate(labels):
            value = matrix[i][j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=14, color=INK if value < 40 else "white")
    ax.set_xticks(range(3), ["correct", "partial", "wrong"])
    ax.set_yticks(range(3), ["correct", "partial", "wrong"])
    ax.set_xlabel("裁判判定")
    ax.set_ylabel("人工标签")
    ax.set_title("裁判偏宽：5 道 partial 被判成 correct", loc="left", fontsize=13, color=INK, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, "6-3-judge-confusion", preview)


def plot_eval_disagreement(preview: bool) -> None:
    faithful = {row["metric"]: float(row["value"]) for row in read_evaluation("faithfulness-summary.csv") if row["value"] not in ("",)}
    pointwise = [row for row in read_evaluation("judge-pointwise.csv") if row["category"] != "refuse"]
    auto_map = {row["id"]: row["auto_correct"] for row in read_rag("generation-audit.csv")}
    human_correct = sum(row["human"] == "correct" for row in pointwise) / len(pointwise)
    auto_correct = sum(1 for row in pointwise if auto_map.get(row["id"]) == "1") / len(pointwise)
    judge_correct = sum(row["strict"] == "correct" for row in pointwise) / len(pointwise)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)
    names = ["人工标签", "字符串判定", "9B 裁判"]
    values = [human_correct, auto_correct, judge_correct]
    colors = [GRAY, BLUE_LIGHT, BLUE]
    bars = left.bar(names, values, color=colors, width=0.56, zorder=3)
    for bar, value in zip(bars, values):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.0%}", ha="center", fontsize=11, color=INK)
    left.set_ylim(0, 1.08)
    left.set_ylabel("判为 correct 的比例")
    left.set_title("正确率：三把尺子最多差 20 个百分点", loc="left", fontsize=13, color=INK, pad=12)
    style_axes(left)
    left.grid(False)

    names = ["程序判定（全句支撑）", "9B 裁判"]
    values = [faithful.get("answers_content_all_supported", 0), faithful.get("judge_faithful", 0)]
    bars = right.bar(names, values, color=[BLUE_LIGHT, BLUE], width=0.46, zorder=3)
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.0%}", ha="center", fontsize=11, color=INK)
    right.set_ylim(0, 1.08)
    right.set_ylabel("判为忠实的比例")
    right.set_title("忠实度：程序 68%，裁判 97%", loc="left", fontsize=13, color=INK, pad=12)
    style_axes(right)
    right.grid(False)
    fig.tight_layout()
    save(fig, "6-3-eval-disagreement", preview)



def plot_ablation(preview: bool) -> None:
    rows = read_evaluation("ablation-retrieval.csv")
    order = ["bm25", "dense", "rrf", "rrf_rerank", "rrf_fixed", "rrf_512"]
    labels = {
        "bm25": "BM25",
        "dense": "稠密",
        "rrf": "RRF",
        "rrf_rerank": "RRF+重排",
        "rrf_fixed": "固定切分",
        "rrf_512": "512 维",
    }
    by_name = {row["variant"]: row for row in rows}
    colors = {"bm25": GRAY, "dense": BLUE_LIGHT, "rrf": BLUE, "rrf_rerank": "#93c5fd", "rrf_fixed": AMBER, "rrf_512": "#fcd34d"}

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    width = 0.36
    for index, (metric, label, offset) in enumerate(
        (("recall@5", "Recall@5", -width / 2), ("mrr@10", "MRR@10", width / 2))
    ):
        values = [float(by_name[name][metric]) for name in order]
        positions = [position + offset for position in range(len(order))]
        metric_color = BLUE if metric == "recall@5" else BLUE_LIGHT
        bars = ax.bar(positions, values, width, color=metric_color, label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", fontsize=8, color=INK)

    ax.set_xticks(range(len(order)), [labels[name] for name in order])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("指标值（越高越好）")
    ax.set_title("换切分掉 10 个点，重排几乎不动：组件的重要性差一个量级", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    save(fig, "6-4-ablation", preview)


def plot_ablation_forest(preview: bool) -> None:
    rows = [
        row
        for row in read_evaluation("ablation-paired.csv")
        if row["metric"] == "mrr@10"
    ]
    labels = {
        "bm25": "BM25",
        "dense": "稠密",
        "rrf_rerank": "RRF+重排",
        "rrf_fixed": "固定切分",
        "rrf_512": "512 维",
    }
    rows.sort(key=lambda row: float(row["delta"]))
    fig, ax = plt.subplots(figsize=(9.6, 3.8), dpi=100)
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows):
        low, high = float(row["ci_low"]), float(row["ci_high"])
        delta = float(row["delta"])
        color = AMBER if low > 0 or high < 0 else MUTED
        ax.plot([low, high], [y, y], color=color, linewidth=2.2)
        ax.plot(delta, y, marker="o", markersize=7, color=color)
        ax.text(high + 0.008, y, f"{delta:+.2f}", va="center", fontsize=9, color=color)

    ax.axvline(0, color=INK, linewidth=1, linestyle="--")
    ax.set_yticks(ys, [labels[row["variant"]] for row in rows])
    ax.set_xlabel("MRR@10 相对 RRF 的差值（95% bootstrap 区间）")
    ax.set_title("重排的区间跨过 0：这项改动在本测试集上分辨不出收益", loc="left", fontsize=14, color=INK, pad=14)
    style_axes(ax)
    ax.grid(axis="x")
    fig.tight_layout()
    save(fig, "6-4-forest", preview)


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
    plot_long_context(args.preview)
    plot_chunking(args.preview)
    plot_ann(args.preview)
    plot_retrieval(args.preview)
    plot_rerank(args.preview)
