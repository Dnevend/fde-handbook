#!/usr/bin/env python3
"""拒答题审计：把每道拒答题当查询检索一遍，看语料里有没有「其实答得上」的段落。

拒答题最容易出的错不是「问得太偏」，而是问到了语料里真有的东西。人写题时凭印象
判断「这份资料不可能有」，但语料里常有专利、环评、事故报告这类细节文本。这个脚本
用第五部分的同一套块与向量，给每道拒答题取 top-3，按相似度标出需要人工复核的题。

判据来自可答题：可答题 top-1 相似度的 5% 分位约 0.67（本机实测），低于它的
可答题已经属于「检索很难命中」的个例。所以把拒答题相似度 ≥0.65 视为「可能答得上」，
必须逐条看 top-1 原文；低于阈值的也可能漏，脚本只负责把范围缩小，不替代人。

用法（在 labs/ 下）：
    uv run 06-evaluation/audit_refuse.py
    uv run 06-evaluation/audit_refuse.py --threshold 0.6 --k 5

产物：
    benchmarks/evaluation/refuse-audit.csv        逐题 top-3 与标记
    benchmarks/evaluation/refuse-audit-summary.csv 可答题与拒答题的相似度分布
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))

import raglib as r  # noqa: E402

CACHE = ROOT / "labs/05-rag/.cache"
OUT = ROOT / "benchmarks/evaluation"
TAG = re.compile(r"<[^>]+>")

# 人工复核结论：只记录被标记的题，避免把「相似度高」误当成「可答」。
REVIEW = {
    "R67": "报告只写了该厂安全管理人员的人数和岗位，没有姓名与电话，不能回答本厂人员",
    "R73": "62 万元是另一家企业燃气事故的罚款，不是本厂隐患排查处罚，不能回答",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", text))).strip()


def snippet(text: str, limit: int = 120) -> str:
    return clean(text)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="拒答题可答性审计")
    parser.add_argument("--file", type=Path, default=ROOT / "eval/qa-v2.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--tag", default="", help="产物后缀，例如 v1 → refuse-audit-v1.csv")
    args = parser.parse_args()

    chunks_path = CACHE / "chunks-structure-real.jsonl"
    vectors_path = CACHE / "vectors-1024-real.npy"
    if not chunks_path.exists() or not vectors_path.exists():
        raise SystemExit("缺第五部分的块与向量缓存，先跑 labs/05-rag/embed_index.py")

    chunks = read_jsonl(chunks_path)
    vectors = np.load(vectors_path)
    rows = read_jsonl(args.file)
    if not rows:
        raise SystemExit(f"{args.file} 是空的")

    questions = [row["question"] for row in rows]
    queries = r.embed_texts(questions, kind="query")
    scores = queries @ vectors.T

    audit: list[dict] = []
    for index, row in enumerate(rows):
        if row["category"] != "refuse":
            continue
        order = np.argsort(-scores[index])[: args.k]
        top = [int(position) for position in order]
        flagged = bool(scores[index, top[0]] >= args.threshold)
        audit.append(
            {
                "id": row["id"],
                "split": row.get("split", ""),
                "question": row["question"],
                "top1_sim": round(float(scores[index, top[0]]), 3),
                "top1_doc": chunks[top[0]]["doc_id"],
                "top1_clause": chunks[top[0]].get("clause_no") or "",
                "top1_snippet": snippet(chunks[top[0]]["text"]),
                "top2_sim": round(float(scores[index, top[1]]), 3) if len(top) > 1 else "",
                "top2_doc": chunks[top[1]]["doc_id"] if len(top) > 1 else "",
                "flagged": int(flagged),
                "review": REVIEW.get(row["id"], ""),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.tag}" if args.tag else ""
    detail_path = OUT / f"refuse-audit{suffix}.csv"
    summary_path = OUT / f"refuse-audit-summary{suffix}.csv"
    fields = list(audit[0]) if audit else []
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)

    answerable = [index for index, row in enumerate(rows) if row["category"] != "refuse"]
    refuse = [index for index, row in enumerate(rows) if row["category"] == "refuse"]
    answer_top1 = scores[answerable].max(axis=1) if answerable else np.array([0.0])
    refuse_top1 = scores[refuse].max(axis=1) if refuse else np.array([0.0])

    def quantile(values: np.ndarray, q: float) -> float:
        return round(float(np.percentile(values, q)), 3)

    summary = [
        ("answerable_min_top1", round(float(answer_top1.min()), 3)),
        ("answerable_p05_top1", quantile(answer_top1, 5)),
        ("answerable_median_top1", quantile(answer_top1, 50)),
        ("refuse_median_top1", quantile(refuse_top1, 50)),
        ("refuse_p90_top1", quantile(refuse_top1, 90)),
        ("refuse_max_top1", round(float(refuse_top1.max()), 3)),
        ("refuse_flagged", sum(item["flagged"] for item in audit)),
        ("refuse_total", len(audit)),
        ("threshold", args.threshold),
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)

    for metric, value in summary:
        print(f"{metric:24s} {value}")
    print(f"\n逐题明细：{detail_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
