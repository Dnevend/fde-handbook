#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2,<3"]
# ///
"""7.5 量化质量对比：同一批抽取题，分别打给 BF16 / FP8 / AWQ 三个端点。

题目复用第四部分的合成工单探针：从文档里取一个字段，答案唯一、判分不需要模型。
这不是 RAG 质量结论——RAG 用第六部分的测试集；这里只回答“量化有没有把
基本抽取能力打坏”。

用法（先起三个端点，模型名保持一致）：
    uv run labs/07-gpu-inference/quant_eval.py \
      --endpoints bf16=http://127.0.0.1:8001/v1,fp8=http://127.0.0.1:8002/v1,awq=http://127.0.0.1:8003/v1 \
      --model qwen3.5-9b --limit 20 \
      --output benchmarks/gpu/quant/quality.csv

前置：合成语料要先生成（不入库）：
    uv run corpus/synthetic/generate.py --clean --validate

判分口径：标准答案去空白后是回答的子串即算对。这个口径偏宽松，
只适合做同题对比；绝对值不要当准确率写进正文。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "labs/04-quantization"))

from gpulib import (  # noqa: E402
    DEFAULT_MODEL,
    RESULTS_DIR,
    env_note,
    parse_endpoints,
    percentile,
    stream_once,
    write_csv,
)
from task_probe import build_items  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="量化模型抽取能力对比")
    parser.add_argument("--endpoints", required=True, help="name=url,name=url；url 带 /v1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="各端点统一的 served-model-name")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0 表示全部")
    parser.add_argument("--max-tokens", type=int, default=32, help="每个回答最多生成 token")
    parser.add_argument("--output", default=str(RESULTS_DIR / "quant" / "quality.csv"), help="逐题 CSV")
    args = parser.parse_args()

    endpoints = parse_endpoints(args.endpoints)
    items = build_items()
    if not items:
        print("找不到合成语料。先跑：uv run corpus/synthetic/generate.py --clean --validate", file=sys.stderr)
        return 2
    if args.limit:
        items = items[: args.limit]

    print(env_note(base_url=", ".join(url for _, url in endpoints), model=args.model))
    print(f"\n端点: {', '.join(name for name, _ in endpoints)}")
    print(f"题目: {len(items)} 道；模型名: {args.model}\n")

    rows: list[dict] = []
    for name, url in endpoints:
        for index, item in enumerate(items, start=1):
            prompt = f"{item['body']}\n\n{item['question']}"
            try:
                result = stream_once(
                    prompt,
                    base_url=url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    seed=42,
                    enable_thinking=False,
                )
                response = result["content"].strip()
                correct = normalize(item["answer"]) in normalize(response)
                rows.append({
                    "endpoint": name,
                    "doc_id": item["doc_id"],
                    "kind": item["kind"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "response": response.replace("\n", " ")[:300],
                    "correct": int(correct),
                    "ttft_s": result["ttft_s"],
                    "total_s": result["total_s"],
                    "tok_per_s": result["tok_per_s"],
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "endpoint": name,
                    "doc_id": item["doc_id"],
                    "kind": item["kind"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "response": "",
                    "correct": 0,
                    "ttft_s": 0.0,
                    "total_s": 0.0,
                    "tok_per_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            print(f"{name} {index}/{len(items)} {item['doc_id']} correct={rows[-1]['correct']}")

    out = Path(args.output)
    write_csv(out, rows)

    summary = []
    for name, _ in endpoints:
        subset = [row for row in rows if row["endpoint"] == name]
        ok = [row for row in subset if not row["error"]]
        summary.append({
            "endpoint": name,
            "questions": len(subset),
            "correct": sum(row["correct"] for row in ok),
            "accuracy": round(sum(row["correct"] for row in ok) / len(ok), 3) if ok else 0.0,
            "ttft_p50_s": round(percentile([row["ttft_s"] for row in ok], 50), 3),
            "tok_per_s_p50": round(percentile([row["tok_per_s"] for row in ok], 50), 2),
            "errors": len(subset) - len(ok),
        })

    summary_out = out.with_name(out.stem + "-summary.csv")
    write_csv(summary_out, summary)
    print(f"\n逐题: {out}")
    print(f"汇总: {summary_out}")
    for row in summary:
        print(row)
    print("\n注意：这是字符串判定的同题对比，不是 RAG 准确率；绝对值不要直接写进正文。")
    return 0 if all(row["errors"] == 0 for row in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
