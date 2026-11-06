#!/usr/bin/env python3
"""8.4 运维摘要：把 traces / feedback / audit 三份 JSONL 收成一页。

不做可视化，不做告警；只回答“过去这批请求，用户体感怎么样、谁在用、
哪些文档被引用、有没有错误”。适合交付时贴进周报/验收记录。

用法：
    uv run labs/08-delivery/observability/report.py
    uv run labs/08-delivery/observability/report.py --json
    uv run labs/08-delivery/observability/report.py --show-comments
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def summarize(base: Path, show_comments: bool = False) -> dict:
    traces = read_jsonl(base / "traces.jsonl")
    feedback = read_jsonl(base / "feedback.jsonl")
    audit = read_jsonl(base / "audit.jsonl")

    ok = [row for row in traces if row.get("ok")]
    cancelled = [row for row in traces if row.get("cancelled")]
    errors = [row for row in traces if not row.get("ok") and not row.get("cancelled")]
    refused = [row for row in traces if row.get("refused")]
    # 拒答/断开可能 tokens=0；延迟统计只取有生成 token 的成功请求
    latency = [row for row in ok if int(row.get("tokens", 0)) > 0]
    tpots = []
    for row in latency:
        tokens = int(row.get("tokens", 0))
        gen = float(row.get("generation_s", 0.0))
        ttft = float(row.get("ttft_s", 0.0))
        if tokens > 1 and gen > ttft:
            tpots.append((gen - ttft) / (tokens - 1) * 1000)
    ratings = [int(row.get("rating", 0)) for row in feedback if row.get("rating")]
    docs = Counter(doc for row in traces for doc in row.get("docs", []))
    users = Counter(row.get("user", "anonymous") for row in traces)

    payload = {
        "requests": len(traces),
        "ok": len(ok),
        "errors": len(errors),
        "cancelled": len(cancelled),
        "refused": len(refused),
        "streams": sum(1 for row in traces if row.get("stream")),
        "users": dict(users),
        "retrieval_ms_p50": round(percentile([row.get("retrieval_ms", 0) for row in latency], 50), 1),
        "retrieval_ms_p95": round(percentile([row.get("retrieval_ms", 0) for row in latency], 95), 1),
        "ttft_s_p50": round(percentile([row.get("ttft_s", 0) for row in latency], 50), 3),
        "ttft_s_p95": round(percentile([row.get("ttft_s", 0) for row in latency], 95), 3),
        "tpot_ms_p50": round(percentile(tpots, 50), 1),
        "tpot_ms_p95": round(percentile(tpots, 95), 1),
        "total_s_p50": round(percentile([row.get("total_s", 0) for row in latency], 50), 3),
        "total_s_p95": round(percentile([row.get("total_s", 0) for row in latency], 95), 3),
        "tokens_total": sum(int(row.get("tokens", 0)) for row in traces),
        "top_docs": docs.most_common(10),
        "feedback": {
            "count": len(feedback),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
            "distribution": dict(sorted(Counter(ratings).items())),
            "low_ratings": [
                {
                    "request_id": row.get("request_id"),
                    "rating": row.get("rating"),
                    "comment": row.get("comment", "") if show_comments else "",
                }
                for row in feedback
                if int(row.get("rating", 5)) <= 2
            ],
        },
        "audit": {
            "rows": len(audit),
            "users": dict(Counter(row.get("user", "anonymous") for row in audit)),
            "denied_requests": sum(1 for row in audit if row.get("allowed_count") == 0),
        },
        "errors_detail": [
            {"request_id": row.get("request_id"), "error": row.get("error", "")[:160]}
            for row in errors[:10]
        ],
    }
    return payload


def print_report(payload: dict) -> None:
    print("# 交付运行摘要")
    print()
    print(
        f"请求：{payload['requests']}（成功 {payload['ok']}，失败 {payload['errors']}，"
        f"取消 {payload['cancelled']}，拒答 {payload['refused']}，流式 {payload['streams']}）"
    )
    print(f"用户：{payload['users']}")
    print(f"检索 p50/p95：{payload['retrieval_ms_p50']} / {payload['retrieval_ms_p95']} ms")
    print(f"TTFT p50/p95：{payload['ttft_s_p50']} / {payload['ttft_s_p95']} s")
    print(f"TPOT p50/p95：{payload['tpot_ms_p50']} / {payload['tpot_ms_p95']} ms")
    print(f"总耗时 p50/p95：{payload['total_s_p50']} / {payload['total_s_p95']} s")
    print(f"生成 token：{payload['tokens_total']}")
    print()
    print("被引用最多的文档：")
    for doc, count in payload["top_docs"]:
        print(f"  {doc}: {count}")
    print()
    fb = payload["feedback"]
    print(f"反馈：{fb['count']} 条，平均 {fb['avg_rating']}；分布 {fb['distribution']}")
    if fb["low_ratings"]:
        print("低分反馈：")
        for row in fb["low_ratings"]:
            print(f"  {row['request_id']} rating={row['rating']} {row['comment']}")
    print()
    audit = payload["audit"]
    print(f"审计：{audit['rows']} 行；用户 {audit['users']}；无可见文档的请求 {audit['denied_requests']}")
    if payload["errors_detail"]:
        print()
        print("错误（前 10 条）：")
        for row in payload["errors_detail"]:
            print(f"  {row['request_id']}: {row['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="8.4 运维摘要")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parents[3] / "benchmarks" / "delivery"),
        help="traces/feedback/audit 所在目录",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--show-comments", action="store_true", help="低分反馈里显示评论")
    args = parser.parse_args()
    payload = summarize(Path(args.dir), show_comments=args.show_comments)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
