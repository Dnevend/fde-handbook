#!/usr/bin/env python3
"""6.3 在 v2 test 的 60 条生成结果上跑裁判，和字符串判定对比。

v2 没有人工标签，所以这里**不算准确率、不算 κ**——只报「自动判定 vs 裁判」
的一致率。这个数字只能说明两把尺子差多少，不能说明谁对；6.3 的正文把它当作
「没有人工校准的裁判分数不可用」的证据。

用法（在 labs/ 下）：
    uv run 06-evaluation/judge_v2.py

产物：
    benchmarks/evaluation/generation-v2-judged.jsonl
    benchmarks/evaluation/judge-v2-summary.csv
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))
sys.path.insert(0, str(ROOT / "labs/06-evaluation"))

import raglib as r  # noqa: E402
from judge_calibration import REFUSE_PROMPT, STRICT_PROMPT, judge, strict_label  # noqa: E402

GEN = ROOT / "benchmarks/evaluation/generation-v2.jsonl"
OUT = ROOT / "benchmarks/evaluation/generation-v2-judged.jsonl"
SUMMARY = ROOT / "benchmarks/evaluation/judge-v2-summary.csv"


def main() -> int:
    rows = [json.loads(line) for line in GEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, start=1):
        if row["category"] == "refuse":
            payload, latency, _ = judge(
                "你只输出 JSON。",
                f"{REFUSE_PROMPT}\n\n问题：{row['question']}\n模型回答：{row['answer']}",
                max_tokens=120,
            )
            label = "correct" if str(payload.get("label", "")).lower() == "correct" else "wrong"
        else:
            payload, latency, _ = judge(
                "你只输出 JSON。",
                f"{STRICT_PROMPT}\n\n问题：{row['question']}\n标准答案：{row['gold']}\n模型回答：{row['answer']}",
                max_tokens=160,
            )
            label = strict_label(payload)
        row["judge_label"] = label
        row["judge_reason"] = str(payload.get("reason", ""))[:60]
        row["judge_latency_s"] = round(latency, 2)
        r.log(f"{number}/{len(rows)} {row['id']} auto={row['auto_correct']} judge={label}")

    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")

    answerable = [row for row in rows if row["category"] != "refuse"]
    auto_ok = sum(row["auto_correct"] for row in rows)
    judge_ok = sum(row["judge_label"] == "correct" for row in rows)
    agree = sum(bool(row["auto_correct"]) == (row["judge_label"] == "correct") for row in rows)
    summary = [
        ("questions", len(rows)),
        ("auto_correct", auto_ok),
        ("auto_correct_rate", round(auto_ok / len(rows), 3)),
        ("judge_correct", judge_ok),
        ("judge_correct_rate", round(judge_ok / len(rows), 3)),
        ("auto_judge_agreement", round(agree / len(rows), 3)),
        ("answerable", len(answerable)),
        ("answerable_auto_rate", round(sum(row["auto_correct"] for row in answerable) / len(answerable), 3)),
        ("answerable_judge_rate", round(sum(row["judge_label"] == "correct" for row in answerable) / len(answerable), 3)),
        ("mean_judge_latency_s", round(sum(row["judge_latency_s"] for row in rows) / len(rows), 2)),
    ]
    SUMMARY.write_text("metric,value\n" + "\n".join(f"{key},{value}" for key, value in summary) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    for key, value in summary:
        print(f"{key:26s} {value}")
    print("按题型：", dict(Counter((row["category"], row["judge_label"]) for row in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
