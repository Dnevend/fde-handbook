#!/usr/bin/env python3
"""同一批 0-100 分，换一个「多少分算 correct」的边界，κ 会怎么变。

裁判打分只给了分数；把分数映射回 correct/partial/wrong 是**你的选择**，不是
模型的选择。这个脚本用 judge-pointwise.csv 里已有的分数，复算两种常见映射：

  - 提示词口径：90-100 correct / 30-89 partial / 0-29 wrong
  - 宽松口径：  80-100 correct / 40-79 partial / 0-39 wrong

产物：benchmarks/evaluation/judge-score-mapping.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POINTWISE = ROOT / "benchmarks/evaluation/judge-pointwise.csv"
OUT = ROOT / "benchmarks/evaluation/judge-score-mapping.csv"


def kappa(human: list[str], judge: list[str]) -> float:
    n = len(human)
    observed = sum(1 for a, b in zip(human, judge) if a == b) / n
    human_dist = {label: human.count(label) / n for label in set(human)}
    judge_dist = {label: judge.count(label) / n for label in set(judge)}
    expected = sum(human_dist.get(label, 0) * judge_dist.get(label, 0) for label in set(human) | set(judge))
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def main() -> int:
    rows = [
        row
        for row in csv.DictReader(POINTWISE.open(encoding="utf-8"))
        if row["category"] != "refuse" and row["score"] not in ("", None)
    ]
    human = [row["human"] for row in rows]
    scores = [int(row["score"]) for row in rows]
    mappings = [
        ("prompt", 90, 30),
        ("lenient", 80, 40),
    ]
    out = []
    for name, correct_cut, partial_cut in mappings:
        labels = [
            "correct" if score >= correct_cut else "partial" if score >= partial_cut else "wrong"
            for score in scores
        ]
        out.append(
            {
                "mapping": name,
                "correct_cut": correct_cut,
                "partial_cut": partial_cut,
                "correct_count": labels.count("correct"),
                "partial_count": labels.count("partial"),
                "wrong_count": labels.count("wrong"),
                "accuracy": round(sum(1 for a, b in zip(human, labels) if a == b) / len(rows), 3),
                "kappa": round(kappa(human, labels), 3),
            }
        )
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for row in out:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
