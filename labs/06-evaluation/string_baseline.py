#!/usr/bin/env python3
"""字符串判定（5.8）在 60 道可答题上的基线：准确率、κ、精确率、召回率。

数据来自第五部分的 generation.jsonl 与 generation-audit.csv，不重新跑模型。
产物：benchmarks/evaluation/string-baseline.csv
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/evaluation/string-baseline.csv"


def kappa(human: list[bool], judge: list[bool]) -> float:
    n = len(human)
    observed = sum(1 for a, b in zip(human, judge) if a == b) / n
    pa = sum(human) / n
    pb = sum(judge) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def main() -> int:
    outputs = {
        row["id"]: row
        for row in (json.loads(line) for line in (ROOT / "benchmarks/rag/generation.jsonl").read_text().splitlines() if line.strip())
        if row["mode"] == "rag"
    }
    audit = list(csv.DictReader((ROOT / "benchmarks/rag/generation-audit.csv").open(encoding="utf-8")))
    pairs = [
        (outputs[row["id"]]["correct"], row["manual_label"] == "correct")
        for row in audit
        if outputs[row["id"]]["category"] != "refuse"
    ]
    auto = [pair[0] for pair in pairs]
    human = [pair[1] for pair in pairs]
    tp = sum(1 for a, h in pairs if a and h)
    fp = sum(1 for a, h in pairs if a and not h)
    fn = sum(1 for a, h in pairs if not a and h)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    rows = [
        ("n", len(pairs)),
        ("auto_correct", sum(auto)),
        ("accuracy", round(sum(1 for a, h in pairs if a == h) / len(pairs), 3)),
        ("kappa", round(kappa(human, auto), 3)),
        ("precision", round(precision, 3)),
        ("recall", round(recall, 3)),
        ("f1", round(f1, 3)),
        ("false_positives", fp),
        ("false_negatives", fn),
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(dict(Counter(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
