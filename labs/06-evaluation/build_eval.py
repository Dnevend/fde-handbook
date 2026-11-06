#!/usr/bin/env python3
"""把四份扩题草稿合并成 eval/qa-v2.jsonl，分配 dev/test 分片，并产出测试集统计。

    # 首次（草稿到位后）：
    uv run labs/06-evaluation/build_eval.py --drafts /path/to/drafts

    # 之后核对（不重写分片）：
    python3 labs/06-evaluation/build_eval.py --check

规则：
  - v2 是 v1（eval/qa.jsonl，80 题）的超集；v1 全部进 dev，因为第五部分已经用它们
    选过切分、维度与融合方式，不能当测试集。
  - 新增题的 dev 配额按类别固定（direct 17 / condition 18 / cross 17 / refuse 19，
    合计 71），其余 150 道进 test。选哪几道进 dev 由题目哈希决定，避免人工挑选偏差。
  - 分片结果写回 qa-v2.jsonl 的 split 字段；重复运行结果不变（幂等）。

统计写到 benchmarks/evaluation/testset-*.csv，正文只引用这两个文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from clause_text import load_clause_texts  # noqa: E402

QA_V1 = ROOT / "eval/qa.jsonl"
QA_V2 = ROOT / "eval/qa-v2.jsonl"
OUT = ROOT / "benchmarks/evaluation"
NEW_DEV = {"direct": 17, "condition": 18, "cross": 17, "refuse": 19}
RETIRED_V1 = {"R12"}
COVERED_DOCS = ["reg-safety-law", "reg-mem-order-13", "gov-ledger-guide"]
CATEGORIES = ["direct", "condition", "cross", "refuse"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def digest(row: dict) -> str:
    return hashlib.sha256(f"{row['id']}\t{row['question']}".encode()).hexdigest()


def assign_split(rows: list[dict], v1_ids: set[str]) -> None:
    """v1 全部 dev；新题按类别取哈希最小的若干道进 dev，其余 test。"""
    for row in rows:
        if row["id"] in v1_ids:
            row["split"] = "dev"

    for category in CATEGORIES:
        fresh = [row for row in rows if row["id"] not in v1_ids and row["category"] == category]
        fresh.sort(key=digest)
        keep = NEW_DEV[category]
        if len(fresh) < keep:
            raise SystemExit(f"{category} 只有 {len(fresh)} 道新题，不够 {keep} 道 dev")
        for index, row in enumerate(fresh):
            row["split"] = "dev" if index < keep else "test"


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[position]


def stats(rows: list[dict]) -> tuple[list[list], list[list]]:
    per_split: list[list] = [["category", "split", "n", "mean_question_chars", "p95_question_chars", "mean_answer_chars", "mean_sources"]]
    for category in CATEGORIES:
        for split in ("dev", "test"):
            subset = [row for row in rows if row["category"] == category and row["split"] == split]
            qlens = [len(row["question"]) for row in subset]
            alens = [len(row["answer"]) for row in subset]
            sources = [len(row["sources"]) for row in subset]
            per_split.append(
                [
                    category,
                    split,
                    len(subset),
                    round(sum(qlens) / len(qlens), 1) if qlens else 0,
                    percentile(qlens, 0.95),
                    round(sum(alens) / len(alens), 1) if alens else 0,
                    round(sum(sources) / len(sources), 2) if sources else 0,
                ]
            )

    texts = load_clause_texts(COVERED_DOCS)
    available = Counter(doc for doc, _ in texts)
    used_citations = Counter(source["doc"] for row in rows for source in row["sources"])
    used_clauses = defaultdict(set)
    for row in rows:
        for source in row["sources"]:
            used_clauses[source["doc"]].add(source["clause"])

    coverage: list[list] = [["doc", "clauses_available", "clauses_used", "citations", "coverage_rate"]]
    for doc in COVERED_DOCS:
        total = available.get(doc, 0)
        used = len(used_clauses.get(doc, set()))
        coverage.append(
            [doc, total, used, used_citations.get(doc, 0), round(used / total, 3) if total else 0]
        )
    return per_split, coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="合并扩题草稿、分配分片、产出统计")
    parser.add_argument("--drafts", type=Path, help="含 draft-*.jsonl 的目录")
    parser.add_argument("--check", action="store_true", help="只核对现有 qa-v2.jsonl，不重写")
    args = parser.parse_args()

    v1_all = read_jsonl(QA_V1)
    v1 = [row for row in v1_all if row["id"] not in RETIRED_V1]
    v1_ids = {row["id"] for row in v1}

    if args.drafts:
        rows = list(v1)
        for category in CATEGORIES:
            path = args.drafts / f"draft-{category}.jsonl"
            if not path.exists():
                raise SystemExit(f"缺少草稿 {path}")
            rows.extend(read_jsonl(path))
        assign_split(rows, v1_ids)
        order = {category: index for index, category in enumerate(CATEGORIES)}
        rows.sort(key=lambda row: (order[row["category"]], int(re.search(r"\d+$", row["id"]).group())))
        write_jsonl(QA_V2, rows)
        print(f"写出 {QA_V2.relative_to(ROOT)}：{len(rows)} 题")
    else:
        rows = read_jsonl(QA_V2)

    missing = {row["id"] for row in v1} - {row["id"] for row in rows}
    if missing:
        raise SystemExit(f"v2 缺少 v1 的题目：{sorted(missing)[:5]}")

    if args.check:
        expected = dict(zip([row["id"] for row in rows], [row.get("split") for row in rows]))
        probe = [dict(row) for row in rows]
        assign_split(probe, v1_ids)
        drift = [row["id"] for row in probe if expected[row["id"]] != row["split"]]
        if drift:
            raise SystemExit(f"分片漂移：{drift[:5]}（重跑 --drafts 或检查哈希规则）")

    per_split, coverage = stats(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        OUT / "testset-split.csv": per_split,
        OUT / "testset-coverage.csv": coverage,
    }
    for path, table in paths.items():
        path.write_text("\n".join(",".join(str(cell) for cell in row) for row in table) + "\n", encoding="utf-8")
        print(f"写出 {path.relative_to(ROOT)}")

    counts = Counter((row["category"], row["split"]) for row in rows)
    print(" ".join(f"{category}/{split}={counts[(category, split)]}" for category in CATEGORIES for split in ("dev", "test")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
