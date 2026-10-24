#!/usr/bin/env python3
"""校验 eval/qa.jsonl 的结构与出处。零依赖，只读。

    python3 eval/validate.py

检查项：
  1. 字段齐全、id 唯一、题目不重复
  2. 四类各 20 题（与 labs/05-rag/config.yaml 的 eval 段一致）
  3. 非拒答题必须有出处；拒答题不得有出处
  4. 出处里的 doc 存在于 corpus/index.json 或 corpus/raw/ 的语料目录
  5. 引用的条款号在 corpus/index.json 里存在
  6. cross 题至少两个出处
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "eval/qa.jsonl"
INDEX = ROOT / "corpus/index.json"
RAW = ROOT / "corpus/raw"
EXPECTED = {"direct": 20, "condition": 20, "cross": 20, "refuse": 20}

errors: list[str] = []
warnings: list[str] = []


def read_rows() -> list[dict]:
    rows = []
    for number, line in enumerate(QA.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"第 {number} 行不是合法 JSON：{exc}")
    return rows


def main() -> int:
    rows = read_rows()
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    clauses = {(source["id"], clause["no"]) for source in index["sources"] for clause in source["clauses"]}
    docs = {source["id"] for source in index["sources"]}
    synthetic_docs = {path.stem for path in (RAW / "synthetic").glob("*/*.md")}
    docs |= synthetic_docs

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    counts = Counter()

    for row in rows:
        qid = row.get("id", "")
        if not qid:
            errors.append("有一行缺 id")
            continue
        if qid in seen_ids:
            errors.append(f"{qid}：id 重复")
        seen_ids.add(qid)

        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        category = row.get("category")
        sources = row.get("sources") or []

        if category not in EXPECTED:
            errors.append(f"{qid}：未知类别 {category}")
            continue
        counts[category] += 1

        if not question or not answer:
            errors.append(f"{qid}：question 或 answer 为空")
        if question in seen_questions:
            errors.append(f"{qid}：题目重复")
        seen_questions.add(question)
        if len(question) > 120:
            warnings.append(f"{qid}：题目偏长（{len(question)} 字）")
        if re.search(r"资料中没有相关信息", question):
            warnings.append(f"{qid}：题目里出现了标准拒答语")

        if category == "refuse":
            if sources:
                errors.append(f"{qid}：拒答题不应有出处")
        else:
            if not sources:
                errors.append(f"{qid}：非拒答题必须有出处")
        if category == "cross" and len(sources) < 2:
            errors.append(f"{qid}：cross 题至少要有两个出处")

        for source in sources:
            doc = source.get("doc")
            clause = source.get("clause")
            if doc not in docs:
                errors.append(f"{qid}：出处 {doc} 不在语料里")
                continue
            if clause and (doc, clause) not in clauses:
                errors.append(f"{qid}：{doc} 没有 {clause}（见 corpus/index.json）")

    for category, expected in EXPECTED.items():
        if counts[category] != expected:
            errors.append(f"{category} 类有 {counts[category]} 题，应为 {expected}")

    print(f"题目 {len(rows)} 道：{dict(counts)}")
    for line in warnings:
        print(f"WARN  {line}")
    for line in errors:
        print(f"ERROR {line}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
