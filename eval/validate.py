#!/usr/bin/env python3
"""校验 eval/qa.jsonl（v1，第五部分用）与 eval/qa-v2.jsonl（v2，第六部分用）。

    python3 eval/validate.py
    python3 eval/validate.py --draft /tmp/draft.jsonl --category direct
    python3 eval/validate.py --no-grounding

检查项：
  1. 字段齐全、id 唯一、题目不重复
  2. 配额：v1 四类各 20；v2 四类各 75，dev/test 各 150
  3. 非拒答题必须有出处；拒答题不得有出处
  4. 出处里的 doc 与条款号存在于 corpus/index.json
  5. cross 题至少两个出处，且每个出处都要在答案里留下原文
  6. 答案有原文支撑：逐句（按；。）必须能在所引条款正文里找到。
     v1 的答案经过改写，只按 4-gram 覆盖率告警；v2 新增题按逐句子串严格判定。
  7. 分片：v1 的题一律 dev；v2 的 test 只放新题，避免第五部分的调参泄漏。
  8. 条款复用不超过 2 题；同一文档不超过总量的 55%。

正文来自 corpus/raw/（不入库）。raw 不在时第 6 项降级为提示，不阻断。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from clause_text import load_clause_occurrences, load_clause_texts

ROOT = Path(__file__).resolve().parents[1]
QA_V1 = ROOT / "eval/qa.jsonl"
QA_V2 = ROOT / "eval/qa-v2.jsonl"
INDEX = ROOT / "corpus/index.json"
RAW = ROOT / "corpus/raw"

CATEGORIES = ("direct", "condition", "cross", "refuse")
V1_QUOTA = {category: 20 for category in CATEGORIES}
V2_QUOTA = {category: 75 for category in CATEGORIES}
V2_SPLIT = {"dev": 150, "test": 150}
V2_NEW_DEV = {"direct": 17, "condition": 18, "cross": 17, "refuse": 19}
RETIRED_V1 = {"R12": "pat-inj-02 里有料筒温度和保压参数，属可答题，v2 不再收录"}
MAX_CLAUSE_REUSE = 4
MAX_CLAUSE_REUSE_PER_CATEGORY = 2
MAX_DOC_SHARE = 0.55
PUNCT = re.compile(r"[\s，。；：、！？（）()「」“”\"'·,.;:!?\[\]【】]")
JUDGEMENT = re.compile(r"^(不符合|不合规|符合|合规|违反|不违反|不能|不可以|应当|需要|不需要)[。，,]?$")


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def ngrams(text: str, n: int = 4) -> set[str]:
    clean = normalize(text)
    return {clean[i : i + n] for i in range(max(len(clean) - n + 1, 0))}


def segments(answer: str) -> list[str]:
    return [part for part in re.split(r"[；;。]", answer) if normalize(part)]


class Checker:
    def __init__(self, grounding: bool) -> None:
        payload = json.loads(INDEX.read_text(encoding="utf-8"))
        self.docs = {source["id"] for source in payload["sources"]}
        self.clauses = {
            (source["id"], clause["no"]) for source in payload["sources"] for clause in source["clauses"]
        }
        self.texts = load_clause_texts() if grounding else {}
        self.occurrences = load_clause_occurrences() if grounding else {}
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def check_rows(
        self,
        rows: list[dict],
        *,
        label: str,
        quota: dict[str, int] | None = None,
        strict_grounding_from: int | None = None,
        check_split: bool = False,
        seen_questions: set[str] | None = None,
        seen_ids: set[str] | None = None,
        reuse_per_category: bool = False,
    ) -> None:
        counts: Counter[str] = Counter()
        splits: Counter[str] = Counter()
        doc_counts: Counter[str] = Counter()
        clause_counts: Counter[tuple[str, str]] = Counter()
        category_clause_counts: Counter[tuple[str, str, str]] = Counter()
        seen_questions = seen_questions if seen_questions is not None else set()
        seen_ids = seen_ids if seen_ids is not None else set()

        for row in rows:
            qid = (row.get("id") or "").strip()
            if not qid:
                self.error(f"{label}：有一行缺 id")
                continue
            if qid in seen_ids:
                self.error(f"{label}：{qid} id 重复")
            seen_ids.add(qid)

            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            category = row.get("category")
            sources = row.get("sources") or []

            if category not in CATEGORIES:
                self.error(f"{label}/{qid}：未知类别 {category}")
                continue
            counts[category] += 1

            if not question or not answer:
                self.error(f"{label}/{qid}：question 或 answer 为空")
            if question in seen_questions:
                self.error(f"{label}/{qid}：题目重复")
            seen_questions.add(question)
            if len(question) > 120:
                self.warn(f"{label}/{qid}：题目偏长（{len(question)} 字）")
            if len(answer) > 160:
                self.warn(f"{label}/{qid}：答案偏长（{len(answer)} 字）")
            if "资料中没有相关信息" in question:
                self.warn(f"{label}/{qid}：题目里出现了标准拒答语")

            split = row.get("split")
            if split is not None:
                splits[split] += 1
                if split not in {"dev", "test"}:
                    self.error(f"{label}/{qid}：split 只能是 dev / test")

            if category == "refuse":
                if sources:
                    self.error(f"{label}/{qid}：拒答题不应有出处")
                continue
            if not sources:
                self.error(f"{label}/{qid}：非拒答题必须有出处")
                continue
            if category == "cross" and len(sources) < 2:
                self.error(f"{label}/{qid}：cross 题至少要有两个出处")
            if not row.get("joint") and category == "cross":
                self.warn(f"{label}/{qid}：cross 题建议 joint=true")

            bodies: dict[tuple[str, str], str] = {}
            for source in sources:
                doc, clause = source.get("doc"), source.get("clause")
                if doc not in self.docs:
                    self.error(f"{label}/{qid}：出处 {doc} 不在语料里")
                    continue
                if clause and (doc, clause) not in self.clauses:
                    self.error(f"{label}/{qid}：{doc} 没有 {clause}（见 corpus/index.json）")
                    continue
                clause_counts[(doc, clause)] += 1
                category_clause_counts[(doc, clause, category)] += 1
                doc_counts[doc] += 1
                body = self.texts.get((doc, clause))
                if body:
                    bodies[(doc, clause)] = body

            if not bodies:
                continue
            strict = strict_grounding_from is not None and self._number(qid) >= strict_grounding_from
            if strict:
                self._check_strict(label, qid, category, answer, sources, bodies)
            else:
                self._check_loose(label, qid, answer, bodies)

        if quota:
            for category, expected in quota.items():
                if counts[category] != expected:
                    self.error(f"{label}：{category} 类有 {counts[category]} 题，应为 {expected}")
        if check_split:
            for split, expected in V2_SPLIT.items():
                if splits[split] != expected:
                    self.error(f"{label}：split={split} 有 {splits[split]} 题，应为 {expected}")

        for (doc, clause), count in clause_counts.items():
            if count > MAX_CLAUSE_REUSE:
                self.warn(f"{label}：{doc} {clause} 被 {count} 题共用（上限 {MAX_CLAUSE_REUSE}）")
        if reuse_per_category:
            for (doc, clause, category), count in category_clause_counts.items():
                if count > MAX_CLAUSE_REUSE_PER_CATEGORY:
                    self.error(
                        f"{label}：新增题里 {doc} {clause} 在 {category} 类用了 {count} 次"
                        f"（上限 {MAX_CLAUSE_REUSE_PER_CATEGORY}）"
                    )
        total = sum(doc_counts.values()) or 1
        for doc, count in doc_counts.items():
            if count / total > MAX_DOC_SHARE:
                self.warn(f"{label}：{doc} 占 {count / total:.0%}，超过 {MAX_DOC_SHARE:.0%}")

    def _check_strict(
        self,
        label: str,
        qid: str,
        category: str,
        answer: str,
        sources: list[dict],
        bodies: dict[tuple[str, str], str],
    ) -> None:
        parts = segments(answer)
        if not parts:
            return
        union = "\n".join(bodies.values())
        if category != "cross":
            for part in parts:
                if JUDGEMENT.match(part):
                    continue
                if normalize(part) not in normalize(union):
                    self.error(f"{label}/{qid}：答案未在所引条款中逐字出现——「{part[:30]}」")
            return
        # cross：每句都要有原文支撑，且每个出处都要留下至少一句
        for part in parts:
            if JUDGEMENT.match(part):
                continue
            if normalize(part) not in normalize(union):
                self.error(f"{label}/{qid}：答案未在所引条款中逐字出现——「{part[:30]}」")
        for key, body in bodies.items():
            if not any(normalize(part) in normalize(body) for part in parts):
                self.error(f"{label}/{qid}：出处 {key[0]} {key[1]} 在答案里没有原文支撑")
        for key, body in bodies.items():
            self._check_occurrence(label, qid, key, body, answer)

    def _check_occurrence(
        self, label: str, qid: str, key: tuple[str, str], merged: str, answer: str
    ) -> None:
        """条号在正文里出现多次时，要求答案只落在其中一段（相同正文重复排版不算）。"""
        occurrences = self.occurrences.get(key) or []
        if len(occurrences) <= 1:
            return
        merged_norm = normalize(merged)
        parts = [
            part
            for part in segments(answer)
            if not JUDGEMENT.match(part) and normalize(part) in merged_norm
        ]
        if not parts:
            return
        # 同一段条文会在不同章节重复排版，后接的指导文字却不同。按「答案所在的
        # 局部上下文」聚类，而不是整段正文：上下文相同才算同一处，上下文不同
        # 才算「同一条号指向两段不同内容」。
        part_norms = [normalize(part) for part in parts]
        windows: list[str] = []
        for body in occurrences:
            text = normalize(body)
            spans = []
            for part in part_norms:
                position = text.find(part)
                if position < 0:
                    spans = []
                    break
                spans.append((position, position + len(part)))
            if not spans:
                continue
            start_at = max(0, min(begin for begin, _ in spans) - 30)
            end_at = min(len(text), max(end for _, end in spans) + 120)
            windows.append(text[start_at:end_at])
        if not windows:
            return
        distinct: list[str] = []
        distinct_grams: list[set[str]] = []
        for window in windows:
            grams = ngrams(window)
            if any(
                grams and existing and len(grams & existing) / len(grams | existing) >= 0.8
                for existing in distinct_grams
            ):
                continue
            distinct.append(window)
            distinct_grams.append(grams)
        if len(distinct) <= 1:
            return
        hits = [window for window in distinct if all(normalize(part) in window for part in parts)]
        if len(hits) != 1:
            self.error(
                f"{label}/{qid}：{key[0]} {key[1]} 在正文里有 {len(distinct)} 段不同内容，"
                f"答案落在 {len(hits)} 段，出处无法唯一定位"
            )

    def _check_loose(
        self, label: str, qid: str, answer: str, bodies: dict[tuple[str, str], str]
    ) -> None:
        union = "\n".join(bodies.values())
        gold, body = ngrams(answer), ngrams(union)
        if not gold:
            return
        coverage = len(gold & body) / len(gold)
        if coverage < 0.45:
            self.warn(f"{label}/{qid}：答案与所引条款的 4-gram 覆盖率仅 {coverage:.0%}")

    @staticmethod
    def _number(qid: str) -> int:
        match = re.search(r"(\d+)$", qid)
        return int(match.group(1)) if match else 0


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number} 不是合法 JSON：{exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="校验测试集")
    parser.add_argument("--draft", type=Path, help="只校验一个草稿文件")
    parser.add_argument("--category", help="草稿文件应全为该类别")
    parser.add_argument("--no-grounding", action="store_true", help="跳过原文核对")
    args = parser.parse_args()

    if not RAW.is_dir() and not args.no_grounding:
        print("提示：corpus/raw 不在，跳过原文核对（先跑 corpus/fetch.sh）", file=sys.stderr)

    checker = Checker(grounding=not args.no_grounding)

    if args.draft:
        rows = read_jsonl(args.draft)
        if args.category:
            wrong = [row.get("id") for row in rows if row.get("category") != args.category]
            if wrong:
                checker.error(f"{args.draft.name}：出现非 {args.category} 题：{', '.join(wrong[:5])}")
        extra = set()
        if QA_V2.exists():
            extra = {row["question"] for row in read_jsonl(QA_V2)}
        if QA_V1.exists():
            extra |= {row["question"] for row in read_jsonl(QA_V1)}
        overlap = [row.get("id") for row in rows if row.get("question") in extra]
        if overlap:
            checker.error(f"{args.draft.name}：题目与现有测试集重复：{', '.join(overlap[:5])}")
        checker.check_rows(rows, label=args.draft.name, strict_grounding_from=0)
        print(f"草稿 {len(rows)} 题：{dict(Counter(row.get('category') for row in rows))}")
    else:
        v1 = read_jsonl(QA_V1)
        checker.check_rows(v1, label="v1", quota=V1_QUOTA)
        v1_ids = {row["id"] for row in v1}
        v1_questions = {row["question"] for row in v1}

        if QA_V2.exists():
            v2 = read_jsonl(QA_V2)
            v2_questions = {row["question"] for row in v2}
            missing = v1_questions - v2_questions
            allowed = {row["question"] for row in v1 if row["id"] in RETIRED_V1}
            unexpected = missing - allowed
            if unexpected:
                checker.error(f"v2：缺少 v1 的 {len(unexpected)} 道题（v2 必须是 v1 的超集）")
            for qid, reason in RETIRED_V1.items():
                if any(row["id"] == qid for row in v2):
                    checker.error(f"v2：{qid} 已退役（{reason}），不应出现在 v2")
            checker.check_rows(
                [row for row in v2 if row["id"] in v1_ids],
                label="v2∩v1",
                strict_grounding_from=10**9,
            )
            checker.check_rows(
                [row for row in v2 if row["id"] not in v1_ids],
                label="v2 新增",
                strict_grounding_from=0,
                reuse_per_category=True,
            )
            checker.check_rows(v2, label="v2", quota=V2_QUOTA, check_split=True)
            for row in v2:
                if row["id"] in v1_ids and row.get("split") != "dev":
                    checker.error(f"v2/{row['id']}：v1 的题必须是 dev，不能进 test")
            per_category = Counter(row["category"] for row in v2 if row["split"] == "dev" and row["id"] not in v1_ids)
            for category, expected in V2_NEW_DEV.items():
                if per_category[category] != expected:
                    checker.error(f"v2：{category} 新增题里 dev 有 {per_category[category]} 道，应为 {expected}")
            kept_v1 = sum(1 for row in v2 if row['id'] in v1_ids)
            print(f"v2 共 {len(v2)} 题（含 v1 {kept_v1} 题，R12 已退役）：{dict(Counter(row['category'] for row in v2))}")
        else:
            print(f"v1 共 {len(v1)} 题：{dict(Counter(row['category'] for row in v1))}")

    for line in checker.warnings:
        print(f"WARN  {line}")
    for line in checker.errors:
        print(f"ERROR {line}")
    print(f"\n校验完成：{len(checker.errors)} error / {len(checker.warnings)} warning")
    return 1 if checker.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
