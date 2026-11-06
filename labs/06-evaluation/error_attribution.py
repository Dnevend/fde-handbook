#!/usr/bin/env python3
"""6.5 错误归因：把错题分到「索引 / 检索 / 拒答 / 引用 / 生成」五类。

输入是已经跑过的生成结果，不重新调模型：

  - v1：60 道可答题的人工标签（10 道非 correct）
  - v2：60 题里裁判判为非 correct 的 8 道可答题

判定顺序就是 6.5 的排查流程图：金标在不在索引 → 前 5 有没有金标 → 是不是
误拒 → 引用有没有支撑 → 剩下的才算生成错误。

产物：benchmarks/evaluation/error-attribution.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "labs/06-evaluation"))

import raglib as r  # noqa: E402
import retrieval_metrics as rm  # noqa: E402

OUT = ROOT / "benchmarks/evaluation/error-attribution.csv"
REFUSAL = ["资料中没有相关信息", "无法回答", "没有相关信息", "资料不足", "未提供相关信息", "未提及", "未说明"]
CITATION = re.compile(r"\[(\d+)\]")
PUNCT = re.compile(r"[\s，。；：、！？（）()「」“”\"'·,.;:!?\[\]【】]")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def ngrams(text: str, n: int = 4) -> set[str]:
    clean = normalize(text)
    return {clean[i : i + n] for i in range(max(len(clean) - n + 1, 0))}


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in REFUSAL)


def support_rate(answer: str, passages: list[dict]) -> tuple[float, int]:
    if not passages:
        return 0.0, 0
    answer = re.sub(r"([。！？])\s*((?:\[\d+\])+)", r"\2\1", answer)
    supported = total = 0
    for sentence in re.split(r"(?<=[。！？])", answer):
        citations = [int(value) for value in CITATION.findall(sentence)]
        if not citations:
            continue
        text = CITATION.sub("", sentence).strip()
        gold = ngrams(text)
        for citation in citations:
            total += 1
            if 1 <= citation <= len(passages):
                passage = passages[citation - 1]
                if len(gold & ngrams(passage["text"])) / max(len(gold), 1) >= 0.5:
                    supported += 1
    return (supported / total if total else 0.0), total


def clause_present(doc: str, clause: str, chunks: list[dict], texts: dict[tuple[str, str], str]) -> bool:
    """条款正文是否在索引里：条号对上，或某块覆盖了条款正文的一半。"""
    body = texts.get((doc, clause), "")
    if not body:
        return False
    for chunk in chunks:
        if chunk["doc_id"] != doc:
            continue
        if chunk.get("clause_no") == clause:
            return True
        if len(ngrams(body) & ngrams(chunk["text"])) / max(len(ngrams(body)), 1) >= 0.5:
            return True
    return False


def gold_in_top(top: list[dict], specs: list[dict], texts: dict[tuple[str, str], str]) -> bool:
    for spec in specs:
        body = texts.get((spec["doc"], spec["clause"]), "")
        if not any(
            passage["doc_id"] == spec["doc"]
            and (
                passage.get("clause_no") == spec["clause"]
                or (
                    body
                    and len(ngrams(body) & ngrams(passage["text"])) / max(len(ngrams(body)), 1) >= 0.5
                )
            )
            for passage in top
        ):
            return False
    return True


def stage_of(
    *,
    index_ok: bool,
    retrieval_ok: bool,
    refused: bool,
    support: float,
    citations: int,
) -> str:
    if not index_ok:
        return "index_gap"
    if not retrieval_ok:
        return "retrieval_miss"
    if refused:
        return "false_refusal"
    if citations == 0 or support < 0.5:
        return "unsupported"
    return "generation_error"


def main() -> int:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    texts = rm.load_clause_texts()
    rows: list[dict] = []

    # ---- v1：人工标签
    v1_qa = {row["id"]: row for row in read_jsonl(ROOT / "eval/qa.jsonl")}
    v1_gen = {row["id"]: row for row in read_jsonl(ROOT / "benchmarks/rag/generation.jsonl") if row["mode"] == "rag"}
    v1_audit = {row["id"]: row for row in csv.DictReader((ROOT / "benchmarks/rag/generation-audit.csv").open(encoding="utf-8"))}
    v1_faith = {row["id"]: row for row in csv.DictReader((ROOT / "benchmarks/evaluation/faithfulness.csv").open(encoding="utf-8"))}
    reranked = json.loads((ROOT / "labs/05-rag/.cache/reranked.json").read_text(encoding="utf-8"))

    for qid, label in v1_audit.items():
        if label["category"] == "refuse" or label["manual_label"] == "correct" or qid not in v1_gen:
            continue
        question_row = v1_qa[qid]
        answer_row = v1_gen[qid]
        specs = rm.relevance_specs(question_row, texts)
        index_ok = all(clause_present(spec["doc"], spec["clause"], chunks, texts) for spec in specs)
        top = [by_id[chunk_id] for chunk_id in reranked[qid][:5]]
        retrieval_ok = gold_in_top(top, specs, texts)
        faith = v1_faith.get(qid, {})
        support = float(faith.get("support_rate") or 0)
        citations = int(faith.get("citations") or 0)
        rows.append(
            {
                "set": "v1",
                "id": qid,
                "category": label["category"],
                "label": label["manual_label"],
                "stage": stage_of(
                    index_ok=index_ok,
                    retrieval_ok=retrieval_ok,
                    refused=is_refusal(answer_row["answer"]),
                    support=support,
                    citations=citations,
                ),
                "index_ok": int(index_ok),
                "retrieval_ok": int(retrieval_ok),
                "refused": int(is_refusal(answer_row["answer"])),
                "citations": citations,
                "support_rate": round(support, 2),
                "overlap": answer_row.get("overlap", ""),
                "note": "",
            }
        )

    # ---- v2：裁判标签
    v2_qa = {row["id"]: row for row in read_jsonl(ROOT / "eval/qa-v2.jsonl")}
    for row in read_jsonl(ROOT / "benchmarks/evaluation/generation-v2-judged.jsonl"):
        if row["category"] == "refuse" or row["judge_label"] == "correct":
            continue
        query = v2_qa[row["id"]]
        specs = rm.relevance_specs(query, texts)
        index_ok = all(clause_present(spec["doc"], spec["clause"], chunks, texts) for spec in specs)
        top = [by_id[chunk_id] for chunk_id in row["passages"]]
        retrieval_ok = gold_in_top(top, specs, texts)
        support, citations = support_rate(row["answer"], top)
        rows.append(
            {
                "set": "v2",
                "id": row["id"],
                "category": row["category"],
                "label": row["judge_label"],
                "stage": stage_of(
                    index_ok=index_ok,
                    retrieval_ok=retrieval_ok,
                    refused=is_refusal(row["answer"]),
                    support=support,
                    citations=citations,
                ),
                "index_ok": int(index_ok),
                "retrieval_ok": int(retrieval_ok),
                "refused": int(is_refusal(row["answer"])),
                "citations": citations,
                "support_rate": round(support, 2),
                "overlap": row.get("overlap", ""),
                "note": row.get("judge_reason", "")[:40],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}")
    from collections import Counter

    print("stage:", dict(Counter(row["stage"] for row in rows)))
    for row in rows:
        print(row["set"], row["id"], row["category"], row["label"], "->", row["stage"], row["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
