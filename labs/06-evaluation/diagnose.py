#!/usr/bin/env python3
"""6.5 诊断卡：给一道错题画出失败环节，别一上来就改提示词。

    uv run 06-evaluation/diagnose.py --set v1 --id X13
    uv run 06-evaluation/diagnose.py --set v2 --id X32
    uv run 06-evaluation/diagnose.py --all            # 只列判定，不展开

判定顺序与 6.5 的流程图一致：索引 → 检索 → 拒答 → 引用 → 生成/标注。
输出是**第一个失败环节**，不是唯一原因；后面的环节也可能同时有问题。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "labs/06-evaluation"))

import raglib as r  # noqa: E402
import retrieval_metrics as rm  # noqa: E402
from error_attribution import (  # noqa: E402
    clause_present,
    gold_in_top,
    is_refusal,
    stage_of,
    support_rate,
)

TIPS = {
    "index_gap": "先修切分/入库；这不是检索器的错，改提示词也没用。",
    "retrieval_miss": "先查切分、向量、查询改写；生成端可能只是没拿到第二段。",
    "false_refusal": "资料已经召回却拒答：查上下文是否够、拒答提示词是否过强。",
    "unsupported": "引用与结论对不上：先约束输出结构（逐句引用/引用校验）。",
    "generation_error": "条款和引用都对，答案仍错：查提示词、模型、标准答案。",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows() -> dict[tuple[str, str], dict]:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    texts = rm.load_clause_texts()
    rows: dict[tuple[str, str], dict] = {}

    v1_qa = {row["id"]: row for row in read_jsonl(ROOT / "eval/qa.jsonl")}
    audit_label = {row["id"]: row["manual_label"] for row in csv.DictReader((ROOT / "benchmarks/rag/generation-audit.csv").open(encoding="utf-8"))}
    v1_gen = {row["id"]: row for row in read_jsonl(ROOT / "benchmarks/rag/generation.jsonl") if row["mode"] == "rag"}
    reranked = json.loads((ROOT / "labs/05-rag/.cache/reranked.json").read_text(encoding="utf-8"))
    for qid, output in v1_gen.items():
        if output["category"] == "refuse":
            continue
        question = v1_qa[qid]
        top = [by_id[chunk_id] for chunk_id in reranked[qid][:5]]
        specs = rm.relevance_specs(question, texts)
        support, citations = support_rate(output["answer"], top)
        rows[("v1", qid)] = {
            "label": audit_label.get(qid, ""),
            "question": question,
            "answer": output["answer"],
            "gold": output["gold"],
            "top": top,
            "specs": specs,
            "texts": texts,
            "support": support,
            "citations": citations,
            "index_ok": all(clause_present(spec["doc"], spec["clause"], chunks, texts) for spec in specs),
            "retrieval_ok": gold_in_top(top, specs, texts),
        }

    v2_qa = {row["id"]: row for row in read_jsonl(ROOT / "eval/qa-v2.jsonl")}
    for output in read_jsonl(ROOT / "benchmarks/evaluation/generation-v2-judged.jsonl"):
        if output["category"] == "refuse":
            continue
        question = v2_qa[output["id"]]
        top = [by_id[chunk_id] for chunk_id in output["passages"]]
        specs = rm.relevance_specs(question, texts)
        support, citations = support_rate(output["answer"], top)
        rows[("v2", output["id"])] = {
            "label": output.get("judge_label", ""),
            "question": question,
            "answer": output["answer"],
            "gold": output["gold"],
            "top": top,
            "specs": specs,
            "texts": texts,
            "support": support,
            "citations": citations,
            "index_ok": all(clause_present(spec["doc"], spec["clause"], chunks, texts) for spec in specs),
            "retrieval_ok": gold_in_top(top, specs, texts),
        }
    return rows


def card(key: tuple[str, str], row: dict) -> str:
    question = row["question"]
    if row.get("label") == "correct":
        return "\n".join(
            [
                f"===== {key[0]} / {key[1]}  [{question['category']}]",
                f"问题：{question['question']}",
                "判定：无错误（标签 correct）",
                f"证据：金标在索引 {'✓' if row['index_ok'] else '✗'} · 前5命中 {'✓' if row['retrieval_ok'] else '✗'} · "
                f"引用 {row['citations']} 条/支撑 {row['support']:.0%} · 拒答 {'是' if is_refusal(row['answer']) else '否'}",
                f"回答：{row['answer'][:160]}",
            ]
        )
    stage = stage_of(
        index_ok=row["index_ok"],
        retrieval_ok=row["retrieval_ok"],
        refused=is_refusal(row["answer"]),
        support=row["support"],
        citations=row["citations"],
    )
    sources = ", ".join(f"{source['doc']} {source['clause']}" for source in question["sources"])
    lines = [
        f"===== {key[0]} / {key[1]}  [{question['category']}]",
        f"问题：{question['question']}",
        f"判定：{stage}",
        f"证据：金标在索引 {'✓' if row['index_ok'] else '✗'} · 前5命中 {'✓' if row['retrieval_ok'] else '✗'} · "
        f"引用 {row['citations']} 条/支撑 {row['support']:.0%} · 拒答 {'是' if is_refusal(row['answer']) else '否'}",
        f"出处：{sources}",
        f"建议：{TIPS[stage]}",
        f"回答：{row['answer'][:160]}",
        f"标准：{row['gold'][:160]}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="错题归因诊断卡")
    parser.add_argument("--set", choices=["v1", "v2"], default="v2")
    parser.add_argument("--id")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.id:
        parser.error("给 --id 或 --all")

    rows = load_rows()
    if args.all:
        for (set_name, qid), row in rows.items():
            if row.get("label") == "correct":
                continue
            stage = stage_of(
                index_ok=row["index_ok"],
                retrieval_ok=row["retrieval_ok"],
                refused=is_refusal(row["answer"]),
                support=row["support"],
                citations=row["citations"],
            )
            print(f"{set_name} {qid} {row['question']['category']} -> {stage}")
        return 0

    key = (args.set, args.id)
    if key not in rows:
        raise SystemExit(f"没有 {args.set} {args.id}；可用 --all 看清单")
    print(card(key, rows[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
