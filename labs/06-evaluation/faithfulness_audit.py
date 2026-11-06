#!/usr/bin/env python3
"""6.3 忠实度：引用段落是否真的支撑了那句话。

两套判定：

  1. 程序判定：把回答按句切分，句子末尾的 [n] 对应第 n 段资料；句子去掉
     引用标记后与第 n 段的 4-gram 覆盖率达到 0.5 才算「有支撑」；
  2. 裁判判定：把同一批资料与回答给本地 9B，让它判断「每个结论是否都能在
     资料里找到依据」，再和程序判定对齐。

同时算 5.8 的「引用相关性」（引用的段落是不是标准答案所在条款），用来
说明「引用相关」与「引用支撑」不是一回事。

用法（在 labs/ 下）：
    uv run 06-evaluation/faithfulness_audit.py
    uv run 06-evaluation/faithfulness_audit.py --limit 10

产物：
    benchmarks/evaluation/faithfulness.csv          逐条：支撑率、裁判结论
    benchmarks/evaluation/faithfulness-summary.csv  汇总：微平均、一致率、κ
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))

import raglib as r  # noqa: E402

GEN = ROOT / "benchmarks/rag/generation.jsonl"
AUDIT = ROOT / "benchmarks/rag/generation-audit.csv"
RERANK = ROOT / "labs/05-rag/.cache/reranked.json"
OUT = ROOT / "benchmarks/evaluation"
PUNCT = re.compile(r"[\s，。；：、！？（）()「」“”\"'·,.;:!?\[\]【】]")
CITATION = re.compile(r"\[(\d+)\]")

PROMPT = """下面有 5 段资料和一个回答。判断回答里的每个事实性结论是否都能在资料中找到依据。
允许措辞不同；数字、主体、适用条件必须一致。引用编号不必正确，只看内容是否被资料支撑。
只输出 JSON，不要解释：{"faithful": true 或 false, "unsupported": ["不被支撑的结论，最多 3 条"]}"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def ngrams(text: str, n: int = 4) -> set[str]:
    clean = normalize(text)
    return {clean[i : i + n] for i in range(max(len(clean) - n + 1, 0))}


def coverage(sentence: str, passage: str) -> float:
    gold = ngrams(sentence)
    return len(gold & ngrams(passage)) / max(len(gold), 1)


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}")


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="忠实度审计")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    chunks = {chunk["chunk_id"]: chunk for chunk in r.cached_chunks("structure", include_synthetic=False)}
    rankings = json.loads(RERANK.read_text(encoding="utf-8"))
    outputs = {row["id"]: row for row in read_jsonl(GEN) if row["mode"] == "rag"}
    audit = {row["id"]: row for row in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    golds_by_id = {
        row["id"]: row.get("sources") or []
        for row in read_jsonl(ROOT / "eval/qa.jsonl")
    }
    ids = [qid for qid in audit if qid in outputs and outputs[qid]["category"] != "refuse"]
    if args.limit:
        ids = ids[: args.limit]

    rows = []
    for number, qid in enumerate(ids, start=1):
        output = outputs[qid]
        top = [chunks[chunk_id] for chunk_id in rankings[qid][:5]]
        # 引用常写在句号后面（「……处理。[1]」）；先把它挪到句号前再切句，
        # 否则会切出一个只有 [1] 的「句子」，把支撑率算成 0。
        answer = re.sub(r"([。！？])\s*((?:\[\d+\])+)", r"\2\1", output["answer"])
        sentences = re.split(r"(?<=[。！？])", answer)
        total = supported = relevant = 0
        content_total = content_supported = 0
        for sentence in sentences:
            citations = [int(value) for value in CITATION.findall(sentence)]
            if not citations:
                continue
            text = CITATION.sub("", sentence).strip()
            content_total += 1
            if any(coverage(text, passage["text"]) >= 0.5 for passage in top):
                content_supported += 1
            for citation in citations:
                total += 1
                if not (1 <= citation <= len(top)):
                    # 越界引用（如只有 5 段却写 [8]）算不支撑，不能静默丢掉
                    continue
                passage = top[citation - 1]
                if coverage(text, passage["text"]) >= 0.5:
                    supported += 1
                golds = golds_by_id.get(qid, [])
                # 5.8 的引用相关性口径：段落是否落在标准答案条款上
                if any(passage["doc_id"] == gold["doc"] and passage.get("clause_no") == gold.get("clause") for gold in golds):
                    relevant += 1

        context = "\n\n".join(
            f"[{index}]（{chunk['doc_id']} {chunk.get('clause_no') or ''}）{chunk['text']}"
            for index, chunk in enumerate(top, start=1)
        )
        start = time.perf_counter()
        raw = r.chat(
            [
                {"role": "system", "content": "你只输出 JSON。"},
                {"role": "user", "content": f"{PROMPT}\n\n资料：\n{context}\n\n回答：{output['answer']}"},
            ],
            max_tokens=200,
        )
        latency = time.perf_counter() - start
        payload = parse_json(raw)
        faithful = payload.get("faithful")
        judge = bool(faithful) if isinstance(faithful, bool) else str(faithful).lower() == "true"
        unsupported = payload.get("unsupported") or []
        if isinstance(unsupported, list):
            unsupported = " / ".join(str(item)[:40] for item in unsupported[:3])
        rows.append(
            {
                "id": qid,
                "category": output["category"],
                "human": audit[qid]["manual_label"],
                "citations": total,
                "supported": supported,
                "support_rate": round(supported / total, 2) if total else "",
                "all_supported": int(bool(total) and supported == total),
                "relevant_rate": round(relevant / total, 2) if total else "",
                "content_total": content_total,
                "content_supported": content_supported,
                "content_rate": round(content_supported / content_total, 2) if content_total else "",
                "content_all_supported": int(bool(content_total) and content_supported == content_total),
                "judge_faithful": int(judge),
                "judge_unsupported": unsupported,
                "latency_s": round(latency, 2),
            }
        )
        r.log(f"{number}/{len(ids)} {qid} support={supported}/{total} judge={int(judge)}")

    write_csv(OUT / "faithfulness.csv", rows)

    total_citations = sum(row["citations"] for row in rows)
    total_supported = sum(row["supported"] for row in rows)
    total_relevant = sum(round(float(row["relevant_rate"] or 0) * row["citations"]) for row in rows)
    program_all = [bool(row["all_supported"]) for row in rows]
    program_80 = [bool(row["citations"]) and float(row["support_rate"] or 0) >= 0.8 for row in rows]
    content_all = [bool(row["content_all_supported"]) for row in rows]
    judge_flags = [bool(row["judge_faithful"]) for row in rows]
    summary = [
        ("answers", len(rows)),
        ("citations", total_citations),
        ("citation_support_micro", round(total_supported / total_citations, 3) if total_citations else ""),
        ("citation_relevance_micro", round(total_relevant / total_citations, 3) if total_citations else ""),
        ("content_support_micro", round(sum(row["content_supported"] for row in rows) / max(1, sum(row["content_total"] for row in rows)), 3)),
        ("answers_content_all_supported", round(sum(content_all) / len(rows), 3)),
        ("answers_all_supported", round(sum(program_all) / len(rows), 3)),
        ("answers_support_ge_80", round(sum(program_80) / len(rows), 3)),
        ("judge_faithful", round(sum(judge_flags) / len(rows), 3)),
        ("judge_vs_all_supported_agreement", round(sum(a == b for a, b in zip(judge_flags, program_all)) / len(rows), 3)),
        ("judge_vs_all_supported_kappa", round(kappa(judge_flags, program_all), 3)),
        ("judge_vs_80_agreement", round(sum(a == b for a, b in zip(judge_flags, program_80)) / len(rows), 3)),
        ("judge_vs_80_kappa", round(kappa(judge_flags, program_80), 3)),
        ("judge_vs_content_agreement", round(sum(a == b for a, b in zip(judge_flags, content_all)) / len(rows), 3)),
        ("judge_vs_content_kappa", round(kappa(judge_flags, content_all), 3)),
        ("mean_latency_s", round(sum(row["latency_s"] for row in rows) / len(rows), 2)),
    ]
    write_csv(OUT / "faithfulness-summary.csv", [{"metric": key, "value": value} for key, value in summary])
    for key, value in summary:
        print(f"{key:36s} {value}")

    # 忠实度与人工正确率的交叉表：不忠实的回答，人工是不是也判错了
    table = Counter((row["human"], "faithful" if row["judge_faithful"] else "unfaithful") for row in rows)
    cross = [
        {"human": human, "judge": judge, "count": table[(human, judge)]}
        for human in ("correct", "partial", "wrong")
        for judge in ("faithful", "unfaithful")
    ]
    write_csv(OUT / "faithfulness-cross.csv", cross)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
