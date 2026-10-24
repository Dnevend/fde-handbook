"""5.8 生成：带出处回答、查不到时拒答。

两种模式：
  rag         —— 给 top-5 资料，要求标注出处、资料不足时拒答；
  closed_book —— 不给资料，只让模型凭参数知识回答（对照，看哪些问题它本来就会）。

判定不靠人眼：
  正确   —— 标准答案的 2-gram 有 ≥60% 出现在回答里，且数字全部对上；
  引用命中 —— 回答里的 [n] 至少有一个指向 gold 块；
  拒答   —— 回答包含「资料中没有相关信息」或同义说法。

产物：
  benchmarks/rag/generation.csv       分模式 × 题型的汇总
  benchmarks/rag/generation.jsonl     逐题输出，供人工复核
  benchmarks/rag/generation-sample.csv 20 条人工抽检样本（manual_label 留空）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402

REFUSAL = ["资料中没有相关信息", "无法回答", "没有相关信息", "资料不足", "未提供相关信息"]
CITATION = re.compile(r"\[(\d+)\]")
NUMERIC = re.compile(r"\d+(?:\.\d+)?%?|[一二三四五六七八九十百千万亿]+(?:万|亿|千|百|十)?(?:元|人|年|日|天|小时|个月|周)?")
PUNCT = re.compile(r"[\s，。；：、！？（）()「」“”\"'·,.;:!?\[\]]")


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def bigrams(text: str) -> set[str]:
    clean = normalize(text)
    return {clean[i:i + 2] for i in range(max(len(clean) - 1, 0))}


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in REFUSAL)


def numbers(text: str) -> set[str]:
    return set(NUMERIC.findall(text))


def answer_score(gold: str, response: str) -> tuple[bool, float]:
    gold_clean, resp_clean = normalize(gold), normalize(response)
    exact = gold_clean in resp_clean
    overlap = len(bigrams(gold) & bigrams(response)) / max(len(bigrams(gold)), 1)
    numeric_ok = numbers(gold).issubset(numbers(response))
    return bool(exact or (overlap >= 0.6 and numeric_ok)), round(overlap, 2)


def build_prompt(question: str, passages: list[dict], with_context: bool) -> list[dict]:
    if with_context:
        context = "\n\n".join(
            f"[{index}]（{chunk['doc_id']} {chunk.get('clause_no') or chunk.get('section') or ''}）{chunk['text']}"
            for index, chunk in enumerate(passages, start=1)
        )
        user = (
            "你是工厂安全与设备资料的问答助手。只能依据下面提供的资料回答，不得使用资料以外的知识。\n\n"
            f"资料：\n{context}\n\n"
            "要求：\n"
            "1. 回答简洁，不超过 150 字。\n"
            "2. 每个结论后用 [编号] 标注来源，编号必须来自上面的资料。\n"
            "3. 如果资料不足以回答，只输出：资料中没有相关信息。\n\n"
            f"问题：{question}"
        )
    else:
        user = (
            "你是工厂安全与设备资料的问答助手。不要引用任何资料，直接凭你已有的知识回答。\n"
            "回答简洁，不超过 150 字；如果不知道，只输出：资料中没有相关信息。\n\n"
            f"问题：{question}"
        )
    return [
        {"role": "system", "content": "你只输出答案本身，不要解释你的推理过程。"},
        {"role": "user", "content": user},
    ]


def main() -> int:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    rows = [json.loads(line) for line in (r.ROOT / "eval/qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    top20 = json.loads((r.CACHE / "hybrid-top20.json").read_text(encoding="utf-8"))
    reranked = json.loads((r.CACHE / "reranked.json").read_text(encoding="utf-8")) if (r.CACHE / "reranked.json").exists() else {}

    r.ensure_llm_server(ctx=8192)
    outputs: list[dict] = []

    for mode in ("rag", "closed_book"):
        for row in rows:
            with_context = mode == "rag"
            ranking = (reranked.get(row["id"]) or top20.get(row["id"], []))[:5]
            passages = [by_id[chunk_id] for chunk_id in ranking]
            answer = r.chat(build_prompt(row["question"], passages, with_context))
            cited = sorted({int(value) for value in CITATION.findall(answer) if 1 <= int(value) <= len(passages)})
            cited_chunks = [passages[index - 1] for index in cited]
            relevant = r.relevant_ids(chunks, row["sources"]) if row["sources"] else set()
            correct, overlap = answer_score(row["answer"], answer) if row["sources"] else (False, 0.0)
            citation_hit = any(chunk["chunk_id"] in relevant for chunk in cited_chunks)
            joint_hit = False
            if row.get("joint") and row["sources"]:
                joint_hit = all(
                    any(r.chunk_is_relevant(chunk, gold) for chunk in cited_chunks)
                    for gold in row["sources"]
                )
            outputs.append(
                {
                    "id": row["id"],
                    "mode": mode,
                    "category": row["category"],
                    "question": row["question"],
                    "joint": bool(row.get("joint")),
                    "gold": row["answer"],
                    "answer": answer,
                    "cited": cited,
                    "correct": correct,
                    "overlap": overlap,
                    "citation_hit": citation_hit,
                    "citation_all": joint_hit,
                    "refused": is_refusal(answer),
                }
            )
            r.log(f"{mode} {row['id']} correct={correct} cited={cited} refused={is_refusal(answer)}")

    summary = []
    for mode in ("rag", "closed_book"):
        for category in ("direct", "condition", "cross", "refuse", "all"):
            subset = [row for row in outputs if row["mode"] == mode and (category == "all" or row["category"] == category)]
            if not subset:
                continue
            answerable = [row for row in subset if row["category"] != "refuse"]
            summary.append(
                {
                    "mode": mode,
                    "category": category,
                    "n": len(subset),
                    "correct_rate": round(float(np.mean([row["correct"] for row in answerable])), 3) if answerable else "",
                    "citation_hit": round(float(np.mean([row["citation_hit"] for row in answerable])), 3) if answerable else "",
                    "citation_all": round(float(np.mean([row["citation_all"] for row in answerable if row["joint"]])), 3) if any(row["joint"] for row in answerable) else "",
                    "refused_rate": round(float(np.mean([row["refused"] for row in subset])), 3),
                    "false_refusal": round(float(np.mean([row["refused"] for row in answerable])), 3) if answerable else "",
                }
            )

    r.write_csv(r.BENCH / "generation.csv", summary)
    r.save_jsonl(r.BENCH / "generation.jsonl", outputs)

    sample = []
    for category in ("direct", "condition", "cross", "refuse"):
        picked = [row for row in outputs if row["mode"] == "rag" and row["category"] == category][:5]
        for row in picked:
            sample.append(
                {
                    "id": row["id"],
                    "category": category,
                    "question": row["question"],
                    "joint": bool(row.get("joint")),
                    "gold": row["gold"],
                    "answer": row["answer"],
                    "cited": " ".join(map(str, row["cited"])),
                    "auto_correct": row["correct"],
                    "manual_label": "",
                }
            )
    r.write_csv(r.BENCH / "generation-sample.csv", sample)
    r.log("写入 generation.csv / generation.jsonl / generation-sample.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
