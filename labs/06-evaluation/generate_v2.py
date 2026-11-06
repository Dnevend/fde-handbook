#!/usr/bin/env python3
"""6.3 在 v2 test 上跑一遍生成，用来看「没有人工标签时裁判数字有多不可信」。

只取 60 题（可答 40 + 拒答 20），检索用 6.2 的 RRF top-5，生成提示词与 5.8 相同。
产物里的 `auto_correct` 是 5.8 的字符串/数字判定，`judge_*` 由 judge_v2.py 另跑。

用法（在 labs/ 下）：
    uv run 06-evaluation/generate_v2.py

产物：
    benchmarks/evaluation/generation-v2.jsonl   逐题输出、引用、检索到的块
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))

import raglib as r  # noqa: E402

EVAL = ROOT / "eval/qa-v2.jsonl"
OUT = ROOT / "benchmarks/evaluation/generation-v2.jsonl"
CITATION = re.compile(r"\[(\d+)\]")
REFUSAL = ["资料中没有相关信息", "无法回答", "没有相关信息", "资料不足", "未提供相关信息"]
PUNCT = re.compile(r"[\s，。；：、！？（）()「」“”\"'·,.;:!?\[\]]")
NUMERIC = re.compile(r"\d+(?:\.\d+)?%?|[一二三四五六七八九十百千万亿]+(?:万|亿|千|百|十)?(?:元|人|年|日|天|小时|个月|周)?")


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def bigrams(text: str) -> set[str]:
    clean = normalize(text)
    return {clean[i : i + 2] for i in range(max(len(clean) - 1, 0))}


def numbers(text: str) -> set[str]:
    return set(NUMERIC.findall(text))


def answer_score(gold: str, response: str) -> tuple[bool, float]:
    gold_clean, resp_clean = normalize(gold), normalize(response)
    exact = gold_clean in resp_clean
    overlap = len(bigrams(gold) & bigrams(response)) / max(len(bigrams(gold)), 1)
    numeric_ok = numbers(gold).issubset(numbers(response))
    return bool(exact or (overlap >= 0.6 and numeric_ok)), round(overlap, 2)


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in REFUSAL)


def build_prompt(question: str, passages: list[dict]) -> list[dict]:
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
    return [
        {"role": "system", "content": "你只输出答案本身，不要解释你的推理过程。"},
        {"role": "user", "content": user},
    ]


def main() -> int:
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [row for row in rows if row.get("split") == "test"]
    answerable = [row for row in test if row["category"] != "refuse"]
    refuse = [row for row in test if row["category"] == "refuse"]

    # 每类取固定数量，答案可复现
    picked = []
    for category, count in (("direct", 14), ("condition", 13), ("cross", 13)):
        picked.extend([row for row in answerable if row["category"] == category][:count])
    picked.extend(refuse[:20])
    picked.sort(key=lambda row: row["id"])

    chunks = r.cached_chunks("structure", include_synthetic=False)
    ids = [chunk["chunk_id"] for chunk in chunks]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    vectors = np.load(r.CACHE / "vectors-1024-real.npy").astype(np.float32)
    queries = r.embed_texts([row["question"] for row in picked], kind="query")
    # raglib 没有 BM25，直接复用 5.6 的实现
    from retrieve import BM25, tokenize  # noqa: PLC0415

    bm25 = BM25([tokenize(chunk["text"]) for chunk in chunks], r.CONFIG["retrieval"]["bm25_k1"], r.CONFIG["retrieval"]["bm25_b"])

    outputs = []
    for number, row in enumerate(picked, start=1):
        dense = [ids[position] for position in np.argsort(-(vectors @ queries[number - 1]))[:20]]
        sparse = [ids[position] for position in np.argsort(-bm25.scores(tokenize(row["question"])))[:20]]
        fused: dict[str, float] = {}
        for ranking in (sparse, dense):
            for position, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (r.CONFIG["retrieval"]["rrf_k"] + position)
        ranking = [chunk_id for chunk_id, _ in sorted(fused.items(), key=lambda pair: -pair[1])][:5]
        passages = [by_id[chunk_id] for chunk_id in ranking]

        answer = r.chat(build_prompt(row["question"], passages))
        cited = sorted({int(value) for value in CITATION.findall(answer) if 1 <= int(value) <= len(passages)})
        if row["category"] == "refuse":
            correct = is_refusal(answer)
            overlap = 0.0
        else:
            correct, overlap = answer_score(row["answer"], answer)
        outputs.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "gold": row["answer"],
                "answer": answer,
                "cited": cited,
                "passages": ranking,
                "auto_correct": correct,
                "overlap": overlap,
                "refused": is_refusal(answer),
            }
        )
        r.log(f"{number}/{len(picked)} {row['id']} auto={correct} cited={cited}")

    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in outputs) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"题目 {len(outputs)}：{dict(Counter(row['category'] for row in outputs))}")
    print(f"自动判定正确 {sum(1 for row in outputs if row['auto_correct'])} / {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
