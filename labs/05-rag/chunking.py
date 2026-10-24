"""5.3 切分：按长度 / 按结构 / 按语义，比较检索效果与表格完整性。

三种策略：
  fixed     —— 固定字符窗口（chunk_size=400, overlap=80）
  structure —— 法规按「第X条」、文档按标题/段落切，超长再按句切
  semantic  —— 先分句，句向量相邻余弦低于阈值处断开，再按长度封顶

公平比较的关键：fixed/semantic 的块没有条款号，不能再用「doc 对就算命中」。
这里按块与标准条款的 4-gram 覆盖率判断相关性（≥0.35 算命中）——
语义切分常把一条条款拆成两块，阈值取 0.5 会把「找到半条」错判成失败。

产物：benchmarks/rag/chunking.csv
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402

SENTENCE = re.compile(r"(?<=[。！？；\n])")
TABLE_ROW = re.compile(r"^\s*\|")


def grams(text: str, size: int = 4) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i:i + size] for i in range(max(len(text) - size + 1, 0))}


def gold_texts(chunks: list[dict]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for chunk in chunks:
        if chunk["clause_no"]:
            key = (chunk["doc_id"], chunk["clause_no"])
            out[key] = out.get(key, "") + chunk["text"]
    return out


def relevant_ids(chunks: list[dict], golds: list[dict], golds_text: dict[tuple[str, str], str]) -> set[str]:
    ids: set[str] = set()
    for gold in golds:
        clause = gold.get("clause")
        if not clause:
            ids |= {chunk["chunk_id"] for chunk in chunks if chunk["doc_id"] == gold["doc"]}
            continue
        target = grams(golds_text.get((gold["doc"], clause), ""))
        if not target:
            continue
        for chunk in chunks:
            if chunk["doc_id"] != gold["doc"]:
                continue
            overlap = len(grams(chunk["text"]) & target) / len(target)
            if overlap >= 0.35:
                ids.add(chunk["chunk_id"])
    return ids


def semantic_chunks(docs: list[dict], threshold: float) -> list[dict]:
    size = r.CONFIG["corpus"]["chunk_size"]
    min_chunk = r.CONFIG["corpus"]["min_chunk"]
    chunks: list[dict] = []
    for doc in docs:
        sentences = [part.strip() for part in SENTENCE.split(doc["text"]) if part.strip()]
        if not sentences:
            continue
        vectors = r.embed_texts(sentences)
        groups: list[list[str]] = []
        current = [sentences[0]]
        for index in range(1, len(sentences)):
            similarity = float(vectors[index - 1] @ vectors[index])
            length = sum(len(item) for item in current)
            if similarity < threshold or length + len(sentences[index]) > size * 2:
                groups.append(current)
                current = [sentences[index]]
            else:
                current.append(sentences[index])
        groups.append(current)
        for group in groups:
            text = "".join(group).strip()
            if len(text) < min_chunk:
                continue
            chunks.append({"doc_id": doc["doc_id"], "clause_no": None, "section": None, "text": text})
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{chunk['doc_id']}#s{index:05d}"
        chunk["char_len"] = len(chunk["text"])
    return chunks


def table_headers() -> dict[str, set[str]]:
    """每份文档里每个表格的第一行（表头），用来判断块是否丢了表头。"""
    headers: dict[str, set[str]] = {}
    for doc in r.load_docs(include_synthetic=False):
        found: set[str] = set()
        in_table = False
        for line in doc["text"].splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                if not in_table:
                    found.add(stripped)
                in_table = True
            else:
                in_table = False
        headers[doc["doc_id"]] = found
    return headers


def broken_tables(chunks: list[dict]) -> int:
    """没有带上表头的数据行数：表格被切断时最直接的损失。"""
    headers = table_headers()
    broken = 0
    for chunk in chunks:
        lines = [line.strip() for line in chunk["text"].splitlines() if line.strip().startswith("|")]
        if not lines:
            continue
        if any(line in headers.get(chunk["doc_id"], set()) for line in lines):
            continue
        broken += sum(1 for line in lines if not re.fullmatch(r"\|[\s\-:|]+\|", line))
    return broken


def evaluate(chunks: list[dict], rows: list[dict], vectors: np.ndarray, queries: np.ndarray, golds_text) -> dict:
    ids = [chunk["chunk_id"] for chunk in chunks]
    scores = queries @ vectors.T
    recalls, mrrs, ndcgs = [], [], []
    for row, row_scores in zip(rows, scores):
        order = np.argsort(-row_scores)[:10]
        ranked = [ids[i] for i in order]
        relevant = relevant_ids(chunks, row["sources"], golds_text)
        recalls.append(r.recall_at_k(ranked, relevant, 5))
        mrrs.append(r.mrr_at_k(ranked, relevant, 10))
        ndcgs.append(r.ndcg_at_k(ranked, relevant, 10))
    return {
        "recall@5": round(float(np.mean(recalls)), 3),
        "mrr@10": round(float(np.mean(mrrs)), 3),
        "ndcg@10": round(float(np.mean(ndcgs)), 3),
    }


def main() -> int:
    rows = [json.loads(line) for line in (r.ROOT / "eval/qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["category"] != "refuse"]
    query_path = r.CACHE / "query-vectors.npy"
    queries = np.load(query_path) if query_path.exists() else r.embed_texts([row["question"] for row in rows], kind="query")
    golds_text = gold_texts(r.cached_chunks("structure", include_synthetic=False))

    out: list[dict] = []
    for strategy in ("structure", "fixed", "semantic"):
        start = time.perf_counter()
        if strategy == "structure":
            chunks = r.cached_chunks("structure", include_synthetic=False)
            cache = r.CACHE / "vectors-1024-real.npy"
            vectors = np.load(cache) if cache.exists() else r.embed_texts([chunk["text"] for chunk in chunks])
        elif strategy == "fixed":
            chunks = r.build_chunks("fixed", include_synthetic=False)
            cache = r.CACHE / "vectors-fixed-real.npy"
            if cache.exists():
                vectors = np.load(cache)
            else:
                vectors = r.embed_texts([chunk["text"] for chunk in chunks])
                np.save(cache, vectors)
        else:
            chunks = semantic_chunks(r.load_docs(include_synthetic=False), r.CONFIG["corpus"]["semantic_threshold"])
            cache = r.CACHE / "vectors-semantic-real.npy"
            if cache.exists():
                vectors = np.load(cache)
            else:
                vectors = r.embed_texts([chunk["text"] for chunk in chunks])
                np.save(cache, vectors)
        elapsed = time.perf_counter() - start

        result = evaluate(chunks, rows, vectors, queries, golds_text)
        lengths = np.array([chunk["char_len"] for chunk in chunks])
        out.append(
            {
                "strategy": strategy,
                "chunks": len(chunks),
                "avg_chars": int(lengths.mean()),
                "p95_chars": int(np.percentile(lengths, 95)),
                "recall@5": result["recall@5"],
                "mrr@10": result["mrr@10"],
                "ndcg@10": result["ndcg@10"],
                "broken_table_rows": broken_tables(chunks),
                "elapsed_s": round(elapsed, 1),
            }
        )
        r.log(f"{strategy}: {len(chunks)} 块 recall@5={result['recall@5']}")

    r.write_csv(r.BENCH / "chunking.csv", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
