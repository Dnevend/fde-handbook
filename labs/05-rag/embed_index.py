"""5.4 Embedding：维度（1024/512/256）与向量精度（f16/q8）对检索的影响。

做法：
  1. 把全部块向量化一次，缓存到 .cache/；
  2. 对每种（维度, 精度）组合，截断/量化后做暴力余弦检索；
  3. 在 eval/qa.jsonl 的非拒答题上算 Recall@10 / MRR@10 / nDCG@10；
  4. 索引体积按 n × dim × 每维字节数计算（f16=2 字节，q8=1 字节）。

产物：benchmarks/rag/embedding.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402


def load_eval() -> list[dict]:
    path = r.ROOT / "eval/qa.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["category"] != "refuse"]


def doc_vectors(chunks: list[dict]) -> tuple[np.ndarray, float]:
    path = r.CACHE / "vectors-1024-real.npy"
    if path.exists():
        return np.load(path), 0.0
    start = time.perf_counter()
    vectors = r.embed_texts([chunk["text"] for chunk in chunks])
    elapsed = time.perf_counter() - start
    np.save(path, vectors)
    (r.CACHE / "vectors-meta.json").write_text(
        json.dumps({"chunks": len(chunks), "seconds": round(elapsed, 1)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return vectors, elapsed


def query_vectors(queries: list[str]) -> np.ndarray:
    path = r.CACHE / "query-vectors.npy"
    if path.exists() and np.load(path).shape[0] == len(queries):
        return np.load(path)
    vectors = r.embed_texts(queries, kind="query")
    np.save(path, vectors)
    return vectors


def rank(scores: np.ndarray, ids: list[str], k: int) -> list[str]:
    order = np.argsort(-scores)[:k]
    return [ids[i] for i in order]


def evaluate(chunks: list[dict], rows: list[dict], docs: np.ndarray, queries: np.ndarray, k: int = 10) -> dict:
    ids = [chunk["chunk_id"] for chunk in chunks]
    recalls, mrrs, ndcgs, joint = [], [], [], []
    for row, qvec in zip(rows, queries):
        ranked = rank(docs @ qvec, ids, k)
        golds = row["sources"]
        relevant = r.relevant_ids(chunks, golds)
        recalls.append(r.recall_at_k(ranked, relevant, k))
        mrrs.append(r.mrr_at_k(ranked, relevant, k))
        ndcgs.append(r.ndcg_at_k(ranked, relevant, k))
        if row.get("joint") and len(golds) > 1:
            covered = {
                (gold["doc"], gold.get("clause"))
                for gold in golds
                if any(
                    chunk["chunk_id"] in ranked
                    and r.chunk_is_relevant(chunk, gold)
                    for chunk in chunks
                )
            }
            joint.append(1.0 if len(covered) == len(golds) else 0.0)
    return {
        "recall@10": round(float(np.mean(recalls)), 3),
        "mrr@10": round(float(np.mean(mrrs)), 3),
        "ndcg@10": round(float(np.mean(ndcgs)), 3),
        "joint_recall@10": round(float(np.mean(joint)), 3) if joint else "",
        "n": len(rows),
    }


def main() -> int:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    rows = load_eval()
    docs, embed_seconds = doc_vectors(chunks)
    queries = query_vectors([row["question"] for row in rows])
    meta = json.loads((r.CACHE / "vectors-meta.json").read_text(encoding="utf-8"))

    out: list[dict] = []
    for dim in r.CONFIG["embedding"]["dims"]:
        doc_slice = r.matryoshka(docs, dim)
        query_slice = r.matryoshka(queries, dim)
        for dtype in ("f16", "q8"):
            if dtype == "q8":
                doc_used = r.quantize_rows(doc_slice)
                query_used = r.quantize_rows(query_slice)
                bytes_per_dim = 1
            else:
                doc_used, query_used = doc_slice, query_slice
                bytes_per_dim = 2
            start = time.perf_counter()
            metrics = evaluate(chunks, rows, doc_used, query_used)
            query_ms = (time.perf_counter() - start) * 1000 / max(len(rows), 1)
            out.append(
                {
                    "dim": dim,
                    "dtype": dtype,
                    "recall@10": metrics["recall@10"],
                    "mrr@10": metrics["mrr@10"],
                    "ndcg@10": metrics["ndcg@10"],
                    "joint_recall@10": metrics["joint_recall@10"],
                    "index_mb": round(len(chunks) * dim * bytes_per_dim / 1e6, 2),
                    "brute_query_ms": round(query_ms, 2),
                    "embed_chunks_s": meta["seconds"],
                    "n_chunks": len(chunks),
                    "n_questions": metrics["n"],
                }
            )
            r.log(f"dim={dim} {dtype}: recall@10={metrics['recall@10']:.3f}")

    r.write_csv(r.BENCH / "embedding.csv", out)
    r.log(f"写入 {r.BENCH / 'embedding.csv'}")

    # 最近邻抽查：确认向量不是噪声
    sample = [0, len(chunks) // 3, len(chunks) - 1]
    lines = []
    for index in sample:
        sims = docs @ docs[index]
        order = np.argsort(-sims)[1:4]
        lines.append(f"块 {chunks[index]['chunk_id']}：{chunks[index]['text'][:50]}")
        for rank_index in order:
            lines.append(f"  {sims[rank_index]:.3f}  {chunks[rank_index]['chunk_id']}  {chunks[rank_index]['text'][:50]}")
    (r.BENCH / "embedding-neighbors.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
