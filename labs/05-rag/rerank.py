"""5.7 重排：双塔召回 20 条，交叉编码器重排后取 5 条。

对照两阶段：hybrid top-20 → 直接取前 5（不重排） vs 重排后取前 5。
模型是 bge-reranker-v2-m3（llama.cpp 的 /v1/rerank 端点）。

产物：benchmarks/rag/rerank.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402


def metrics(rows: list[dict], chunks: list[dict], ranking_of, k: int = 5) -> tuple[float, float, float]:
    recalls, mrrs, ndcgs = [], [], []
    for row in rows:
        ranked = ranking_of(row)[:k]
        relevant = r.relevant_ids(chunks, row["sources"])
        recalls.append(r.recall_at_k(ranked, relevant, k))
        mrrs.append(r.mrr_at_k(ranked, relevant, 10))
        ndcgs.append(r.ndcg_at_k(ranked, relevant, 10))
    return float(np.mean(recalls)), float(np.mean(mrrs)), float(np.mean(ndcgs))


def main() -> int:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    rows = [json.loads(line) for line in (r.ROOT / "eval/qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["category"] != "refuse"]
    top20 = json.loads((r.CACHE / "hybrid-top20.json").read_text(encoding="utf-8"))

    reranked: dict[str, list[str]] = {}
    latencies: list[float] = []
    for row in rows:
        candidates = top20[row["id"]]
        start = time.perf_counter()
        results = r.rerank(row["question"], [by_id[chunk_id]["text"] for chunk_id in candidates], top_n=len(candidates))
        latencies.append((time.perf_counter() - start) * 1000)
        order = sorted(results, key=lambda item: -item["relevance_score"])
        reranked[row["id"]] = [candidates[item["index"]] for item in order]
        r.log(f"{row['id']} 重排完成")

    before = metrics(rows, chunks, lambda row: top20[row["id"]])
    after = metrics(rows, chunks, lambda row: reranked[row["id"]])

    def changes() -> int:
        count = 0
        for row in rows:
            if top20[row["id"]][0] != reranked[row["id"]][0]:
                count += 1
        return count

    out = [
        {
            "stage": "before",
            "n": len(rows),
            "recall@5": round(before[0], 3),
            "mrr@10": round(before[1], 3),
            "ndcg@10": round(before[2], 3),
            "p50_ms": "",
            "p95_ms": "",
            "top1_changed": "",
        },
        {
            "stage": "after",
            "n": len(rows),
            "recall@5": round(after[0], 3),
            "mrr@10": round(after[1], 3),
            "ndcg@10": round(after[2], 3),
            "p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "top1_changed": changes(),
        },
    ]
    r.write_csv(r.BENCH / "rerank.csv", out)
    (r.CACHE / "reranked.json").write_text(json.dumps(reranked, ensure_ascii=False), encoding="utf-8")
    r.log(f"before {before} → after {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
