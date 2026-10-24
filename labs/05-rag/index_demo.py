"""5.5 向量索引：暴力检索 vs HNSW 的参数扫描。

两组数据：
  real   —— 本书语料（约 800 块）的真实向量，看小规模下索引是否值得；
  random —— 2 万条 1024 维随机单位向量，只看索引机制（延迟、召回随 efSearch 的变化），
            不代表任何真实语料的检索质量。

产物：benchmarks/rag/ann.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402


def brute_force(vectors: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    scores = queries @ vectors.T
    return [list(np.argsort(-row)[:k]) for row in scores]


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def time_calls(fn, repeats: int = 3) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def run_dataset(name: str, vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> list[dict]:
    n, dim = vectors.shape
    truth = brute_force(vectors, queries, k)
    rows: list[dict] = []

    start = time.perf_counter()
    brute_ids = brute_force(vectors, queries, k)
    brute_total = (time.perf_counter() - start) * 1000
    rows.append(
        {
            "dataset": name,
            "n": n,
            "dim": dim,
            "method": "brute",
            "M": "",
            "ef_construction": "",
            "ef_search": "",
            "recall@10": 1.0,
            "p50_ms": round(brute_total / len(queries), 3),
            "p95_ms": round(brute_total / len(queries), 3),
            "build_s": 0.0,
            "index_mb": round(n * dim * 4 / 1e6, 1),
        }
    )

    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=n,
        M=r.CONFIG["hnsw"]["M"],
        ef_construction=r.CONFIG["hnsw"]["ef_construction"],
        random_seed=r.CONFIG["hnsw"]["seed"],
    )
    index.add_items(vectors, np.arange(n))
    build_s = time.perf_counter() - start

    for ef in (10, 20, 40, 80, 160):
        index.set_ef(ef)
        labels, _ = index.knn_query(queries, k=k)
        recall = float(np.mean([len(set(map(int, got)) & set(gold)) / k for got, gold in zip(labels, truth)]))
        samples: list[float] = []
        for query in queries:
            start = time.perf_counter()
            index.knn_query(query, k=k)
            samples.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "dataset": name,
                "n": n,
                "dim": dim,
                "method": "hnsw",
                "M": r.CONFIG["hnsw"]["M"],
                "ef_construction": r.CONFIG["hnsw"]["ef_construction"],
                "ef_search": ef,
                "recall@10": round(recall, 3),
                "p50_ms": round(percentile(samples, 50), 3),
                "p95_ms": round(percentile(samples, 95), 3),
                "build_s": round(build_s, 2),
                "index_mb": round((n * dim * 4 + n * r.CONFIG["hnsw"]["M"] * 2 * 4) / 1e6, 1),
            }
        )
        r.log(f"{name} ef={ef}: recall@10={recall:.3f}")
    del brute_ids
    return rows


def main() -> int:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    cache = r.CACHE / "vectors-1024-real.npy"
    if not cache.exists():
        raise SystemExit("先跑 embed_index.py，生成块向量")
    vectors = np.load(cache)
    rows_eval = [json.loads(line) for line in (r.ROOT / "eval/qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_eval = [row for row in rows_eval if row["category"] != "refuse"]
    queries = r.embed_texts([row["question"] for row in rows_eval], kind="query")
    out = run_dataset("real", vectors.astype(np.float32), queries.astype(np.float32))

    rng = np.random.default_rng(r.CONFIG["hnsw"]["seed"])
    random_vectors = rng.normal(size=(20000, vectors.shape[1])).astype(np.float32)
    random_vectors = r.l2_normalize(random_vectors)
    query_count = 50
    random_queries = r.l2_normalize(rng.normal(size=(query_count, vectors.shape[1])).astype(np.float32))
    out += run_dataset("random-20k", random_vectors, random_queries)

    r.write_csv(r.BENCH / "ann.csv", out)
    r.log(f"写入 {r.BENCH / 'ann.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
