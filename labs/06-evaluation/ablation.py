#!/usr/bin/env python3
"""6.4 消融：每次只改一个变量，看每一步的收益和区间。

基线是 BM25；然后依次加稠密检索（RRF 融合）、加重排，得到一条「逐步加组件」
的链。再在 RRF 上单独换两个变量：切分策略（structure → fixed）与 embedding
维度（1024 → 512）。所有指标在 v2 test 的同一批题上算，配对 bootstrap。

用法（在 labs/ 下）：
    uv run 06-evaluation/ablation.py

产物：
    benchmarks/evaluation/ablation-retrieval.csv   各变体的 Recall@5 / MRR / nDCG / joint
    benchmarks/evaluation/ablation-paired.csv      相对 RRF 的配对差值与区间
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))
sys.path.insert(0, str(ROOT / "labs/06-evaluation"))
sys.path.insert(0, str(ROOT / "eval"))

import raglib as r  # noqa: E402
import retrieval_metrics as rm  # noqa: E402
from retrieve import BM25, tokenize  # noqa: E402

OUT = ROOT / "benchmarks/evaluation"
K_MAX = 20
BASELINE = "rrf"


def ranking_from_scores(scores: np.ndarray, ids: list[str], k: int = K_MAX) -> list[str]:
    return [ids[position] for position in np.argsort(-scores)[:k]]


def fuse(rankings: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: -pair[1])][:K_MAX]


def main() -> int:
    structure = r.cached_chunks("structure", include_synthetic=False)
    fixed = r.cached_chunks("fixed", include_synthetic=False)
    structure_ids = [chunk["chunk_id"] for chunk in structure]
    fixed_ids = [chunk["chunk_id"] for chunk in fixed]
    structure_by_id = {chunk["chunk_id"]: chunk for chunk in structure}
    fixed_by_id = {chunk["chunk_id"]: chunk for chunk in fixed}

    vectors = np.load(r.CACHE / "vectors-1024-real.npy").astype(np.float32)
    if len(vectors) != len(structure):
        raise SystemExit(f"结构块与向量数量不一致：{len(structure)}/{len(vectors)}")

    # 5.4 的 fixed 向量缓存与当前 chunk_fixed 的顺序不一定一致（重跑过切分就会错位），
    # 消融实验里宁可重算一次，也不拿错位的向量算出 0.06 的假召回。
    fixed_cache = ROOT / "labs/06-evaluation/.cache/fixed-vectors.npy"
    cached_ok = False
    if fixed_cache.exists() and np.load(fixed_cache).shape[0] == len(fixed):
        sample = [0, len(fixed) // 2, len(fixed) - 1]
        probe = r.embed_texts([fixed[index]["text"] for index in sample], kind="doc")
        cached = np.load(fixed_cache)[sample]
        cached_ok = bool(np.allclose((cached * probe).sum(axis=1), 1.0, atol=1e-3))
    if cached_ok:
        fixed_vectors = np.load(fixed_cache).astype(np.float32)
    else:
        print(f"重新向量化 {len(fixed)} 个 fixed 块（缓存缺失或与当前块不对齐）……", file=sys.stderr)
        fixed_vectors = r.embed_texts([chunk["text"] for chunk in fixed], kind="doc")
        fixed_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(fixed_cache, fixed_vectors)

    rows = [row for row in rm.read_jsonl(ROOT / "eval/qa-v2.jsonl") if row.get("split") == "test" and row["category"] != "refuse"]
    texts = rm.load_clause_texts()
    kept, excluded = [], []
    for row in rows:
        specs = rm.relevance_specs(row, texts)
        missing = [
            spec
            for spec in specs
            if not rm.source_present([chunk for chunk in structure if chunk["doc_id"] == spec["doc"]], spec)
        ]
        if missing:
            excluded.append(row["id"])
        else:
            kept.append(row)
    print(f"test 可答 {len(rows)}，金标在索引 {len(kept)}，排除 {excluded}", file=sys.stderr)

    queries = r.embed_texts([row["question"] for row in kept], kind="query")
    structure_bm25 = BM25([tokenize(chunk["text"]) for chunk in structure], r.CONFIG["retrieval"]["bm25_k1"], r.CONFIG["retrieval"]["bm25_b"])
    fixed_bm25 = BM25([tokenize(chunk["text"]) for chunk in fixed], r.CONFIG["retrieval"]["bm25_k1"], r.CONFIG["retrieval"]["bm25_b"])
    dense_512 = r.matryoshka(vectors, 512)
    query_512 = r.matryoshka(queries, 512)

    rankings: dict[str, list[list[str]]] = {name: [] for name in ("bm25", "dense", "rrf", "rrf_rerank", "rrf_fixed", "rrf_512")}
    relevance: dict[str, list[set[str]]] = {"structure": [], "fixed": []}
    specs_all: list[list[dict]] = []

    for index, row in enumerate(kept):
        specs = rm.relevance_specs(row, texts)
        specs_all.append(specs)
        relevance["structure"].append(rm.relevant_chunks(structure, specs, "answer", 0.5))
        relevance["fixed"].append(rm.relevant_chunks(fixed, specs, "answer", 0.5))

        bm25_rank = ranking_from_scores(structure_bm25.scores(tokenize(row["question"])), structure_ids)
        dense_rank = ranking_from_scores(vectors @ queries[index], structure_ids)
        rrf_rank = fuse([bm25_rank, dense_rank], r.CONFIG["retrieval"]["rrf_k"])

        candidates = [structure_by_id[chunk_id] for chunk_id in rrf_rank]
        reranked = r.rerank(row["question"], [chunk["text"] for chunk in candidates], top_n=5)
        ordered = sorted(reranked, key=lambda item: -item["relevance_score"])
        rerank_ids = [candidates[item["index"]]["chunk_id"] for item in ordered][:5]
        rerank_ids += [chunk_id for chunk_id in rrf_rank if chunk_id not in rerank_ids]

        fixed_dense = ranking_from_scores(fixed_vectors @ queries[index], fixed_ids)
        fixed_sparse = ranking_from_scores(fixed_bm25.scores(tokenize(row["question"])), fixed_ids)
        fixed_rank = fuse([fixed_sparse, fixed_dense], r.CONFIG["retrieval"]["rrf_k"])

        dense512_rank = ranking_from_scores(dense_512 @ query_512[index], structure_ids)
        rrf512_rank = fuse([bm25_rank, dense512_rank], r.CONFIG["retrieval"]["rrf_k"])

        rankings["bm25"].append(bm25_rank)
        rankings["dense"].append(dense_rank)
        rankings["rrf"].append(rrf_rank)
        rankings["rrf_rerank"].append(rerank_ids)
        rankings["rrf_fixed"].append(fixed_rank)
        rankings["rrf_512"].append(rrf512_rank)
        r.log(f"{index + 1}/{len(kept)} {row['id']}")

    metric_rows = []
    per_question: dict[str, dict[str, list[float]]] = {}
    per_joint: dict[str, list[float]] = {}
    for name, ranked_all in rankings.items():
        chunk_set = fixed if name == "rrf_fixed" else structure
        relevant_set = relevance["fixed" if name == "rrf_fixed" else "structure"]
        values = {key: [] for key in ("recall@5", "mrr@10", "ndcg@10")}
        joint = []
        for index, ranked in enumerate(ranked_all):
            relevant = relevant_set[index]
            values["recall@5"].append(rm.recall_at(ranked, relevant, 5))
            values["mrr@10"].append(rm.mrr_at(ranked, relevant, 10))
            values["ndcg@10"].append(rm.ndcg_at(ranked, relevant, 10))
            if kept[index]["category"] == "cross":
                joint.append(rm.joint_at(ranked, chunk_set, specs_all[index], 5, 0.5))
        per_question[name] = values
        per_joint[name] = joint
        metric_rows.append(
            {
                "variant": name,
                "n": len(kept),
                "recall@5": round(sum(values["recall@5"]) / len(kept), 3),
                "mrr@10": round(sum(values["mrr@10"]) / len(kept), 3),
                "ndcg@10": round(sum(values["ndcg@10"]) / len(kept), 3),
                "joint_recall@5": round(sum(joint) / len(joint), 3) if joint else "",
                "joint_n": len(joint),
            }
        )
    rm.write_csv(OUT / "ablation-retrieval.csv", metric_rows)

    paired = []
    for name in rankings:
        if name == BASELINE:
            continue
        for metric in ("recall@5", "mrr@10", "ndcg@10", "joint_recall@5"):
            if metric == "joint_recall@5" and (not per_joint[name] or not per_joint[BASELINE]):
                continue
            if metric == "joint_recall@5":
                left, right = per_joint[name], per_joint[BASELINE]
                deltas = [a - b for a, b in zip(left, right)]
                low, high = rm.bootstrap_ci(deltas)
                paired.append(
                    {
                        "variant": name,
                        "baseline": BASELINE,
                        "metric": metric,
                        "value": round(sum(left) / len(left), 3),
                        "baseline_value": round(sum(right) / len(right), 3),
                        "delta": round(sum(deltas) / len(deltas), 3),
                        "ci_low": round(low, 3),
                        "ci_high": round(high, 3),
                        "mcnemar_p": round(
                            rm.mcnemar(
                                [1 if a and not b else 0 for a, b in zip(left, right)],
                                [1 if b and not a else 0 for a, b in zip(left, right)],
                            ),
                            4,
                        ),
                    }
                )
                continue
            left = per_question[name][metric]
            right = per_question[BASELINE][metric]
            deltas = [a - b for a, b in zip(left, right)]
            low, high = rm.bootstrap_ci(deltas)
            row_out = {
                "variant": name,
                "baseline": BASELINE,
                "metric": metric,
                "value": round(sum(left) / len(left), 3),
                "baseline_value": round(sum(right) / len(right), 3),
                "delta": round(sum(deltas) / len(deltas), 3),
                "ci_low": round(low, 3),
                "ci_high": round(high, 3),
            }
            if metric == "recall@5":
                row_out["mcnemar_p"] = round(
                    rm.mcnemar(
                        [1 if a and not b else 0 for a, b in zip(left, right)],
                        [1 if b and not a else 0 for a, b in zip(left, right)],
                    ),
                    4,
                )
            paired.append(row_out)
    rm.write_csv(OUT / "ablation-paired.csv", paired)

    # 固定切分没有条款号，相关性随块重算；阈值 0.5 与 5.3 的 0.35 不同，报一次敏感性
    threshold_rows = []
    for threshold in (0.35, 0.5):
        fixed_relevance = [rm.relevant_chunks(fixed, specs, "answer", threshold) for specs in specs_all]
        recalls = [rm.recall_at(ranked, relevant, 5) for ranked, relevant in zip(rankings["rrf_fixed"], fixed_relevance)]
        mrrs = [rm.mrr_at(ranked, relevant, 10) for ranked, relevant in zip(rankings["rrf_fixed"], fixed_relevance)]
        threshold_rows.append(
            {
                "threshold": threshold,
                "recall@5": round(sum(recalls) / len(recalls), 3),
                "mrr@10": round(sum(mrrs) / len(mrrs), 3),
            }
        )
    rm.write_csv(OUT / "ablation-fixed-threshold.csv", threshold_rows)

    for row in metric_rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
