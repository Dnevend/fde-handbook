#!/usr/bin/env python3
"""6.2 检索指标：在 v2 测试集上重算 Recall@k / MRR / nDCG，并给配对置信区间。

三件事：

  1. 同一批 734 个块、同一个 embedding，只换检索策略（BM25 / 稠密 / RRF），
     看三个指标各自怎么变；
  2. 用两种「相关」定义各算一遍：按条款号匹配（第五部分的口径）与按答案原文匹配
     （v2 的口径）。差异就是 6.1 说的「条号不唯一」带来的偏差；
  3. 对策略差异做配对 bootstrap 与 McNemar，避免把噪声当成提升。

用法（在 labs/ 下）：
    uv run 06-evaluation/retrieval_metrics.py

产物：
    benchmarks/evaluation/retrieval-metrics.csv    策略 × 相关性口径 × 题型
    benchmarks/evaluation/retrieval-k-curve.csv    Recall@k 曲线（k=1..20）
    benchmarks/evaluation/retrieval-paired.csv    策略两两比较的 delta、CI、p 值
    benchmarks/evaluation/retrieval-relevance.csv  两种口径下每题的相关块数
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))
sys.path.insert(0, str(ROOT / "eval"))

import raglib as r  # noqa: E402
from clause_text import load_clause_texts  # noqa: E402

EVAL = ROOT / "eval/qa-v2.jsonl"
CACHE = ROOT / "labs/06-evaluation/.cache"
OUT = ROOT / "benchmarks/evaluation"
K_MAX = 20
BOOTSTRAP = 2000
SEED = 42
STRATEGIES = ["bm25", "dense", "rrf"]
LABELS = {"bm25": "BM25", "dense": "稠密", "rrf": "RRF"}
STOPWORDS = set("的了和及与或在为对按应当由及其之而也就都把被让给从到于".split())
PUNCT = re.compile(r"[\s，。；：、！？（）()「」“”\"'·,.;:!?\[\]【】]")


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def ngrams(text: str, n: int = 4) -> set[str]:
    clean = normalize(text)
    return {clean[i : i + n] for i in range(max(len(clean) - n + 1, 0))}


def segments(answer: str) -> list[str]:
    return [part for part in re.split(r"[；;。]", answer) if normalize(part)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in jieba.cut_for_search(text.lower()):
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isalnum():
            continue
        tokens.append(token)
    return tokens


class BM25:
    """与 5.6 相同的实现；参数来自 config.yaml。"""

    def __init__(self, corpus: list[list[str]], k1: float, b: float) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.lengths = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avg = float(self.lengths.mean()) if len(corpus) else 0.0
        self.doc_freq: Counter[str] = Counter()
        for doc in corpus:
            self.doc_freq.update(set(doc))
        self.n = len(corpus)

    def scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in query:
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for index, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * len(doc) / (self.avg or 1))
                scores[index] += idf * tf * (self.k1 + 1) / norm
        return scores


def coverage(segment: str, text: str) -> float:
    """答案片段有多少 4-gram 落在块里；片段很短时退回子串判断。"""
    clean = normalize(segment)
    if len(clean) < 8:
        return 1.0 if clean in normalize(text) else 0.0
    gold = ngrams(segment)
    return len(gold & ngrams(text)) / max(len(gold), 1)


def relevance_specs(row: dict, texts: dict[tuple[str, str], str]) -> list[dict]:
    specs = []
    for source in row["sources"]:
        body = texts.get((source["doc"], source["clause"]), "")
        picked = [part for part in segments(row["answer"]) if normalize(part) in normalize(body)]
        if not picked:
            picked = [row["answer"]]
        specs.append(
            {
                "doc": source["doc"],
                "clause": source.get("clause"),
                "segments": picked,
            }
        )
    return specs


def relevant_chunks(chunks: list[dict], specs: list[dict], mode: str, threshold: float = 0.5) -> set[str]:
    ids: set[str] = set()
    for chunk in chunks:
        for spec in specs:
            if chunk["doc_id"] != spec["doc"]:
                continue
            if mode == "clause":
                if chunk.get("clause_no") == spec["clause"]:
                    ids.add(chunk["chunk_id"])
                continue
            if any(coverage(segment, chunk["text"]) >= threshold for segment in spec["segments"]):
                ids.add(chunk["chunk_id"])
    return ids


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def joint_at(ranked: list[str], chunks: list[dict], specs: list[dict], k: int = 5, threshold: float = 0.5) -> float:
    if len(specs) < 2:
        return float("nan")
    top = set(ranked[:k])
    covered = 0
    for spec in specs:
        hit = any(
            chunk["chunk_id"] in top
            and chunk["doc_id"] == spec["doc"]
            and any(coverage(segment, chunk["text"]) >= threshold for segment in spec["segments"])
            for chunk in chunks
        )
        if hit:
            covered += 1
    return 1.0 if covered == len(specs) else 0.0


def quantile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def bootstrap_ci(values: list[float], rounds: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return quantile(means, 2.5), quantile(means, 97.5)


def mcnemar(better: list[float], worse: list[float]) -> float:
    """better = A 对 B 错，worse = B 对 A 错；双侧精确检验。"""
    b, c = int(sum(better)), int(sum(worse))
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


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



def source_present(doc_chunks: list[dict], spec: dict, threshold: float = 0.5) -> bool:
    """金标段落是否真在索引里：只认块文本，不认条号标签。

    条号可能被切分器带到相邻块上（甚至漏块），按标签判断会把「块里没有答案」
    误当成「索引里有」。这里用与相关性相同的 0.5 覆盖率，保证留下的题至少
    有一个块能覆盖答案的一半。
    """
    return any(
        any(coverage(segment, chunk["text"]) >= threshold for segment in spec["segments"])
        for chunk in doc_chunks
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 检索指标")
    parser.add_argument("--threshold", type=float, default=0.5, help="答案 4-gram 覆盖率阈值")
    parser.add_argument("--split", default="test", choices=["test", "dev", "all"], help="用哪个分片算指标")
    args = parser.parse_args()

    chunks = r.cached_chunks("structure", include_synthetic=False)
    ids = [chunk["chunk_id"] for chunk in chunks]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    vectors = np.load(r.CACHE / "vectors-1024-real.npy").astype(np.float32)
    texts = load_clause_texts()

    all_rows = [row for row in read_jsonl(EVAL) if row["category"] != "refuse"]
    if args.split != "all":
        all_rows = [row for row in all_rows if row.get("split") == args.split]
    CACHE.mkdir(parents=True, exist_ok=True)
    doc_chunks: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk["doc_id"]].append(chunk)

    kept, excluded = [], []
    for row in all_rows:
        specs = relevance_specs(row, texts)
        missing = [
            f"{spec['doc']} {spec['clause']}"
            for spec in specs
            if not source_present(doc_chunks.get(spec["doc"], []), spec)
        ]
        if missing:
            excluded.append({"id": row["id"], "category": row["category"], "missing": "; ".join(missing)})
        else:
            kept.append(row)

    coverage_rows = [
        {"metric": "questions_total", "value": len(all_rows)},
        {"metric": "indexed", "value": len(kept)},
        {"metric": "excluded", "value": len(excluded)},
        {"metric": "excluded_ids", "value": " ".join(item["id"] for item in excluded)},
    ]
    write_csv(OUT / "retrieval-coverage.csv", coverage_rows)
    write_csv(OUT / "retrieval-excluded.csv", excluded)
    print(f"索引覆盖 {len(kept)}/{len(all_rows)}，排除 {len(excluded)} 题", file=sys.stderr)

    rows = kept
    cache = CACHE / "query-vectors-v2.npy"
    if cache.exists() and np.load(cache).shape[0] == len(all_rows):
        queries = np.load(cache).astype(np.float32)
        keep_ids = {row["id"] for row in kept}
        queries = np.array([q for q, row in zip(queries, all_rows) if row["id"] in keep_ids], dtype=np.float32)
    else:
        queries = r.embed_texts([row["question"] for row in rows], kind="query")
        np.save(cache, queries)

    bm25 = BM25([tokenize(chunk["text"]) for chunk in chunks], r.CONFIG["retrieval"]["bm25_k1"], r.CONFIG["retrieval"]["bm25_b"])
    print(f"块 {len(chunks)}，题 {len(rows)}，BM25 建索引完成", file=sys.stderr)

    rankings: dict[str, list[list[str]]] = {strategy: [] for strategy in STRATEGIES}
    relevance: dict[str, list[set[str]]] = {"answer": [], "clause": []}
    specs_all: list[list[dict]] = []

    for index, row in enumerate(rows):
        specs = relevance_specs(row, texts)
        specs_all.append(specs)
        for mode in relevance:
            relevance[mode].append(relevant_chunks(chunks, specs, mode, args.threshold))

        dense_scores = vectors @ queries[index]
        dense_rank = [ids[position] for position in np.argsort(-dense_scores)[:K_MAX]]
        bm25_rank = [ids[position] for position in np.argsort(-bm25.scores(tokenize(row["question"])))[:K_MAX]]
        fused: dict[str, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for position, item in enumerate(ranking, start=1):
                fused[item] = fused.get(item, 0.0) + 1.0 / (r.CONFIG["retrieval"]["rrf_k"] + position)
        rrf_rank = [item for item, _ in sorted(fused.items(), key=lambda pair: -pair[1])][:K_MAX]

        rankings["bm25"].append(bm25_rank)
        rankings["dense"].append(dense_rank)
        rankings["rrf"].append(rrf_rank)

    # ---- 两种相关性口径的差异
    relevance_rows = []
    for mode in relevance:
        counts = [len(relevant) for relevant in relevance[mode]]
        relevance_rows.append(
            {
                "mode": mode,
                "questions": len(counts),
                "mean_relevant_chunks": round(sum(counts) / len(counts), 2),
                "zero_relevant": sum(1 for count in counts if count == 0),
                "p95_relevant_chunks": int(np.percentile(counts, 95)),
            }
        )
    write_csv(OUT / "retrieval-relevance.csv", relevance_rows)

    # ---- 指标表
    metric_rows = []
    per_question: dict[tuple[str, str, str], list[float]] = {}
    categories = ["all", "direct", "condition", "cross"]
    for strategy in STRATEGIES:
        for mode in relevance:
            for category in categories:
                subset = [index for index, row in enumerate(rows) if category == "all" or row["category"] == category]
                values = {k: [] for k in (1, 3, 5, 10, 20)}
                mrrs, ndcgs, joints = [], [], []
                for index in subset:
                    ranked = rankings[strategy][index]
                    relevant = relevance[mode][index]
                    for k in values:
                        values[k].append(recall_at(ranked, relevant, k))
                    mrrs.append(mrr_at(ranked, relevant, 10))
                    ndcgs.append(ndcg_at(ranked, relevant, 10))
                    if rows[index]["category"] == "cross":
                        joints.append(joint_at(ranked, chunks, specs_all[index], 5, args.threshold))
                row_out = {
                    "strategy": strategy,
                    "relevance": mode,
                    "category": category,
                    "n": len(subset),
                }
                for k in values:
                    row_out[f"recall@{k}"] = round(sum(values[k]) / len(values[k]), 3)
                    if category == "all":
                        per_question[(strategy, mode, f"recall@{k}")] = values[k]
                row_out["mrr@10"] = round(sum(mrrs) / len(mrrs), 3)
                row_out["ndcg@10"] = round(sum(ndcgs) / len(ndcgs), 3)
                row_out["joint_recall@5"] = round(sum(joints) / len(joints), 3) if joints else ""
                if category == "all":
                    per_question[(strategy, mode, "mrr@10")] = mrrs
                    per_question[(strategy, mode, "ndcg@10")] = ndcgs
                metric_rows.append(row_out)
    write_csv(OUT / "retrieval-metrics.csv", metric_rows)

    # ---- Recall@k 曲线（全体，answer 口径）
    curve_rows = []
    for strategy in STRATEGIES:
        for k in range(1, K_MAX + 1):
            values = [recall_at(rankings[strategy][index], relevance["answer"][index], k) for index in range(len(rows))]
            low, high = bootstrap_ci(values)
            curve_rows.append(
                {
                    "strategy": strategy,
                    "k": k,
                    "recall": round(sum(values) / len(values), 3),
                    "ci_low": round(low, 3),
                    "ci_high": round(high, 3),
                }
            )
    write_csv(OUT / "retrieval-k-curve.csv", curve_rows)

    # ---- 配对比较（answer 口径）
    paired_rows = []
    comparisons = [("dense", "bm25"), ("rrf", "bm25"), ("rrf", "dense")]
    for left, right in comparisons:
        for metric in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
            a = per_question[(left, "answer", metric)]
            b = per_question[(right, "answer", metric)]
            deltas = [x - y for x, y in zip(a, b)]
            low, high = bootstrap_ci(deltas)
            row_out = {
                "left": left,
                "right": right,
                "metric": metric,
                "left_value": round(sum(a) / len(a), 3),
                "right_value": round(sum(b) / len(b), 3),
                "delta": round(sum(deltas) / len(deltas), 3),
                "ci_low": round(low, 3),
                "ci_high": round(high, 3),
            }
            if metric.startswith("recall"):
                row_out["mcnemar_p"] = round(mcnemar([1 if x and not y else 0 for x, y in zip(a, b)],
                                                     [1 if y and not x else 0 for x, y in zip(a, b)]), 4)
                row_out["discordant"] = int(sum(1 for x, y in zip(a, b) if x != y))
            paired_rows.append(row_out)
    write_csv(OUT / "retrieval-paired.csv", paired_rows)

    # ---- 相关性阈值敏感性：同一批检索结果，只换「算不算相关」的阈值
    threshold_rows = []
    for threshold in (0.35, 0.5, 0.65):
        for strategy in STRATEGIES:
            relevances = [
                relevant_chunks(chunks, specs_all[index], "answer", threshold)
                for index in range(len(rows))
            ]
            recalls = [recall_at(rankings[strategy][index], relevances[index], 5) for index in range(len(rows))]
            mrrs = [mrr_at(rankings[strategy][index], relevances[index], 10) for index in range(len(rows))]
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "strategy": strategy,
                    "recall@5": round(sum(recalls) / len(recalls), 3),
                    "mrr@10": round(sum(mrrs) / len(mrrs), 3),
                }
            )
    write_csv(OUT / "retrieval-threshold.csv", threshold_rows)

    overall = next(row for row in metric_rows if row["strategy"] == "dense" and row["relevance"] == "answer" and row["category"] == "all")
    print(
        f"稠密 / answer：Recall@5={overall['recall@5']} MRR@10={overall['mrr@10']} nDCG@10={overall['ndcg@10']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
