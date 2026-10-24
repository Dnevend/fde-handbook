"""5.6 稀疏检索与混合检索：BM25 / 纯向量 / RRF。

BM25 的中文分词用 jieba，丢掉单字虚词；稠密检索用 5.4 缓存好的 Qwen3-Embedding 向量。
RRF 只融合排名，不比较分数——这是它能跨检索器使用的原因（Cormack et al., 2009）。

产物：
  benchmarks/rag/retrieval.csv    各策略 × 题型的指标
  .cache/hybrid-top20.json        混合检索的候选，供 5.7/5.8 复用
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import jieba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402

STOPWORDS = set("的了和及与或在为对按应当由及其之而也就都把被让给从到于".split())

# 标识符类查询单独统计：型号、编号、数字是稀疏检索的主场，不能混进语义题里平均掉。
ID_QUERIES = [
    ("北京市人民政府令第 285 号规定的是什么？", "gov-ledger-guide"),
    ("应急管理部令第 13 号是哪部规定？", "reg-mem-order-13"),
    ("《安全生产法》第二十四条要求哪些单位设置安全生产管理机构？", "reg-safety-law"),
    ("环评报告里 130T 注塑机有几台？", "eia-injection-01"),
    ("DB44 标准管的是哪类污染物排放？", "eia-injection-01"),
    ("事故报告里 7 号吹瓶机是什么设备？", "acc-fangsu-0518"),
    ("《工贸企业有限空间作业安全规定》从什么时候施行？", "reg-mem-order-13"),
    ("《安全生产法》第一百一十四条规定一般事故罚款多少？", "reg-safety-law"),
]


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


def rank_from_scores(scores: np.ndarray, ids: list[str], k: int) -> list[str]:
    order = np.argsort(-scores)[:k]
    return [ids[i] for i in order]


def rrf(rankings: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: -pair[1])]


def load_eval() -> list[dict]:
    rows = [json.loads(line) for line in (r.ROOT / "eval/qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["category"] != "refuse"]
    for index, (question, doc) in enumerate(ID_QUERIES, start=1):
        rows.append(
            {
                "id": f"I{index:02d}",
                "category": "identifier",
                "question": question,
                "answer": "",
                "sources": [{"doc": doc}],
                "synthetic": False,
                "joint": False,
            }
        )
    return rows


def main() -> int:
    chunks = r.cached_chunks("structure", include_synthetic=False)
    ids = [chunk["chunk_id"] for chunk in chunks]
    rows = load_eval()

    vectors = np.load(r.CACHE / "vectors-1024-real.npy").astype(np.float32)
    query_cache = r.CACHE / "query-vectors.npy"
    if query_cache.exists() and np.load(query_cache).shape[0] == len(rows):
        queries = np.load(query_cache).astype(np.float32)
    else:
        queries = r.embed_texts([row["question"] for row in rows], kind="query")
        np.save(query_cache, queries)

    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25(corpus_tokens, r.CONFIG["retrieval"]["bm25_k1"], r.CONFIG["retrieval"]["bm25_b"])
    top_k = r.CONFIG["retrieval"]["top_k"]
    candidate_k = r.CONFIG["retrieval"]["candidate_k"]
    rrf_k = r.CONFIG["retrieval"]["rrf_k"]

    metrics = {
        strategy: {"direct": [], "condition": [], "cross": [], "identifier": [], "all": []}
        for strategy in ("bm25", "dense", "hybrid")
    }
    joint = {"bm25": [], "dense": [], "hybrid": []}
    top20: dict[str, list[str]] = {}

    for row, qvec in zip(rows, queries):
        query_tokens = tokenize(row["question"])
        sparse = rank_from_scores(bm25.scores(query_tokens), ids, candidate_k)
        dense_scores = vectors @ qvec
        dense = rank_from_scores(dense_scores, ids, candidate_k)
        hybrid = rrf([sparse, dense], rrf_k)[:candidate_k]
        if row["category"] != "identifier":
            top20[row["id"]] = hybrid

        relevant = r.relevant_ids(chunks, row["sources"])
        for strategy, ranking in (("bm25", sparse), ("dense", dense), ("hybrid", hybrid)):
            metrics[strategy][row["category"]].append(
                (
                    r.recall_at_k(ranking, relevant, top_k),
                    r.mrr_at_k(ranking, relevant, 10),
                    r.ndcg_at_k(ranking, relevant, 10),
                )
            )
            if row["category"] != "identifier":
                metrics[strategy]["all"].append(metrics[strategy][row["category"]][-1])

        if row.get("joint") and len(row["sources"]) > 1:
            for strategy, ranking in (("bm25", sparse), ("dense", dense), ("hybrid", hybrid)):
                covered = {
                    (gold["doc"], gold.get("clause"))
                    for gold in row["sources"]
                    if any(chunk["chunk_id"] in ranking[:top_k] and r.chunk_is_relevant(chunk, gold) for chunk in chunks)
                }
                joint[strategy].append(1.0 if len(covered) == len(row["sources"]) else 0.0)

    out = []
    for strategy, per_category in metrics.items():
        for category, values in per_category.items():
            if not values:
                continue
            array = np.array(values)
            out.append(
                {
                    "strategy": strategy,
                    "category": category,
                    "n": len(values),
                    "recall@5": round(float(array[:, 0].mean()), 3),
                    "mrr@10": round(float(array[:, 1].mean()), 3),
                    "ndcg@10": round(float(array[:, 2].mean()), 3),
                    "joint_recall@5": round(float(np.mean(joint[strategy])), 3) if joint[strategy] else "",
                }
            )

    r.write_csv(r.BENCH / "retrieval.csv", out)
    (r.CACHE / "hybrid-top20.json").write_text(json.dumps(top20, ensure_ascii=False), encoding="utf-8")
    for row in out:
        if row["category"] == "all":
            r.log(f"{row['strategy']}: recall@5={row['recall@5']} ndcg@10={row['ndcg@10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
