"""5.9 元数据、权限过滤与增量更新。

三件事：
  1. 不过滤 / 后过滤 / 前过滤 三种权限策略下，越权结果数与允许文档的召回；
  2. 文档修改后，全量重建与增量更新的耗时差；
  3. 删除文档后，旧内容是否还会被检索到（陈旧结果）。

权限口径写在 config.yaml 的 access 段：受限文档对「内部」用户不可见，对「受限」用户可见。
事故报告含个人信息，按受限处理；这是演示用的简化模型，不是真实 ACL。

产物：benchmarks/rag/metadata.csv
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

QUERIES = [
    ("事故的直接原因是什么？", "acc-fangsu-0518"),
    ("这起机械伤害事故造成几人死亡？", "acc-fangsu-0518"),
    ("涉事吹瓶机的型号和厂家是什么？", "acc-fangsu-0518"),
    ("电池托盘上盖的保压压力是多少？", "PC-007"),
    ("M-2004 模具用哪种材料牌号？", "PC-007"),
    ("PC-007 的料筒温度区间是多少？", "PC-007"),
    ("ZJ-03 的液压油多久检测一次？", "MP-003"),
    ("注塑机开机前要检查什么？", "PC-001"),
    ("注塑机多久润滑一次？", "MP-001"),
    ("RO-003 的停机时间是多少？", "RO-003"),
    ("日常点检要检查哪些项目？", "DI-001"),
    ("PC-002 的模温是多少？", "PC-002"),
]


def ids_matching(chunks: list[dict], doc_id: str) -> set[str]:
    return {chunk["chunk_id"] for chunk in chunks if chunk["doc_id"] == doc_id}


def top_chunks(scores: np.ndarray, chunks: list[dict], k: int) -> list[dict]:
    order = np.argsort(-scores)[:k]
    return [chunks[index] for index in order]


def main() -> int:
    chunks = r.cached_chunks("structure")
    all_vectors = r.CACHE / "vectors-1024.npy"
    if not all_vectors.exists():
        np.save(all_vectors, r.embed_texts([chunk["text"] for chunk in chunks]))
    vectors = np.load(all_vectors).astype(np.float32)
    restricted_docs = set(r.CONFIG["access"]["restricted_docs"])
    restricted = {chunk["chunk_id"] for chunk in chunks if chunk["doc_id"] in restricted_docs}
    allowed_mask = np.array([chunk["chunk_id"] not in restricted for chunk in chunks])

    queries = r.embed_texts([question for question, _ in QUERIES], kind="query")
    rows: list[dict] = []

    counts = {"no_filter": [], "post_filter": [], "pre_filter": []}
    latencies = {"no_filter": [], "post_filter": [], "pre_filter": []}
    allowed_recall = {"no_filter": [], "post_filter": [], "pre_filter": []}
    restricted_recall = {"no_filter": [], "post_filter": [], "pre_filter": []}

    for (question, expected_doc), qvec in zip(QUERIES, queries):
        relevant = ids_matching(chunks, expected_doc)
        is_restricted = expected_doc in restricted_docs

        start = time.perf_counter()
        naive = top_chunks(vectors @ qvec, chunks, 5)
        latencies["no_filter"].append((time.perf_counter() - start) * 1000)
        counts["no_filter"].append(sum(chunk["chunk_id"] in restricted for chunk in naive))
        target = restricted_recall if is_restricted else allowed_recall
        target["no_filter"].append(r.recall_at_k([chunk["chunk_id"] for chunk in naive], relevant, 5))

        start = time.perf_counter()
        wide = top_chunks(vectors @ qvec, chunks, 20)
        post = [chunk for chunk in wide if chunk["chunk_id"] not in restricted][:5]
        latencies["post_filter"].append((time.perf_counter() - start) * 1000)
        counts["post_filter"].append(sum(chunk["chunk_id"] in restricted for chunk in post))
        if not is_restricted:
            allowed_recall["post_filter"].append(
                r.recall_at_k([chunk["chunk_id"] for chunk in post], relevant, 5)
            )

        start = time.perf_counter()
        filtered_vectors = vectors[allowed_mask]
        filtered_chunks = [chunk for chunk, keep in zip(chunks, allowed_mask) if keep]
        pre = top_chunks(filtered_vectors @ qvec, filtered_chunks, 5)
        latencies["pre_filter"].append((time.perf_counter() - start) * 1000)
        counts["pre_filter"].append(sum(chunk["chunk_id"] in restricted for chunk in pre))
        if not is_restricted:
            allowed_recall["pre_filter"].append(
                r.recall_at_k([chunk["chunk_id"] for chunk in pre], relevant, 5)
            )

    for strategy in counts:
        rows.append(
            {
                "experiment": "permission",
                "strategy": strategy,
                "metric": "leaked_results@5",
                "value": round(float(np.mean(counts[strategy])), 3),
            }
        )
        if allowed_recall[strategy]:
            rows.append(
                {
                    "experiment": "permission",
                    "strategy": strategy,
                    "metric": "recall_allowed@5",
                    "value": round(float(np.mean(allowed_recall[strategy])), 3),
                }
            )
        if restricted_recall[strategy]:
            rows.append(
                {
                    "experiment": "permission",
                    "strategy": strategy,
                    "metric": "recall_restricted@5",
                    "value": round(float(np.mean(restricted_recall[strategy])), 3),
                }
            )
        rows.append(
            {
                "experiment": "permission",
                "strategy": strategy,
                "metric": "latency_ms",
                "value": round(float(np.mean(latencies[strategy])), 3),
            }
        )

    # ---- 增量更新：改一条 MP-003 的液压油标准
    target = next(chunk for chunk in chunks if chunk["doc_id"] == "MP-003")
    old_text = target["text"]
    new_text = old_text.replace("2000 h", "1000 h").replace("NAS 9 级", "NAS 7 级")
    if new_text == old_text:
        raise SystemExit("MP-003 里没找到要替换的文本，检查语料是否变了")

    embed_times = []
    for _ in range(5):
        start = time.perf_counter()
        new_vector = r.embed_texts([new_text])[0]
        embed_times.append(time.perf_counter() - start)
    embed_one = float(np.median(embed_times))
    embed_min, embed_max = min(embed_times), max(embed_times)

    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(chunks) + 8, M=r.CONFIG["hnsw"]["M"], ef_construction=200, random_seed=42)
    index.add_items(vectors, np.arange(len(chunks)))
    index.set_ef(64)

    position = chunks.index(target)
    start = time.perf_counter()
    index.mark_deleted(position)
    index.add_items(new_vector.reshape(1, -1), np.array([len(chunks)]))
    update_index = time.perf_counter() - start

    query = r.embed_texts(["MP-003 的液压油更换标准"], kind="query")[0]
    updated_chunks = [dict(chunk) for chunk in chunks]
    updated_chunks[position]["text"] = new_text
    old_top = top_chunks(vectors @ query, chunks, 5)

    # 真正查更新后的 HNSW，而不是查暴力向量。
    labels, _ = index.knn_query(query, k=5)
    new_top = []
    for label in labels[0]:
        label = int(label)
        if label == len(chunks):
            new_top.append({"chunk_id": f"{target['chunk_id']}-new", "doc_id": target["doc_id"], "text": new_text})
        else:
            new_top.append(chunks[label])

    # 全量重建的参照值来自 2026-10-24 对 794 块（含合成件）的一次完整向量化。
    full_rebuild = 46.7
    rows += [
        {"experiment": "incremental", "strategy": "embed", "metric": "seconds_per_chunk_median", "value": round(embed_one, 3)},
        {"experiment": "incremental", "strategy": "embed", "metric": "seconds_per_chunk_max", "value": round(embed_max, 3)},
        {"experiment": "incremental", "strategy": "hnsw", "metric": "update_ms", "value": round(update_index * 1000, 1)},
        {"experiment": "incremental", "strategy": "full_rebuild", "metric": "embed_seconds", "value": round(full_rebuild, 1)},
        {
            "experiment": "incremental",
            "strategy": "update_hnsw",
            "metric": "new_text_in_top5",
            "value": sum("NAS 7 级" in chunk["text"] for chunk in new_top),
        },
        {
            "experiment": "incremental",
            "strategy": "update_hnsw",
            "metric": "stale_text_hits@5",
            "value": sum("NAS 9 级" in chunk["text"] and chunk["doc_id"] == "MP-003" for chunk in new_top),
        },
        {
            "experiment": "incremental",
            "strategy": "bruteforce_before",
            "metric": "old_text_hits@5",
            "value": sum("NAS 9 级" in chunk["text"] and chunk["doc_id"] == "MP-003" for chunk in old_top),
        },
    ]

    # ---- 删除：重建一个同样的索引，再标记删除并查询 #
    delete_index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    delete_index.init_index(max_elements=len(chunks), M=r.CONFIG["hnsw"]["M"], ef_construction=200, random_seed=42)
    delete_index.add_items(vectors, np.arange(len(chunks)))
    delete_index.set_ef(64)
    start = time.perf_counter()
    delete_index.mark_deleted(position)
    after_labels, _ = delete_index.knn_query(query, k=5)
    delete_ms = (time.perf_counter() - start) * 1000
    after_delete = [chunks[int(label)] for label in after_labels[0] if int(label) != len(chunks)]
    rows += [
        {"experiment": "delete", "strategy": "hnsw_delete", "metric": "delete_ms", "value": round(delete_ms, 2)},
        {
            "experiment": "delete",
            "strategy": "hnsw_delete",
            "metric": "stale_hits@5",
            "value": sum(chunk["doc_id"] == "MP-003" for chunk in after_delete),
        },
    ]

    r.write_csv(r.BENCH / "metadata.csv", rows)
    (r.CACHE / "incremental.json").write_text(
        json.dumps(
            {"old": old_text, "new": new_text, "old_top": [c["chunk_id"] for c in old_top], "new_top": [c["chunk_id"] for c in new_top]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    r.log("写入 metadata.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
