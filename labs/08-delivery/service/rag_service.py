"""8.1 RAG 服务核心：检索 → 重排 → 提示词 → 流式生成。

复用第五部分的 `raglib` / `retrieve` / `generate`：
- 切分、向量、BM25、RRF、重排的口径和 5.6/5.7 一致；
- 提示词和 5.8 一致，避免“实验里一套、服务里另一套”。

这里只做编排，不重新发明检索算法。服务层（app.py）负责 HTTP/SSE，
本文件保持可单独测试：给一个问题，返回 passages 和流式事件。
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs/05-rag"))

import raglib as rag  # noqa: E402
import retrieve as retrieval  # noqa: E402
from generate import build_prompt  # noqa: E402

from openai import OpenAI  # noqa: E402

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Passage:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    clause_no: str | None
    section: str | None
    text: str
    score: float | None = None

    def citation(self) -> dict:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "clause_no": self.clause_no,
            "section": self.section,
            "text": self.text[:160],
        }


class RagEngine:
    """检索 + 重排 + 生成。线程内串行使用；FastAPI 每个请求调用一次。"""

    def __init__(
        self,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        rerank: bool = True,
        allowed_docs: set[str] | None = None,
    ) -> None:
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rerank_enabled = rerank
        self.allowed_docs = allowed_docs

        self.chunks = rag.cached_chunks("structure", include_synthetic=False)
        self.ids = [chunk["chunk_id"] for chunk in self.chunks]
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.vectors = self._load_vectors()
        self.bm25 = retrieval.BM25(
            [retrieval.tokenize(chunk["text"]) for chunk in self.chunks],
            rag.CONFIG["retrieval"]["bm25_k1"],
            rag.CONFIG["retrieval"]["bm25_b"],
        )
        self.llm_port: int | None = None

    def _load_vectors(self) -> np.ndarray:
        path = rag.CACHE / "vectors-1024-real.npy"
        if path.exists():
            vectors = np.load(path).astype(np.float32)
            if vectors.shape[0] == len(self.chunks):
                return vectors
        rag.log("缓存向量缺失或与块数不一致，重新计算（需要 embedding 服务）")
        vectors = rag.embed_texts([chunk["text"] for chunk in self.chunks], kind="doc")
        np.save(path, vectors)
        return vectors

    def warmup(self) -> None:
        """显式拉起三个本地服务；失败时让启动失败，而不是等第一个请求才炸。"""
        rag.ensure_embedding_server()
        if self.rerank_enabled:
            rag.ensure_reranker_server()
        self.llm_port = rag.ensure_llm_server(ctx=8192)

    def retrieve(self, question: str) -> tuple[list[Passage], float]:
        started = time.perf_counter()
        query_vector = rag.embed_texts([question], kind="query")[0]

        # 权限过滤在候选池上做，而不是对 top-20 事后过滤：
        # 否则受限文档很多的用户可能因为候选被截断而“误拒”。
        # 注意 rank_from_scores 不会跳过 -inf，所以 k 要封顶到允许块数。
        allowed_mask = None
        allowed_count = len(self.ids)
        if self.allowed_docs is not None:
            allowed_mask = np.array(
                [self.by_id[chunk_id]["doc_id"] in self.allowed_docs for chunk_id in self.ids],
                dtype=bool,
            )
            allowed_count = int(allowed_mask.sum())
            if allowed_count == 0:
                return [], (time.perf_counter() - started) * 1000

        rank_k = min(self.candidate_k, allowed_count)
        dense_scores = (self.vectors @ query_vector).astype(np.float32, copy=True)
        sparse_scores = self.bm25.scores(retrieval.tokenize(question)).astype(np.float32, copy=True)
        if allowed_mask is not None:
            dense_scores[~allowed_mask] = -np.inf
            sparse_scores[~allowed_mask] = -np.inf

        dense = retrieval.rank_from_scores(dense_scores, self.ids, rank_k)
        sparse = retrieval.rank_from_scores(sparse_scores, self.ids, rank_k)
        hybrid = retrieval.rrf([sparse, dense], rag.CONFIG["retrieval"]["rrf_k"])[:rank_k]

        if allowed_mask is not None:
            allowed_ids = {chunk_id for chunk_id, ok in zip(self.ids, allowed_mask) if ok}
            hybrid = [chunk_id for chunk_id in hybrid if chunk_id in allowed_ids]

        if not hybrid:
            return [], (time.perf_counter() - started) * 1000

        effective_top_k = min(self.top_k, len(hybrid))
        if self.rerank_enabled and len(hybrid) > effective_top_k:
            results = rag.rerank(
                question,
                [self.by_id[chunk_id]["text"] for chunk_id in hybrid],
                top_n=len(hybrid),
            )
            order = sorted(results, key=lambda item: -item["relevance_score"])
            ranking = [hybrid[item["index"]] for item in order][:effective_top_k]
        else:
            ranking = hybrid[:effective_top_k]

        passages = []
        for index, chunk_id in enumerate(ranking, start=1):
            chunk = self.by_id[chunk_id]
            passages.append(
                Passage(
                    index=index,
                    chunk_id=chunk_id,
                    doc_id=chunk["doc_id"],
                    title=chunk.get("title", ""),
                    clause_no=chunk.get("clause_no"),
                    section=chunk.get("section"),
                    text=chunk["text"],
                )
            )
        return passages, (time.perf_counter() - started) * 1000

    def _messages(self, question: str, passages: list[Passage]) -> list[dict]:
        chunks = [
            {
                "doc_id": passage.doc_id,
                "clause_no": passage.clause_no,
                "section": passage.section,
                "text": passage.text,
            }
            for passage in passages
        ]
        return build_prompt(question, chunks, with_context=True)

    def stream(self, question: str, *, max_tokens: int = 384, model: str | None = None):
        """生成事件流：retrieval → delta* → citations → done。

        model 是后端 served-model-name；对外暴露的模型名由 app.py 处理。
        vLLM 会校验这个名字，不能把 `rag-service` 直接传进来。
        """
        backend_model = model or os.environ.get("BACKEND_MODEL", "qwen3.5-9b")
        passages, retrieval_ms = self.retrieve(question)
        yield {
            "type": "retrieval",
            "retrieval_ms": round(retrieval_ms, 1),
            "passages": [passage.citation() for passage in passages],
        }

        if not passages:
            yield {
                "type": "delta",
                "text": "资料中没有相关信息。",
            }
            yield {
                "type": "done",
                "ttft_s": 0.0,
                "total_s": 0.0,
                "tokens": 0,
                "citations": [],
            }
            return

        port = self.llm_port or rag.ensure_llm_server(ctx=8192)
        client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-local")
        started = time.perf_counter()
        first = None
        parts: list[str] = []
        tokens = 0
        usage_tokens = 0
        prompt_tokens = 0

        stream = client.chat.completions.create(
            model=backend_model,
            messages=self._messages(question, passages),
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=42,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_tokens = int(getattr(chunk.usage, "completion_tokens", 0) or 0)
                prompt_tokens = int(getattr(chunk.usage, "prompt_tokens", 0) or 0)
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            if not text:
                continue
            if first is None:
                first = time.perf_counter()
            parts.append(text)
            tokens += 1
            yield {"type": "delta", "text": text}

        total = time.perf_counter() - started
        answer = "".join(parts)
        used = sorted({int(value) for value in CITATION.findall(answer)})
        citations = [
            passage.citation()
            for passage in passages
            if passage.index in used
        ] or [passage.citation() for passage in passages]
        yield {
            "type": "done",
            "ttft_s": round((first - started) if first else total, 4),
            "total_s": round(total, 4),
            "tokens": usage_tokens or tokens,
            "prompt_tokens": prompt_tokens,
            "citations": citations,
        }

    def answer(self, question: str, *, max_tokens: int = 384, model: str | None = None) -> dict:
        """非流式包装，给 /v1/chat/completions?stream=false 用。"""
        content: list[str] = []
        citations: list[dict] = []
        retrieval_ms = 0.0
        usage = {}
        for event in self.stream(question, max_tokens=max_tokens, model=model):
            if event["type"] == "retrieval":
                retrieval_ms = event["retrieval_ms"]
            elif event["type"] == "delta":
                content.append(event["text"])
            elif event["type"] == "done":
                citations = event["citations"]
                usage = event
        return {
            "content": "".join(content),
            "citations": citations,
            "retrieval_ms": retrieval_ms,
            "ttft_s": usage.get("ttft_s", 0.0),
            "total_s": usage.get("total_s", 0.0),
            "tokens": usage.get("tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
        }
