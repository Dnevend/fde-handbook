#!/usr/bin/env python3
"""8.1 可上线的 RAG 服务：OpenAI 兼容接口 + SSE + 引用 + 反馈 + 基础指标。

这不是“再写一个 demo”，而是把第五部分的链路放进一个服务壳：
- `/v1/chat/completions`：兼容 OpenAI 的流式/非流式接口；
- `/v1/models`、`/health`：给客户端和探针用；
- `/feedback`：把用户反馈按 request_id 落盘；
- `/metrics`：最小 Prometheus 文本，8.4 会扩展。

运行（先按 5.x 准备好模型与索引）：
    cd labs && uv run 08-delivery/service/app.py

    或从 service 目录起 uvicorn（目录名带连字符，不能用 08-delivery.service 这种导入路径）：
    cd labs/08-delivery/service && uv run uvicorn app:app --host 127.0.0.1 --port 8090

环境变量：
    SERVICE_HOST=127.0.0.1  SERVICE_PORT=8090
    SERVICE_WARMUP=1        启动时拉起 embedding/rerank/llm
    RERANK=1                是否重排
    ALLOWED_DOCS=           逗号分隔的 doc_id；空表示不限制（8.5 的权限钩子）
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_service import RagEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FEEDBACK = ROOT / "benchmarks" / "delivery" / "feedback.jsonl"

logger = logging.getLogger("rag-service")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


def log_event(**fields: Any) -> None:
    logger.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}, ensure_ascii=False))


class Metrics:
    """够用的进程内指标；多进程部署时每个 worker 各一份，8.4 再换 Prometheus。"""

    def __init__(self) -> None:
        self.lock = Lock()
        self.requests = 0
        self.errors = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.total_sum = 0.0
        self.tokens = 0
        self.retrieval_sum = 0.0
        self.retrieval_count = 0
        self.feedback = 0

    def observe(self, *, ok: bool, ttft: float, total: float, tokens: int, retrieval_ms: float) -> None:
        with self.lock:
            self.requests += 1
            if not ok:
                self.errors += 1
            if ttft:
                self.ttft_sum += ttft
                self.ttft_count += 1
            self.total_sum += total
            self.tokens += tokens
            if retrieval_ms:
                self.retrieval_sum += retrieval_ms
                self.retrieval_count += 1

    def add_feedback(self) -> None:
        with self.lock:
            self.feedback += 1

    def render(self) -> str:
        with self.lock:
            lines = [
                "# HELP rag_requests_total 请求总数",
                "# TYPE rag_requests_total counter",
                f"rag_requests_total {self.requests}",
                "# HELP rag_errors_total 错误总数",
                "# TYPE rag_errors_total counter",
                f"rag_errors_total {self.errors}",
                "# HELP rag_tokens_total 生成 token 总数",
                "# TYPE rag_tokens_total counter",
                f"rag_tokens_total {self.tokens}",
                "# HELP rag_feedback_total 反馈条数",
                "# TYPE rag_feedback_total counter",
                f"rag_feedback_total {self.feedback}",
            ]
            if self.ttft_count:
                lines += [
                    "# HELP rag_ttft_seconds 首字延迟",
                    "# TYPE rag_ttft_seconds summary",
                    f"rag_ttft_seconds_sum {self.ttft_sum:.4f}",
                    f"rag_ttft_seconds_count {self.ttft_count}",
                ]
            if self.retrieval_count:
                lines += [
                    "# HELP rag_retrieval_ms 检索+重排耗时",
                    "# TYPE rag_retrieval_ms summary",
                    f"rag_retrieval_ms_sum {self.retrieval_sum:.1f}",
                    f"rag_retrieval_ms_count {self.retrieval_count}",
                ]
            return "\n".join(lines) + "\n"


metrics = Metrics()
ENGINE: RagEngine | None = None
ENGINE_LOCK = Lock()


def build_engine() -> RagEngine:
    allowed = {item.strip() for item in os.environ.get("ALLOWED_DOCS", "").split(",") if item.strip()}
    engine = RagEngine(
        rerank=os.environ.get("RERANK", "1") != "0",
        allowed_docs=allowed or None,
    )
    if os.environ.get("SERVICE_WARMUP", "1") != "0":
        engine.warmup()
    return engine


def get_engine() -> RagEngine:
    global ENGINE
    if ENGINE is None:
        with ENGINE_LOCK:
            if ENGINE is None:
                ENGINE = build_engine()
    return ENGINE


app = FastAPI(title="FDE Handbook RAG Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        item.strip()
        for item in os.environ.get(
            "CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
        ).split(",")
        if item.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log_event(
        event="http",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return response


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "rag-service"
    messages: list[Message]
    stream: bool = False
    max_tokens: int = Field(default=384, ge=1, le=8192)
    temperature: float = 0.0


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    tags: list[str] = []


def last_user_message(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    raise ValueError("messages 里没有非空 user 消息")


def sse(event: str | None, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def chunk_payload(request_id: str, model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


def stream_events(question: str, request_id: str, model: str, max_tokens: int):
    started = time.perf_counter()
    done: dict = {}
    retrieval_ms = 0.0
    ok = False
    try:
        engine = get_engine()
        for event in engine.stream(
            question,
            max_tokens=max_tokens,
            model=os.environ.get("BACKEND_MODEL", "qwen3.5-9b"),
        ):
            if event["type"] == "retrieval":
                retrieval_ms = float(event["retrieval_ms"])
                yield sse(
                    "retrieval",
                    {
                        "request_id": request_id,
                        "retrieval_ms": event["retrieval_ms"],
                        "passages": event["passages"],
                    },
                )
            elif event["type"] == "delta":
                yield sse(None, chunk_payload(request_id, model, event["text"]))
            elif event["type"] == "done":
                done = event
                ok = True
                yield sse(None, {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                yield sse("citations", {
                    "request_id": request_id,
                    "citations": event.get("citations", []),
                })
                yield sse("done", {
                    "request_id": request_id,
                    "ttft_s": event.get("ttft_s", 0.0),
                    "total_s": event.get("total_s", 0.0),
                    "tokens": event.get("tokens", 0),
                    "prompt_tokens": event.get("prompt_tokens", 0),
                })
        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001 - 错误要变成 SSE 事件，不能只写日志
        log_event(event="error", request_id=request_id, error=f"{type(exc).__name__}: {exc}")
        yield sse("error", {
            "request_id": request_id,
            "error": "internal_error",
            "message": "生成失败，请用 request_id 查日志。",
        })
        yield "data: [DONE]\n\n"
    finally:
        # 注意：finally 里不能再 yield。客户端断开时生成器会被 close，
        # 在 finally 里 yield 会抛 GeneratorExit/RuntimeError；这里只记指标和日志。
        total = time.perf_counter() - started
        metrics.observe(
            ok=ok,
            ttft=float(done.get("ttft_s", 0.0)),
            total=total,
            tokens=int(done.get("tokens", 0)),
            retrieval_ms=retrieval_ms,
        )
        log_event(
            event="chat",
            request_id=request_id,
            ok=ok,
            total_s=round(total, 3),
            ttft_s=done.get("ttft_s"),
            tokens=done.get("tokens"),
        )


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "chunks": len(engine.chunks),
        "model": os.environ.get("BACKEND_MODEL", "qwen3.5-9b"),
        "rerank": engine.rerank_enabled,
        "allowed_docs": sorted(engine.allowed_docs) if engine.allowed_docs else None,
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "rag-service", "object": "model", "owned_by": "fde-handbook"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: Request, payload: ChatRequest):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    try:
        question = last_user_message(payload.messages)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    max_chars = int(os.environ.get("MAX_QUESTION_CHARS", "8000"))
    total_chars = sum(len(message.content) for message in payload.messages)
    if total_chars > max_chars:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": f"messages 总长度超过 {max_chars} 字符上限；请缩短或先切分文档。",
                    "type": "request_too_large",
                }
            },
        )

    if payload.stream:
        return StreamingResponse(
            stream_events(question, request_id, payload.model, payload.max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    engine = get_engine()
    started = time.perf_counter()
    try:
        result = engine.answer(
            question,
            max_tokens=payload.max_tokens,
            model=os.environ.get("BACKEND_MODEL", "qwen3.5-9b"),
        )
        metrics.observe(
            ok=True,
            ttft=float(result.get("ttft_s", 0.0)),
            total=time.perf_counter() - started,
            tokens=int(result.get("tokens", 0)),
            retrieval_ms=float(result.get("retrieval_ms", 0.0)),
        )
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }
            ],
            "citations": result["citations"],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result["tokens"],
                "total_tokens": result.get("prompt_tokens", 0) + result["tokens"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        metrics.observe(ok=False, ttft=0.0, total=time.perf_counter() - started, tokens=0, retrieval_ms=0.0)
        log_event(event="error", request_id=request_id, error=f"{type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "生成失败，请用 request_id 查日志。", "type": "internal_error"}},
        )


@app.post("/feedback")
def feedback(payload: FeedbackRequest, request: Request):
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": payload.request_id,
        "rating": payload.rating,
        "comment": payload.comment,
        "tags": payload.tags,
        "reported_by": request.headers.get("x-user-id", ""),
        "feedback_request_id": getattr(request.state, "request_id", ""),
    }
    with FEEDBACK.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics.add_feedback()
    log_event(event="feedback", **row)
    return {"ok": True, "request_id": payload.request_id}


@app.get("/metrics")
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


def main() -> int:
    import uvicorn

    host = os.environ.get("SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("SERVICE_PORT", "8090"))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
