"""5.1 长上下文 vs RAG：把答案埋进不同长度的上下文，再看模型能否找到。

三档上下文长度 × 三个插入位置（开头/中间/结尾），与 RAG（top-5）对比：
  命中（答案给出「设机构或配专职人员」）、首字延迟、上下文 token 数。

这是"塞进长上下文"这条路的实测；语料不够 32k token 时用政府文件正文补足。
产物：benchmarks/rag/long-context.csv
"""

from __future__ import annotations

import argparse
import sys
import uuid
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402

NEEDLE = (
    "补充记录：依照《安全生产法》第二十四条，从业人员超过一百人的其他生产经营单位，"
    "应当设置安全生产管理机构或者配备专职安全生产管理人员。"
)
QUESTION = "从业人员超过一百人的其他生产经营单位，应当如何配备安全生产管理人员？"


def hit(answer: str) -> bool:
    normalized = answer.replace(" ", "")
    return "设置安全生产管理机构" in normalized or ("专职" in normalized and "机构" in normalized)


def build_context(target_chars: int) -> str:
    chunks = [chunk for chunk in r.cached_chunks("structure", include_synthetic=False) if chunk["doc_id"] in {"reg-safety-law", "gov-ledger-guide"}]
    parts: list[str] = []
    length = 0
    for chunk in chunks:
        parts.append(chunk["text"])
        length += len(chunk["text"])
        if length >= target_chars:
            break
    return "\n".join(parts)[:target_chars]


def insert(text: str, position: float) -> str:
    index = int(len(text) * position)
    return text[:index] + "\n" + NEEDLE + "\n" + text[index:]


def prompt(context: str, nonce: str) -> str:
    # nonce 放在最前面，避免服务复用上一次的 KV 前缀，TTFT 才是冷启动的预填充时间。
    return (
        f"会话编号：{nonce}\n"
        "只根据下面提供的资料回答问题，不要使用资料以外的知识。"
        "回答不超过 40 字；资料中没有答案就回答「不知道」。\n\n"
        f"资料：\n{context}\n\n问题：{QUESTION}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑一档，先看耗时")
    args = parser.parse_args()

    port = r.ensure_llm_server(ctx=32768, port=8083)
    sizes = [4000] if args.quick else [4000, 16000, 40000]
    positions = [0.5] if args.quick else [0.0, 0.5, 1.0]

    rows: list[dict] = []
    for size in sizes:
        filler = build_context(size)
        for position in positions:
            context = insert(filler, position)
            text = prompt(context, uuid.uuid4().hex)
            tokens = r.count_tokens(text, port=port)
            r.log(f"上下文 {size} 字 / {tokens} token，位置 {position:.0%}")
            start = time.perf_counter()
            answer, ttft = r.chat_stream([{"role": "user", "content": text}], max_tokens=48, port=port)
            total = time.perf_counter() - start
            correct = hit(answer)
            rows.append(
                {
                    "mode": "long_context",
                    "context_chars": len(context),
                    "context_tokens": tokens,
                    "position": position,
                    "ttft_s": round(ttft, 2),
                    "total_s": round(total, 1),
                    "correct": int(correct),
                    "answer": answer.replace("\n", " ")[:120],
                }
            )
            r.log(f"  → {'命中' if correct else '未命中'}，TTFT {ttft:.1f}s")

    # RAG 对照：稠密检索 top-5
    chunks = r.cached_chunks("structure", include_synthetic=False)
    vectors = np.load(r.CACHE / "vectors-1024-real.npy")
    query = r.embed_texts([QUESTION], kind="query")[0]
    order = np.argsort(-(vectors @ query))[:5]
    context = "\n".join(chunks[index]["text"] for index in order)
    text = prompt(context, uuid.uuid4().hex)
    tokens = r.count_tokens(text, port=port)
    answer, ttft = r.chat_stream([{"role": "user", "content": text}], max_tokens=48, port=port)
    rows.append(
        {
            "mode": "rag_top5",
            "context_chars": len(context),
            "context_tokens": tokens,
            "position": "",
            "ttft_s": round(ttft, 2),
            "total_s": "",
            "correct": int(hit(answer)),
            "answer": answer.replace("\n", " ")[:120],
        }
    )

    r.write_csv(r.BENCH / "long-context.csv", rows)
    r.log("写入 long-context.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
