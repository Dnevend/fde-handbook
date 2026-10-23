"""3.6 并发与批处理：一个人用很流畅，两个人用就排队，瓶颈在哪。

做法：槽位数取 1 / 2 / 4，每种情况下同时发 CONCURRENCY 个请求。
每个请求记录首字延迟和总时长，最后算聚合吞吐。

关键观察：
- 槽位是硬上限。同时到达的请求数超过槽位，超出的在服务端排队，
  表现为首字延迟随并发线性上升，而单请求 decode 速度不变。
- 每个槽位分到的上下文是 CTX/NP。NP 开大，单请求能用的上下文变小，
  这是并发能力的隐形代价。
- GPU 上的连续批处理（第 7 部分）能把多个请求塞进同一次前向，
  吞吐提升明显；CPU/Metal 上没有同等效果，不要拿这里的曲线外推。

用法：
    # 方式一：脚本自己按 NP=1,2,4 起服务（端口 8082）
    uv run python labs/03-deployment/concurrency.py
    # 方式二：连已启动的服务，只测当前配置
    BASE_URL=http://127.0.0.1:8080/v1 uv run python labs/03-deployment/concurrency.py --external
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from serverlib import MODEL_NAME, client, env_note, start_server, stop_server, write_csv, REPO_ROOT

SLOT_CONFIGS = [1, 2, 4]
CONCURRENCY = 8
MAX_TOKENS = 64
CTX = 8192
PORT = 8082
RESULTS_DIR = REPO_ROOT / "benchmarks" / "concurrency"
PROMPT = "用一句话说明什么是 prefill，什么是 decode。"


def one_request(i: int, port: int | None = None) -> dict:
    """发一个流式请求，记录首字延迟、总时长、生成分片数。"""
    c = client(port)
    started = time.perf_counter()
    ttft = None
    tokens = 0
    stream = c.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"{PROMPT}（第 {i} 个请求）"}],
        max_tokens=MAX_TOKENS,
        temperature=0,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if ttft is None:
                ttft = time.perf_counter() - started
            tokens += 1
    total = time.perf_counter() - started
    return {"ttft_s": ttft or total, "total_s": total, "gen_tokens": tokens}


def run_round(slots: int, port: int | None = None) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(lambda i: one_request(i, port), range(CONCURRENCY)))
    wall = time.perf_counter() - started

    total_tokens = sum(r["gen_tokens"] for r in results)
    return {
        "slots": slots,
        "concurrency": CONCURRENCY,
        "wall_s": round(wall, 2),
        "aggregate_tok_per_s": round(total_tokens / wall, 1),
        "ttft_p50_s": round(sorted(r["ttft_s"] for r in results)[len(results) // 2], 2),
        "ttft_max_s": round(max(r["ttft_s"] for r in results), 2),
        "per_req_tok_per_s": round(
            sum(r["gen_tokens"] / max(r["total_s"] - r["ttft_s"], 1e-6) for r in results) / len(results), 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", action="store_true",
                        help="不自己起服务，直接测 BASE_URL 指向的实例")
    args = parser.parse_args()

    print(env_note())
    rows = []

    if args.external:
        rows.append(run_round(slots=-1))
    else:
        for slots in SLOT_CONFIGS:
            print(f"\n== 启动 NP={slots}（每槽位上下文 {CTX // slots}）")
            proc = start_server(slots=slots, port=PORT, ctx=CTX,
                                log_path=RESULTS_DIR / f"server-np{slots}.log")
            try:
                rows.append(run_round(slots, PORT))
            finally:
                stop_server(proc)
            print(rows[-1])

    write_csv(RESULTS_DIR / "concurrency.csv", rows)
    (RESULTS_DIR / "env.txt").write_text(env_note() + "\n", encoding="utf-8")
    print("\n读法：aggregate_tok_per_s 是机器总产出，per_req_tok_per_s 是单个用户体感。")
    print("槽位数从 1 加到 2，若 aggregate 明显上涨而 per_req 不降，说明批处理吃到了并行；")
    print("若 ttft 随并发暴涨而 aggregate 不涨，就是纯排队。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
