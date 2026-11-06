#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2,<3"]
# ///
"""7.4 / 7.6 / 7.7 的并发压测：从 OpenAI 兼容接口测 TTFT、TPOT、吞吐。

和第三部分的 bench_client.py 的区别：
- 这里打的是 vLLM，不是 llama-server；
- 支持一次给多个 base_url（2 个副本），请求轮询分配；
- 支持并发梯度，一次跑完 1/2/4/8/16/32；
- 输出逐请求 CSV + 汇总 CSV，方便后面画图。

用法（先起服务，见 serve_vllm.sh）：
    uv run labs/07-gpu-inference/bench_openai.py \
      --base-url http://127.0.0.1:8000/v1 \
      --concurrency 1,2,4,8,16,32 --requests 64 \
      --prompt-chars 1200 --max-tokens 128 --label qwen-awq

    # 两个副本：请求轮询到两个端口
    uv run labs/07-gpu-inference/bench_openai.py \
      --base-urls http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1 \
      --concurrency 8 --requests 64 --label two-replicas

注意：
- 默认给每个请求加唯一前缀，避免命中 Prefix Caching；测缓存收益时加 --no-unique-prefix。
- 每个并发点先 warmup 一次，不计入统计。
- TTFT 和 TPOT 都按流式分片算；非流式测不了首字延迟。
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulib import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RESULTS_DIR,
    env_note,
    percentile,
    stream_once,
    write_csv,
)

# 固定语料，只用来生成不同长度的 prompt；改动会让历史数据失去可比性。
BASE_TEXT = (
    "设备维护记录显示，三号产线的液压站需要每五百小时更换一次液压油。"
    "如果油温连续两小时超过七十摄氏度，应当先检查冷却水路，再判断是否需要停机。"
    "维修完成后，操作员需要在工单上填写停机时长、更换件编号和复检结果。"
)


def build_prompt(chars: int, nonce: str) -> str:
    """按字符数重复固定语料；实际 token 数由服务端 usage 报告。"""
    if chars <= 0:
        return nonce + "用一句话说明什么是首字延迟。"
    repeats = chars // len(BASE_TEXT) + 1
    return nonce + (BASE_TEXT * repeats)[:chars]


def one_request(
    *,
    base_url: str,
    index: int,
    model: str,
    prompt_chars: int,
    max_tokens: int,
    unique_prefix: bool,
) -> dict:
    nonce = f"[请求 {index}] " if unique_prefix else ""
    prompt = build_prompt(prompt_chars, nonce)
    started = time.perf_counter()
    try:
        result = stream_once(
            prompt,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=42,
            enable_thinking=False,
        )
        tpot_ms = 0.0
        if result["gen_tokens"] > 1:
            tpot_ms = (result["total_s"] - result["ttft_s"]) / (result["gen_tokens"] - 1) * 1000
        return {
            "base_url": base_url,
            "index": index,
            "ttft_s": result["ttft_s"],
            "total_s": result["total_s"],
            "gen_tokens": result["gen_tokens"],
            "tok_per_s": result["tok_per_s"],
            "tpot_ms": round(tpot_ms, 2),
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - 记录错误，不让一轮压测整体崩掉
        return {
            "base_url": base_url,
            "index": index,
            "ttft_s": 0.0,
            "total_s": round(time.perf_counter() - started, 4),
            "gen_tokens": 0,
            "tok_per_s": 0.0,
            "tpot_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def show(path: Path) -> str:
    """优先显示仓库相对路径；用户给绝对路径时原样显示。"""
    try:
        return str(path.relative_to(RESULTS_DIR.parent.parent))
    except ValueError:
        return str(path)


def summarize(label: str, concurrency: int, rows: list[dict], wall_s: float) -> dict:
    ok = [row for row in rows if not row["error"]]
    ttfts = [row["ttft_s"] for row in ok]
    tpots = [row["tpot_ms"] for row in ok]
    speeds = [row["tok_per_s"] for row in ok]
    prompts = [row["prompt_tokens"] for row in ok]
    total_tokens = sum(row["gen_tokens"] for row in ok)
    return {
        "label": label,
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "wall_s": round(wall_s, 3),
        "aggregate_tok_per_s": round(total_tokens / wall_s, 1) if wall_s > 0 else 0.0,
        "request_per_s": round(len(ok) / wall_s, 2) if wall_s > 0 else 0.0,
        "ttft_p50_s": round(percentile(ttfts, 50), 3),
        "ttft_p95_s": round(percentile(ttfts, 95), 3),
        "tpot_p50_ms": round(percentile(tpots, 50), 2),
        "tpot_p95_ms": round(percentile(tpots, 95), 2),
        "per_req_tok_s_p50": round(percentile(speeds, 50), 2),
        "prompt_tokens_p50": int(percentile(prompts, 50)),
    }


def run_round(
    *,
    label: str,
    base_urls: list[str],
    concurrency: int,
    requests: int,
    model: str,
    prompt_chars: int,
    max_tokens: int,
    unique_prefix: bool,
    warmup: bool,
) -> tuple[dict, list[dict]]:
    if warmup:
        for url in base_urls:
            one_request(
                base_url=url,
                index=-1,
                model=model,
                prompt_chars=min(prompt_chars, 256),
                max_tokens=8,
                unique_prefix=True,
            )

    def call(index: int) -> dict:
        return one_request(
            base_url=base_urls[index % len(base_urls)],
            index=index,
            model=model,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            unique_prefix=unique_prefix,
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(call, range(requests)))
    wall = time.perf_counter() - started
    return summarize(label, concurrency, rows, wall), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM 并发压测（TTFT / TPOT / 吞吐）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="单个 vLLM /v1 地址")
    parser.add_argument("--base-urls", default="", help="多个地址，逗号分隔；给了就覆盖 --base-url")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served-model-name")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32", help="并发梯度，逗号分隔")
    parser.add_argument("--requests", type=int, default=64, help="每个并发点的请求数")
    parser.add_argument("--prompt-chars", type=int, default=1200, help="prompt 字符数（实际 token 看 usage）")
    parser.add_argument("--max-tokens", type=int, default=128, help="每个请求最多生成 token")
    parser.add_argument("--label", default="bench", help="输出文件前缀")
    parser.add_argument("--no-unique-prefix", action="store_true", help="不加唯一前缀，测 Prefix Caching")
    parser.add_argument("--no-warmup", action="store_true", help="跳过 warmup")
    parser.add_argument("--output", default="", help="汇总 CSV；默认 benchmarks/gpu/bench/<label>-summary.csv")
    args = parser.parse_args()

    base_urls = [u.strip() for u in args.base_urls.split(",") if u.strip()] or [args.base_url]
    concurrency_list = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if not concurrency_list:
        parser.error("--concurrency 不能为空")

    out = Path(args.output) if args.output else RESULTS_DIR / "bench" / f"{args.label}-summary.csv"
    per_request_out = out.with_name(f"{args.label}-requests.csv")
    env_out = RESULTS_DIR / "env" / f"{args.label}-bench.txt"

    endpoint_note = ", ".join(base_urls)
    print(env_note(base_url=endpoint_note, model=args.model))
    print(f"\n端点: {endpoint_note}")
    print(f"模型: {args.model}；prompt_chars={args.prompt_chars}；max_tokens={args.max_tokens}")
    print(f"并发梯度: {concurrency_list}；每点 {args.requests} 请求\n")

    summaries: list[dict] = []
    all_rows: list[dict] = []
    for concurrency in concurrency_list:
        requests = max(args.requests, concurrency)
        summary, rows = run_round(
            label=args.label,
            base_urls=base_urls,
            concurrency=concurrency,
            requests=requests,
            model=args.model,
            prompt_chars=args.prompt_chars,
            max_tokens=args.max_tokens,
            unique_prefix=not args.no_unique_prefix,
            warmup=not args.no_warmup,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"C={concurrency:>3}  errors={summary['errors']:>2}  "
            f"TTFT p50/p95={summary['ttft_p50_s']:.2f}/{summary['ttft_p95_s']:.2f}s  "
            f"TPOT p50/p95={summary['tpot_p50_ms']:.1f}/{summary['tpot_p95_ms']:.1f}ms  "
            f"聚合={summary['aggregate_tok_per_s']:.0f} tok/s  "
            f"单请求={summary['per_req_tok_s_p50']:.1f} tok/s"
        )
        if summary["errors"] == requests:
            print("  这一轮全部失败；先看服务日志，别继续加并发。", file=sys.stderr)

    write_csv(out, summaries)
    write_csv(per_request_out, all_rows)
    env_out.parent.mkdir(parents=True, exist_ok=True)
    env_out.write_text(env_note(base_url=endpoint_note, model=args.model) + "\n", encoding="utf-8")
    print(f"\n汇总: {show(out)}")
    print(f"逐请求: {show(per_request_out)}")
    print(f"环境: {show(env_out)}")

    if all(row["errors"] == row["requests"] for row in summaries):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
