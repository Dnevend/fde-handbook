"""3.7 性能测量方法：自己测出来的数字，怎么才可信。

`llama-bench` 测的是引擎的裸能力（固定 prompt、无网络、无模板、无排队）。
你从 API 测到的是整条链路：HTTP、模板渲染、tokenize、排队、前缀缓存、流式回传。
两者差多少，这篇脚本就是用来量出来的。

三组对照，每种长度各跑 REPEATS 次：
1. 冷请求（首次出现的前缀）：首字延迟里包含完整的 prefill；
2. 热请求（同一 prompt 紧接着再发）：前缀缓存命中，prefill 几乎归零；
3. 长短不同的 prompt：看首字延迟随输入长度怎么变。

"冷"和"热"必须分开报。只报热请求的首字延迟，等于把缓存当成了引擎性能。

用法（先起服务，见 serve.sh）：
    BASE_URL=http://127.0.0.1:8080/v1 uv run python labs/03-deployment/bench_client.py
"""

from __future__ import annotations

import statistics
import sys

from serverlib import REPO_ROOT, env_note, stream_once, write_csv

RESULTS_DIR = REPO_ROOT / "benchmarks" / "client-bench"
REPEATS = 3
MAX_TOKENS = 96

# 长短由重复同一句话得到
# 固定为 2026-10 实测时的语料：它只用来生成不同长度的 prompt，
# 改动会让 benchmarks/client-bench 的记录失去可比性。与正文的贯穿案例无关。
SENTENCE = "设备维护记录显示三号产线的液压站需要每五百小时更换一次液压油。"
PROMPTS = {
    "short": "用一句话解释什么是首字延迟。",
    "medium": "请阅读下面的背景材料，然后用两句话总结它的要点：" + SENTENCE * 8,
    "long": "请阅读下面的背景材料，然后用两句话总结它的要点：" + SENTENCE * 40,
}


def measure(prompt: str, nonce: str = "") -> dict:
    """nonce 加在 prompt 最前面，用来制造缓存未命中。"""
    return stream_once(nonce + prompt, max_tokens=MAX_TOKENS, temperature=0,
                       enable_thinking=False)


def summarize(label: str, runs: list[dict]) -> dict:
    ttfts = [r["ttft_s"] for r in runs]
    totals = [r["total_s"] for r in runs]
    speeds = [r["tok_per_s"] for r in runs]
    return {
        "case": label,
        "prompt_tokens": (runs[0]["usage"] or {}).get("prompt_tokens", 0),
        "ttft_median_s": round(statistics.median(ttfts), 3),
        "total_median_s": round(statistics.median(totals), 3),
        "tok_per_s_median": round(statistics.median(speeds), 1),
        "ttft_spread_s": round(max(ttfts) - min(ttfts), 3),
        "gen_tokens": runs[0]["gen_tokens"],
    }


def main() -> int:
    print(env_note())
    rows = []

    for name, prompt in PROMPTS.items():
        cold, warm = [], []
        for i in range(REPEATS):
            # 冷：每次换一个前导标记，前缀缓存必然失效
            cold.append(measure(prompt, nonce=f"[第{i}次] "))
            # 热：第一次跑完后，紧接着用完全相同的 prompt 再发
            warm.append(measure(prompt))

        cold_row = summarize(f"{name}-cold", cold)
        warm_row = summarize(f"{name}-warm", warm)
        rows.extend([cold_row, warm_row])
        print(cold_row)
        print(warm_row)

    write_csv(RESULTS_DIR / "api-bench.csv", rows)
    (RESULTS_DIR / "env.txt").write_text(env_note() + "\n", encoding="utf-8")

    print("\n结论要点：")
    print("- 冷请求的首字延迟 = 固定开销 + prompt_tokens × 每 token prefill 时间。")
    print("- 热请求的首字延迟接近固定开销，说明前缀缓存把 prefill 省掉了；"
          "评估生产延迟时必须按冷请求算，或者明确说明命中率假设。")
    print("- ttft_spread 是这台机器的噪声底，小于它的差异不要当结论。")
    print("- 和 llama-bench 的 tg128 比：API 的 tok_per_s 通常更低，"
          "差额里含 HTTP、流式分片和模板开销。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
