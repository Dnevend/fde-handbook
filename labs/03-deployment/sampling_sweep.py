"""3.4 采样参数：temperature / top-p / seed 到底在改什么。

做法：固定问题、固定 seed，扫几组参数，每组采样 3 次。
看三件事：
1. temperature = 0 时多次采样是否逐字一致（可复现性的边界）；
2. temperature 升高后，输出的开头是否开始发散；
3. top-p 收窄能否压住高温带来的发散。

用法（先起服务）：
    uv run python labs/03-deployment/sampling_sweep.py
"""

from __future__ import annotations

import sys

from serverlib import REPO_ROOT, env_note, stream_once, write_csv

QUESTION = "用一句话说明为什么本地推理的速度受内存带宽限制。"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "sampling"

# (temperature, top_p)；seed 固定，重复 3 次
GRID = [(0.0, None), (0.7, 0.95), (1.0, 0.95), (1.0, 0.5)]
REPEATS = 3
MAX_TOKENS = 96


def main() -> int:
    print(env_note())
    rows = []

    for temp, top_p in GRID:
        for run in range(1, REPEATS + 1):
            r = stream_once(
                QUESTION,
                max_tokens=MAX_TOKENS,
                temperature=temp,
                top_p=top_p,
                seed=42,
                enable_thinking=False,
            )
            rows.append({
                "temperature": temp,
                "top_p": "" if top_p is None else top_p,
                "run": run,
                "gen_tokens": r["gen_tokens"],
                "tok_per_s": r["tok_per_s"],
                "output": r["content"].replace("\n", " ")[:160],
            })
            print(f"T={temp:<4} top_p={top_p or '-':<5} run{run} "
                  f"{r['gen_tokens']:3d} 分片  {r['tok_per_s']:5.1f} tok/s  {rows[-1]['output'][:60]}")
        print()

    write_csv(RESULTS_DIR / "sweep.csv", rows)
    (RESULTS_DIR / "env.txt").write_text(env_note() + "\n", encoding="utf-8")

    # 同一组参数内部是否逐字相同
    groups: dict[tuple, list[str]] = {}
    for r in rows:
        groups.setdefault((r["temperature"], r["top_p"]), []).append(r["output"])

    print("同一参数下 3 次输出不同的组：")
    for (temp, top_p), outs in groups.items():
        print(f"- T={temp} top_p={top_p or '-'}: {'不同' if len(set(outs)) > 1 else '逐字相同'}")

    print("\n注意：seed 相同时，同一参数组内应可复现；跨参数组之间不可比，"
          "因为采样分布变了。这条是 1.2 节可复现性实验的延伸。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
