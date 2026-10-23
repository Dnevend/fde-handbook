"""3.3 对话模板：模板错了会怎样，思考模式值不值这个钱。

两组对比，都问同一个问题：

A. 正确模板 vs 错误模板
   起两个服务，一个用模型自带的 jinja 模板，一个强制用 chatml。
   错误模板不一定会崩，更常见的表现是"能用但变差"：角色标记错位、
   说话人混在一起、该停不停。这比报错难查得多。

B. 思考模式开 vs 关
   同一服务，靠模板参数 enable_thinking 切换。记录首字延迟、生成量、是否答对。

用法（脚本自己起服务，用完就关）：
    uv run python labs/03-deployment/template_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from serverlib import REPO_ROOT, env_note, start_server, stop_server, stream_once, write_csv

QUESTION = "一家工厂的三条产线各停机 2 小时，总共损失多少机时？只给数字和算式。"
MAX_TOKENS = 400
PORT = 8081
RESULTS_DIR = REPO_ROOT / "benchmarks" / "template"


def rendered_prompt(port: int) -> str:
    """把服务端实际拼出来的 prompt 取回来。

    这是本节最有说服力的一张证据：模板错没错，看渲染结果，不要猜。
    """
    import json
    import urllib.request

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/apply-template",
        data=json.dumps({"messages": [{"role": "user", "content": "你好"}]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp).get("prompt", "")
    except Exception as exc:  # 端点不存在或模板报错，都是一种结果
        return f"<取不到渲染结果：{exc}>"


def run_variant(label: str, port: int, extra_args: tuple[str, ...], thinking: bool | None) -> dict:
    proc = start_server(port, *extra_args, ctx=4096, slots=1,
                        log_path=RESULTS_DIR / f"server-{label}.log")
    try:
        rendered = rendered_prompt(port)
        r = stream_once(QUESTION, max_tokens=MAX_TOKENS, temperature=0,
                        enable_thinking=thinking, port=port)
    finally:
        stop_server(proc)

    return {
        "variant": label,
        "thinking": thinking,
        "rendered": rendered.replace("\n", "\\n")[:120],
        "ttft_s": r["ttft_s"],
        "total_s": r["total_s"],
        "gen_tokens": r["gen_tokens"],
        "tok_per_s": r["tok_per_s"],
        "has_reasoning": bool(r["reasoning"]),
        "reasoning_chars": len(r["reasoning"]),
        "answer_chars": len(r["content"]),
        "answer": r["content"].replace("\n", " ")[:200],
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(env_note(PORT))
    print(f"问题：{QUESTION}\n")

    rows = []

    # A、B 各跑一次；模板错误这一项如果服务起不来（模板被拒），单独记为加载失败
    rows.append(run_variant("default-think-on", 8081, (), True))
    rows.append(run_variant("default-think-off", 8081, (), False))
    try:
        rows.append(run_variant("chatml-think-off", 8081, ("--chat-template", "chatml"), False))
    except (RuntimeError, TimeoutError) as exc:
        print(f"chatml 模板起服务失败：{exc}")
        print("这本身是一类结果：模板不兼容时服务可能直接拒绝启动。")

    write_csv(RESULTS_DIR / "template.csv", rows)
    (RESULTS_DIR / "env.txt").write_text(env_note(PORT) + "\n", encoding="utf-8")

    print("\n结果：")
    for row in rows:
        print(f"- {row['variant']:20s} 首字 {row['ttft_s']:6.2f}s  "
              f"生成 {row['gen_tokens']:4d} 分片  答案 {row['answer_chars']:4d} 字  "
              f"思考 {'有' if row['has_reasoning'] else '无'}")
        print(f"  渲染后的 prompt：{row['rendered']}")

    print("\n要记录的结论：错误模板下是否还「看得懂但答不对」；"
          "`enable_thinking` 这类模板参数在错误模板下会不会静默失效；思考模式多花了多少 token。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
