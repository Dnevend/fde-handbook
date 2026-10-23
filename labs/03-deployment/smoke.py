"""3.5 服务化：证明"只改 base_url 就能接上"。

做三件事：
1. 非流式请求一次，打印正文、思考内容、token 用量；
2. 流式请求一次，打印首字延迟和 decode 速度；
3. 用同一个 SDK 再问一次，验证多轮上下文（messages 里带历史）能正常工作。

用法（先起服务，见 serve.sh）：
    uv run python labs/03-deployment/smoke.py
"""

from __future__ import annotations

import os
import sys

from serverlib import BASE_URL, MODEL_NAME, client, env_note, stream_once


def main() -> None:
    print(env_note())

    c = client()

    # 1. 非流式：和调用 OpenAI 完全一样，只有 base_url 不同
    resp = c.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
        max_tokens=256,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    print("\n[非流式]")
    print("正文:", msg.content)
    print("思考:", (getattr(msg, "reasoning_content", None) or "")[:80] or "（无）")
    print("用量:", resp.usage.model_dump() if resp.usage else None)

    # 2. 流式：首字延迟和总时长分开看
    print("\n[流式，思考模式用默认设置]")
    result = stream_once("用三句话说明本地部署大模型的主要成本。", max_tokens=256)
    print(f"首字延迟 {result['ttft_s']:.2f}s  "
          f"生成 {result['gen_tokens']} 个分片  "
          f"decode {result['tok_per_s']:.1f} tok/s  "
          f"总时长 {result['total_s']:.2f}s")
    print(f"其中思考内容 {len(result['reasoning'])} 字符，正文 {len(result['content'])} 字符")
    if not result["content"]:
        print("正文为空：思考把 max_tokens 用完了。这不是故障，是 3.3 节要量的成本。")
    else:
        print(result["content"])

    # 3. 多轮：服务端不保存会话，历史由客户端每次带上
    print("\n[多轮]")
    history = [{"role": "user", "content": "记住这个数字：417。"},
               {"role": "assistant", "content": "好的，我记住了 417。"},
               {"role": "user", "content": "刚才的数字是多少？只回答数字。"}]
    resp = c.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        max_tokens=32,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("回答:", resp.choices[0].message.content)

    print(f"\n服务地址 {BASE_URL}，模型名 {MODEL_NAME}。"
          "换到 GPU 机器时只改 OPENAI_BASE_URL 和 LLM_MODEL。")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        os._exit(130)
