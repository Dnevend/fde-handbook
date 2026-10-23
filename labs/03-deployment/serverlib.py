"""第三部分实验共用的工具：起服务、停服务、流式计时。

只依赖标准库和 openai SDK。脚本之间共享的原因很简单：
"怎么算首字延迟"必须只有一个定义，否则各节数字不可比。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from openai import OpenAI

# 默认指向本机 llama-server。换成 GPU 机器时只改环境变量。
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")
# llama-server 不校验模型名，但 OpenAI SDK 要求传一个，用 --alias 对齐即可。
MODEL_NAME = os.environ.get("LLM_MODEL", "qwen3.5-9b")
REPO_ROOT = Path(__file__).resolve().parents[2]


def client(port: int | None = None) -> OpenAI:
    """返回指向本地服务的 SDK 客户端。

    传 port 时连 http://127.0.0.1:<port>/v1，否则用 BASE_URL。
    api_key 是占位值，本地服务默认不校验。
    """
    base_url = f"http://127.0.0.1:{port}/v1" if port else BASE_URL
    return OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "sk-local"))


def model_path() -> str:
    """取 $MODEL；未设置时执行 labs/env.sh 兜底。"""
    path = os.environ.get("MODEL")
    if path and Path(path).is_file():
        return path

    env_sh = REPO_ROOT / "labs" / "env.sh"
    result = subprocess.run(
        ["bash", "-c", f'source "{env_sh}" >/dev/null && printf %s "$MODEL"'],
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    if not path:
        sys.exit("找不到模型路径。先在仓库根目录执行 source labs/env.sh（见 0.3 节）。")
    return path


def free_port(port: int) -> bool:
    """端口上是否已经有服务在跑。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_ready(port: int, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if free_port(port):
            return
        time.sleep(0.5)
    raise TimeoutError(f"服务在 {timeout:.0f}s 内没有就绪，检查端口 {port} 的日志")


def start_server(
    port: int,
    *extra_args: str,
    ctx: int = 8192,
    slots: int = 1,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """后台起一个 llama-server，等它就绪后返回进程句柄。

    只在实验需要切换启动参数（模板、槽位数）时使用；
    平时直接用 serve.sh 手工起服务，脚本连上去即可。
    """
    if free_port(port):
        raise RuntimeError(f"端口 {port} 已有服务在跑，先停掉或换端口")

    cmd = [
        "llama-server",
        "-m", model_path(),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(ctx),
        "-np", str(slots),
        "-ngl", "99",
        "--no-webui",
        "--no-slots",
        "--alias", MODEL_NAME,
        *extra_args,
    ]
    log = subprocess.DEVNULL
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(port)
    except TimeoutError:
        stop_server(proc)
        raise
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def stream_once(
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = 42,
    enable_thinking: bool | None = None,
    port: int | None = None,
) -> dict:
    """发一次流式请求，返回计时和内容。

    返回字段：
      ttft_s        首字延迟：从请求发出到第一个内容分片到达
      total_s       端到端耗时
      gen_tokens    生成的分片数（≈ token 数，特殊 token 不计）
      tok_per_s     gen_tokens / (total_s - ttft_s)，即纯 decode 速度
      content       正文
      reasoning     思考内容（关闭思考时为空）
      usage         服务端报告的 token 计数，用于交叉核对
    """
    kwargs: dict = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed
    if enable_thinking is not None:
        # Qwen3.5 通过对话模板参数控制思考模式；llama-server 原样透传给模板
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    c = client(port)
    started = time.perf_counter()
    ttft = None
    content: list[str] = []
    reasoning: list[str] = []
    gen_tokens = 0
    usage = None

    stream = c.chat.completions.create(**kwargs)
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 思考内容单独计数：它同样消耗时间，但用户看不到
        piece = getattr(delta, "reasoning_content", None) or delta.content
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - started
            gen_tokens += 1
            if getattr(delta, "reasoning_content", None):
                reasoning.append(piece)
            else:
                content.append(piece)
    total = time.perf_counter() - started
    ttft = ttft if ttft is not None else total
    decode_window = max(total - ttft, 1e-6)

    return {
        "ttft_s": round(ttft, 4),
        "total_s": round(total, 4),
        "gen_tokens": gen_tokens,
        "tok_per_s": round(gen_tokens / decode_window, 2),
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """把结果写成 CSV，第一行是表头。列顺序取第一个 dict 的键顺序。"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入 {path}（{len(rows)} 行）")


def env_note(port: int | None = None) -> str:
    """一行环境指纹，写进每份数据文件，避免数字脱离硬件。"""
    import platform

    where = f"port {port}" if port else f"base_url {BASE_URL}"
    return f"{platform.platform()} | python {platform.python_version()} | {where}"
