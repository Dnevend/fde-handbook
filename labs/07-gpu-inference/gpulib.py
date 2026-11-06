"""第七部分实验共用的工具：连 vLLM、流式计时、写 CSV、记录 GPU 环境。

只依赖标准库和 openai SDK。所有脚本共用同一个“首字延迟”定义，
否则各节的数字不可比：

  TTFT = 从请求发出到第一个非空 content/reasoning 分片到达
  TPOT = (总耗时 - TTFT) / (生成 token 数 - 1)

vLLM 的 OpenAI 兼容接口默认监听 127.0.0.1:8000，base_url 要带 /v1。
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "gpu"
DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b-awq")


@lru_cache(maxsize=16)
def client(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> "OpenAI":
    """返回指向 vLLM 的 SDK 客户端。本地服务不校验 key，占位值即可。

    用 lru_cache 复用连接池；否则每个请求新建客户端，TTFT 会把握手开销算进去。
    openai 延迟导入，这样只跑 monitor_gpu.py 时不需要装它。
    """
    from openai import OpenAI

    return OpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-local"),
        timeout=timeout,
    )


def wait_ready(base_url: str | None = None, timeout: float = 600.0) -> None:
    """轮询 /v1/models，直到服务就绪。"""
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}/models"
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        time.sleep(1.0)
    raise TimeoutError(f"vLLM 在 {timeout:.0f}s 内没有就绪：{url}（最后一次错误：{last}）")


def stream_once(
    prompt: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int | None = 42,
    enable_thinking: bool | None = False,
    extra_body: dict | None = None,
) -> dict:
    """发一次流式请求，返回计时、内容和 usage。

    返回字段：
      ttft_s       首字延迟（第一个非空 content/reasoning 分片）
      total_s      端到端耗时
      gen_tokens   生成 token 数（优先用 usage.completion_tokens）
      tok_per_s    gen_tokens / (total_s - ttft_s)
      content      正文
      reasoning    思考内容（关闭思考时为空）
      usage        prompt_tokens / completion_tokens
    """
    kwargs: dict = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        kwargs["seed"] = seed
    body: dict = dict(extra_body or {})
    if enable_thinking is not None:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking
    if body:
        kwargs["extra_body"] = body

    started = time.perf_counter()
    ttft: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    chunks = 0
    usage = None

    stream = client(base_url).chat.completions.create(**kwargs)
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        thought = getattr(delta, "reasoning_content", None)
        if text:
            if ttft is None:
                ttft = time.perf_counter() - started
            content.append(text)
            chunks += 1
        elif thought:
            if ttft is None:
                ttft = time.perf_counter() - started
            reasoning.append(thought)

    total = time.perf_counter() - started
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    gen_tokens = completion_tokens or chunks
    decode_s = max(total - (ttft or total), 1e-9)
    return {
        "ttft_s": round(ttft or total, 4),
        "total_s": round(total, 4),
        "gen_tokens": gen_tokens,
        "tok_per_s": round(gen_tokens / decode_s, 2) if gen_tokens else 0.0,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            "completion_tokens": completion_tokens,
        },
    }


def percentile(values: list[float], q: float) -> float:
    """最近秩百分位；空列表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def write_csv(path: Path, rows: list[dict]) -> None:
    """写 CSV；字段顺序取第一行。空结果不创建文件。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_endpoints(spec: str) -> list[tuple[str, str]]:
    """把 name=url,name=url 解析成列表；url 里可以带 /v1。"""
    endpoints = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"endpoint 要写成 name=url：{item}")
        name, url = item.split("=", 1)
        endpoints.append((name.strip(), url.strip()))
    if not endpoints:
        raise ValueError("至少给一个 --endpoints name=url")
    return endpoints


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    """只取 stdout；命令不存在或失败时返回空串，避免把 traceback 写进环境记录。"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (result.stdout or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _nvidia_query(fields: str) -> str:
    primary = _run([
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader",
    ])
    if primary:
        return primary
    return _run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ])


def gpu_snapshot() -> dict:
    """记录 GPU 查询、拓扑和详细状态；7.7 用 topo 判断是不是 PCIe。"""
    return {
        "query": _nvidia_query(
            "index,name,driver_version,memory.total,power.limit,clocks.max.memory"
        ),
        "topo": _run(["nvidia-smi", "topo", "-m"]),
        "detail": _run(["nvidia-smi", "-q", "-d", "MEMORY,POWER,TEMPERATURE,CLOCK"], timeout=30),
    }


def env_note(base_url: str | None = None, model: str | None = None) -> str:
    """返回一份可贴进数据文件头部的环境说明；不写本机绝对路径。"""
    lines = [
        "# 实验环境",
        "",
        "## 主机",
        f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 系统: {platform.platform()}",
        f"- Python: {sys.version.split()[0]}",
        "",
        "## GPU",
    ]
    query = _nvidia_query(
        "index,name,driver_version,memory.total,power.limit,clocks.max.memory"
    )
    if query:
        lines.extend(f"- {line}" for line in query.splitlines())
        cuda = _run(["nvidia-smi"])
        if "CUDA Version:" in cuda:
            version = cuda.split("CUDA Version:", 1)[1].strip().split()[0]
            lines.append(f"- 驱动支持的 CUDA 上限: {version}")
    else:
        lines.append("- nvidia-smi 不可用（不在 GPU 机器上）")

    lines.extend(["", "## 软件"])
    docker = _run(["docker", "--version"])
    lines.append(f"- Docker: {docker or '未安装'}")
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        lines.append(f"- nvcc: {nvcc.splitlines()[-1]}")
    else:
        lines.append("- nvcc: 未安装（容器内 CUDA 版本见镜像）")
    torch = _run([
        sys.executable, "-c",
        "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())",
    ])
    lines.append(f"- PyTorch: {torch or '未安装'}")
    vllm = _run([sys.executable, "-c", "import vllm; print(vllm.__version__)"])
    lines.append(f"- vLLM: {vllm or '未安装（容器内版本见启动日志）'}")

    lines.extend(["", "## 模型"])
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        path = Path(model_dir)
        lines.append(f"- 目录: {path.name}")
        if path.is_dir():
            total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            lines.append(f"- 大小: {total / 1024**3:.2f} GiB")
    else:
        lines.append("- MODEL_DIR 未设置")
    lines.append(f"- 服务: {base_url or DEFAULT_BASE_URL}")
    lines.append(f"- 模型名: {model or DEFAULT_MODEL}")
    return "\n".join(lines)


def have_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None
