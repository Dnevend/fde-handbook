#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch"]
# ///
"""7.1 硬件实测：HBM 拷贝带宽 + FP16 矩阵乘算力，和厂商规格对一下。

为什么用这两个测试：
- copy 测的是“读 + 写”大块显存，接近 decode 搬权重的访问模式；
- 4096³ 的 FP16 矩阵乘测的是 Tensor Core 的稳定吞吐，接近 prefill 的算力上限。

它们都不是 vLLM 的端到端性能；端到端看 7.4。这里只回答“卡是不是在正常状态”。

用法（GPU 机器上）：
    uv run labs/07-gpu-inference/gpu_probe.py \
      --spec-bandwidth 1008 --spec-tflops 165.2 \
      --output benchmarks/gpu/hardware/probe.json

    # 也可以进 vLLM 容器跑（容器里有 CUDA 版 torch；注意覆盖 entrypoint）
    docker run --rm --gpus all --entrypoint python \
      -v "$PWD:/work" -w /work vllm/vllm-openai:latest \
      labs/07-gpu-inference/gpu_probe.py --output benchmarks/gpu/hardware/probe.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch


def measure_copy(size_mb: int, iters: int) -> float:
    """HBM 拷贝带宽（GB/s）：读 + 写都算，所以乘 2。"""
    elements = size_mb * 1024 * 1024 // 2  # fp16
    src = torch.randn(elements, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    bytes_moved = 2 * src.numel() * src.element_size() * iters
    return bytes_moved / elapsed / 1e9


def measure_matmul(size: int, iters: int) -> float:
    """FP16 矩阵乘算力（TFLOPS）：2*M*N*K 是乘加 FLOPs。"""
    a = torch.randn(size, size, dtype=torch.float16, device="cuda")
    b = torch.randn(size, size, dtype=torch.float16, device="cuda")
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    flops = 2 * size**3 * iters
    return flops / elapsed / 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU 硬件实测：带宽与 FP16 算力")
    parser.add_argument("--size-mb", type=int, default=512, help="拷贝测试的单块大小（MB）")
    parser.add_argument("--matmul-size", type=int, default=4096, help="矩阵乘的边长")
    parser.add_argument("--iters", type=int, default=50, help="每个测试的迭代次数")
    parser.add_argument("--spec-bandwidth", type=float, default=0.0, help="厂商规格带宽 GB/s，用于算比值")
    parser.add_argument("--spec-tflops", type=float, default=0.0, help="厂商规格 FP16 稠密 TFLOPS，用于算比值")
    parser.add_argument("--output", default="benchmarks/gpu/hardware/probe.json", help="输出 JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() = False：这个脚本需要 NVIDIA GPU。")
        return 2

    props = torch.cuda.get_device_properties(0)
    copy_gbps = measure_copy(args.size_mb, args.iters)
    matmul_tflops = measure_matmul(args.matmul_size, args.iters)

    payload = {
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu": {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
            "memory_bus_width_bits": getattr(props, "memory_bus_width", None),
        },
        "measured": {
            "copy_gbps": round(copy_gbps, 1),
            "fp16_matmul_tflops": round(matmul_tflops, 1),
        },
        "spec": {
            "bandwidth_gbps": args.spec_bandwidth or None,
            "fp16_dense_tflops": args.spec_tflops or None,
        },
        "ratio": {
            "bandwidth": round(copy_gbps / args.spec_bandwidth, 3) if args.spec_bandwidth else None,
            "fp16_matmul": round(matmul_tflops / args.spec_tflops, 3) if args.spec_tflops else None,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"GPU: {props.name}（{props.total_memory / 1024**3:.1f} GB）")
    print(f"HBM 拷贝带宽: {copy_gbps:.1f} GB/s" + (
        f"（规格 {args.spec_bandwidth:.0f}，比值 {payload['ratio']['bandwidth']:.2f}）"
        if args.spec_bandwidth else ""
    ))
    print(f"FP16 矩阵乘: {matmul_tflops:.1f} TFLOPS" + (
        f"（规格 {args.spec_tflops:.1f}，比值 {payload['ratio']['fp16_matmul']:.2f}）"
        if args.spec_tflops else ""
    ))
    print(f"写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
