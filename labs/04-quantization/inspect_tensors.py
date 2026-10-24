#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf>=0.10", "numpy>=2"]
# ///
"""读本地 GGUF 的真实权重，量出各数值格式的舍入误差、分组量化的误差，以及离群通道。

产出（默认 benchmarks/quantization/）：
  tensor-types.csv    每种量化类型的张量个数、参数量、真实位宽
  format-error.csv    同一批权重在 fp16/bf16/int8/int4 下的舍入误差
  group-size.csv      4/8 位、不同分组大小、对称与非对称的归一化 RMSE
  outliers.csv        权重矩阵逐输入通道的 absmax 分布（离群通道有多离谱）

用法：
  source labs/env.sh
  uv run labs/04-quantization/inspect_tensors.py            # 用 $MODEL
  uv run labs/04-quantization/inspect_tensors.py --model x.gguf --rows 64

口径提醒：参考值是**反量化后的 Q4 权重**，不是训练时的 fp32。fp16/bf16 的误差
比 Q4 小两三个数量级，所以拿它当参照不影响格式间的比较；但正文里必须写明这一点。
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

# 采样张量：覆盖三类结构——注意力的 qkv、FFN 的 down、以及词嵌入。
SAMPLE_PATTERNS = ("attn_qkv.weight", "ffn_down.weight", "ffn_up.weight", "token_embd.weight")


def to_bf16(x: np.ndarray) -> np.ndarray:
    """fp32 → bf16（保留 8 位尾数），按就近舍入到偶数。"""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.view(np.float32).astype(np.float32)


def quantize_symmetric(x: np.ndarray, bits: int, group: int | None) -> np.ndarray:
    """对称仿射量化：q = clip(round(x / s), -2^(b-1)+1, 2^(b-1)-1)，反量化 q * s。"""
    qmax = 2 ** (bits - 1) - 1
    if group is None:
        # 逐张量：一个 scale 管全部权重
        s = np.max(np.abs(x)) / qmax
        s = np.where(s == 0, 1.0, s)
        return np.clip(np.round(x / s), -qmax - 1, qmax) * s

    flat = x.reshape(-1)
    pad = (-flat.size) % group
    padded = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = padded.reshape(-1, group)
    s = np.max(np.abs(blocks), axis=1, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.round(blocks / s), -qmax - 1, qmax)
    return (q * s).reshape(-1)[: flat.size].reshape(x.shape)


def quantize_asymmetric(x: np.ndarray, bits: int, group: int) -> np.ndarray:
    """非对称仿射量化：s = (max - min) / (2^b - 1)，zero-point 让 0 落在整数格上。"""
    qmax = 2**bits - 1
    flat = x.reshape(-1)
    pad = (-flat.size) % group
    padded = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = padded.reshape(-1, group)
    lo, hi = blocks.min(axis=1, keepdims=True), blocks.max(axis=1, keepdims=True)
    s = (hi - lo) / qmax
    s = np.where(s == 0, 1.0, s)
    z = np.round(-lo / s)
    q = np.clip(np.round(blocks / s) + z, 0, qmax)
    return ((q - z) * s).reshape(-1)[: flat.size].reshape(x.shape)


def metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    diff = got.astype(np.float64) - ref.astype(np.float64)
    scale = ref.astype(np.float64).std() or 1.0
    return {
        "rmse_norm": float(np.sqrt(np.mean(diff**2)) / scale),
        "max_abs_err": float(np.max(np.abs(diff))),
        "mean_abs_err": float(np.mean(np.abs(diff))),
    }


def load_rows(reader: GGUFReader, name: str, rows: int) -> np.ndarray:
    tensor = next(t for t in reader.tensors if t.name == name)
    raw = tensor.data[:rows]
    w = dequantize(raw, tensor.tensor_type)
    return np.asarray(w, dtype=np.float32).reshape(-1, w.shape[-1])


def pick_tensors(reader: GGUFReader, rows: int) -> dict[str, np.ndarray]:
    picked: dict[str, np.ndarray] = {}
    names = [t.name for t in reader.tensors]
    for pattern in SAMPLE_PATTERNS:
        candidates = [n for n in names if n.endswith(pattern)]
        if not candidates:
            continue
        # 取中间层，避开第一层（embedding 附近的分布偏）和最后一层
        name = candidates[len(candidates) // 2]
        picked[name] = load_rows(reader, name, rows)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--out", default="benchmarks/quantization")
    ap.add_argument("--rows", type=int, default=64, help="每个采样张量取前多少行")
    args = ap.parse_args()

    if not args.model:
        print("缺少 --model，先 source labs/env.sh", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(args.model)

    # 1. 类型分布与真实位宽（位宽 = 张量字节数 × 8 / 参数量）
    with (out / "tensor-types.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "tensors", "params", "bytes", "bits_per_weight"])
        types: dict[str, list[int]] = {}
        for t in reader.tensors:
            n = int(np.prod(t.shape))
            row = types.setdefault(str(t.tensor_type.name), [0, 0, 0])
            row[0] += 1
            row[1] += n
            row[2] += t.data.nbytes
        for name, (count, params, nbytes) in sorted(types.items(), key=lambda kv: -kv[1][1]):
            w.writerow([name, count, params, nbytes, f"{nbytes * 8 / params:.4f}"])

    samples = pick_tensors(reader, args.rows)
    print(f"采样张量：{', '.join(samples)}")

    # 2. 各格式的舍入误差
    with (out / "format-error.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tensor", "format", "rmse_norm", "max_abs_err", "mean_abs_err"])
        for name, ref in samples.items():
            variants = {
                "fp16": ref.astype(np.float16).astype(np.float32),
                "bf16": to_bf16(ref),
                "int8-per-tensor": quantize_symmetric(ref, 8, None),
                "int4-per-tensor": quantize_symmetric(ref, 4, None),
                "int4-g32": quantize_symmetric(ref, 4, 32),
            }
            for fmt, got in variants.items():
                m = metrics(ref, got)
                w.writerow([name, fmt, f"{m['rmse_norm']:.6f}", f"{m['max_abs_err']:.6e}", f"{m['mean_abs_err']:.6e}"])

    # 3. 分组大小 vs 误差（取一个 2D 权重，逐行分块）
    name, ref = next(iter(samples.items()))
    with (out / "group-size.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tensor", "bits", "group", "scheme", "rmse_norm", "max_abs_err"])
        for bits in (4, 8):
            for group in (16, 32, 64, 128, 256, 512, 1024):
                m = metrics(ref, quantize_symmetric(ref, bits, group))
                w.writerow([name, bits, group, "symmetric", f"{m['rmse_norm']:.6f}", f"{m['max_abs_err']:.6e}"])
            m = metrics(ref, quantize_symmetric(ref, bits, None))
            w.writerow([name, bits, "per-tensor", "symmetric", f"{m['rmse_norm']:.6f}", f"{m['max_abs_err']:.6e}"])
            for group in (32, 128):
                m = metrics(ref, quantize_asymmetric(ref, bits, group))
                w.writerow([name, bits, group, "asymmetric", f"{m['rmse_norm']:.6f}", f"{m['max_abs_err']:.6e}"])

    # 4. 离群通道：逐列 absmax 与中位数之比
    with (out / "outliers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tensor", "channels", "absmax_max", "absmax_median", "ratio", "p99_over_median"])
        for name, ref in samples.items():
            col = np.max(np.abs(ref), axis=0)
            med = float(np.median(col))
            w.writerow(
                [
                    name,
                    col.size,
                    f"{float(col.max()):.6f}",
                    f"{med:.6f}",
                    f"{float(col.max() / med):.1f}",
                    f"{float(np.percentile(col, 99) / med):.1f}",
                ]
            )

    print(f"写出 {out}/tensor-types.csv、format-error.csv、group-size.csv、outliers.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
