#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf>=0.10"]
# ///
"""KV Cache 量化对困惑度的影响。

在同一个模型、同一段文本上，换 `-ctk/-ctv` 与上下文长度，跑 llama-perplexity，
把 PPL、耗时和按元数据算出的 KV 体积写成 CSV。

用法：
  source labs/env.sh
  uv run labs/04-quantization/kv_quant.py --text /tmp/ppl-text.txt

注意：
- V 量化要求 flash attention，脚本固定带 `-fa on`；不带它 llama.cpp 会直接报错。
- 短上下文下 KV 量化的影响落在 PPL 的置信区间里，测不出差别是正常现象。
  要看代价必须把上下文拉到几千 token 以上，这也是本节要讲的结论之一。
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import re
import subprocess
import sys
import time

from gguf import GGUFReader

PPL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


def kv_bytes_per_token(reader: GGUFReader, kbits: float, vbits: float) -> float:
    """每个 token 的 KV 体积（字节）。

    Qwen3.5 是混合结构：每 full_attention_interval 层才有一层是全注意力，
    其余是线性注意力（状态固定大小，不随 token 增长）。所以只数全注意力层。
    """
    f = reader.fields
    layers = int(f["qwen35.block_count"].contents())
    interval = int(f["qwen35.full_attention_interval"].contents())
    n_full = layers // interval
    kv_heads = int(f["qwen35.attention.head_count_kv"].contents())
    k_len = int(f["qwen35.attention.key_length"].contents())
    v_len = int(f["qwen35.attention.value_length"].contents())
    return n_full * (kv_heads * k_len * kbits + kv_heads * v_len * vbits) / 8


def run(model: str, text: str, ctx: int, chunks: int, ctk: str, ctv: str) -> tuple[float, float, float, str]:
    cmd = [
        "llama-perplexity",
        "-m", model,
        "-f", text,
        "-c", str(ctx),
        "--chunks", str(chunks),
        "-fa", "on",
        "-ctk", ctk,
        "-ctv", ctv,
    ]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    out = proc.stdout + proc.stderr
    m = PPL.search(out)
    if not m:
        tail = "\n".join(out.strip().splitlines()[-6:])
        raise SystemExit(f"没解析到 PPL：{' '.join(cmd)}\n{tail}")
    # 记录可复现的命令，但不把本机绝对路径写进入库的 CSV
    shown = f"llama-perplexity -m $MODEL -f <calib.txt> -c {ctx} --chunks {chunks} -fa on -ctk {ctk} -ctv {ctv}"
    return float(m.group(1)), float(m.group(2)), elapsed, shown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="benchmarks/quantization/kv-quant.csv")
    args = ap.parse_args()

    if not args.model:
        print("缺少 --model，先 source labs/env.sh", file=sys.stderr)
        return 1

    reader = GGUFReader(args.model)
    configs = [
        ("f16", "f16", 16.0, 16.0),
        ("q8_0", "q8_0", 8.5, 8.5),
        ("q8_0", "q4_0", 8.5, 4.5),
        ("q4_0", "q4_0", 4.5, 4.5),
    ]
    contexts = [(512, 8), (16384, 1)]

    rows = []
    for ctx, chunks in contexts:
        for ctk, ctv, kbits, vbits in configs:
            ppl, err, elapsed, cmd = run(args.model, args.text, ctx, chunks, ctk, ctv)
            kv_mib = kv_bytes_per_token(reader, kbits, vbits) * ctx / 2**20
            rows.append([ctx, chunks, ctk, ctv, f"{ppl:.4f}", f"{err:.4f}", f"{kv_mib:.1f}", f"{elapsed:.1f}", cmd])
            print(f"ctx={ctx:<6} ctk={ctk:<5} ctv={ctv:<5} PPL={ppl:.4f} ±{err:.4f}  KV≈{kv_mib:.0f} MiB  {elapsed:.0f}s")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ctx", "chunks", "ctk", "ctv", "ppl", "ppl_stderr", "kv_mib_est", "seconds", "cmd"])
        w.writerows(rows)
    print(f"写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
