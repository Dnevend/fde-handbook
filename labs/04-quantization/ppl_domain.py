#!/usr/bin/env python3
"""同一模型、同样参数，换语料域时的困惑度。

用来证明 4.6 的一个主张：困惑度没有跨语料的可比性，只有"同一段文本上，
两个模型（或两档量化）谁更低"才有意义。

用法：
  source labs/env.sh
  uv run labs/04-quantization/ppl_domain.py
"""

from __future__ import annotations

import csv
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PPL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")

DOMAINS = [
    ("政府文件", "corpus/raw/gov-ledger-guide/full.txt"),
    ("事故报告", "corpus/raw/acc-fangsu-0518/full.txt"),
    ("招股书", "corpus/raw/ipo-borch/full.txt"),
    ("专利", "corpus/raw/pat-inj-01/full.txt"),
    ("合成工单", "corpus/raw/synthetic"),
]


def collect(path: str, limit: int) -> str:
    p = ROOT / path
    if p.is_dir():
        text = "\n".join(f.read_text(encoding="utf-8") for f in sorted(p.rglob("*.md")))
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
    return text[:limit]


def main() -> int:
    model = os.environ.get("MODEL")
    if not model:
        print("先 source labs/env.sh", file=sys.stderr)
        return 1

    out = ROOT / "benchmarks" / "quantization" / "ppl-domain.csv"
    rows = []
    for label, path in DOMAINS:
        text = collect(path, 40000)
        tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"fde-ppl-{label}.txt"
        tmp.write_text(text, encoding="utf-8")
        proc = subprocess.run(
            ["llama-perplexity", "-m", model, "-f", str(tmp), "-c", "512", "--chunks", "4", "-fa", "on"],
            capture_output=True,
            text=True,
        )
        m = PPL.search(proc.stdout + proc.stderr)
        if not m:
            print(f"{label}: 解析失败", file=sys.stderr)
            continue
        rows.append([label, f"{float(m.group(1)):.4f}", f"{float(m.group(2)):.4f}", len(text), path])
        print(f"{label:<8} PPL={m.group(1)} ±{m.group(2)}  样本 {len(text)} 字符")

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "ppl", "ppl_stderr", "chars", "source"])
        w.writerows(rows)
    print(f"写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
