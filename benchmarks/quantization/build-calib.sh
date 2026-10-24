#!/usr/bin/env bash
# 生成本节实验用的困惑度语料（不入库，跑一次重建）。
#
# 为什么不入库：正文语料按 corpus/README.md 的规定只提交清单。这里拼出来的是
# 政府文件全文（可全文入库档）与合成工单，仍然只作为本地实验输入，不进 git。
#
# 用法：
#   benchmarks/quantization/build-calib.sh [输出路径]
#   uv run labs/04-quantization/kv_quant.py --text <输出路径>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/fde-ppl-calib.txt}"

: >"$OUT"
# 政府文件（full 档）+ 内部制度 + 合成工单。顺序固定，保证可复现。
cat "$ROOT/corpus/raw/gov-ledger-guide/full.txt" >>"$OUT"
cat "$ROOT/corpus/raw/inst-equip-repair/full.txt" >>"$OUT"
cat "$ROOT"/corpus/raw/synthetic/*/*.md >>"$OUT"

# 压掉空行，减少 tokenizer 的空转
sed -i '' -e '/^[[:space:]]*$/d' "$OUT" 2>/dev/null || sed -i -e '/^[[:space:]]*$/d' "$OUT"

echo "写出 $OUT"
wc -c "$OUT"
shasum -a 256 "$OUT"
