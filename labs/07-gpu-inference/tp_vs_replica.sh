#!/usr/bin/env bash
# 7.7 双卡对照：TP=2（一个模型切两张卡）vs 2 个副本（每张卡一个完整模型）。
#
# 用法（2×4090 同机，仓库根目录）：
#   bash labs/07-gpu-inference/tp_vs_replica.sh \
#     --model-dir models/qwen3.5-9b-awq --model-name qwen3.5-9b-awq \
#     --concurrency 8 --requests 64
#
# 先干跑看命令：
#   bash labs/07-gpu-inference/tp_vs_replica.sh --dry-run
#
# 说明：
# - TP=2 用一个容器、两张卡；2 副本用两个容器、各一张卡。
# - 两组用同一个模型、同一批请求、同一个并发；只改并行方式。
# - 4090 没有 NVLink，TP=2 走 PCIe；结论只代表 PCIe 场景。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen3.5-9b-awq}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-9b-awq}"
CONCURRENCY=8
REQUESTS=64
MAX_TOKENS=128
PROMPT_CHARS=1200
OUT_DIR="$ROOT/benchmarks/gpu/multi-gpu"
DRY_RUN=0
NAME_PREFIX="part7"

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --prompt-chars) PROMPT_CHARS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --name-prefix) NAME_PREFIX="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  PY=(uv run)
else
  PY=(python3)
fi

TP_NAME="${NAME_PREFIX}-tp2"
R0_NAME="${NAME_PREFIX}-r0"
R1_NAME="${NAME_PREFIX}-r1"

cleanup() {
  bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --stop --name "$TP_NAME" >/dev/null 2>&1 || true
  bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --stop --name "$R0_NAME" >/dev/null 2>&1 || true
  bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --stop --name "$R1_NAME" >/dev/null 2>&1 || true
}

if [ "$DRY_RUN" = "1" ]; then
  echo "# TP=2"
  bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --dry-run --tp 2 --port 8001 --name "$TP_NAME" --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME"
  printf '%q ' "${PY[@]}" "$ROOT/labs/07-gpu-inference/bench_openai.py" \
    --base-url http://127.0.0.1:8001/v1 --concurrency "$CONCURRENCY" \
    --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" --prompt-chars "$PROMPT_CHARS" \
    --label tp2 --output "$OUT_DIR/tp2-summary.csv"
  printf '\n\n# 2 个副本'
  echo
  bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --dry-run --gpus device=0 --port 8002 --name "$R0_NAME" --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME"
  bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --dry-run --gpus device=1 --port 8003 --name "$R1_NAME" --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME"
  printf '%q ' "${PY[@]}" "$ROOT/labs/07-gpu-inference/bench_openai.py" \
    --base-urls http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1 \
    --concurrency "$CONCURRENCY" --requests "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" --prompt-chars "$PROMPT_CHARS" \
    --label replicas --output "$OUT_DIR/replicas-summary.csv"
  printf '\n'
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi 不可用：这个脚本只在 2 卡 NVIDIA 机器上跑。" >&2
  exit 2
fi
GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [ "$GPU_COUNT" -lt 2 ]; then
  echo "需要同一台机器上的 2 张 GPU，当前只有 $GPU_COUNT 张。" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
trap cleanup EXIT

echo "== TP=2：一个模型切两张卡"
bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" \
  --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME" \
  --tp 2 --port 8001 --name "$TP_NAME"
"${PY[@]}" "$ROOT/labs/07-gpu-inference/bench_openai.py" \
  --base-url http://127.0.0.1:8001/v1 \
  --concurrency "$CONCURRENCY" --requests "$REQUESTS" \
  --max-tokens "$MAX_TOKENS" --prompt-chars "$PROMPT_CHARS" \
  --label tp2 --output "$OUT_DIR/tp2-summary.csv"
bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" --stop --name "$TP_NAME"

echo
echo "== 2 个副本：每张卡一个完整模型"
bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" \
  --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME" \
  --gpus device=0 --port 8002 --name "$R0_NAME"
bash "$ROOT/labs/07-gpu-inference/serve_vllm.sh" \
  --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME" \
  --gpus device=1 --port 8003 --name "$R1_NAME"
"${PY[@]}" "$ROOT/labs/07-gpu-inference/bench_openai.py" \
  --base-urls http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1 \
  --concurrency "$CONCURRENCY" --requests "$REQUESTS" \
  --max-tokens "$MAX_TOKENS" --prompt-chars "$PROMPT_CHARS" \
  --label replicas --output "$OUT_DIR/replicas-summary.csv"

echo
echo "产物：$OUT_DIR/tp2-summary.csv、$OUT_DIR/replicas-summary.csv"
echo "读法：先看聚合吞吐，再看 TTFT p95 和单请求速度；TP=2 的通信开销会体现在单请求速度上。"
