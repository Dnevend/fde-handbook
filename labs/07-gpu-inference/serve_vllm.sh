#!/usr/bin/env bash
# 启动 / 停止 vLLM 的 OpenAI 兼容服务（Docker）。
#
# 用法（在仓库根目录）：
#   bash labs/07-gpu-inference/serve_vllm.sh \
#     --model-dir models/qwen3.5-9b-awq --model-name qwen3.5-9b-awq
#
#   # 双卡张量并行
#   bash labs/07-gpu-inference/serve_vllm.sh --tp 2 --port 8001 --name vllm-tp2
#
#   # 两个副本：一个容器一张卡
#   bash labs/07-gpu-inference/serve_vllm.sh --gpus device=0 --port 8002 --name vllm-r0
#   bash labs/07-gpu-inference/serve_vllm.sh --gpus device=1 --port 8003 --name vllm-r1
#
#   # 停止
#   bash labs/07-gpu-inference/serve_vllm.sh --stop --name vllm-tp2
#
# 说明：
# - 模型目录是宿主机路径，容器内只读挂载到 /models；不要把密钥写进脚本。
# - 端口只绑 127.0.0.1，远程访问用 SSH 隧道，不要暴露公网。
# - 不同 vLLM 版本对 --enable-prefix-caching / --enable-chunked-prefill 的
#   默认值和 flag 名可能不同；先用 --dry-run 看命令，再按版本 --help 调整。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen3.5-9b-awq}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-9b-awq}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
GPUS="${GPUS:-all}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
DTYPE="${DTYPE:-auto}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-$MODEL_NAME}"
DRY_RUN=0
STOP=0

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --prefix-caching) PREFIX_CACHING=1; shift ;;
    --no-prefix-caching) PREFIX_CACHING=0; shift ;;
    --chunked-prefill) CHUNKED_PREFILL=1; shift ;;
    --no-chunked-prefill) CHUNKED_PREFILL=0; shift ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --name) CONTAINER_NAME="$2"; shift 2 ;;
    --stop) STOP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$STOP" = "1" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker 不可用。" >&2
    exit 2
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
    echo "已停止并删除容器：$CONTAINER_NAME"
  else
    echo "容器不存在：$CONTAINER_NAME"
  fi
  exit 0
fi

ARGS=(
  --model /models
  --served-model-name "$MODEL_NAME"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --dtype "$DTYPE"
  --disable-log-requests
)
if [ "$PREFIX_CACHING" = "1" ]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi
if [ "$CHUNKED_PREFILL" = "1" ]; then
  ARGS+=(--enable-chunked-prefill)
else
  ARGS+=(--no-enable-chunked-prefill)
fi

CMD=(
  docker run -d
  --name "$CONTAINER_NAME"
  --gpus "$GPUS"
  --ipc=host
  -p "127.0.0.1:$PORT:8000"
  -v "$MODEL_DIR:/models:ro"
  -e HF_HUB_OFFLINE=1
  "$IMAGE"
  "${ARGS[@]}"
)

if [ "$DRY_RUN" = "1" ]; then
  echo "# 模型目录：models/$(basename "$MODEL_DIR")"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker 不可用；先按附录 F 安装 Docker 和 nvidia-container-toolkit。" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi 不可用：这个脚本只在 NVIDIA GPU 机器上跑。" >&2
  exit 2
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "模型目录不完整：$MODEL_DIR（找不到 config.json）" >&2
  echo "先把权重下载到该目录，见附录 F。" >&2
  exit 2
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [ "$TP" -gt "$GPU_COUNT" ]; then
  echo "TP=$TP，但机器上只有 $GPU_COUNT 张 GPU。" >&2
  exit 2
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "容器 $CONTAINER_NAME 已存在；先 --stop，或换 --name。" >&2
  exit 1
fi

LOG_DIR="$ROOT/benchmarks/gpu/env"
mkdir -p "$LOG_DIR"
SAFE_CMD=("${CMD[@]}")
for i in "${!SAFE_CMD[@]}"; do
  if [ "${SAFE_CMD[$i]}" = "$MODEL_DIR:/models:ro" ]; then
    SAFE_CMD[$i]="models/$(basename "$MODEL_DIR"):/models:ro"
  fi
done
{
  echo "# vLLM 启动记录"
  echo "# 模型目录只写 basename，避免把本机路径带进仓库"
  echo "MODEL_DIR=models/$(basename "$MODEL_DIR")"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "PORT=$PORT TP=$TP GPUS=$GPUS"
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEM_UTIL=$GPU_MEM_UTIL"
  echo "MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  echo "PREFIX_CACHING=$PREFIX_CACHING CHUNKED_PREFILL=$CHUNKED_PREFILL"
  echo "IMAGE=$IMAGE"
  echo
  printf '%q ' "${SAFE_CMD[@]}"
  printf '\n'
} > "$LOG_DIR/${CONTAINER_NAME}-serve.txt"

echo "启动：${CONTAINER_NAME}（TP=$TP，端口 $PORT）"
"${CMD[@]}" >/dev/null

READY=0
for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "容器已退出，最后 100 行日志：" >&2
    docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "600 秒内没有就绪，最后 100 行日志：" >&2
  docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
  exit 1
fi

echo "就绪：http://127.0.0.1:$PORT/v1"
echo "模型：$MODEL_NAME；日志：docker logs -f $CONTAINER_NAME"
echo "停止：bash labs/07-gpu-inference/serve_vllm.sh --stop --name $CONTAINER_NAME"
