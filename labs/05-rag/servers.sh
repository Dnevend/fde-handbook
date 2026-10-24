#!/usr/bin/env bash
# 起停第五部分用到的三个 llama.cpp 服务。
#
#   labs/05-rag/servers.sh start     # 嵌入(:8081)、重排(:8082)、生成(:8080)
#   labs/05-rag/servers.sh stop
#   labs/05-rag/servers.sh status
#
# 生成服务的上下文由 CTX 控制，默认 8192；做 5.1 的长上下文实验时用 32768：
#   CTX=32768 labs/05-rag/servers.sh start llm
#
# 端口与模型文件名来自 labs/05-rag/config.yaml，改这里之前先改 config。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${FDE_MODEL_DIR:-$HOME/.cache/fde-handbook/models}"
EMBED="$MODEL_DIR/Qwen3-Embedding-0.6B-Q8_0.gguf"
RERANK="$MODEL_DIR/bge-reranker-v2-m3-Q8_0.gguf"
LOG_DIR="$ROOT/labs/05-rag/.cache/logs"
CTX="${CTX:-8192}"
mkdir -p "$LOG_DIR"

healthy() { curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

start_embed() {
  healthy 8081 && { echo "embedding 已在 8081"; return; }
  [ -f "$EMBED" ] || { echo "缺 $EMBED，先跑 download-models.sh" >&2; exit 1; }
  nohup llama-server -m "$EMBED" --host 127.0.0.1 --port 8081 \
    --embeddings --pooling last -c 4096 -b 2048 -ub 2048 -ngl 99 \
    --alias qwen3-embedding >"$LOG_DIR/embedding.log" 2>&1 &
  echo "embedding 启动中（:8081）"
}

start_rerank() {
  healthy 8082 && { echo "reranker 已在 8082"; return; }
  [ -f "$RERANK" ] || { echo "缺 $RERANK，先跑 download-models.sh" >&2; exit 1; }
  nohup llama-server -m "$RERANK" --host 127.0.0.1 --port 8082 \
    --reranking -c 2048 -b 2048 -ub 2048 -ngl 99 \
    --alias bge-reranker >"$LOG_DIR/reranker.log" 2>&1 &
  echo "reranker 启动中（:8082）"
}

start_llm() {
  healthy 8080 && { echo "llm 已在 8080"; return; }
  source "$ROOT/labs/env.sh" >/dev/null || { echo "找不到生成模型，先按 0.3 节准备" >&2; exit 1; }
  nohup llama-server -m "$MODEL" --host 127.0.0.1 --port 8080 \
    -c "$CTX" -ngl 99 --jinja --alias qwen3.5-9b >"$LOG_DIR/llm.log" 2>&1 &
  echo "llm 启动中（:8080，ctx=$CTX）"
}

wait_all() {
  for port in 8081 8082 8080; do
    for _ in $(seq 1 240); do
      healthy "$port" && break
      sleep 1
    done
    healthy "$port" && echo "  :$port ok" || echo "  :$port 未就绪，看 $LOG_DIR/" >&2
  done
}

case "${1:-}" in
  start)
    if [ -n "${2:-}" ]; then
      "start_$2"
    else
      start_embed; start_rerank; start_llm; wait_all
    fi
    ;;
  stop)
    pkill -f "llama-server" || true
    echo "已停止所有 llama-server"
    ;;
  status)
    for pair in "8081 embedding" "8082 reranker" "8080 llm"; do
      set -- $pair
      if healthy "$1"; then echo ":$1 $2 ok"; else echo ":$1 $2 down"; fi
    done
    ;;
  *)
    echo "用法：$0 {start|stop|status} [embedding|rerank|llm]" >&2
    exit 1
    ;;
esac
