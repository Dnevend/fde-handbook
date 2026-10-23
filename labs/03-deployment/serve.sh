#!/usr/bin/env bash
# 起一个本地 OpenAI 兼容服务（llama-server）。
#
# 用法：
#   labs/03-deployment/serve.sh            # 默认 1 个槽位、8192 上下文、端口 8080
#   NP=4 labs/03-deployment/serve.sh       # 4 个槽位，做并发实验用
#   CTX=16384 PORT=8081 labs/03-deployment/serve.sh
#
# 参数：
#   NP    并发槽位数（-np），默认 1
#   CTX   上下文总量（-c），默认 8192。注意：显式设置 -np N 时，每个槽位分到 CTX/N
#   PORT  端口，默认 8080
#   ALIAS 模型别名，脚本里默认 qwen3.5-9b，客户端填什么都行
#
# 前台运行，Ctrl-C 停止。服务日志就是终端输出，不要重定向掉——出错时第一现场在这里。
set -euo pipefail

cd "$(dirname "$0")/../.."
source labs/env.sh

NP="${NP:-1}"
CTX="${CTX:-8192}"
PORT="${PORT:-8080}"
ALIAS="${ALIAS:-qwen3.5-9b}"

echo "模型: ${MODEL}"
echo "槽位: ${NP}  上下文: ${CTX}（每槽位 $((CTX / NP))）  端口: ${PORT}"
echo "客户端配置: OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1  OPENAI_API_KEY=sk-local"
echo

exec llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -c "$CTX" \
  -np "$NP" \
  -ngl 99 \
  --alias "$ALIAS" \
  --metrics \
  --jinja "$@"
