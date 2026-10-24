#!/usr/bin/env bash
# 下载第五部分需要的两个本地模型，并校验 sha256。
#
# 用法：
#   labs/05-rag/download-models.sh
#   FDE_MODEL_DIR=/data/models labs/05-rag/download-models.sh
#
# 说明：
# - 默认下载到 ~/.cache/fde-handbook/models，与 labs/05-rag/config.yaml 一致。
# - 需要联网；国内网络可先 export HTTPS_PROXY=...。
# - 模型不入库（见 .gitignore 的 *.gguf）。
set -euo pipefail

DIR="${FDE_MODEL_DIR:-$HOME/.cache/fde-handbook/models}"
mkdir -p "$DIR"

fetch() {
  local url="$1" name="$2" want="$3"
  local dest="$DIR/$name"
  if [ -f "$dest" ] && [ "$(shasum -a 256 "$dest" | cut -d' ' -f1)" = "$want" ]; then
    echo "已存在且校验通过：$name"
    return
  fi
  echo "下载 $name"
  curl -L --fail --retry 3 -o "$dest.part" "$url"
  local got
  got="$(shasum -a 256 "$dest.part" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "sha256 不匹配：$name" >&2
    echo "  期望 $want" >&2
    echo "  实际 $got" >&2
    rm -f "$dest.part"
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "完成：$name"
}

fetch \
  "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  "Qwen3-Embedding-0.6B-Q8_0.gguf" \
  "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"

fetch \
  "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf" \
  "bge-reranker-v2-m3-Q8_0.gguf" \
  "a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3"

echo
echo "模型目录：$DIR"
echo "下一步：labs/05-rag/servers.sh start"
