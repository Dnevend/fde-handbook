# 定位本书主力模型（Qwen3.5-9B Q4_K_M），导出环境变量 MODEL。
#
# 用法（在仓库根目录，或用绝对路径）：
#   source labs/env.sh
#
# 说明：
# - 已经设置 MODEL 且文件存在时直接沿用，不覆盖。
# - 找不到文件时报错并给出下载命令，不会让后续命令以"模型加载失败"收场。
# - 兼容 bash 与 zsh，所以返回用 return，被直接执行时退回 exit。

if [ -n "${MODEL:-}" ] && [ -f "${MODEL}" ]; then
  echo "MODEL=$MODEL"
  return 0 2>/dev/null || exit 0
fi

_hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL=$(
  find "$_hf_home/hub/models--unsloth--Qwen3.5-9B-GGUF" \
    -name 'Qwen3.5-9B-Q4_K_M.gguf' 2>/dev/null | head -1
)

# 兜底：llama.cpp 的 LLAMA_CACHE 指向别处时也找一次
if [ -z "$MODEL" ] && [ -n "${LLAMA_CACHE:-}" ]; then
  MODEL=$(find "$LLAMA_CACHE" -name 'Qwen3.5-9B-Q4_K_M.gguf' 2>/dev/null | head -1)
fi

unset _hf_home

if [ -z "${MODEL:-}" ]; then
  echo "找不到 Qwen3.5-9B-Q4_K_M.gguf。" >&2
  echo "先按 0.3 节下载（首次需要联网，约 5.7 GB）：" >&2
  echo '  llama-cli -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M -st -p "你好" -n 32' >&2
  return 1 2>/dev/null || exit 1
fi

export MODEL
echo "MODEL=$MODEL"
