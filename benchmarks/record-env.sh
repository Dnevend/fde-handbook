#!/usr/bin/env bash
# 打印实验环境的可复现信息：机器、内存、系统、引擎版本。
# 用法：
#   bash benchmarks/record-env.sh > benchmarks/<主题>/env.txt
#
# 每份测量数据开头都应有这份文件的内容，正文里的"实测"才有意义。
# 注意：模型只打印文件名与哈希前 12 位，不打印绝对路径——
# 仓库里不允许出现本机路径，别人也不需要知道你的家目录长什么样。
set -u

hash_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -c1-12
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -c1-12
  else
    echo "无 sha256 工具"
  fi
}

print_model() {
  echo "## 模型"
  if [ -n "${MODEL:-}" ] && [ -e "$MODEL" ]; then
    echo "- 文件: $(basename "$MODEL")"
    echo "- 大小: $(ls -lL "$MODEL" | awk '{printf "%.2f GiB", $5/1024/1024/1024}')"
    echo "- sha256: $(hash_file "$MODEL")…"
  else
    echo "- MODEL 未设置（先 source labs/env.sh）"
  fi
}

case "$(uname -s)" in
  Darwin)
    echo "# 实验环境"
    echo
    echo "## 硬件"
    echo "- CPU: $(sysctl -n machdep.cpu.brand_string)"
    echo "- 核心数: $(sysctl -n hw.ncpu)"
    echo "- 内存: $(sysctl -n hw.memsize | awk '{printf "%.1f GB", $1/1024/1024/1024}')"
    echo
    echo "## 系统"
    echo "- $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "- 采集时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "## 推理引擎"
    if command -v llama-cli >/dev/null 2>&1; then
      llama-cli --version 2>&1 | sed 's/^/- /'
    else
      echo "- llama.cpp: 未安装（brew install llama.cpp）"
    fi
    if command -v python3 >/dev/null 2>&1; then
      echo "- Python: $(python3 --version 2>&1 | sed 's/^Python //')"
    fi
    echo
    print_model
    ;;

  Linux)
    echo "# 实验环境"
    echo
    echo "## 硬件"
    CPU=$(awk -F: '/model name/{gsub(/^ +/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null)
    echo "- CPU: ${CPU:-未知}"
    echo "- 核心数: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 未知)"
    MEM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)
    echo "- 内存: $(awk -v kb="${MEM_KB:-0}" 'BEGIN{printf "%.1f GB", kb/1024/1024}')"
    echo
    echo "## 系统"
    OS_NAME=$(. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-unknown}")
    echo "- ${OS_NAME:-未知}"
    echo "- 内核: $(uname -r)"
    echo "- 采集时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "## GPU"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit,clocks.max.memory \
        --format=csv,noheader 2>/dev/null | sed 's/^/- /'
      CUDA=$(nvidia-smi 2>/dev/null | awk -F'CUDA Version: ' '/CUDA Version/{print $2}' | awk '{print $1}')
      echo "- 驱动支持的 CUDA 上限: ${CUDA:-未知}"
      echo "- 拓扑:"
      nvidia-smi topo -m 2>/dev/null | sed 's/^/  /'
    else
      echo "- nvidia-smi: 未安装"
    fi
    echo
    echo "## 软件"
    if command -v python3 >/dev/null 2>&1; then
      echo "- Python: $(python3 --version 2>&1 | sed 's/^Python //')"
    else
      echo "- Python: 未安装"
    fi
    if command -v nvcc >/dev/null 2>&1; then
      echo "- nvcc: $(nvcc --version 2>&1 | tail -1)"
    else
      echo "- nvcc: 未安装（容器内 CUDA 版本见镜像）"
    fi
    if command -v docker >/dev/null 2>&1; then
      echo "- Docker: $(docker --version 2>&1)"
    else
      echo "- Docker: 未安装"
    fi
    if command -v vllm >/dev/null 2>&1; then
      echo "- vLLM: $(vllm --version 2>&1 | head -1)"
    else
      echo "- vLLM: 未安装（容器内版本见启动日志）"
    fi
    TORCH=$(python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>/dev/null)
    echo "- PyTorch: ${TORCH:-未安装}"
    echo
    print_model
    ;;

  *)
    echo "不支持的系统：$(uname -s)" >&2
    exit 1
    ;;
esac
