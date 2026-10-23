#!/usr/bin/env bash
# 打印实验环境的可复现信息：机器、内存、系统、引擎版本。
# 用法：
#   bash benchmarks/record-env.sh > benchmarks/<主题>/env.txt
#
# 每份测量数据开头都应有这份文件的内容，正文里的"实测"才有意义。
# 注意：模型只打印文件名与哈希前 12 位，不打印绝对路径——
# 仓库里不允许出现本机路径，别人也不需要知道你的家目录长什么样。
set -u

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
echo "## 模型"
if [ -n "${MODEL:-}" ] && [ -e "$MODEL" ]; then
  echo "- 文件: $(basename "$MODEL")"
  echo "- 大小: $(ls -lL "$MODEL" | awk '{printf "%.2f GiB", $5/1024/1024/1024}')"
  echo "- sha256: $(shasum -a 256 "$MODEL" | cut -c1-12)…"
else
  echo "- MODEL 未设置（先 source labs/env.sh）"
fi
