#!/usr/bin/env bash
# 删掉测量文件里的本机绝对路径，只保留文件名。
#
# 用法：
#   bash benchmarks/sanitize-paths.sh benchmarks/**/*.csv benchmarks/**/*.txt
#
# 为什么需要：llama-bench 的 CSV 会把模型完整路径写进 model_filename 列，
# 进程输出里也可能带路径。这些路径对你没用，对别人是隐私，而且会让仓库
# 的"不写本机路径"约定破功。正文引用的数字不受影响。
set -eu

if [ "$#" -eq 0 ]; then
  echo "用法: bash benchmarks/sanitize-paths.sh <文件>..." >&2
  exit 1
fi

python3 - "$@" <<'PY'
import os
import re
import sys

# 匹配以 / 开头、连续多级的路径前缀，替换为空，只留下文件名。
pattern = re.compile(r'(?<![\w/])/(?:[^/\s"]+/)+')

for path in sys.argv[1:]:
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        text = f.read()
    new = pattern.sub("", text)
    if new != text:
        with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(new)
        print(f"已清理 {path}")
PY
