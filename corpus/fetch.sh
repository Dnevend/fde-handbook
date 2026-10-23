#!/usr/bin/env bash
# 语料抓取的入口，逻辑在 fetch.py。
#
# 只下载 redistributable: full 且 url 非空的条目；excerpt 与 none 只打印人工清单。
#
# 用法：
#   corpus/fetch.sh --list
#   corpus/fetch.sh --dry-run
#   corpus/fetch.sh
#   corpus/fetch.sh --only reg-mem-order-13
#
# 依赖：uv（fetch.py 自带 PEP 723 声明，首次运行会装 pyyaml）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "需要 uv：https://docs.astral.sh/uv/" >&2
  exit 1
fi

exec uv run --quiet "$ROOT/corpus/fetch.py" "$@"
