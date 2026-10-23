#!/usr/bin/env bash
# 把 figures/src/*.json 渲染成站点用的 SVG。
#
# 做三件事：
#   1. 调 fireworks-tech-graph 的 render（JSON 里带显式坐标）；
#   2. 把字体栈换成 fontconfig 认得的写法——否则 cairosvg 导 PNG 时中文全是豆腐块，
#      而 PNG 是这张图唯一的目视验收手段；
#   3. 跑 check（文字截断、标注重叠、连线弯折、构图）。
#
# 用法：
#   figures/render.sh            # 全部
#   figures/render.sh 1-3 2-3    # 只处理名字以这些前缀开头的图
#   figures/render.sh --png      # 额外导出 1200px 宽的审阅图到 figures/out/preview/
#
# 依赖：python3；skill 装在 $HOME/.agents/skills/fireworks-tech-graph，
# 换位置时用 FIREWORKS_GRAPH_SKILL 覆盖。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_ROOT="${FIREWORKS_GRAPH_SKILL:-$HOME/.agents/skills/fireworks-tech-graph}"
CLI="$SKILL_ROOT/scripts/fireworks.py"
SRC_DIR="$ROOT/figures/src"
OUT_DIR="$ROOT/docs/public/figures"
REPORT_DIR="$ROOT/figures/out/report"
PREVIEW_DIR="$ROOT/figures/out/preview"

# 拉丁字体在前、CJK 字体第二段，且第一段必须是 cairosvg 能解析到的字体，
# 否则导出 PNG 时整段中文退化成方框。Hiragino Sans GB / Noto Sans CJK 覆盖 macOS 与 Linux。
FONT_STACK="'Hiragino Sans GB', 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif"

want_png=0
filters=()
for arg in "$@"; do
  case "$arg" in
    --png) want_png=1 ;;
    *) filters+=("$arg") ;;
  esac
done

if [ ! -f "$CLI" ]; then
  echo "找不到 fireworks-tech-graph：$CLI" >&2
  echo "设置 FIREWORKS_GRAPH_SKILL 指向 skill 目录后再试。" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$REPORT_DIR"
if [ "$want_png" = "1" ]; then mkdir -p "$PREVIEW_DIR"; fi

shopt -s nullglob
sources=("$SRC_DIR"/*.json)
if [ "${#sources[@]}" -eq 0 ]; then
  echo "figures/src/ 下没有 JSON" >&2
  exit 1
fi

selected=()
for src in "${sources[@]}"; do
  name="$(basename "$src" .json)"
  if [ "${#filters[@]}" -gt 0 ]; then
    hit=0
    for prefix in "${filters[@]}"; do
      case "$name" in "$prefix"*) hit=1 ;; esac
    done
    [ "$hit" = "1" ] || continue
  fi
  selected+=("$src")
done

if [ "${#selected[@]}" -eq 0 ]; then
  echo "没有匹配的图源" >&2
  exit 1
fi

for src in "${selected[@]}"; do
  name="$(basename "$src" .json)"
  mode="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("template_type") or d.get("mode"))' "$src")"
  svg="$OUT_DIR/$name.svg"

  echo "== $name ($mode)"
  python3 "$CLI" render "$mode" "$src" "$svg" --report "$REPORT_DIR/$name.json" >/dev/null

  python3 - "$svg" "$FONT_STACK" <<'PY'
import re, sys
path, stack = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
new, count = re.subn(r"font-family:[^;]+;", f"font-family: {stack};", text, count=1)
if count != 1:
    raise SystemExit("字体栈修补失败：没找到唯一的 font-family 声明")

# 正文栏宽 688px，960 宽的画布会缩到 0.72 倍；标题类文字按缩小后的可读尺寸反推。
for selector, size in ((".title", 24), (".subtitle", 15)):
    replacement = rf"({re.escape(selector)} \{{ font-size: )[\d.]+px"
    new, hits = re.subn(replacement, rf"\g<1>{size}px", new, count=1)
    if hits != 1:
        raise SystemExit(f"字号修补失败：{selector}")

# 没有任何文字的节点（括号端头、坐标轴这类纯图形）会被生成器塞一个省略号，
# 2px 宽的矩形上就会出现三个小点。空标签直接删掉。
new, _ = re.subn(r'\s*<text[^>]*data-full-text=""[^>]*>[^<]*</text>', "", new)

open(path, "w", encoding="utf-8").write(new)
PY

  python3 "$CLI" check "$svg" | python3 -c '
import json, sys
report = json.load(sys.stdin)
if not report.get("ok"):
    print(json.dumps(report, ensure_ascii=False, indent=1))
    raise SystemExit(1)
score = report["checks"]["composition"].get("score")
print("   check ok" + (f" · 构图分 {score}" if score is not None else ""))
'

  if [ "$want_png" = "1" ]; then
    python3 "$CLI" export-png "$svg" "$PREVIEW_DIR/$name.png" --width 1200 >/dev/null
    echo "   审阅图 figures/out/preview/$name.png"
  fi
done
