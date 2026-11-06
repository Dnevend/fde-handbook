#!/usr/bin/env bash
# 构建离线 bundle：代码 + 模型 + 语料/索引 + Python wheels + 校验清单。
#
# 用法（在仓库根目录）：
#   bash labs/08-delivery/offline/bundle.sh
#   bash labs/08-delivery/offline/bundle.sh --skip-models --skip-wheels
#   bash labs/08-delivery/offline/bundle.sh --out /tmp/fde-offline --dry-run
#
# 重要：wheels 和二进制必须和**目标机**的平台/架构/Python 版本一致。
# 在 Mac 上构建的 bundle 只能给同架构 Mac 用；给 Linux GPU 服务器构建时，
# 在 Linux x86_64 的联网机器上跑本脚本，或把这一步放进 CI。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$ROOT/dist/offline-$STAMP}"
MODELS_DIR="${FDE_MODEL_DIR:-$HOME/.cache/fde-handbook/models}"
LLM=""
SKIP_MODELS=0
SKIP_WHEELS=0
SKIP_CORPUS=0
DRY_RUN=0

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --llm) LLM="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-wheels) SKIP_WHEELS=1; shift ;;
    --skip-corpus) SKIP_CORPUS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$LLM" ]; then
  # labs/env.sh 会导出 ${MODEL}；找不到也不在这里报错，后面按 --skip-models 处理
  set +e
  # shellcheck disable=SC1091
  source "$ROOT/labs/env.sh" >/dev/null 2>&1
  LLM="${MODEL:-}"
  set -e
fi

echo "构建离线 bundle：$OUT"
if [ "$DRY_RUN" = "1" ]; then
  echo "  dry-run：只打印步骤，不复制文件"
fi

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "缺命令：$1" >&2; exit 2; }
}
need rsync
need python3

if [ ! -d "$ROOT/labs/05-rag" ]; then
  echo "找不到 labs/05-rag；请在仓库根目录运行。" >&2
  exit 2
fi

if [ "$SKIP_MODELS" = "0" ]; then
  EMBED="$MODELS_DIR/Qwen3-Embedding-0.6B-Q8_0.gguf"
  RERANK="$MODELS_DIR/bge-reranker-v2-m3-Q8_0.gguf"
  for file in "$EMBED" "$RERANK" "$LLM"; do
    if [ ! -f "$file" ]; then
      echo "缺模型：${file}（或加 --skip-models 只验证结构）" >&2
      exit 2
    fi
  done
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "  代码：labs/05-rag、labs/08-delivery、labs/env.sh"
  echo "  语料：corpus/sources.yaml、corpus/index.json、corpus/raw（skip=${SKIP_CORPUS}）"
  echo "  索引：labs/05-rag/.cache 里的 chunks/vectors"
  echo "  模型：${MODELS_DIR}（skip=${SKIP_MODELS}）"
  echo "  wheels：uv export + pip download（skip=${SKIP_WHEELS}）"
  echo "  输出：$OUT"
  exit 0
fi

mkdir -p "$OUT/labs" "$OUT/corpus" "$OUT/benchmarks" "$OUT/models"

# 代码：排除缓存和日志，保留脚本本身
rsync -a --exclude '.cache' --exclude '__pycache__' "$ROOT/labs/05-rag" "$OUT/labs/"
rsync -a --exclude '__pycache__' "$ROOT/labs/08-delivery" "$OUT/labs/"
cp "$ROOT/labs/env.sh" "$OUT/labs/"
cp "$ROOT/labs/pyproject.toml" "$ROOT/labs/uv.lock" "$OUT/labs/"

# 语料与索引：离线现场不能下载，必须带原文/节选
if [ "$SKIP_CORPUS" = "0" ]; then
  rsync -a "$ROOT/corpus/sources.yaml" "$ROOT/corpus/index.json" "$OUT/corpus/"
  if [ -d "$ROOT/corpus/raw" ]; then
    rsync -a "$ROOT/corpus/raw" "$OUT/corpus/"
  else
    echo "警告：corpus/raw 不存在，bundle 里没有语料原文。" >&2
  fi
  mkdir -p "$OUT/eval"
  cp "$ROOT/eval/qa.jsonl" "$OUT/eval/"
else
  echo "按 --skip-corpus 跳过语料。"
fi

# 已切好的块和向量：省去现场重跑切分/嵌入的时间（缺失时服务会重建）
mkdir -p "$OUT/labs/05-rag/.cache"
for name in chunks-structure-real.jsonl vectors-1024-real.npy hybrid-top20.json; do
  if [ -f "$ROOT/labs/05-rag/.cache/$name" ]; then
    cp "$ROOT/labs/05-rag/.cache/$name" "$OUT/labs/05-rag/.cache/"
  else
    echo "提示：缺 ${name}，现场首次启动会重建。" >&2
  fi
done

# 模型：拷贝并改成服务期望的文件名
if [ "$SKIP_MODELS" = "0" ]; then
  cp -L "$EMBED" "$OUT/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
  cp -L "$RERANK" "$OUT/models/bge-reranker-v2-m3-Q8_0.gguf"
  cp -L "$LLM" "$OUT/models/Qwen3.5-9B-Q4_K_M.gguf"
else
  cat > "$OUT/models/README.txt" <<'EOF'
本 bundle 使用 --skip-models 构建，没有模型文件。
现场部署前把以下文件放进本目录：
  Qwen3-Embedding-0.6B-Q8_0.gguf
  bge-reranker-v2-m3-Q8_0.gguf
  Qwen3.5-9B-Q4_K_M.gguf
EOF
fi

# Python wheels：在目标平台同架构/Python 版本的机器上构建
if [ "$SKIP_WHEELS" = "0" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "缺 uv；先安装，或加 --skip-wheels。" >&2
    exit 2
  fi
  uv export --project "$ROOT/labs" --format requirements.txt --no-hashes --no-emit-project \
    -o "$OUT/requirements.txt"
  python3 -m pip download --dest "$OUT/wheels" -r "$OUT/requirements.txt"
else
  cat > "$OUT/requirements.txt" <<'EOF'
# 使用 --skip-wheels 构建；现场需自行准备依赖，或把本文件替换成 uv export 的结果。
EOF
fi

cat > "$OUT/config.env" <<'EOF'
# 离线 bundle 的运行配置；按现场改端口/权限。
export FDE_MODEL_DIR="$BUNDLE/models"
export MODEL="$BUNDLE/models/Qwen3.5-9B-Q4_K_M.gguf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SERVICE_HOST=127.0.0.1
export SERVICE_PORT=8090
export BACKEND_MODEL=qwen3.5-9b
export RERANK=1
export ACCESS_POLICY="$BUNDLE/labs/08-delivery/service/access.yaml"
export LOG_FILE="$BUNDLE/benchmarks/delivery/service.log"
# export ALLOWED_DOCS=reg-safety-law,reg-mem-order-13
EOF

cat > "$OUT/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$BUNDLE/config.env"
cd "$BUNDLE"
PYTHON="$BUNDLE/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "没有 .venv；先跑 bash install.sh，或用 PYTHON=... 指定解释器。" >&2
  exit 2
fi
exec "$PYTHON" labs/08-delivery/service/app.py
EOF
chmod +x "$OUT/run.sh"

cp "$ROOT/labs/08-delivery/offline/verify.sh" "$OUT/verify.sh"
chmod +x "$OUT/verify.sh"

cat > "$OUT/install.sh" <<'EOF'
#!/usr/bin/env bash
# 离线安装：只用 bundle/wheels，不访问网络。
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
cd "$BUNDLE"
if [ ! -d wheels ] || [ -z "$(ls -A wheels 2>/dev/null)" ]; then
  echo "wheels/ 为空：这个 bundle 不能离线安装；请重建（不要 --skip-wheels）。" >&2
  exit 2
fi
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links wheels -r requirements.txt
if ! .venv/bin/python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "依赖安装后仍无法 import fastapi/uvicorn；检查 wheels 是否匹配目标平台。" >&2
  exit 2
fi
EOF
chmod +x "$OUT/install.sh"

cat > "$OUT/README.md" <<'EOF'
# 离线 bundle

1. 校验：`bash verify.sh --mode check`
2. 安装：`bash install.sh`（只用 `wheels/`）
3. 启动：`bash run.sh`
4. 断网验证：Linux 上 `bash verify.sh --mode unshare`（root）；Mac 上 `bash verify.sh --mode proxy`

模型在 `models/`，语料在 `corpus/`，索引缓存在 `labs/05-rag/.cache/`。
EOF

# 清单：逐文件 sha256，流式读，避免把大模型读进内存
SKIP_MODELS="$SKIP_MODELS" SKIP_WHEELS="$SKIP_WHEELS" SKIP_CORPUS="$SKIP_CORPUS" \
python3 - "$OUT" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys
import time

root = pathlib.Path(sys.argv[1])
meta = {
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "skip_models": bool(int(os.environ.get("SKIP_MODELS", "0"))),
    "skip_wheels": bool(int(os.environ.get("SKIP_WHEELS", "0"))),
    "skip_corpus": bool(int(os.environ.get("SKIP_CORPUS", "0"))),
}
(root / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = []
count = 0
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "MANIFEST.sha256":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    lines.append(f"{digest.hexdigest()}  {path.relative_to(root)}")
    count += 1
(root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"清单：{count} 个文件 → MANIFEST.sha256")
PY

echo "完成：$OUT"
echo "下一步：bash \"$OUT/verify.sh\" --mode check"
