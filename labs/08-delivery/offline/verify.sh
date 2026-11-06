#!/usr/bin/env bash
# 验证离线 bundle：校验清单 → 检查必需文件 → 起服务 → 问一个已知问题 → 检查引用/拒答/指标。
#
# 用法：
#   bash verify.sh --mode check      # 只校验清单和文件（不启动服务）
#   bash verify.sh --mode proxy      # 代理黑洞，近似断网（Mac 可跑）
#   bash verify.sh --mode local      # 不隔离网络，只做功能冒烟
#   sudo bash verify.sh --mode unshare  # Linux 真断网（新 net namespace）
#
# 说明：unshare 会递归调用 --mode local；需要 root 和 iproute2。
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="check"
SKIP_MANIFEST=0
QUESTION="${QUESTION:-生产经营单位的主要负责人对本单位的安全生产工作负什么责任？}"
REFUSAL_QUESTION="${REFUSAL_QUESTION:-资料里有没有关于火星基地的规定？}"
TIMEOUT="${TIMEOUT:-180}"
LOG="${LOG:-$BUNDLE/verify.log}"

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --skip-manifest) SKIP_MANIFEST=1; shift ;;
    --question) QUESTION="$2"; shift 2 ;;
    --refusal-question) REFUSAL_QUESTION="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  check|proxy|local|unshare) ;;
  *) echo "--mode 只能是 check/proxy/local/unshare" >&2; exit 2 ;;
esac

if [ "$MODE" = "unshare" ]; then
  if [ "$(id -u)" -ne 0 ]; then
    echo "unshare 模式需要 root；用 sudo bash verify.sh --mode unshare" >&2
    exit 2
  fi
  exec unshare -n bash -c "
    ip link set lo up
    exec bash '$BUNDLE/verify.sh' --mode local --skip-manifest \
      --question '$QUESTION' --refusal-question '$REFUSAL_QUESTION' --timeout '$TIMEOUT'
  "
fi

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "PASS: $*"; }

for required in \
  config.env run.sh install.sh requirements.txt \
  labs/env.sh \
  labs/05-rag/raglib.py labs/05-rag/config.yaml \
  labs/05-rag/.cache/chunks-structure-real.jsonl \
  labs/08-delivery/service/app.py labs/08-delivery/service/rag_service.py \
  labs/08-delivery/service/access.py labs/08-delivery/service/access.yaml \
  labs/08-delivery/observability/report.py \
  corpus/sources.yaml; do
  [ -e "$BUNDLE/$required" ] || fail "缺 $required"
done
ok "必需文件齐全"

if [ "$SKIP_MANIFEST" = "0" ]; then
  python3 - "$BUNDLE" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = root / "MANIFEST.sha256"
if not manifest.exists():
    print("FAIL: 缺 MANIFEST.sha256", file=sys.stderr)
    raise SystemExit(1)
missing, bad = [], []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    digest, rel = line.split("  ", 1)
    path = root / rel
    if not path.is_file():
        missing.append(rel)
        continue
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != digest:
        bad.append(rel)
if missing:
    print("FAIL: 清单里有缺失文件：" + ", ".join(missing[:5]), file=sys.stderr)
    raise SystemExit(1)
if bad:
    print("FAIL: 校验不通过：" + ", ".join(bad[:5]), file=sys.stderr)
    raise SystemExit(1)
print("PASS: MANIFEST.sha256 校验通过")
PY
else
  ok "按 --skip-manifest 跳过清单校验"
fi

if [ "$MODE" = "check" ]; then
  if [ -f "$BUNDLE/bundle.json" ]; then
    SKIP_WHEELS=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("skip_wheels", 0)))' "$BUNDLE/bundle.json")
    if [ "$SKIP_WHEELS" = "1" ]; then
      echo "WARN: 这个 bundle 是 --skip-wheels 构建的，不能离线安装；check 通过不代表可安装。" >&2
    fi
  fi
  ok "check 模式完成（未启动服务）"
  exit 0
fi

if [ ! -f "$BUNDLE/models/Qwen3.5-9B-Q4_K_M.gguf" ]; then
  fail "缺 models/Qwen3.5-9B-Q4_K_M.gguf；这个 bundle 只能做 check"
fi

PYTHON="${PYTHON:-$BUNDLE/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || true)"
fi
[ -n "$PYTHON" ] || fail "找不到 Python；先跑 bash install.sh"

export BUNDLE
# shellcheck disable=SC1091
source "$BUNDLE/config.env"
export SERVICE_WARMUP=1
export SERVICE_PORT="${VERIFY_PORT:-8091}"
if [ "$MODE" = "proxy" ]; then
  export HTTP_PROXY="http://127.0.0.1:9"
  export HTTPS_PROXY="http://127.0.0.1:9"
  export ALL_PROXY="http://127.0.0.1:9"
  export NO_PROXY="127.0.0.1,localhost"
  export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY" no_proxy="$NO_PROXY"
  ok "代理黑洞已启用（127.0.0.1:9），本机回环直连"
fi

cleanup() {
  [ -n "${SERVICE_PID:-}" ] && kill "$SERVICE_PID" 2>/dev/null || true
  # 只杀本 bundle 拉起的 llama-server（命令里含 bundle 的模型路径）
  pkill -f "$BUNDLE/models" 2>/dev/null || true
}
trap cleanup EXIT

"$PYTHON" "$BUNDLE/labs/08-delivery/service/app.py" >"$LOG" 2>&1 &
SERVICE_PID=$!
for _ in $(seq 1 "$TIMEOUT"); do
  if curl -sf -m 2 "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    tail -60 "$LOG" >&2 || true
    fail "服务启动失败"
  fi
  sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null || {
  tail -60 "$LOG" >&2 || true
  fail "服务 ${TIMEOUT}s 内没有就绪"
}
ok "服务已就绪：http://127.0.0.1:$SERVICE_PORT"

answer_json="$BUNDLE/verify-answer.json"
curl -sS -m 300 -X POST "http://127.0.0.1:$SERVICE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -H 'X-User-ID: auditor' \
  -d "{\"model\":\"rag-service\",\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}],\"stream\":false,\"max_tokens\":128}" \
  > "$answer_json"

python3 - "$answer_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
content = payload["choices"][0]["message"]["content"]
citations = payload.get("citations", [])
if not content.strip():
    raise SystemExit("FAIL: 回答为空")
if not citations:
    raise SystemExit("FAIL: 没有引用")
print(f"PASS: 回答 {len(content)} 字，引用 {len(citations)} 条")
PY

refusal_json="$BUNDLE/verify-refusal.json"
curl -sS -m 300 -X POST "http://127.0.0.1:$SERVICE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -H 'X-User-ID: anonymous' \
  -d "{\"model\":\"rag-service\",\"messages\":[{\"role\":\"user\",\"content\":\"$REFUSAL_QUESTION\"}],\"stream\":false,\"max_tokens\":64}" \
  > "$refusal_json"

python3 - "$refusal_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
content = payload["choices"][0]["message"]["content"]
if "资料中没有相关信息" not in content:
    raise SystemExit(f"FAIL: 拒答不符合预期：{content[:80]}")
print("PASS: 拒答正确")
PY

curl -sS -m 10 -X POST "http://127.0.0.1:$SERVICE_PORT/feedback" \
  -H 'Content-Type: application/json' -H 'X-User-ID: auditor' \
  -d '{"request_id":"verify","rating":5,"comment":"offline verify","tags":["verify"]}' >/dev/null
ok "反馈接口可用"

curl -sS -m 10 "http://127.0.0.1:$SERVICE_PORT/metrics" | grep -q "rag_requests_total" \
  || fail "metrics 缺 rag_requests_total"
ok "metrics 可用"

echo
echo "离线验证通过（mode=${MODE}）。日志：$LOG"
echo "注意：proxy 模式只黑洞 HTTP(S) 代理；真正的 net namespace 验证用 --mode unshare。"
