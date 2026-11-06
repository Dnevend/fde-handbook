# labs/08-delivery

第八部分「工程化与交付」的实验代码。目标是回答一个问题：**换一个人来，照文档能不能把系统重装一遍？**

## 依赖

在 `labs/` 下安装（新增 `fastapi` / `uvicorn`，只用于 8.1 的服务壳）：

```bash
cd labs && uv sync
```

模型与索引沿用第五部分：

```bash
labs/05-rag/download-models.sh
corpus/fetch.sh
uv run corpus/synthetic/generate.py --clean --validate
labs/05-rag/servers.sh start          # embedding :8081、rerank :8082、llm :8080
```

## 8.1 服务

```bash
cd labs
uv run 08-delivery/service/app.py
# 默认 127.0.0.1:8090；环境变量见 service/app.py 顶部
# 后端 served-model-name 用 BACKEND_MODEL 指定（vLLM 会校验，不能传 rag-service）
```

端点：

| 端点 | 用途 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 兼容；默认非流式（和 OpenAI 一致），`stream=true` 走 SSE |
| `GET /v1/models` | 客户端探测 |
| `GET /health` | 块数、模型、是否重排、权限过滤 |
| `POST /feedback` | 按 request_id 记录评分/评论 |
| `GET /metrics` | 最小 Prometheus 文本 |

试一下：

```bash
curl -s http://127.0.0.1:8090/health

curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"主要负责人对本单位安全生产工作负什么责任？"}],"stream":false,"max_tokens":128}'

curl -sN -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"应急预案多久演练一次？"}],"stream":true,"max_tokens":64}'
```

SSE 事件顺序：

```text
event: retrieval    # 检索到的 passages（先给前端展示出处）
data: {...}

data: {...}         # OpenAI 兼容的 delta chunk，可被标准客户端消费
...

event: citations    # 回答里 [n] 对应的出处
data: {...}

event: done         # TTFT、总耗时、token 数
data: {...}

data: [DONE]
```

## 8.4 可观测性

服务默认把 JSON 日志写到 stderr；设置 `LOG_FILE` 后同时写文件。每个请求有 `X-Request-ID`，并落三份数据：

- `benchmarks/delivery/traces.jsonl`：请求级 span（retrieval、generation、TTFT、token），不记问题原文；
- `benchmarks/delivery/feedback.jsonl`：评分/评论；
- `benchmarks/delivery/audit.jsonl`：权限审计（用户、角色、允许文档数、引用文档 id）。

```bash
cd labs
LOG_FILE=../benchmarks/delivery/service.log uv run 08-delivery/service/app.py
# 另开终端发几个请求后（在仓库根目录）：
uv run labs/08-delivery/observability/report.py
uv run labs/08-delivery/observability/report.py --json
```

`/metrics` 提供 `rag_requests_total`、`rag_errors_total`、`rag_ttft_seconds`、`rag_tpot_seconds`、`rag_retrieval_ms`、`rag_tokens_total`、`rag_feedback_total`。

## 8.5 安全与权限

`service/access.yaml` 是演示用 ACL：`alice`（internal）、`bob`（contractor）、`auditor`（全部）、未带 `X-User-ID` 的 anonymous（默认拒绝）。服务在检索前按用户过滤候选池，不是对 top-5 事后过滤。

```bash
# alice 能看法规，但看不到事故报告/客户资料
curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-User-ID: alice' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"主要负责人对本单位安全生产工作负什么责任？"}],"max_tokens":64}'

# anonymous 没有可见文档，应拒答
curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"应急预案多久演练一次？"}],"max_tokens":64}'
```

生产要接 SSO/OIDC、正式权限系统和审计存储；`X-User-ID` 只是实验里的身份入口。

## 8.7 交付物

`delivery/templates/` 有 8 份核心文档模板：README、架构、部署、运维、验收、培训、AI 使用规范、许可证。`delivery/check_delivery.py` 做交付前自检：文件、关键小节、未填占位符、明显密钥。

```bash
# 模板自检（允许【占位符】）
python3 labs/08-delivery/delivery/check_delivery.py --allow-placeholders --allow-missing-manifest

# 真实交付包（占位符会失败）
python3 labs/08-delivery/delivery/check_delivery.py --dir dist/delivery-<客户>-<版本>
```

## 8.2 离线 bundle

```bash
# 在联网机器上构建；默认跳过模型拷贝（先验证结构）
bash labs/08-delivery/offline/bundle.sh --skip-models --skip-wheels

# 在 bundle 目录里做静态检查
bash dist/offline-*/verify.sh --mode check

# 代理黑洞（Mac 上可跑，近似断网）
bash dist/offline-*/verify.sh --mode proxy
```

真断网在 Linux 目标机上用 `--mode unshare`（需要 root）。详见 `offline/README.md`。

## 数据

- 反馈：`benchmarks/delivery/feedback.jsonl`（可能含用户数据，已 gitignore）
- 压测/验收数据：`benchmarks/delivery/`（后续 8.3 使用）
- 离线 bundle 产物：`dist/`（不入库）
