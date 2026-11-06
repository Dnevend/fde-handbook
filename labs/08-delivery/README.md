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
