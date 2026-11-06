---
title: 服务架构与流式输出
---

# 服务架构与流式输出

> **本章要回答的问题**：一个能上线的 RAG 服务由哪些部件组成？流式输出在架构上要改哪里？

第五部分跑通的是“一条链路”：一个脚本从头到尾把检索、重排、生成串起来。上线要的是“一个服务”：有接口契约、有错误码、有并发、有超时、有日志、有反馈，还要能被别人部署。**Demo 和服务之间差的不是模型，是工程。**

这一节先给架构，再讲流式输出的协议和坑；最小实现放在 `labs/08-delivery/service/`，本机已经跑通。

## 一张架构图

```text
客户端（Web / IM / CLI）
   │  HTTPS + SSE
   ▼
API 网关（鉴权、限流、审计、TLS）
   ▼
RAG 编排（app.py）
   ├─ 检索：BM25 + 向量 + RRF → rerank → top-k
   ├─ 提示词：系统约束 + 资料 + 引用要求
   ├─ 生成：llama.cpp / vLLM 的 OpenAI 兼容流
   └─ 反馈/指标：feedback.jsonl、/metrics
   ▼
SSE：retrieval → delta* → citations → done → [DONE]
```

各部件的最小职责：

| 部件 | 职责 | 出错时 |
|---|---|---|
| 客户端 | 渲染流式文本、引用、停止/重试 | 断线重连；用 request_id 报障 |
| 网关 | 鉴权、限流、审计、TLS | 401/429/413；不要透传内部错误 |
| 编排 | 检索、重排、提示词、生成、引用 | 400/500 + request_id；不要泄漏堆栈 |
| 检索 | BM25 + 向量 + RRF | 降级到单路；返回空结果要能拒答 |
| 重排 | 交叉编码器 top-k | 超时降级为不重排 |
| 生成 | 流式 token | 超时/断流；保留已生成部分 |
| 存储 | 语料、索引、反馈 | 读写失败要告警；不要静默丢数据 |

这张图里没有“向量数据库”这个框，是因为本书的规模（几百到几万块）用本地索引就够；换成 Qdrant/Milvus 时，检索接口不变，编排层不用改。

## 请求生命周期

一个 `/v1/chat/completions` 请求按顺序做：

1. **鉴权与权限**：确认用户能看哪些文档；把 `allowed_doc_ids` 传给检索（8.5 展开）。
2. **参数校验**：`messages` 里必须有非空 user 消息；`max_tokens` 有上限；`stream` 只接受布尔值。
3. **问题归一化**：去首尾空白、限制长度；超长直接 413（最小实现默认 8000 字符，`MAX_QUESTION_CHARS` 可改），不要让它进模型。
4. **检索**：BM25 + 向量 → RRF → 候选 top-20；权限过滤在重排前做，避免把无权文档送给重排模型。
5. **重排**：交叉编码器取 top-5；超时降级为不重排。
6. **提示词**：系统约束 + 资料（带编号）+ 引用要求 + 拒答要求（和 5.8 一致）。
7. **生成**：流式读 token；记录 TTFT、总耗时、token 数。
8. **引用**：把回答里的 `[n]` 映射回 top-5 的出处；没有引用时给出全部候选或明确“无引用”。
9. **反馈**：客户端可以用 request_id 提交评分/评论；落盘供 8.4 的闭环使用。
10. **审计**：记录 who/what/when/request_id；不要记录完整敏感正文。

**顺序很重要**：权限过滤必须在重排和生成之前；引用必须在生成之后；审计要覆盖失败请求。

## 流式输出：SSE 还是 WebSocket

RAG 的流是**单向**的：服务端不断推 token，客户端偶尔发一个“停止”。这种场景用 SSE（Server-Sent Events）就够了（引用 [\[2\]](#ref-2)）：

- 基于 HTTP，网关、日志、鉴权都复用；
- 浏览器原生 `EventSource`，断线自动重连；
- 文本协议，调试时 `curl -N` 就能看。

WebSocket 适合双向、低延迟的交互（协同编辑、语音），RAG 不需要。**不要为了“看起来高级”上 WebSocket**；多一套协议就多一套运维。

流式响应的格式可以兼容 OpenAI 的 chunk（引用 [\[1\]](#ref-1)）：

```json
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"每年"},"finish_reason":null}]}
```

再在前后加自定义事件：

```text
event: retrieval
data: {"request_id":"...","passages":[...]}

data: {...}          # 标准 chunk，可直接被 OpenAI SDK 消费

event: citations
data: {"request_id":"...","citations":[...]}

event: done
data: {"request_id":"...","ttft_s":1.2,"total_s":8.4,"tokens":188}

data: [DONE]
```

标准客户端只认 `data:` 里的 chunk，会忽略 `event:` 自定义事件；需要引用/耗时的客户端再解析自定义事件。**兼容性和信息量可以同时要。**

## 流式的五个坑

**第一，代理缓冲。** Nginx 默认会缓冲响应，客户端看到的是“一次性吐完”。加 `proxy_buffering off;`，服务端加 `X-Accel-Buffering: no`。

**第二，首字延迟不是总耗时。** 用户感知的是 TTFT；总耗时影响的是“什么时候能复制答案”。两个指标都要记，别只报总耗时。

**第三，客户端断开。** 用户点“停止”或关页面，服务端要能取消生成，释放 KV/连接；否则 GPU 还在为一个没人看的回答烧钱。当前最小实现没有做真正的取消：Starlette 会关闭生成器，但同步生成器里的 llama-server 请求不保证立刻停；本机实测 `curl --max-time 1` 断开后，服务端仍把整段回答生成完（日志 `ok=false`，耗时继续增长）。生产要用异步客户端/后台任务，压测时看 GPU 是否还在跑。

**第四，错误发生在流中间。** HTTP 200 已经发出去了，后面出错只能发一个 `event: error`，不能改状态码。客户端要能处理“半截回答 + 错误”。

**第五，背压。** 生成速度远快于网络时，服务端要按客户端消费速度推，不能无限缓冲；否则内存涨、延迟涨。

## 引用与拒答

RAG 的引用不是“装饰”，是**可核查性**：

- 检索 top-5 带编号（`[1]`…`[5]`）传给模型；
- 提示词要求每个结论后标 `[n]`；
- 服务端把 `[n]` 映射回 `chunk_id`/`doc_id`/`clause_no`；
- 客户端点击引用能定位到原文。

拒答同样重要：资料里没有答案时，模型必须说“资料中没有相关信息”，而不是编。提示词里写死拒答句，服务端用字符串检查兜底；拒答率/误拒率进 6.x 的评估。

**引用和拒答都要在服务层可观测**：没有引用/没有拒答的请求，应该能被统计出来，而不是靠人抽查。

## 并发模型

最小实现用 FastAPI 的同步端点：每个请求在 threadpool 里跑，检索/重排/生成都是阻塞调用。这对单机、低并发够用；上 GPU 后要重新评估：

- **一个模型服务**：llama.cpp/vLLM 自己会批处理；应用层不要再开一堆并发生成。
- **CPU 密集**：BM25、重排、分词会占 CPU；不要放在 async 事件循环里。
- **超时与队列**：请求进来先排队，超过队列上限直接 429；不要无限接。
- **多 worker**：每个 worker 都会建一份索引/连接；内存翻倍。先用单 worker，压测后再扩。

8.3 会用压测把“能接多少并发”量出来，而不是拍脑袋。

## 最小实现

`labs/08-delivery/service/app.py` 提供：

- `POST /v1/chat/completions`：流式/非流式；
- `GET /v1/models`、`GET /health`；
- `POST /feedback`：按 request_id 落盘；
- `GET /metrics`：最小 Prometheus 文本。

它复用第五部分的 `raglib`/`retrieve`/`generate`：检索、重排、提示词的口径和 5.6/5.7/5.8 一致，不重新发明一套。HTTP 层用 FastAPI，流式响应用 Starlette 的 `StreamingResponse`（引用 [\[3\]](#ref-3)、[\[4\]](#ref-4)）。

对外模型名和后端 `served-model-name` 要做映射：客户端传 `rag-service`，服务端用 `BACKEND_MODEL`（默认 `qwen3.5-9b`）去调 llama.cpp/vLLM。vLLM 会校验这个名字，直接把 `rag-service` 传过去会 404。

本机（M4/24GB，llama.cpp 0.7.0）已经跑通：非流式回答带引用、流式事件顺序为 `retrieval → delta → citations → done → [DONE]`、语料外问题返回“资料中没有相关信息”。这些是冒烟结果，延迟数字等 GPU 验收再写进 8.3。

## 动手实验

::: tip 动手实验：把服务跑起来并检查流式契约
**假设**：非流式回答有引用；流式事件顺序正确；语料外问题能拒答；feedback 和 metrics 可用。

**命令**（先按 5.x 起三个 llama-server，再起服务）：

```bash
cd labs
uv run 08-delivery/service/app.py

# 另开终端
curl -s http://127.0.0.1:8090/health
curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"主要负责人对本单位安全生产工作负什么责任？"}],"stream":false,"max_tokens":128}'

curl -sN -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"资料里有没有关于火星基地的规定？"}],"stream":true,"max_tokens":32}' \
  | grep -E '^event:|^data: \[DONE\]'

# 断开测试：curl 1 秒后自己退出，观察服务端是否继续生成/日志
curl --max-time 1 -sN -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rag-service","messages":[{"role":"user","content":"把刚才的回答再详细展开一遍。"}],"stream":true,"max_tokens":256}' \
  | head -5
```

**记录**：request_id、TTFT、总耗时、token 数、引用条数、是否拒答、事件顺序，以及断开后服务端是否继续生成。数据存 `benchmarks/delivery/`。**不要只测 happy path**；至少测一次语料外拒答和一次客户端断开。
:::

## 工程含义

**接口契约比模型重要。** 客户端只依赖 `/v1/chat/completions` 的形状和 SSE 事件；换模型、换检索、换 GPU，契约不变。

**request_id 是排障的锚点。** 日志、反馈、指标、审计都用同一个 id 串起来；没有它，用户说“刚才那个回答不对”时你只能猜。

**流式要能停、能错、能重试。** 只实现“正常流”不够；断线、超时、半截错误、取消，才是上线后每天发生的事。

**先单 worker，再压测扩。** 多 worker 会重复建索引/连接；没量过之前不要开。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>OpenAI. *Chat Completions streaming*. https://platform.openai.com/docs/api-reference/chat/streaming
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>MDN. *Using server-sent events*. https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>FastAPI. *FastAPI documentation*. https://fastapi.tiangolo.com/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Starlette. *StreamingResponse*. https://www.starlette.io/responses/
