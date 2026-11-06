---
title: 可观测性：日志、链路追踪、反馈闭环
---

# 可观测性：日志、链路追踪、反馈闭环

> **本章要回答的问题**：上线后出了问题，日志和链路追踪要记到什么粒度才够用？

可观测性不是“多打日志”，而是**回答问题的能力**：

- 用户说“刚才那个回答不对”，你能不能凭一个 request_id 找到当时检索了什么、生成了什么、引用了哪条？
- 系统变慢，你能不能分清是检索、重排、排队，还是模型降频？
- 一批反馈进来，你能不能判断是数据问题、提示词问题，还是模型能力问题？

这一节给四类信号：日志、指标、追踪、反馈。RAG 还要额外记**检索/引用/拒答**，否则出了错只能看到“模型胡说”，看不到“资料没找到”。

## 四类信号

| 信号 | 回答什么问题 | 粒度 | 典型存储 |
|---|---|---|---|
| 日志 | 这次请求发生了什么？ | 请求/事件 | JSONL、日志系统 |
| 指标 | 整体趋势和告警？ | 分钟级聚合 | Prometheus |
| 追踪 | 时间花在哪个 span？ | 请求内 span | OTel/Jaeger |
| 反馈 | 用户觉得好不好？ | 请求 + 评分 | 反馈表/工单 |

四类缺一不可：只有日志，没法告警；只有指标，没法定位单次请求；只有追踪，没有业务质量；只有反馈，没有现场。

## 日志：结构化，但不记全文

最小实现把日志写成 JSON（引用 [\[1\]](#ref-1) 的 span/attribute 思路），一行一个事件：

```json
{"ts":"2026-11-07T01:06:18+0800","event":"chat","request_id":"9be818eb0f9d","user":"alice","ok":true,"total_s":8.33,"ttft_s":6.86,"tokens":117,"allowed_count":14}
```

建议字段：

| 字段 | 为什么 |
|---|---|
| `request_id` | 串起日志/追踪/反馈/审计 |
| `user` / `roles` | 权限与行为分析；不要记密码/密钥 |
| `event` / `span` | `http`、`retrieval`、`generation`、`error` |
| `ms` / `ttft_s` / `total_s` | 定位慢在哪 |
| `tokens` | 成本与吞吐 |
| `allowed_count` / `docs` | 权限与引用是否合理 |
| `question_len` / `question_sha256` | 关联问题，但不存原文 |
| `error` | 类型 + 简短消息；不要堆栈进日志正文 |

**不要记完整 prompt、完整文档、完整回答。** 它们可能含客户数据、个人信息、密钥；需要复现时用 `question_sha256` + 文档 id + 采样开关，而不是默认全存。

## 指标：RED + RAG 专用

通用部分是 RED：Rate（请求量）、Errors（错误率）、Duration（延迟）（引用 [\[3\]](#ref-3)）。RAG 再加四个：

- **检索耗时/召回**：retrieval p50/p95；命中率、空结果率（引擎侧的 TTFT/TPOT 见引用 [\[4\]](#ref-4)）。
- **生成质量代理**：引用率、拒答率、无引用回答数。
- **成本**：token/s、每百万 token 成本（7.9）。
- **反馈**：平均评分、低分率、按标签分布。

Prometheus 的文本格式和标签基数要控制（引用 [\[2\]](#ref-2)）：

```text
rag_requests_total 5
rag_errors_total 0
rag_ttft_seconds_sum 23.73
rag_ttft_seconds_count 4
rag_tpot_seconds_sum 0.26
rag_tpot_seconds_count 4
rag_retrieval_ms_sum 5169.8
rag_retrieval_ms_count 5
```

**不要给 `request_id`/`user_id` 做指标标签**：基数会爆炸。需要按用户排查时用日志/追踪，不要用 Prometheus。

## 追踪：请求内的 span

一次 RAG 请求拆成：

```text
request
├─ auth / acl          （可选）
├─ retrieval           BM25 + 向量 + RRF
├─ rerank              交叉编码器
├─ prompt              模板渲染
├─ generation          首 token / 逐 token
└─ citation             引用映射
```

最小实现把 `retrieval`/`generation` 两个 span 写进 JSON 日志（`log_event(event="span")`），每个请求另写一行聚合到 `benchmarks/delivery/traces.jsonl`；生产用 OpenTelemetry 的 span/attribute 模型（引用 [\[1\]](#ref-1)），接 Jaeger/Tempo。

追踪要能回答三个问题：

1. **慢在哪**：retrieval 1.2 s，generation 7 s，还是排队 10 s？
2. **为什么慢**：候选太多、重排模型慢、prompt 太长、KV 压力、降频？
3. **影响了谁**：哪些用户/哪些文档/哪些请求类型？

采样是必须的：全量追踪会拖垮存储；至少对错误、慢请求、低分反馈做**强制采样**。

## 反馈闭环

反馈不是“收集评分”，是**把用户信号变成下一轮改进**：

1. **收集**：request_id + 评分 + 可选评论/标签；不要强迫用户写长文。
2. **分诊**：按错误类型（检索没找到、引用错、答非所问、拒答过多、格式问题）分类。
3. **进测试集**：把确认的 bad case 补进 6.x 的测试集，标注出处和标准答案。
4. **回归**：改检索/提示词/模型后，用同一批题重跑；指标没变好就不上线。
5. **回访**：告诉用户“你反馈的问题已修复/已加入评估集”，否则下次没人愿意反馈。

这和第六部分的评估是同一个闭环：**反馈产生题，题驱动评估，评估决定上线**。没有分诊和回归，反馈表只是情绪收纳箱。

## 最小实现

`labs/08-delivery/service/app.py` 已经带：

- JSON 日志（stderr，可选 `LOG_FILE`）；
- `X-Request-ID` 响应头；
- `benchmarks/delivery/traces.jsonl` 请求追踪（含 `ok`/`refused`/`cancelled`，拒答和客户端断开分开统计）；
- `benchmarks/delivery/audit.jsonl` 权限审计（8.5）；
- `/metrics`：请求、错误、TTFT、TPOT、检索耗时、token、反馈；
- `POST /feedback` + `benchmarks/delivery/feedback.jsonl`。

`labs/08-delivery/observability/report.py` 把三份 JSONL 收成一页摘要：

```bash
uv run labs/08-delivery/observability/report.py
# 请求数、成功/失败/取消/拒答、用户分布、TTFT/TPOT p50/p95、Top 文档、反馈、审计
```

本机（M4/24GB，llama.cpp 0.7.0）用 5 个请求做了冒烟：`auditor` 能答事故题，`alice`/`bob`/`anonymous` 拒答；traces/audit 都带用户、`allowed_count`、引用文档 id；报告能汇总出用户分布和引用 Top 文档。证据在 `benchmarks/delivery/observability-smoke.json`；GPU/生产延迟待 8.3。

## 常见坑

**日志记全文。** 最危险的一种“可观测性”：把客户文档、问题、回答全打进日志，然后日志被同步到第三方。默认脱敏，采样才存原文。

**指标标签爆炸。** `user_id`、`request_id`、`doc_id` 不要做 Prometheus 标签；用日志/追踪查。

**只看平均延迟。** 平均 TTFT 会掩盖长尾；必须看 p95/p99，按请求类型/上下文长度分组。

**追踪没有采样策略。** 全量存不起，随机采样又会漏掉错误；错误/慢请求/低分反馈强制采样。

**反馈没有分诊和回归。** 收集了 100 条评论，没有分类、没有进测试集、没有复测，等于没做。

**多 worker 指标各算各的。** 进程内 metrics 在多 worker 下会不一致；生产用 Prometheus 客户端/聚合，或固定单 worker 先跑。

## 工程含义

**先能回答“这一次”，再谈“整体趋势”。** request_id + traces 是排障的底座；没有它，指标只能告诉你“坏了”，不能告诉你“为什么”。

**指标要挂 SLO，不要挂“看起来高”。** TTFT/TPOT/错误率/队列长度才是用户体感；GPU 利用率只是解释。

**反馈是评估的入口，不是客服的附件。** 每条低分反馈都要能追到检索结果、引用和配置版本，否则无法复现。

**可观测性本身要可运维。** 日志轮转、保留期、采样率、脱敏规则、访问权限，都要写进交付文档。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>OpenTelemetry. *Documentation*（traces、spans、attributes）. https://opentelemetry.io/docs/
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Prometheus. *Documentation*（metrics、labels、instrumentation）. https://prometheus.io/docs/
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Google SRE. *Monitoring Distributed Systems*（SRE Book 第 6 章）. https://sre.google/sre-book/monitoring-distributed-systems/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>vLLM Project. *Metrics*（TTFT、TPOT、吞吐、KV Cache）. https://docs.vllm.ai/en/latest/usage/metrics.html
