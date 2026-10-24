---
title: 服务化：OpenAI 兼容接口
---

# 服务化：OpenAI 兼容接口

> **本章要回答的问题**：怎么把本地模型变成 OpenAI 一样的接口，让现有代码不加改动就能接上？

## 为什么要"兼容 OpenAI"

OpenAI 的 `/v1/chat/completions` 事实上成了行业接口标准。兼容它有两个实际好处：

- **生态直接可用**：OpenAI 官方 SDK、LangChain、LlamaIndex、各类前端（Open WebUI 等）都只认这一个协议；
- **切换成本可控**：本地跑通之后，上云、换引擎、换模型，客户端只改 `base_url` 和模型名。

llama-server 从设计上就提供这套端点（引用 [\[1\]](#ref-1)）：

| 端点 | 用途 |
|---|---|
| `POST /v1/chat/completions` | 对话，主流入口 |
| `POST /v1/completions` | 纯文本补全，老接口 |
| `POST /v1/embeddings` | 向量化，第五部分做 RAG 要用 |
| `GET /v1/models` | 模型列表与别名 |
| `GET /health` | 健康检查，部署脚本用 |
| `GET /props` | 运行参数（上下文、默认采样值、模板），排查用 |
| `GET /slots` | 槽位状态（每个槽位在处理什么） |
| `GET /metrics` | Prometheus 指标，需 `--metrics` 开启 |

## 客户端只改两个环境变量

```python
from openai import OpenAI

client = OpenAI()   # 读 OPENAI_BASE_URL 和 OPENAI_API_KEY

resp = client.chat.completions.create(
    model="qwen3.5-9b",                 # 对应 llama-server 的 --alias
    messages=[{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
    max_tokens=256,
    temperature=0,
)
print(resp.choices[0].message.content)
```

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-local          # 本地服务不校验，但 SDK 要求非空
```

这段代码不需要任何 llama.cpp 专有逻辑。换到 vLLM 或云端时，只改这两个环境变量和模型名。

**`model` 字段本地不校验**，填什么都能通。但别随便填：它会在日志和指标里出现，也是你以后接多个模型做路由时的唯一标识。用 `--alias` 起一个有意义的名字。

## 流式：用户感知的快慢在这里

生成是逐 token 的，服务端没必要等全部生成完再返回。开启 `stream=True` 后，响应变成 Server-Sent Events（SSE）：每个事件推一个增量（`delta`），最后一个事件带原因和用量（引用 [\[2\]](#ref-2)）。

```python
stream = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "列出三条本地部署的隐性成本。"}],
    stream=True,
)
for chunk in stream:
    piece = chunk.choices[0].delta.content if chunk.choices else None
    if piece:
        print(piece, end="", flush=True)
```

流式带来的第一个可观测指标是**首字延迟（TTFT）**：从请求发出到第一个增量到达。本机实测（见 `benchmarks/client-bench/`）：短 prompt 冷请求约 0.26 s，约 700 token 的长 prompt 冷请求约 3.6 s；同一 prompt 紧接着再发，首字降到 0.1 s 左右——因为前缀缓存命中了。这条差异在 3.7 节展开。

把一次流式请求画成时序图（图 3.5-1）：TTFT 覆盖「发请求 → 套模板 → prefill → 第一个 token 回来」这几步，之后每个分片才是一个 TPOT。

<Figure
  id="3.5-1"
  src="/figures/3-5-streaming-sequence.svg"
  alt="流式请求时序：客户端发 POST，服务端套模板并 prefill，引擎返回第一个 token，服务端随后持续推送 SSE 分片，TTFT 覆盖到第一个分片到达为止"
  caption="用户感知的快慢由竖线左侧那一段决定；右侧的分片再多，也只是在填已经开始的回答。"
  source="时序按 llama-server 的实际行为绘制；TTFT 的具体数值见 benchmarks/client-bench/api-bench.csv（实测）。"
/>

## 非标准字段：思考内容

`Qwen3.5` 默认开思考，思考内容不能混进正文，否则前端会把推理过程当答案显示。llama-server 的做法是把思考拆到 `reasoning_content` 字段：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "3 × 2 = 6",
      "reasoning_content": "Thinking Process: ..."
    }
  }]
}
```

这个字段不在 OpenAI 规范里，属于 llama.cpp / vLLM 等引擎的共同约定。工程上要注意两点：

- 客户端要能容忍它缺失（换成不返回该字段的引擎时不崩）；
- **把历史消息发回去时不要带上思考内容**。模型卡明确建议多轮对话里只保留最终回答（引用 [\[3\]](#ref-3)），否则每轮都在重放推理过程，上下文和成本一起涨。

本机实测（见 `benchmarks/template/template.csv`）：同一道简单算术题，开思考生成 263 个分片、16.7 秒，关思考 7 个分片、0.81 秒，答案相同。用 256 的 `max_tokens` 试时，思考把预算吃光、正文为空——这是服务化之后最常见的"模型没回答"投诉来源。

## 超时、重试与上下文

三件事必须在客户端配置里定死，否则上线就被动：

**超时。** 一个 500 token 的回答在本机要 30 秒左右。SDK 默认超时比这短，长回答会被客户端主动断开，表现为"模型偶尔答一半"。按最慢路径设：`timeout = 首字上限 + 最大输出 token 数 ÷ 最低速度`，再留一倍余量。

**重试。** 只重试幂等的失败（连接错误、5xx），不要重试已经生成一半的流。重试会重复计费、重复占用槽位，在单机服务上还会把队列压垮。

**上下文。** 服务端不保存会话，历史每次由客户端带上。上下文是硬上限：超过 `-c` 之后，llama.cpp 的行为取决于是否开启上下文平移（`--context-shift`，默认关闭），关着的时候会报错或截断。多轮对话要在客户端做预算：system prompt + 历史 + 本轮输入 + 预期输出，四项加起来不能超过窗口。

## 部署时的三个开关

```bash
# 给服务加一道门槛：不带 key 的请求直接 401
llama-server ... --api-key "$(cat ~/.config/local-llm.key)"

# 只监听本机，别把 8080 暴露到局域网
--host 127.0.0.1

# 监控指标：Prometheus 拉 /metrics
--metrics
```

默认配置是 `--host 127.0.0.1`、无鉴权、CORS 允许所有来源。一旦改成本机之外的地址，**先加 `--api-key`，再考虑反向代理**。这是交付时的最低要求，第八部分会展开。

## 工程含义

**服务化把"模型"变成"接口"，也把故障点从模型转移到链路。** 出现"答得慢"时，先看 `/slots`（是不是在排队）、再看 `/metrics`（prefill 还是 decode 慢）、最后才怀疑模型。

**模板和采样参数要在服务端固定。** 客户端每次传，就会出现"某个前端传了奇怪的温度"这类难查的问题。固定服务端默认值，客户端只传必要的覆盖项。

**别用 `/v1/completions` 做对话。** 它不套模板，模型看到的是裸文本，行为与对话接口完全不同（3.3 节实测过：裸补全会直接开始写 `<think>`）。老代码迁移时最容易踩这个坑。

## 动手实验

::: tip 动手实验：把本地模型接进现有代码
**假设**：只改 `base_url` 和模型名，OpenAI SDK 的对话、流式、多轮三个场景都能跑通；思考模式会让 `max_tokens` 被提前耗尽。

**命令**：

```bash
# 终端 1：起服务（前台，日志就在眼前）
labs/03-deployment/serve.sh

# 终端 2：跑冒烟脚本
cd labs
uv sync                                   # 首次：装 openai SDK
uv run python 03-deployment/smoke.py
```

脚本做三件事：非流式请求、流式请求（打印首字延迟与 decode 速度）、多轮对话。它会故意保留默认思考模式跑一次流式，让你看到"正文为空"这种现象。

**记录**：三个场景是否跑通、首字延迟、decode 速度、`usage` 里的 prompt/completion token 数、`cached_tokens` 是否大于 0（说明前缀缓存生效）。数据存 `benchmarks/api/`。

**追加一组**：故意把 `max_tokens` 设成 32，开思考再跑一次，记录正文是否为空。这条实验能让你在客户报"模型不回答"时，第一时间想到预算问题而不是模型坏了。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp server 文档*（OpenAI 兼容端点、`/props`、`/slots`、`/metrics`、`--api-key`）. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>OpenAI. *Chat Completions API 参考*（请求/响应结构、流式增量、`usage`）. https://platform.openai.com/docs/api-reference/chat
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡*（思考模式、`enable_thinking`、多轮历史不带思考内容）. https://huggingface.co/Qwen/Qwen3.5-9B
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>OpenAI. *openai-python*（`base_url`、`timeout`、重试策略）. https://github.com/openai/openai-python
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>WHATWG. *Server-Sent Events*. https://html.spec.whatwg.org/multipage/server-sent-events.html
