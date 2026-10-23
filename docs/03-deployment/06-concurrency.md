---
title: 并发与批处理：并发槽位、连续批处理
---

# 并发与批处理：并发槽位、连续批处理

> **本章要回答的问题**：一个人用很流畅，两个人同时问就排队，瓶颈在哪？并发槽位和连续批处理在解决什么？

## 一个请求占着什么

要理解排队，先看一个请求在服务端占用的资源：

- **一个槽位（slot）**：一串 KV Cache，加上"我正在生成"的状态；
- **一份上下文额度**：这个槽位最多能装多少 token；
- **计算时间**：prefill 一次性吃满 GPU/CPU，decode 每个 token 一次。

llama-server 的并发模型很直白：`-np N` 就是 N 个槽位。请求进来，有空槽就进，没有就排队（引用 [\[1\]](#ref-1)）。

这里有一个容易忽略的算术：**显式设置 `-np N` 时，`-c` 是总量，每槽位分到 `-c / N`**。本机启动日志写得很清楚：

```text
-np 2 -c 8192  →  n_slots = 2, n_ctx_slot = 4096, kv_unified = 'false'
-np 4 -c 8192  →  n_slots = 4, n_ctx_slot = 2048
```

所以"把并发从 1 加到 4"不是免费的：每个用户能用的上下文从 8192 降到 2048。长文档场景下，加并发等于把文档截断得更狠。`-np -1`（自动）时走统一 KV 池（`kv_unified`），内存分配方式不同，行为要单独验证。

## 连续批处理在解决什么

最朴素的批处理是"攒一批，一起跑完再收下一批"。问题是每个回答长度不同：一个 20 token 的回答要等旁边 500 token 的回答跑完，GPU 大量空转。

连续批处理（continuous batching，也叫动态批处理）改成"按 token 步调度"：每一步，把当前所有活跃请求打包成一次前向；谁生成完了就退出，空出的位置立刻给排队的新请求（引用 [\[2\]](#ref-2)）。llama-server 默认开启（`-cb`），vLLM 的整个设计就是围绕它展开的。

在 GPU 上，这个优化的收益很大：矩阵乘的批维度一变大，单位 token 的成本就摊薄。在 CPU / Metal 上，收益小得多——内存带宽是共享的，批处理省下的是调度开销，不是内存读取。这一点后面有实测数字。

## 实测：8 个并发请求，槽位 1 / 2 / 4

本机（M4/24GB，llama.cpp 0.7.0）用 8 个并发请求打同一服务，`-c 8192`，每个请求最多生成 64 个 token（实测，见 `benchmarks/concurrency/concurrency.csv`）：

| 槽位 | 总耗时 | 聚合吞吐 | 首字延迟 p50 | 首字延迟最大 | 单请求 decode 速度 |
|---|---|---|---|---|---|
| 1 | 24.3 s | 14.1 tok/s | 11.5 s | 20.6 s | 16.3 tok/s |
| 2 | 17.2 s | 19.8 tok/s | 8.0 s | 13.3 s | 12.8 tok/s |
| 4 | 12.7 s | 27.0 tok/s | 5.1 s | 8.2 s | 9.8 tok/s |

同一批数据画成甘特图，排队时间就是每行前面的灰段（图 3.6-1）：槽位越少，灰段越长，首字延迟越高。

<Figure
  id="3.6-1"
  src="/figures/3-6-slot-queue-gantt.svg"
  alt="8 个并发请求在 1、2、4 个槽位下的甘特图：灰色为排队等待，蓝色为生成，槽位越多每行灰段越短、总耗时越短"
  caption="槽位就是排队长度。灰段越长，用户看到的空白越久——这正是首字延迟的体感来源。"
  source="实测，见 benchmarks/concurrency/concurrency.csv（M4 / 24GB，llama.cpp 0.7.0，-c 8192，每请求最多 64 token，8 并发）。"
/>

四件事：

**第一，槽位是最直接的排队长度。** 槽位从 1 加到 2，首字延迟中位数从 11.5 秒降到 8.0 秒——因为队列短了一半。8 个请求、1 个槽位时，每个请求要跑四秒上下（64 个 token，约 16 tok/s），排在后面的自然要等前面三四个跑完，这就是十几秒的来源。

**第二，吞吐涨了，但用户体感慢了。** 聚合吞吐从 14.1 涨到 27.0（约 1.9 倍），代价是单请求 decode 从 16.3 掉到 9.8（约 0.6 倍）。这不是 bug：同一块内存带宽要喂给更多并发。要不要这个交换，取决于你的场景是"很多人偶尔问一句"还是"一个人问要快"。

**第三，收益递减明显。** 槽位 1→2 吞吐涨 40%，2→4 只涨 36%，而单请求速度每次都在掉。拐点在哪，取决于硬件和生成长度，必须实测，不能照抄。

**第四，上下文在偷偷缩水。** 上表每一行的 `-c` 都是 8192，但槽位 4 时每个请求只有 2048 可用。如果测的是长文档问答，槽位 4 那一行测的根本不是同一个任务。

## 怎么估并发容量

两个方向，都要算：

**内存方向**（能开几个槽位）：按 1.6 节的公式，`权重 + 槽位 × 单请求 KV + 运行时开销 ≤ 可用内存`。本机 9B 在 8K 上下文下单槽位约 8.9 GB（估算），4 个槽位不是简单乘 4——KV 随每槽位上下文变化，要按 `-c/N` 重算。

**延迟方向**（该开几个槽位）：先定首字延迟和总时长的上限（SLO），再用实测曲线找对应的槽位数。上表里，如果要求首字 ≤ 5 秒，本机只能开 4 个槽位；要求 ≤ 3 秒，8 并发就满足不了，得加机器或降负载。

真实容量规划还要乘一个"同时在线率"：10 个注册用户不等于 10 个并发，通常峰值并发是日活的几个百分点。第八部分会把这条接到压测上。

## 工程含义

**并发数和上下文长度是一根绳的两头。** 客户端要长上下文，服务端就少开槽位；要扛并发，就把单请求窗口压小。这个取舍必须写进配置评审，不能留给默认值。

**排队要能被观测。** `/slots` 能看到每个槽位在干什么，`/metrics` 能拉队列长度和 token 计数。没有这两个，用户说"卡"的时候你只能猜。

**流式让排队更容易接受。** 排队时间体现在首字延迟上，用户看到第一个字之后就有耐心。关掉流式，用户面对的是十几秒空白，主观感受差一个量级。

**不要拿 CPU/Metal 的并发曲线推 GPU。** 这里的 1.9 倍是共享带宽下的结果；GPU 上连续批处理的收益通常大得多，但依赖显存、KV 分页和请求长度分布。第七部分会在 4090 上重做这张表。

## 待实测

::: tip 待实测：并发曲线与槽位代价
**假设**：聚合吞吐随槽位增加而上升但边际递减；首字延迟随并发/槽位比线性变化；每槽位上下文随 `-np` 反比缩小。

**命令**（脚本自动按 NP=1/2/4 起停服务）：

```bash
cd labs
uv run python 03-deployment/concurrency.py

# 只测已启动的实例
BASE_URL=http://127.0.0.1:8080/v1 uv run python 03-deployment/concurrency.py --external
```

**记录**：每种槽位配置下的总耗时、聚合吞吐、首字延迟 p50/最大、单请求 decode 速度，以及启动日志里的 `n_ctx_slot`。数据存 `benchmarks/concurrency/`。

**建议追加两组**：
1. 并发数从 8 改成 2，看"并发不超过槽位"时是否还有排队；
2. 上下文从 8192 改成 4096，观察同一槽位数下吞吐是否变化（KV 变小，prefill 变快，差别可能比你预期的小）。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp server 文档*（`-np`、`-cb`、`-kvu`、`--kv-unified-per-slot`）. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Yu, G.-I. et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*（连续批处理）. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Agrawal, A. et al. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. arXiv:2403.02310. https://arxiv.org/abs/2403.02310
