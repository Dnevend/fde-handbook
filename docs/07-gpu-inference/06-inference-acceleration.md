---
title: 推理加速：Prefix Caching（RAG 场景收益最大）、Chunked Prefill、投机解码
---

# 推理加速：Prefix Caching（RAG 场景收益最大）、Chunked Prefill、投机解码

> **本章要回答的问题**：RAG 每次都带同样的长前缀，Prefix Caching 能省多少？投机解码又是什么？

到了 7.4，vLLM 已经能跑、能扛并发了。这一节讲三个“再快一点”的杠杆：**Prefix Caching** 省重复的 prefill，**Chunked Prefill** 让长请求不堵短请求，**投机解码** 用一个小模型猜下一个 token。三个都能提速，但收益高度依赖负载；RAG 场景的优先级通常是 Prefix Caching > Chunked Prefill > 投机解码。

## Prefix Caching：复用相同前缀

RAG 的请求通常长这样：

```text
[系统提示 + 安全规范 + 输出格式] + [检索到的文档] + [用户问题]
```

系统提示和安全规范每次都一样，检索文档在同一个会话里也可能重复。Prefix Caching 把前缀的 KV Cache 按块缓存起来；下一个请求如果前缀命中，直接复用这些块，不再重新 prefill（引用 [\[1\]](#ref-1)）。

收益体现在 **TTFT**：前缀越长、命中率越高，省下的 prefill 越多。代价是：

- 缓存要占显存；块被占用后，KV 池的可用容量变小；
- 前缀必须**逐 token 完全一致**才命中；多一个空格、换一个模板，缓存就失效；
- 缓存有淘汰策略；并发高、前缀多时，命中率会下降。

**RAG 是 Prefix Caching 的最佳场景**：系统提示、few-shot 示例、固定的输出格式都是长前缀。开放域聊天、每个请求都完全不同的场景，收益很小。

## Chunked Prefill：让长请求不堵短请求

一个 8K prompt 的 prefill 会占住 GPU 一段时间；这期间短请求只能排队，TTFT 飙升。Chunked Prefill 把长 prefill 切成小块，和 decode 交错执行：长请求分几步做完 prefill，短请求可以在中间插进来（引用 [\[2\]](#ref-2)）。

它改善的是**公平性和尾延迟**：TTFT p95 更稳，短请求不会被长请求“堵死”。它**不减少总计算量**：prefill 的 FLOPs 不变，只是排得更细。

关键参数是 `--max-num-batched-tokens`：块太小，调度开销大；块太大，长请求又会堵住别人。要在“短请求延迟”和“长请求完成时间”之间找平衡。vLLM 的开关和默认值见优化文档（引用 [\[5\]](#ref-5)）。

## 投机解码：用小模型猜 token

decode 是一个 token 一个 token 生成的，每一步都要读一遍权重，带宽是瓶颈。投机解码（speculative decoding）的思路是：

1. 用一个小的 draft 模型（或 n-gram）先猜接下来几个 token；
2. 大模型一次性验证这几个 token；
3. 猜对的接受，猜错的从第一个错的位置重新生成。

如果 draft 模型猜得准（接受率高），一次验证能产出多个 token，decode 速度提升；如果猜得不准，验证的算力白花，反而更慢（引用 [\[3\]](#ref-3)）。

它适合**输出可预测**的场景：代码补全、格式化抽取、固定模板的回答。创意写作、高 temperature、开放域聊天，接受率低，收益小。代价是额外的 draft 模型显存和验证算力；vLLM 的投机解码配置见官方文档（V1 用 `--speculative-config` JSON，旧版用 `--speculative-model`；引用 [\[4\]](#ref-4)）。

## RAG 场景的优先级

| 杠杆 | 主要收益 | 主要代价 | RAG 优先级 |
|---|---|---|---|
| Prefix Caching | TTFT 下降，prefill 省算 | 缓存占显存，前缀必须一致 | 高 |
| Chunked Prefill | TTFT p95 更稳，长短混排更公平 | 调度开销，块大小要调 | 中 |
| 投机解码 | decode 速度提升 | draft 模型显存/算力，接受率依赖场景 | 低（可选） |

先开 Prefix Caching，因为 RAG 的前缀重复是结构性的；再开 Chunked Prefill，解决长短请求混排；投机解码最后考虑，除非你的输出高度可预测。

## 怎么测

三个实验都要**开/关对照，其他变量不变**：

- **Prefix Caching**：同一段 4K 前缀，连续发 N 个请求；对比开/关的 TTFT、prefill 时间、缓存命中率。注意第一个请求是冷启动，不要把它算进热请求。
- **Chunked Prefill**：长短请求混合（例如 8K 长请求 + 200 token 短请求），对比开/关的短请求 TTFT p95、长请求完成时间、聚合吞吐。
- **投机解码**：同一批 prompt，对比开/关的接受率、decode token/s、总耗时；接受率低就关掉。

记录要包括：prompt 长度分布、并发、`max-num-batched-tokens`、缓存命中率/接受率、TTFT/TPOT、显存。**只报“快了多少”不报命中率/接受率，别人无法判断能不能复现。**

## 常见坑

**版本默认值不一样。** vLLM 不同版本对 Prefix Caching / Chunked Prefill 的默认开关不同；先 `vllm serve --help` 确认，别把“默认开启”当成“关闭”来对比。

**前缀差一个 token 就不命中。** 系统提示里的时间戳、用户名、随机 ID 都会让缓存失效；RAG 的前缀要设计成稳定部分在前、变化部分在后。

**Chunked Prefill 不是免费的。** 块切得太小，调度和 kernel 启动开销会吃掉收益；`max-num-batched-tokens` 要实测。

**投机解码的 draft 模型要同 tokenizer。** 词表不一致，猜出来的 token 无法验证；draft 模型太大，显存和算力又不划算。

**缓存/投机都可能影响质量。** Prefix Caching 本身不改输出（命中是精确复用）；投机解码理论上等价，但 draft 模型/实现有 bug 时会偏。上线前用同一批题对比。

## 动手实验

::: tip 动手实验：三个杠杆的开/关对照
**假设**：4K 共享前缀下，Prefix Caching 能把热请求的 TTFT 降低一个量级；Chunked Prefill 改善短请求 p95，但不改变总吞吐量级；投机解码在格式化抽取上接受率高，在开放写作上收益小。

**命令**（先按 7.4 起服务；每次只改一个开关）：

```bash
# 1) Prefix Caching：同一段 4K 前缀，连续发 20 个请求
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 1 --requests 20 --prompt-chars 6000 --max-tokens 64 \
  --no-unique-prefix --label prefix-on

# 关掉 Prefix Caching 重启，再跑一次
# bash labs/07-gpu-inference/serve_vllm.sh --no-prefix-caching ...
# uv run labs/07-gpu-inference/bench_openai.py ... --label prefix-off

# 2) Chunked Prefill：长短混合，看短请求 p95
# 8K prompt + 32 生成需要 max-model-len ≥ 8224；先把服务开到 16384，
# 或把 --prompt-chars 降到 9000。
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 8 --requests 32 --prompt-chars 12000 --max-tokens 32 --label chunked-on

# 3) 投机解码：按 vLLM 版本的配置起服务（V1 用 --speculative-config JSON），再跑同一批题
```

**记录**：TTFT p50/p95、TPOT、聚合吞吐、缓存命中率、接受率、显存；启动参数和 vLLM 版本。数据存 `benchmarks/gpu/prefix/`。**冷请求和热请求分开报**；只报热请求会把缓存收益夸大。
:::

## 工程含义

**先优化结构性的重复，再优化调度，最后才猜 token。** RAG 的前缀重复是结构性的，Prefix Caching 收益最稳；Chunked Prefill 解决混排；投机解码是锦上添花。

**加速开关不是越多越好。** 每个开关都有显存/调度/质量代价；一次只开一个，测完再决定是否保留。

**用命中率和接受率解释收益。** 没有这两个数，“快了多少”无法迁移到别人的负载；把 prompt 分布和开关一起写进报告。

**版本升级要重测。** vLLM 的调度和缓存实现会变；升级后旧曲线可能失效，重新跑一遍对照。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>vLLM Project. *Prefix Caching*. https://docs.vllm.ai/en/latest/features/prefix_caching.html
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Agrawal, A. et al. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. arXiv:2403.02310. https://arxiv.org/abs/2403.02310
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Leviathan, Y. et al. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023. arXiv:2211.17192. https://arxiv.org/abs/2211.17192
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>vLLM Project. *Speculative Decoding*. https://docs.vllm.ai/en/latest/features/spec_decode.html
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>vLLM Project. *Optimization*（chunked prefill 等）. https://docs.vllm.ai/en/latest/configuration/optimization.html
