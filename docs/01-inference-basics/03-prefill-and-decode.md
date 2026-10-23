---
title: Prefill 与 Decode：首字延迟和生成速度分别由什么决定
---

# Prefill 与 Decode：首字延迟和生成速度分别由什么决定

> **本章要回答的问题**：第一次出字为什么要等，之后为什么快一些？首字延迟和生成速度分别被什么卡住？

## 一次请求被切成两段

模型处理一次对话，内部其实分成两个性质完全不同的阶段。

**Prefill（预填充）**：把整段提示词一次性喂进去，算出所有位置的 Key/Value，并得到第一个输出 token 的概率分布。这一段可以并行——512 个 token 的 prompt，在矩阵乘里是一次 `[512, hidden] × [hidden, ...]` 的大运算，GPU 的算力能被吃满。

**Decode（解码）**：从第二个 token 开始，每次只处理一个新 token。它要读全部的模型权重，却只做一次"矩阵 × 向量"的乘法，算力利用率极低。每生成一个 token 都要重复一遍这个过程，直到结束。

两段的耗时特征完全不同，所以性能指标也必须分开看（图 1.3-1）：

<Figure
  id="1.3-1"
  src="/figures/1-3-prefill-decode-timeline.svg"
  alt="一次请求的时间轴：排队、Prefill、Decode 三段，标注 TTFT 覆盖排队与 prefill，TPOT 是两个相邻 token 的间隔，端到端时延为 TTFT 加 TPOT 乘输出 token 数"
  caption="示意，非等比。TTFT 由排队与 prefill 决定，总时长还要再加「输出 token 数 × TPOT」。"
  source="指标定义见本节参考资料 [3]；图中不含实测数字。"
/>

- **首字延迟（Time To First Token，TTFT）**：从请求发出到第一个 token 返回。主要由排队、分词、prefill 决定。
- **每 token 时间（Time Per Output Token，TPOT）**，也叫 token 间延迟（ITL）：decode 阶段平均每个 token 花多久。这两个指标的定义以服务引擎的指标文档为准（引用 [\[3\]](#ref-3)）。
- **端到端时延**：TTFT + TPOT × 输出 token 数。

一个常见误区是只看"每秒多少 token"。同样是 20 token/s，首字等 0.5 秒和等 8 秒，用户体验是两个产品。

## 为什么 prefill 快、decode 慢

用算术强度（arithmetic intensity）解释最清楚：一个运算每读取一字节内存，能做多少次浮点运算。

- Prefill 是矩阵乘：一次读入权重，参与几百上千个 token 的计算，算术强度高，瓶颈在算力。
- Decode 是矩阵向量乘：读一遍权重，只算一个 token，算术强度低，瓶颈在内存带宽（引用 [\[2\]](#ref-2)）。

这解释了三个现象：

1. **同样一台机器，prefill 每秒能处理的 token 数远高于 decode 每秒生成的 token 数。** 前者在"算"，后者在"等内存"。
2. **prompt 越长，首字越慢，但近似线性。** 512 token 的 prompt 大约是 256 token 的两倍 prefill 时间（估算，忽略固定开销）。
3. **decode 速度几乎不随上下文长度线性下降，但会缓慢变差。** 因为每一步还要读取越来越长的 KV Cache（1.4 节）。

## 排队、批处理与吞吐

单用户单请求时，TTFT 基本等于 prefill 时间。真实系统里还要加上排队。

服务端通常会把多个请求合批（batching）：在 decode 阶段，把同一时刻要生成 token 的多个请求拼成一个 batch，一次读权重、服务多个用户。这样**吞吐（每秒总 token 数）大幅上升，但单请求的 TPOT 会略微变长**——因为每次要算的向量变多了。

传统的静态批处理要求一个 batch 里的请求一起开始、一起结束，短请求被长请求拖住，GPU 空转。连续批处理（continuous batching）在每一步 decode 后重新组批，完成的请求立即退出、新请求立即加入，显著提升利用率（引用 [\[1\]](#ref-1)）。配合分页的 KV 管理（PagedAttention，引用 [\[4\]](#ref-4)），显存碎片也被压下去。这也是 vLLM、TensorRT-LLM 这类服务引擎的核心能力之一。

对本书的场景，这条结论很重要：**并发数和单请求延迟是互相挤压的**。给客户承诺"50 人同时用"之前，必须明确是"50 人同时在线"还是"50 人同时提问"，以及能接受多长的等待。

## 各阶段的优化手段

| 阶段 | 瓶颈 | 有效手段 | 代价 |
|---|---|---|---|
| 排队 | 并发调度 | 连续批处理、优先级队列 | 实现复杂，短请求仍可能被长请求影响 |
| Prefill | 算力 | 缩短 prompt、减少检索片段、前缀缓存 | 上下文变少，可能丢信息 |
| Decode | 内存带宽 | 更低比特量化、更小模型、投机解码 | 质量损失、实现复杂度 |
| 两者 | KV Cache 读写 | KV 量化、GQA/MQA 架构、分页管理 | 精度损失、工程复杂度 |
| 端到端 | 输出长度 | 限制 `max_tokens`、关掉思考模式、要求简短 | 可能牺牲推理质量 |

注意"缩短 prompt"和"保证检索质量"是一对矛盾。RAG 系统里这两件事天天打架，第六部分的评估就是用来裁决的：减少片段后，答对率掉了多少？如果没掉，那就是白省。

## 一个容易被忽略的细节：前缀缓存

如果多个请求的开头完全相同（比如固定的 system prompt、固定的对话模板），引擎可以把这段的 KV Cache 缓存下来，下次直接复用，跳过重复的 prefill。llama.cpp 默认会做 prompt cache，vLLM 有 prefix caching 选项。

RAG 场景特别适合这个优化：system prompt 和模板通常一字不变，变的是检索片段。启用前缀缓存后，TTFT 能明显下降（估算，降幅取决于固定前缀占比）。这也是第三部分配置服务时要检查的一项。

## 待实测

::: tip 待实测：分开测 prefill 与 decode
**假设**：在同一台机器上，`pp512`（每秒处理 512 个 prompt token）显著高于 `tg128`（每秒生成 128 个 token）；上下文加深后 `tg` 会下降。

**命令**：

```bash
# 默认上下文深度
llama-bench -m "$MODEL" -p 512 -n 128 -r 5 -o md

# 分别在不同深度下测 decode（-d 为上下文深度）
llama-bench -m "$MODEL" -p 512 -n 128 -d 0,4096,16384 -r 3 -o csv
```

**记录**：`pp512`、`tg128` 的均值与标准差，以及 `-d` 变化时 `tg` 的下降幅度。CSV 存到 `benchmarks/prefill-decode/`，同目录放一份 `bash benchmarks/record-env.sh` 的输出（机型、macOS 版本、引擎版本）。`-r 5` 指每个配置重复 5 次，`±` 后面的数字就是这 5 次的方差，方差大说明机器当时不干净（后台任务、热降频），重跑一次再下结论。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Yu, G.-I. et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Pope, R. et al. *Efficiently Scaling Transformer Inference*. arXiv:2211.05102. https://arxiv.org/abs/2211.05102
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>vLLM Project. *Metrics*（TTFT、TPOT/ITL 的定义与采集方式）. https://docs.vllm.ai/en/latest/design/metrics.html
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
