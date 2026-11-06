---
title: vLLM 部署：PagedAttention、连续批处理、关键参数与调参顺序
---

# vLLM 部署：PagedAttention、连续批处理、关键参数与调参顺序

> **本章要回答的问题**：vLLM 凭什么比朴素实现快几倍？PagedAttention 和连续批处理各解决了什么？

第三部分的 llama.cpp 是为“一个人用”设计的：一个请求、一段上下文、生成完就结束。到了服务器上，50 个人同时问、每个人上下文不同、有人问一句就走、有人要写长文，问题变成两个：**KV Cache 怎么放**，**请求怎么排**。vLLM 的两个核心设计——PagedAttention 和连续批处理——就是回答这两个问题的。

这一节先讲原理，再讲参数和调参顺序；具体的并发曲线和数字在本节的动手实验里填。

## KV Cache 的两笔浪费

朴素实现里，每个请求启动时按 `max_model_len` 预留一整块连续显存给 KV Cache。这带来两笔浪费：

1. **内部碎片**：请求实际只用了 2K，但预留了 8K，剩下 6K 别人不能用。
2. **外部碎片**：请求长短不一，连续内存块分配久了会碎，最后明明有总量却拼不出一块连续的。

并发越高、请求长度越不均匀，浪费越大。PagedAttention 的思路借自操作系统：把 KV Cache 切成固定大小的块（block），用页表（page table）把逻辑上的连续 KV 映射到物理上不连续的块（引用 [\[1\]](#ref-1)）。

## PagedAttention 解决了什么

PagedAttention 把每个请求的 KV Cache 变成一组块：

- **按需分配**：用到哪块分配哪块，不再按 `max_model_len` 一次性预留；
- **块可以离散**：物理块不需要连续，消除外部碎片；
- **块可以共享**：相同前缀的请求可以共享物理块（Prefix Caching 的基础，7.6 会讲）；
- **块大小可调**：块太大浪费，太小页表开销大；vLLM 默认块大小是一个需要理解的参数。

它没有让计算变快，而是让**同样显存能装下更多请求**。对吞吐的影响是间接的：能开更多并发，GPU 的 batch 才能更大，单位 token 的成本才摊得薄。

## 连续批处理解决了什么

朴素批处理是“攒一批，一起跑完再收下一批”。问题是每个回答长度不同：一个 20 token 的回答要等旁边 500 token 的回答跑完，GPU 大量空转。

连续批处理（continuous batching，也叫 iteration-level scheduling）改成“按 token 步调度”：每一步，把当前所有活跃请求打包成一次前向；谁生成完了就退出，空出的位置立刻给排队的新请求（引用 [\[2\]](#ref-2)）。vLLM 的调度器就是围绕这个设计的。

它和 PagedAttention 是互补的：PagedAttention 让显存能装下更多请求，连续批处理让这些请求能动态进出。两者一起，才有“高并发下吞吐明显上升”的效果。

## 启动一个服务

实验室里用脚本起服务，参数写在命令行上：

```bash
bash labs/07-gpu-inference/serve_vllm.sh \
  --model-dir models/qwen3.5-9b-awq \
  --model-name qwen3.5-9b-awq \
  --max-model-len 8192 \
  --gpu-mem-util 0.90 \
  --max-num-seqs 64
```

等价的核心参数（完整命令见 `benchmarks/gpu/env/*-serve.txt`）：

```bash
vllm serve /models \
  --served-model-name qwen3.5-9b-awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192
```

## 关键参数表

| 参数 | 作用 | 调大 | 调小 |
|---|---|---|---|
| `--max-model-len` | 单请求最大上下文 | 能放更长文档，但 KV 占用线性上涨 | 省显存，但长文档会被截断/拒绝 |
| `--gpu-memory-utilization` | vLLM 预分配显存的比例 | KV 池更大，并发更高；太贴近 1.0 容易 OOM | 留更多余量，但浪费显存 |
| `--max-num-seqs` | 同时调度的请求数上限 | 吞吐可能上升，单请求延迟可能变差 | 延迟更稳，吞吐上不去 |
| `--max-num-batched-tokens` | 一次前向最多处理多少 token | prefill 更快，但 decode 可能被挤 | decode 更稳，长 prompt 首字更慢 |
| `--tensor-parallel-size` | 张量并行卡数 | 大模型放得下，但通信开销上升 | 单卡简单，模型可能放不下 |
| `--enable-prefix-caching` | 复用相同前缀的 KV 块 | RAG 场景省 prefill | 前缀不重复时没收益，还多占块 |
| `--enable-chunked-prefill` | 把长 prefill 切块，和 decode 交错 | 长 prompt 不阻塞短请求 | 可能增加调度开销 |

`--dtype` 和 `--quantization` 不是“调大调小”的参数：前者选精度（`float16`/`bfloat16`/`auto`），后者选量化方式（AWQ/GPTQ/FP8）。它们主要影响显存和质量，放在 7.5 讲。

参数的具体默认值随 vLLM 版本变化，启动前用 `vllm serve --help` 确认（引用 [\[3\]](#ref-3)）。**不要照抄博客里的默认值**，版本一换就变。

## 调参顺序

一次只改一个变量，按这个顺序：

1. **先让模型放下**：选量化、`--max-model-len`、`--gpu-memory-utilization`，目标是能稳定启动，不 OOM。
2. **再定上下文**：按业务最长文档定 `--max-model-len`；不要为了“以后可能用”开到 32K，KV 会按长度吃显存。
3. **再定并发**：用 `--max-num-seqs` 和 `--gpu-memory-utilization` 找“显存放得下 + SLO 达标”的最大并发；7.9 的公式先估，本节的压测再校准。
4. **再调 batch**：`--max-num-batched-tokens` 影响 prefill 和 decode 的平衡；长 prompt 多就调大，decode 延迟敏感就调小。
5. **最后开加速**：Prefix Caching、Chunked Prefill、投机解码，按 7.6 的收益排序，不要一开始全开。它们的具体行为见 vLLM 优化文档（引用 [\[4\]](#ref-4)）。
6. **多卡最后考虑**：单卡能放下、SLO 能达标，就不要 TP；多副本通常更简单（7.7）。

这个顺序的逻辑是：**先保证能跑，再保证放得下，再保证延迟，最后才榨吞吐。** 反过来先开一堆优化，出了问题不知道是谁的锅。

## 日志怎么读

vLLM 启动日志里有几个关键信息：

- **模型加载**：权重格式、量化方式、dtype；确认加载的是你以为的那个模型。
- **KV Cache 容量**：V1 日志会打印 `GPU KV cache size: ... tokens`；块数可以用 tokens / block_size 反推（block_size 见启动日志或引擎参数）。V0 的 `# GPU blocks` 格式不同，按版本看。
- **最大并发**：V1 日志会打印 `Maximum concurrency for ...`；和 7.9 的 `C_vram` 对一下。
- **CUDA graph / 编译**：首次启动可能花几十秒做 capture；不是卡死。
- **警告**：`max_num_seqs` 超过 KV 能力、`max_model_len` 超过模型支持、prefix caching 不兼容等；看到警告先处理，不要忽略。

如果日志里出现 `No available memory for the cache blocks`，说明 KV 池不够；先降 `max_model_len`、降并发、或降 `gpu_memory_utilization` 之外的占用（7.8）。

## 常见误区

**“`gpu-memory-utilization` 调到 1.0 能用满显存。”** 不能。还要留激活、CUDA 上下文、碎片和系统进程的余量；0.85–0.92 是经验区间，具体看卡和模型。

**“`max-model-len` 只是上限，设大一点没关系。”** 有关系。vLLM 的 KV 池由 `gpu-memory-utilization` 扣掉权重/激活后的余量决定，`max-model-len` 决定单请求最多能占多少；设太大，单请求可占的块变多，总并发下降。长上下文是拿并发换的。

**“`max-num-seqs` 越大吞吐越高。”** 到一定程度会反过来：KV 读取量随并发线性上涨，单请求速度下降；超过带宽/算力拐点后，聚合吞吐也可能不涨（7.1 的公式）。

**“Prefix Caching 开了就快。”** 只有前缀重复时才快；前缀不重复时，它只是多占块、多算哈希。RAG 场景收益大，开放域聊天收益小。

**“Chunked Prefill 能减少计算量。”** 不能。它只是把长 prefill 切块，让短请求不被长请求堵住；总 FLOPs 不变（引用 [\[5\]](#ref-5)）。

## 动手实验

::: tip 动手实验：并发曲线与参数扫描
**假设**：聚合吞吐随并发上升但边际递减；TTFT 随并发/上下文上升；`max-num-seqs` 和 `gpu-memory-utilization` 共同决定 OOM 边界。

**命令**（先按 7.4 起服务，记录启动参数）：

```bash
# 并发梯度：固定 prompt 和输出长度
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 1,2,4,8,16,32 --requests 64 \
  --prompt-chars 1200 --max-tokens 128 --label vllm-awq

# 参数扫描：一次只改一个
# 1) max-num-seqs 32/64/128
# 2) gpu-memory-utilization 0.80/0.90/0.95
# 3) max-model-len 4096/8192/16384
```

**记录**：每个并发点的 TTFT p50/p95、TPOT p50/p95、聚合吞吐、单请求速度、显存峰值、OOM/错误数；启动日志里的 KV 块数和 max concurrency。显存峰值另开 `monitor_gpu.py` 采，或读 vLLM 的 metrics。数据存 `benchmarks/gpu/bench/`。**一次只改一个参数**，否则曲线无法归因。
:::

## 工程含义

**vLLM 的收益来自“显存利用率 × 调度”，不是某个魔法参数。** PagedAttention 让显存不碎，连续批处理让请求不空等；参数只是把这两个能力调到你的负载上。

**先测再调。** 没有自己的并发曲线，调参就是猜。把 `bench_openai.py` 和启动命令留在仓库里，换模型/换卡时重跑一遍。

**版本要钉住。** vLLM 的参数名、默认值、调度行为在版本间会变；交付时把镜像 tag 和启动命令写进部署文档，不要用 `latest` 上线。

**单卡优先，多副本优先于 TP。** 9B 这种单卡放得下的模型，两个副本通常比 TP=2 简单、延迟更稳；只有模型放不下或单卡确实不够时，才上 TP（7.7）。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Yu, G.-I. et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>vLLM Project. *Engine Arguments*. https://docs.vllm.ai/en/latest/configuration/engine_args.html
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>vLLM Project. *Optimization*（chunked prefill、prefix caching 等）. https://docs.vllm.ai/en/latest/configuration/optimization.html
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Agrawal, A. et al. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. arXiv:2403.02310. https://arxiv.org/abs/2403.02310
