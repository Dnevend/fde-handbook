---
title: 多卡：张量并行、流水线并行、多副本三者的取舍；单机上 LLM / Embedding / Reranker 怎么分配
---

# 多卡：张量并行、流水线并行、多副本三者的取舍；单机上 LLM / Embedding / Reranker 怎么分配

> **本章要回答的问题**：两张卡该怎么用——张量并行、流水线并行还是各跑一个副本？

手里有两张卡，不等于模型会自动变快。三种用法解决的是不同问题：

- **张量并行（tensor parallelism，TP）**：把同一层的矩阵切到多张卡上，一起算一层；
- **流水线并行（pipeline parallelism，PP）**：把不同层放到不同卡上，像流水线一样传激活；
- **多副本（replicas）**：每张卡跑一个完整模型，请求分给不同副本。

对 9B 这种单卡放得下的模型，**默认应该是多副本**；TP 是模型放不下或单卡确实不够时的手段，不是“两张卡一定比一张快”。

## 三种并行

| 方式 | 切什么 | 卡间通信 | 适合 | 主要代价 |
|---|---|---|---|---|
| TP | 同一层的权重/注意力头 | 每层 all-reduce，通信频繁 | 单卡放不下、延迟敏感 | 通信开销；需要高带宽互联 |
| PP | 不同层 | 只传激活，通信少 | 超大模型、吞吐优先 | 流水线气泡；调参复杂 |
| 多副本 | 不切，每卡一个完整模型 | 不通信 | 单卡放得下的模型、吞吐优先 | 每卡一份权重；需要负载均衡 |

TP 的通信量随层数增长：每一层都要把部分结果汇总，才能继续下一层。TP 和 PP 的经典实现见 Megatron-LM（引用 [\[1\]](#ref-1)）；vLLM 的 `--tensor-parallel-size`/`--pipeline-parallel-size` 见引擎参数文档（引用 [\[2\]](#ref-2)）。NVLink 的带宽和拓扑见 NVIDIA 官方页面（引用 [\[3\]](#ref-3)），H100 SXM 的 NVLink 是 900 GB/s（引用 [\[4\]](#ref-4)）；它比 PCIe 高一个数量级（7.1 的表），所以 TP 对互联最敏感。4090 没有 NVLink，两张 4090 做 TP=2 只能走 PCIe，通信开销会吃掉一部分收益；这正是本节要实测的。

PP 把模型按层切开，卡之间只传激活值，通信少；但流水线有气泡：前面的卡在算，后面的卡在等，反之亦然。微批（micro-batch）越大，气泡占比越小，但延迟也越高。

多副本最简单：每个副本独立处理请求，卡之间不通信。缺点是每张卡都要放一份完整权重；对 9B 4-bit（约 5.7 GB）来说，两张 24GB 卡各放一份很轻松。

## 怎么选

按顺序问：

1. **模型一张卡放得下吗？** 放得下，优先多副本；放不下，才考虑 TP/PP。
2. **要的是延迟还是吞吐？** 延迟敏感、模型又放不下，TP 更合适；吞吐优先、模型很大，PP 可能更划算。
3. **互联够好吗？** TP 需要高带宽；PCIe 消费卡做 TP，收益可能不如两个副本。NVLink 是加钱项，不是默认项。
4. **运维复杂度能接受吗？** 多副本需要负载均衡和健康检查；TP/PP 需要 NCCL、拓扑、`/dev/shm` 调优。先用简单的。

对本书的 2×4090：**先测 2 个副本，再测 TP=2**。如果 2 个副本的聚合吞吐更高、TTFT 更稳，就没有理由上 TP。

## 单机上 LLM / Embedding / Reranker 怎么分配

RAG 系统不只一个模型：LLM、Embedding、Reranker 都要占显存和算力。单机多卡有三种分配方式：

1. **全挤在一张卡上**：LLM 占大头，Embedding/Reranker 用剩下的小块。简单，但互相抢显存/带宽；LLM 高并发时，Embedding 延迟会被拖累。
2. **LLM 独占一张卡，Embedding/Reranker 放另一张**：隔离性好，LLM 的 KV 池不被小模型挤占；另一张卡的算力可能闲置，适合 LLM 是瓶颈的场景。
3. **按副本分配**：LLM 两张卡各一个副本，Embedding/Reranker 放 CPU 或第三张卡；适合 LLM 吞吐是绝对瓶颈、Embedding/Reranker 很轻的场景。

怎么选取决于瓶颈：如果 LLM 的 TTFT/TPOT 是主矛盾，优先给它独占；如果检索链路慢，给 Embedding/Reranker 留资源。**不要默认“小模型不占资源”**：Embedding/Reranker 的 batch 一大，也会抢带宽。

## 拓扑先看，再调参

多卡机器先跑：

```bash
nvidia-smi topo -m
```

它告诉你卡之间是 PCIe 还是 NVLink、跨不跨 NUMA、有没有通过 PCIe switch。拓扑不对，TP 的通信开销会把收益吃光；这时候调 `--max-num-seqs` 没用。

容器里跑 TP 还要注意共享内存：加 `--ipc=host` 或 `--shm-size=16g`；`/dev/shm` 太小会让 NCCL/算子报错或变慢。

## 怎么测

用同一台 2 卡机器、同一个模型、同一批请求、同一个并发，只改并行方式：

```bash
bash labs/07-gpu-inference/tp_vs_replica.sh \
  --model-dir models/qwen3.5-9b-awq --model-name qwen3.5-9b-awq \
  --concurrency 8 --requests 64
```

脚本会先起 TP=2 的一个容器，再起两个各占一张卡的副本，分别压测。记录：

- 聚合吞吐、单请求 token/s、TTFT p50/p95、TPOT p50/p95；
- 显存峰值、`nvidia-smi topo -m`、NCCL 日志里有没有 fallback；
- 启动时间、容器日志里的 TP/PP 配置。

**只改并行方式，其他变量不变。** 模型、量化、`max-model-len`、`max-num-seqs`、请求分布都要一致，否则数字不可比。

## 常见坑

**小模型上 TP 反而更慢。** TP 每层都要通信；模型小、计算少时，通信占比高。9B 单卡放得下，TP=2 的收益可能为负。

**4090 没有 NVLink。** 两张 4090 的 TP 走 PCIe，甚至可能落到主机内存；不要拿 H100 的 TP 曲线外推。

**异构卡不能随便 TP。** TP 要求各卡显存/算力/驱动一致；一张 4090 + 一张 A6000 做 TP，轻则慢，重则报错。

**`/dev/shm` 太小。** 容器默认 64MB 时，TP/NCCL 可能直接失败；先 `df -h /dev/shm`，再加 `--ipc=host`。

**副本没有负载均衡。** 两个副本各自监听端口，前面要有轮询/最少连接；否则一个副本忙死，一个闲着。

**多模型抢显存。** LLM 的 KV 池按 `gpu-memory-utilization` 预分配；Embedding/Reranker 再挤进来，可能把 LLM 的余量吃光。先算总账（1.6、7.9）。

## 工程含义

**默认多副本，TP 是例外。** 单卡放得下的模型，多副本更简单、延迟更稳、没有卡间通信；只有模型放不下或单卡确实不够时，才考虑 TP/PP。

**NVLink 是选型问题，不是调参问题。** 有 NVLink 的机器贵；先确认 TP 是不是真的需要，再决定是否为互联加钱。本书计划在 2×4090 上实测；跑完后只代表 PCIe 场景，NVLink 的数字引用厂商规格。

**多模型分配按瓶颈来。** LLM 是瓶颈就给它独占；检索链路是瓶颈就给 Embedding/Reranker 留资源。不要平均分，也不要默认小模型不占资源。

**拓扑和共享内存先检查。** 多卡问题一半在拓扑和 `/dev/shm`；先看 `nvidia-smi topo -m` 和 `df -h /dev/shm`，再调引擎参数。

## 动手实验

::: tip 动手实验：TP=2 vs 2 个副本
**假设**：9B 4-bit 在 4090 上单卡放得下；TP=2 的聚合吞吐不一定高于 2 个副本，单请求速度可能更差；瓶颈在 PCIe 通信和 KV 读取。

**命令**（2×4090 同机，仓库根目录）：

```bash
# 先干跑看命令
bash labs/07-gpu-inference/tp_vs_replica.sh --dry-run

# 正式跑：TP=2 和 2 副本各压一轮
bash labs/07-gpu-inference/tp_vs_replica.sh \
  --model-dir models/qwen3.5-9b-awq --model-name qwen3.5-9b-awq \
  --concurrency 8 --requests 64

# 拓扑和共享内存
nvidia-smi topo -m
df -h /dev/shm
```

**记录**：两种方式的聚合吞吐、单请求 token/s、TTFT/TPOT、显存峰值、启动日志、拓扑。数据存 `benchmarks/gpu/multi-gpu/`。如果 TP=2 不如 2 副本，结论就写“在这个模型/这台机器上，副本更划算”，不要硬推 TP。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Shoeybi, M. et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*. arXiv:1909.08053. https://arxiv.org/abs/1909.08053
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>vLLM Project. *Engine Arguments*（`tensor-parallel-size`、`pipeline-parallel-size`）. https://docs.vllm.ai/en/latest/configuration/engine_args.html
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>NVIDIA. *NVLink and NVLink Switch*. https://www.nvidia.com/en-us/data-center/nvlink/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA H100 Tensor Core GPU 规格页*（NVLink 900 GB/s）. https://www.nvidia.com/en-us/data-center/h100/
