---
title: 容量规划：并发数 × 上下文长度 → 需要什么卡、几张
---

# 容量规划：并发数 × 上下文长度 → 需要什么卡、几张

> **本章要回答的问题**：50 个人同时用、上下文 8K，我需要几张什么卡？怎么算出来的？

“50 人同时用”不是一张卡能回答的问题。50 个并发、每个 8K 上下文，和 50 个并发、每个 1K 上下文，KV 需求差 8 倍；50 个并发、要求首字 1 秒，和允许首字 5 秒，选卡也完全不同。这一节把 1.6 的显存公式、7.1 的带宽/算力公式、7.4 的实测曲线收成一套算法，最后给出 4090 上的算例。

**先说明口径**：本节公式给出的是**上限估算**，不是承诺。实际能开多少并发、跑多快，取决于引擎、模型结构、KV 精度、请求长度分布和调度策略；7.4 会用同一套脚本测出真实曲线，再把这里的估算替换成实测。

## 四个约束，不是一张卡

容量规划要同时过四关：

| 约束 | 问题 | 主要公式 | 不够时 |
|---|---|---|---|
| 显存 | 权重 + KV 放得下吗？ | `W + C × KV ≤ VRAM × util` | 启动 OOM，或并发/上下文被压缩 |
| 带宽 | 单请求 decode 够快吗？ | `t_step ≈ (W + C × KV_request) / BW` | 生成慢，GPU 利用率不高 |
| 算力 | prefill 和大 batch 够快吗？ | `t_prefill ≈ 2PL / (F × MFU)` | 首字慢；高并发时生成速度掉 |
| SLO | 首字和每 token 延迟达标吗？ | `TTFT ≤ SLO`、`TPOT ≤ SLO` | 加卡、降上下文、换模型或降并发 |

四个约束里，**显存决定“能不能”，带宽和算力决定“多快”，SLO 决定“值不值”**。只算显存，会得到“能跑但没人愿意用”的配置；只算吞吐，会得到“便宜但首字 10 秒”的配置。

## 第一步：算 KV 账

每 token 的 KV Cache 字节数：

```text
K = 2 × layers × kv_heads × head_dim × bytes_per_element
```

`2` 是 K 和 V 两份。`Qwen3.5-9B` 的 32 层里只有 8 层是全注意力，按 f16 算是 32 KiB/token（估算，推导见 1.6 节）。这不是通用值：标准 14B 模型是 160 KiB/token，混合注意力模型要逐层算，不能套一个公式。

每个请求的 KV：

```text
KV_request = K × L
```

8K 上下文、32 KiB/token：`32 KiB × 8192 ≈ 0.27 GB`（估算）。如果 KV 用 FP8，字节数减半；如果上下文从 8K 拉到 32K，KV 变成 4 倍。**上下文长度是线性乘数，不是“稍微长一点”。**

## 第二步：显存能开多少并发

vLLM 启动时会按 `gpu_memory_utilization` 预分配显存（引用 [\[2\]](#ref-2)）。扣掉权重和运行时开销，剩下的才是 KV 池：

```text
KV_budget = VRAM × util - W - overhead
C_vram = floor(KV_budget / KV_request)
```

以 4090 24GB（引用 [\[3\]](#ref-3)）、util 0.9、9B 4 位（W ≈ 5.7 GB）、运行时开销 1.5 GB、8K 上下文为例：

```text
KV_budget = 24 × 0.9 - 5.7 - 1.5 = 14.4 GB
C_vram = floor(14.4 / 0.27) ≈ 53
```

这是**显存上限**。实际会因 PagedAttention 的块大小、碎片、激活峰值和 `max_num_seqs` 而更低（引用 [\[1\]](#ref-1)）；如果引擎按块分配，最后一个块装不满也会浪费。7.4 会用 `--max-num-seqs` 和实际 OOM 点校准这个数。

## 第三步：带宽和算力决定多快

decode 的一步时间取带宽和算力的较大值：

```text
t_step(C) ≈ max( (W + C × KV_request) / BW, 2 × P × C / (F × eff) )
```

- 第一项是“搬权重和 KV”的时间；
- 第二项是“算这一批 token”的时间；
- `eff` 是 decode 阶段的实际算力效率，通常远低于 prefill 的 MFU。

把 9B、4 位、4090、8K 上下文代进去（估算）：

| 并发 C | 带宽项 | 算力项（eff=1.0） | 谁先撞墙 | 单请求速度（上限） |
|---|---|---|---|---|
| 1 | 5.92 ms | 0.11 ms | 带宽 | 约 169 token/s |
| 50 | 18.97 ms | 5.45 ms | 带宽 | 约 53 token/s |
| 100 | 32.2 ms | 10.9 ms | 带宽 | 约 31 token/s |

这张表解释了一个反直觉的现象：**并发从 1 加到 50，单请求速度从 169 掉到 53 token/s**。原因是 8K 上下文的 KV 读取量已经超过权重：50 个请求的 KV 一共 13.5 GB，是 5.7 GB 权重的两倍多。权重只读一遍省下的那点时间，被 KV 吃掉了。

交叉点 `B*` 在 8K 下不存在（带宽一直占主导）；把上下文缩到 1–2K，才会在几十（低效率）到一百多（理想效率）并发时看到算力接管。7.1 给了公式（引用 [\[5\]](#ref-5)）。

实际数字会低不少：注意力算子、KV 读取、调度和碎片都会吃掉效率。7.4 的实测曲线才是最终依据；这张表只用来判断“瓶颈大概在哪”。

## 第四步：首字延迟

prefill 的时间：

```text
t_prefill ≈ 2 × P × L / (F × MFU)
```

9B、8K prompt、4090 的 165.2 TFLOPS、MFU 30%：约 3.0 秒（估算）。如果 SLO 是首字 ≤ 1 秒，4090 在 8K prompt 上单卡做不到；要么缩短 prompt、要么换算力更高的卡（H100 SXM 的 3.35 TB/s 和更高算力是另一个量级，引用 [\[4\]](#ref-4)）、要么把 prefill 拆开（chunked prefill 改善的是排队和公平性，不减少总计算量，7.6 会讲）。

这个公式只算了线性部分。长上下文还要加注意力的 `O(L²)` 项，所以 16K、32K 的实际 TTFT 会比这个估算更差。**8K 以上不要只信线性外推，必须实测。**

## 第五步：算钱

容量规划最后要落到“每百万 token 多少钱”：

```text
¥/M tokens = GPU 每小时价格 / (聚合 token/s × 3600) × 1,000,000
```

按 4090 云租赁价约 ¥2/小时（估算，按 2026-11 的公开价取整；实际以下单页为准）：

| 聚合吞吐 | ¥/M tokens（估算） |
|---|---|
| 1,000 token/s | 约 ¥0.56 |
| 3,000 token/s | 约 ¥0.19 |
| 5,000 token/s | 约 ¥0.11 |

这张表比“每小时多少钱”有用得多：同样的卡，吞吐翻 5 倍，单位成本就降到五分之一。**容量规划的目标不是买最便宜的卡，而是把每百万 token 的成本压到可接受，同时守住 SLO。**

## 一张容量表

把上面的账合起来，看三个典型场景（9B 4 位、8K 上下文、4090 24GB、util 0.9、开销 1.5 GB；全部为估算）：

| 场景 | 并发 × 上下文 | 总 KV | 显存小计 | 4090 能否 | 备注 |
|---|---|---|---|---|---|
| 小 | 10 × 8K | 2.7 GB | 9.9 GB | 能 | 带宽和算力都宽裕 |
| 中 | 50 × 8K | 13.5 GB | 20.7 GB | 勉强 | 接近 0.9 利用率；TTFT 约 3 秒；单请求约 53 token/s、聚合约 2.6k token/s（估算）；必须实测 |
| 大 | 100 × 8K | 27 GB | 34.2 GB | 不能 | 2 卡、FP8 KV 或降到 4K |

“能”只表示显存放得下，不表示 SLO 达标。中等场景里，显存刚够，但 8K prompt 的 prefill 已经要几秒；如果客户要求首字 1 秒，这张表要重新算。

如果换成 2×4090 做 TP=2，显存和算力大致翻倍，但 TP 的通信开销会吃掉一部分收益；7.7 会实测“TP=2 vs 2 个副本”。对 9B 这种单卡放得下的模型，**两个副本通常比 TP=2 更划算**——副本之间不通信，调度也更简单。

## 峰值并发不是日活

容量规划最常见的错误，是拿“注册用户数”当并发数。10 个注册用户不等于 10 个并发；真实系统的峰值并发通常是日活的几个百分点（估算，取决于场景），而且集中在少数几个时段。3.6 节在 Mac 上已经看到：并发超过槽位后，首字延迟线性上升，聚合吞吐虽然还在涨，但边际收益递减。

所以容量表要写三列：**日活、峰值并发、峰值持续时间**。只写“50 人同时用”，没法选卡；写成“日活 500、峰值并发 50、每天集中 2 小时”，才能算钱。

还要留余量。显存、带宽、算力三笔账都是按理想情况估的；实际会有碎片、重试、长尾请求和模型版本变化。**建议在峰值并发上再留 20–30% 的余量**（估算），并用 7.8 的 OOM/卡顿排查验证边界。

## 动手实验

::: tip 动手实验：用实测曲线校准容量表
**假设**：显存公式预测的最大并发与实测 OOM 点的偏差在 20% 以内；带宽公式预测的单请求速度高于实测 30–50%；TTFT 随 prompt 长度超线性增长。

**命令**（先按 7.4 起服务，记录 `--max-num-seqs`、`--max-model-len`、`--gpu-memory-utilization`）：

```bash
# 用同一套参数算一遍
uv run labs/07-gpu-inference/capacity.py \
  --model qwen3.5-9b-awq --vram 24 --ctx 8192 --concurrency 50 --gpu-price 1.92

# 再用 7.4 的压测脚本扫 10/50/100 并发，记录 TTFT、TPOT、OOM。
# 8K prompt + 128 生成需要 max-model-len ≥ 8320；先按 7.4 起服务时用
# --max-model-len 16384，或把 --prompt-chars 降到 9000。
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 10,50,100 --requests 100 \
  --prompt-chars 12000 --max-tokens 128 \
  --label capacity-8k
```

**记录**：预测的最大并发、实测 OOM 点、各并发下的 TTFT/TPOT、聚合吞吐、每百万 token 成本。数据存 `benchmarks/gpu/capacity/`。如果预测和实测差得远，先查 KV 精度、块大小、`max_num_seqs` 和运行时开销，再改公式。
:::

## 工程含义

**先定 SLO，再选卡。** “能跑”不是目标；首字 ≤ 2 秒、每 token ≤ 50 ms、每百万 token ≤ ¥1，这些才是。SLO 一变，卡型和数量都可能变。

**显存、带宽、算力分开算。** 显存不够是 OOM，带宽不够是单请求慢和高并发掉速，算力不够是首字慢。三种现象对应三种解法，不要用“加显存”解决“带宽不够”。

**小模型优先副本，大模型才考虑 TP。** 9B 这种单卡放得下的模型，2 个副本通常比 TP=2 简单、便宜、延迟更稳；只有模型放不下一张卡，或者单卡算力/带宽确实不够时，TP 才值得。

**容量表要跟着模型和引擎更新。** 换量化、换 KV 精度、换 vLLM 版本、换上下文长度，表都要重算。把 `capacity.py` 和实测 CSV 留在仓库里，比记住一个数字可靠。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>vLLM Project. *Engine Arguments*（`gpu_memory_utilization`、`max_model_len`、`max_num_seqs`）. https://docs.vllm.ai/en/latest/configuration/engine_args.html
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>NVIDIA. *GeForce RTX 4090 规格页*（24 GB、1008 GB/s）. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA H100 Tensor Core GPU 规格页*（80 GB HBM3、3.35 TB/s、NVLink 900 GB/s）. https://www.nvidia.com/en-us/data-center/h100/
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Williams, S., Waterman, A., Patterson, D. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM, 2009. https://dl.acm.org/doi/10.1145/1498765.1498785
