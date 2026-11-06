---
title: GPU 硬件常识：显存 / 带宽 / 算力三者的关系，消费卡与数据中心卡，NVLink 与 PCIe
---

# GPU 硬件常识：显存 / 带宽 / 算力三者的关系，消费卡与数据中心卡，NVLink 与 PCIe

> **本章要回答的问题**：显存、带宽、算力三者怎么互相制约？消费卡和数据中心卡差在哪？NVLink 值不值得加钱？

买卡时最容易犯的错，是只问“多少 TFLOPS”。推理不是训练：单请求逐 token 生成时，GPU 大部分时间在等显存把权重搬过来，算力反而闲着。先建立三个数的关系，再看卡型，最后才轮到 NVLink。

## 三个数，三种瓶颈

把一次推理拆成两个阶段（1.3 节）：prefill 一次读完整段 prompt，decode 一个 token 一个 token 地生成。两个阶段的瓶颈不一样。

| 资源 | 决定什么 | 不够时先看到什么 |
|---|---|---|
| 显存（VRAM） | 模型能不能放下，KV Cache 能开多大 | 启动直接 OOM；或上下文/并发被压得很小 |
| 显存带宽（BW） | 单请求 decode 能有多快 | 首字正常，但生成慢；GPU 利用率不高 |
| 算力（FLOPS） | prefill 有多快，大 batch 的吞吐上限 | 首字慢；并发一高，生成速度掉得厉害 |

三个数不是并列的。**显存是门槛，带宽是单请求和长上下文并发的速度，算力是 prefill 和短上下文大 batch 的天花板。** 下面把这句话拆成公式。

## decode 为什么看带宽

decode 每一步只生成一个 token，但要把这一步用到的权重和这个请求的 KV 都从显存读一遍。单个请求的一步可以近似写成：

```text
t_step ≈ (W + K × L) / BW
```

- `W`：这一步要读的权重字节数。对 9B 的 4 位量化模型，约 5.7 GB（估算，按 1.6 节的 `Q4_K_M` 文件大小取同一量级）；
- `K`：这个请求每 token 的 KV Cache 字节数。`Qwen3.5-9B` 在 f16 KV 下约 32 KiB/token（估算，推导见 1.6 节）；
- `L`：当前上下文长度（token 数）。上下文很短时 `K × L` 可以忽略；8K 时不能；
- `BW`：显存带宽。RTX 4090 为 1008 GB/s（引用 [\[1\]](#ref-1)）。

代进去：

- 上下文只有 1 个 token：`(5.7 GB + 32 KiB) / 1008 GB/s ≈ 5.65 ms/token`，约 177 token/s；
- 8K 上下文：`(5.7 GB + 0.27 GB) / 1008 GB/s ≈ 5.9 ms/token`，约 169 token/s。

这两个数都是**理论上限**（估算）。0.3 节用带宽推出的 177 token/s 是短上下文的乐观值；上下文一长，KV 就开始吃带宽。实际还会因算子开销、显存碎片而更低，但“带宽决定单请求速度”这个方向不会变。

批量请求会改变分母。一个 batch 里，权重只读一遍，但每个请求的 KV 都要读；每个请求的 KV 是 `K × L`：

```text
t_step(B, L) ≈ (W + B × K × L) / BW
吞吐(B, L) = B / t_step(B, L)
```

当 `B × K × L` 还远小于 `W` 时，吞吐几乎随 batch 线性上涨；当 KV 项超过权重项，吞吐就撞到 `BW / (K × L)` 的天花板。这时再加 batch，只会让每个请求更慢。上下文越长，这个天花板越低——8K 上下文的 KV 读取量是 1K 的 8 倍。

## 算力用在哪：prefill 和大 batch

prefill 要把整段 prompt 过一次前向，计算量约 `2 × P × L`（P 为参数量，L 为 prompt token 数）。时间可以粗估为：

```text
t_prefill ≈ 2 × P × L / (F × MFU)
```

`F` 是 FP16/BF16 稠密算力，`MFU`（Model FLOPs Utilization）是实际达到的比例。以 9B、8K prompt、4090 的 165.2 TFLOPS（引用 [\[2\]](#ref-2)）、MFU 30% 估算：

```text
2 × 9e9 × 8192 / (165.2e12 × 0.3) ≈ 3.0 s
```

这就是长上下文首字延迟的来源。它和 decode 的 5.9 ms/token（8K 上下文）是两笔账：**首字慢，先看 prefill 的算力和 prompt 长度；生成慢，先看带宽和 batch。**

大 batch 的 decode 也可能从带宽瓶颈转向算力瓶颈。每一步的计算量约 `2 × P × B`，时间约 `2PB / (F × 效率)`。把两边相等，可以得到一个粗略的交叉点：

```text
B* ≈ (W / BW) / (2P / (F × 效率) - (K × L) / BW)
```

如果分母小于等于 0，说明在给定上下文下带宽一直占主导，不存在交叉点。9B、4 位、4090、8K 上下文代进去，KV 项比算力项大，decode 全程是带宽瓶颈；只有把上下文缩到 1–2K，才会在几十（低效率）到一百多（理想效率）并发时看到算力接管。**上下文越长，越像带宽问题；上下文越短、batch 越大，越像算力问题。**

以 8K 上下文为例（估算）：

| 并发 | KV 总量 | 带宽项 | 算力项（eff=1.0） | 谁先撞墙 | 单请求上限 |
|---|---|---|---|---|---|
| 1 | 0.27 GB | 5.9 ms | 0.11 ms | 带宽 | 约 169 token/s |
| 50 | 13.5 GB | 18.9 ms | 5.45 ms | 带宽 | 约 53 token/s |
| 100 | 27 GB | 32.2 ms | 10.9 ms | 带宽 | 约 31 token/s |

这张表解释了一个反直觉的现象：**并发从 1 加到 50，单请求速度从 169 掉到 53 token/s**。原因是 8K 上下文的 KV 读取量已经超过权重：50 个请求的 KV 一共 13.5 GB，是 5.7 GB 权重的两倍多。权重只读一遍省下的那点时间，被 KV 吃掉了。7.9 会用这张表做容量规划。这套“先看带宽、再看算力”的算法就是 roofline 模型（引用 [\[6\]](#ref-6)）。

## 消费卡与数据中心卡差在哪

把三张有代表性的卡放在一起（数字均为厂商公开规格）：

| 卡 | 显存 | 带宽 | FP16 Tensor 稠密 | 互联 | 功耗 / 形态 |
|---|---|---|---|---|---|
| RTX 4090 | 24 GB GDDR6X | 1008 GB/s（引用 [\[1\]](#ref-1)） | 165.2 TFLOPS（引用 [\[2\]](#ref-2)） | PCIe 4.0 x16；**无 NVLink**（引用 [\[1\]](#ref-1)） | 450 W；3 槽消费卡（引用 [\[1\]](#ref-1)） |
| L40S | 48 GB GDDR6 ECC | 864 GB/s（引用 [\[3\]](#ref-3)） | 362.05 TFLOPS（引用 [\[3\]](#ref-3)） | PCIe 4.0 x16 64 GB/s；无 NVLink（引用 [\[3\]](#ref-3)） | 350 W；双槽被动（引用 [\[3\]](#ref-3)） |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s（引用 [\[4\]](#ref-4)） | 989 TFLOPS（稠密；引用 [\[4\]](#ref-4)） | NVLink 900 GB/s；PCIe 5.0 128 GB/s（引用 [\[4\]](#ref-4)） | 700 W；SXM（引用 [\[4\]](#ref-4)） |

读这张表要注意三件事。

**第一，算力数字的口径不一样。** 4090 的官方页面写的是 1321 AI TOPS（引用 [\[1\]](#ref-1)），那是 FP8 稀疏口径；上表统一换成 FP16 稠密。厂商页面经常在“稠密/稀疏”“FP16/FP8/INT8”之间切换，稀疏通常翻倍。**不统一口径就比算力，等于拿公斤和磅直接比大小。**

**第二，数据中心卡贵的不只是显存。** L40S 有 ECC 显存、被动散热、NEBS Level 3、Secure Boot（引用 [\[3\]](#ref-3)）；H100 有 NVLink、MIG、更大的 HBM 带宽和 7×24 的企业支持（引用 [\[4\]](#ref-4)）。这些特性在“跑个 demo”时看不出来，在“交付给工厂连续跑三年”时决定你能不能睡着觉。

**第三，消费卡的物理形态不是为机柜设计的。** 4090 是 3 槽、304 mm、450 W、850 W 系统电源（引用 [\[1\]](#ref-1)），轴向风扇把热吹进机箱；把它塞进 1U/2U 服务器，风道和供电都要重新算。云厂商能把它装进机器，不代表你买一张塞进客户的机柜也能稳定跑。

还有一类“4090 48G”魔改卡：显存是改出来的，保修、驱动、供电都不在厂商支持范围内。**做实验可以，写进交付方案要非常谨慎。** 本书的基准始终是原厂 4090 24GB。

## NVLink 与 PCIe

单卡不存在互联问题；一旦模型放不下一张卡，或者想用两张卡各跑一个副本，互联就开始影响结果。

| 互联 | 双向带宽 | 典型场景 |
|---|---|---|
| PCIe 4.0 x16 | 64 GB/s（引用 [\[3\]](#ref-3)） | RTX 4090、L40S |
| PCIe 5.0 x16 | 128 GB/s（引用 [\[4\]](#ref-4)） | H100 PCIe / NVL |
| NVLink 3 | 600 GB/s（引用 [\[4\]](#ref-4)） | H100 NVL |
| NVLink 4 | 900 GB/s（引用 [\[4\]](#ref-4)） | H100 SXM |

RTX 4090 的规格页明确写着 NVLink 为 No（引用 [\[1\]](#ref-1)）。所以两张 4090 做张量并行（tensor parallelism，TP）时，通信只能走 PCIe，甚至可能落到主机内存。7.7 会实测这条路径的代价。

NVLink 的拓扑和代际差异见 NVIDIA 官方页面（引用 [\[5\]](#ref-5)）。它什么时候值钱？看通信发生在哪里：

- **张量并行**：每一层都要做 all-reduce，通信频繁、量大；NVLink 的收益最明显。
- **流水线并行**（pipeline parallelism，PP）：按层切分，卡之间只传激活值，通信少，但有流水线气泡；NVLink 的收益小于 TP。
- **多副本**：每个副本独立处理请求，卡之间不通信；这时 NVLink 一分钱都不值。

所以“NVLink 值不值得加钱”没有统一答案，先问三个问题：

1. 模型一张卡放得下吗？放得下，优先多副本，不需要 NVLink。
2. 放不下，必须 TP/PP 吗？是，NVLink 的收益开始出现；TP 比 PP 更吃互联。
3. 延迟敏感还是吞吐敏感？延迟敏感的大模型更值得为 NVLink 付钱；吞吐敏感的小模型，多副本通常更划算。

本书计划在 2×4090（PCIe，无 NVLink）上实测；NVLink 的带宽数字引用厂商规格，不写成实测。这样反而更贴近大多数 FDE 的现场：客户机器是 PCIe 消费卡，NVLink 是选型时的加钱项，不是默认项。

## 一张选卡清单

按顺序问，不要一上来就看算力排名：

1. **显存**：权重 + KV × 并发 + 运行时开销，放得下吗？公式在 1.6 和 7.9。
2. **带宽**：单请求 decode 的目标速度是多少？`BW ≥ (W + K × L) × 目标 token/s`。
3. **算力**：8K prompt 的首字延迟目标是多少？`F ≥ 2PL / (TTFT × MFU)`。
4. **互联**：模型一张卡放得下吗？放得下优先多副本；放不下再考虑 TP/PP 和 NVLink。
5. **可靠性**：要不要 ECC、被动散热、机柜供电、厂商支持？要，就选数据中心卡。
6. **成本**：最后算每百万 token 的成本，而不是每卡每小时的价格。7.9 给公式。

这张清单的顺序是有意的：**先排除“放不下”，再排除“太慢”，最后才比较“谁更便宜”。** 反过来做，很容易买回一张显存不够或带宽不够的卡。

## 动手实验

::: tip 动手实验：把厂商规格和实测对上
**假设**：4090 的实测 HBM 拷贝带宽能达到规格的 80–90%；FP16 矩阵乘的实测算力在 60–80% 之间；如果明显低于这个区间，先查功耗、温度和时钟，而不是怀疑代码。

**命令**（在 GPU 机器上，仓库根目录）：

```bash
# 1. 先看卡和驱动
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.memory --format=csv

# 2. 测 HBM 带宽和 FP16 矩阵乘（脚本会打印与规格的比值）
uv run labs/07-gpu-inference/gpu_probe.py \
  --spec-bandwidth 1008 --spec-tflops 165.2 \
  --output benchmarks/gpu/hardware/probe.json

# 3. 双卡时记录拓扑
nvidia-smi topo -m
```

如果 `uv` 装到的是 CPU 版 torch，进 vLLM 容器跑（容器里有 CUDA 版，注意覆盖 entrypoint）：

```bash
docker run --rm --gpus all --entrypoint python \
  -v "$PWD:/work" -w /work vllm/vllm-openai:latest \
  labs/07-gpu-inference/gpu_probe.py \
  --spec-bandwidth 1008 --spec-tflops 165.2 \
  --output benchmarks/gpu/hardware/probe.json
```

**记录**：GPU 型号、驱动版本、CUDA 版本、PyTorch 版本、实测带宽、实测 TFLOPS、与规格的比值。数据存 `benchmarks/gpu/hardware/`。如果实测带宽只有规格的一半，先看 `nvidia-smi` 的功耗和温度是不是撞墙，再看是不是被其他进程占了显存。
:::

## 工程含义

**先看带宽，再看算力。** 单请求的体感由带宽决定；算力决定首字和短上下文大 batch 的上限。买卡前先算这两笔账，别被 TFLOPS 排名带走。

**显存是门槛，不是速度。** 一张显存更大但带宽更低的卡，可能装得下更大的模型，却生成得更慢。容量和速度要分开评估。

**消费卡可以试点，交付要看全生命周期。** ECC、散热、供电、保修、驱动支持，这些在 demo 里看不见，在客户现场会变成故障单。魔改卡不要写进交付方案。

**NVLink 只在跨卡通信时值钱。** 小模型、单卡、多副本，都不需要它；大模型、TP、低延迟，才需要。先确定并行策略，再决定是否为互联加钱。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>NVIDIA. *GeForce RTX 4090 规格页*（24 GB GDDR6X、PCIe Gen 4、NVLink: No、450 W）. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA Ada GPU Architecture*（RTX 4090 的 FP16 Tensor 165.2 TFLOPS 稠密 / 330.3 稀疏口径）. https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA L40S GPU 规格页*（48 GB GDDR6 ECC、864 GB/s、PCIe Gen4 x16 64 GB/s、FP16 Tensor 362.05/733 TFLOPS、350 W、NVLink: No）. https://www.nvidia.com/en-us/data-center/l40s/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA H100 Tensor Core GPU 规格页*（80 GB HBM3、3.35 TB/s、NVLink 900 GB/s、PCIe Gen5 128 GB/s、FP16 Tensor 1,979 TFLOPS 稀疏）. https://www.nvidia.com/en-us/data-center/h100/
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>NVIDIA. *NVLink and NVLink Switch*. https://www.nvidia.com/en-us/data-center/nvlink/
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>Williams, S., Waterman, A., Patterson, D. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM, 2009. https://dl.acm.org/doi/10.1145/1498765.1498785
