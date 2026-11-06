---
title: 监控与诊断：nvidia-smi、nvtop、DCGM；利用率、显存、功耗、温度与降频
---

# 监控与诊断：nvidia-smi、nvtop、DCGM；利用率、显存、功耗、温度与降频

> **本章要回答的问题**：nvidia-smi 里哪个数字才说明 GPU 真的忙？显存占满但利用率很低是怎么回事？

先记住一句话：**GPU 利用率 100% 不等于算力打满。** `nvidia-smi` 的 `utilization.gpu` 只表示“这一秒有 kernel 在跑”，不表示这些 kernel 在高效地算。decode 阶段大量时间在等显存，利用率可能很高，算力却闲着。监控要看四个数，再结合引擎自己的指标。

## 先看四个数

| 指标 | `nvidia-smi` 字段 | 说明什么 | 异常时先怀疑 |
|---|---|---|---|
| GPU 利用率 | `utilization.gpu` | 有 kernel 在跑的时间占比 | 低：CPU/网络/排队；高：不一定算力打满 |
| 显存 | `memory.used` / `memory.total` | 权重、KV Cache、激活、碎片 | 接近上限：KV 压力、并发过高、泄漏 |
| 功耗 | `power.draw` / `power.limit` | 是否撞到功耗墙 | 长期贴上限：降频、散热不足 |
| 温度 / 时钟 | `temperature.gpu`、`clocks.current.sm` | 是否热降频 | 温度高、时钟低：风道/风扇/环境温度 |

四个数要一起看。只看利用率，会把“内存带宽瓶颈”误判成“算力瓶颈”；只看显存，会把 vLLM 的正常预分配误判成泄漏。

## nvidia-smi 的正确用法

一次性快照：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.sm \
  --format=csv,noheader,nounits
```

按秒采样：

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 1
```

看拓扑（多卡必做）：

```bash
nvidia-smi topo -m
```

看降频原因和详细状态：

```bash
nvidia-smi -q -d PERFORMANCE
nvidia-smi -q -d POWER,TEMPERATURE,CLOCK,MEMORY
```

字段名必须用点号形式（`utilization.gpu`、`memory.used`、`power.draw`、`temperature.gpu`、`clocks.current.sm`），不能写成 `gpu_util_pct` 这种自定义名字；`nvidia-smi` 会直接报 `Field ... is not a valid field to query`（引用 [\[1\]](#ref-1)）。

## 利用率为什么骗人

`utilization.gpu` 的定义是“过去采样周期内，至少有一个 kernel 在 GPU 上执行的时间比例”。它不区分：

- 这个 kernel 是在做矩阵乘，还是在等显存；
- 是在做有用的 prefill，还是在做低效的逐 token decode；
- 是一个大 kernel，还是一堆小 kernel 在排队。

所以会出现两种反直觉的现象：

**现象一：利用率 95%，但 token/s 很低。** 单请求 decode 是典型的内存带宽瓶颈：每个 token 都要把权重读一遍，SM 大部分时间在等 HBM。此时利用率高是正常的，但算力利用率很低。判断标准不是利用率，而是**实测 token/s 和理论带宽上限的比值**（7.1 给了公式）。

**现象二：利用率 30%，但吞吐不低。** 高并发时，vLLM 的调度、KV 管理、CPU 侧的分词/网络都可能让 GPU 有空隙；只要 TTFT/TPOT 达标，30% 不一定是问题。反过来，如果队列在涨、TTFT 在涨，30% 就说明瓶颈不在 GPU 算力，而在上游。

**结论：利用率是“有没有在跑”，不是“跑得好不好”。** 真正的健康指标是引擎侧的 TTFT、TPOT、吞吐、队列长度和 KV Cache 使用率（引用 [\[3\]](#ref-3)）。

## 显存占满但利用率低

vLLM 启动时会按 `gpu_memory_utilization` 预分配显存，把大部分剩余显存划给 KV Cache 池（引用 [\[6\]](#ref-6)）。所以“显存占满”在 vLLM 里通常是**设计行为**，不是泄漏。

要区分三种情况：

1. **启动后就占满，服务正常**：预分配，正常。看日志里的 KV cache blocks 和 max concurrency。
2. **运行中逐渐占满，TTFT/TPOT 变差**：KV Cache 被长上下文/高并发吃满，新请求排队；看 `max_num_seqs`、请求长度分布、KV Cache 使用率。
3. **显存占满且服务 OOM/崩溃**：可能是 `max_model_len` 太大、`gpu_memory_utilization` 太高、并发超过 KV 池，或别的进程占了卡。看 7.8。

**不要用 `nvidia-smi` 的显存数字判断泄漏。** 进程退出后显存不一定会立刻释放；CUDA 上下文、缓存分配器、容器都可能持有显存。要看进程和引擎日志。

## 功耗、温度与降频

GPU 有三条常见的“墙”：

- **功耗墙**：`power.draw` 长期贴 `power.limit`，时钟会被压下来。
- **温度墙**：温度接近上限，风扇/环境温度不够，触发降频。
- **电源/供电墙**：瞬态功耗超过电源能力，表现为时钟抖动、Xid、甚至掉卡。

`nvidia-smi -q -d PERFORMANCE` 会列出 `Clocks Throttle Reasons`：`SW Power Cap`、`HW Slowdown`、`HW Thermal Slowdown`、`SW Thermal Slowdown` 等。看到这些，先查散热和功耗，不要先改模型参数。

消费卡塞进服务器尤其要注意：4090 是 3 槽、450 W、轴向风扇（引用 [\[4\]](#ref-4)），风道和供电不是为 1U/2U 机柜设计的。云厂商能把它装进机器，不代表你的机柜也能稳定跑。

## DCGM 和 nvtop

**DCGM**（Data Center GPU Manager）适合长期监控和健康检查：它能采利用率、显存、功耗、温度、ECC、XID 等指标，并可通过 `dcgm-exporter` 导出到 Prometheus（引用 [\[2\]](#ref-2)）。但 GeForce 卡不是所有字段都支持；先跑 `dcgmi discovery -l` 看能发现什么，再决定采集哪些字段。不要假设“DCGM 有，GeForce 就一定有”。

**nvtop** 是交互式查看器，适合快速看多卡、进程和显存；它不负责长期记录（引用 [\[5\]](#ref-5)）。临时排查用它，正式数据用 `nvidia-smi --query-gpu` 或 DCGM 落 CSV。

## 一张诊断表

| 现象 | 先看 | 常见原因 |
|---|---|---|
| 利用率低、TTFT 高 | 队列长度、CPU、网络、请求长度 | 请求排队、prefill 太长、上游慢 |
| 利用率高、token/s 低 | 带宽上限、batch、KV 长度 | 内存带宽瓶颈、KV 太大、batch 太小 |
| 显存接近上限、服务变慢 | KV Cache 使用率、`max_num_seqs` | 并发/上下文超过 KV 池 |
| 显存占满、进程崩溃 | 启动日志、`max_model_len`、`gpu_memory_utilization` | 启动 OOM、KV 分配失败 |
| 功耗贴上限、时钟下降 | `nvidia-smi -q -d PERFORMANCE` | 功耗墙、散热不足、电源不够 |
| 温度高、风扇满转 | 风道、环境温度、灰尘 | 热降频 |
| 多卡吞吐不线性 | `nvidia-smi topo -m`、NCCL 日志 | PCIe/无 NVLink、TP 通信开销 |

## 动手实验

::: tip 动手实验：四种状态下的 GPU 时序
**假设**：空载、prefill 重、decode 重、显存将满四种状态下，利用率、功耗、温度和时钟的形态明显不同；只看利用率无法区分 prefill 和 decode。

**命令**（GPU 机器上，另开终端）：

长 prompt 需要 `max-model-len ≥ prompt + max_tokens`；先把服务开到 16384，或把 `--prompt-chars` 降到 9000，否则 8K 那一组会直接 400。

```bash
# 空载 30 秒
uv run labs/07-gpu-inference/monitor_gpu.py --duration 30 --interval 1 --label idle

# prefill 重：长 prompt、短输出
uv run labs/07-gpu-inference/monitor_gpu.py --duration 60 --interval 1 --label prefill &
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 1 --requests 4 --prompt-chars 12000 --max-tokens 16 --label prefill

# decode 重：短 prompt、长输出
uv run labs/07-gpu-inference/monitor_gpu.py --duration 60 --interval 1 --label decode &
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 8 --requests 32 --prompt-chars 200 --max-tokens 256 --label decode

# 显存接近上限：48 × 8K ≈ 12.9 GB KV，加上权重/开销约 20 GB
uv run labs/07-gpu-inference/monitor_gpu.py --duration 60 --interval 1 --label memory &
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 48 --requests 64 --prompt-chars 12000 --max-tokens 64 --label memory
```

**记录**：四段时序的 GPU 利用率、显存、功耗、温度、SM 时钟均值/峰值；同一时段的 TTFT、TPOT、聚合吞吐；`nvidia-smi topo -m`。数据存 `benchmarks/gpu/monitor/`。如果某一段的利用率和吞吐对不上，回到 7.1 的带宽/算力公式找原因，不要只看利用率。
:::

## 工程含义

**监控 SLO，不要监控风扇。** 用户感知的是首字延迟、每 token 延迟和错误率；GPU 利用率只是解释这些指标的证据。告警应该挂在 TTFT/TPOT/队列长度/OOM 上，而不是“利用率低于 50%”。

**先记录，再解释。** 没有时序数据，看到 100% 也说不清是算力打满还是内存瓶颈。把 `nvidia-smi --query-gpu` 落成 CSV，和压测的 TTFT/TPOT 对齐，才能下结论。

**降频是硬件问题，不是模型问题。** 功耗墙、温度墙、供电不足，调 `max_num_seqs` 只能缓解，不能根治。先看 `Clocks Throttle Reasons`。

**多卡先看拓扑。** `nvidia-smi topo -m` 会告诉你两张卡是 PCIe 还是 NVLink、跨不跨 NUMA。拓扑不对，TP 的通信开销会把收益吃光（7.7）。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>NVIDIA. *nvidia-smi*（查询字段、`--query-gpu`、`-q -d`）. https://docs.nvidia.com/deploy/nvidia-smi/index.html
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA DCGM Documentation*. https://docs.nvidia.com/datacenter/dcgm/latest/
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>vLLM Project. *Metrics*（TTFT、TPOT、吞吐、KV Cache 使用率）. https://docs.vllm.ai/en/latest/usage/metrics.html
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>NVIDIA. *GeForce RTX 4090 规格页*（450 W、3 槽）. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Syllo. *nvtop*（交互式 GPU 监控）. https://github.com/Syllo/nvtop
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>vLLM Project. *Engine Arguments*（`gpu_memory_utilization`、`max_num_seqs`）. https://docs.vllm.ai/en/latest/configuration/engine_args.html
