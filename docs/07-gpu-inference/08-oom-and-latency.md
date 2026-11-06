---
title: 卡顿与 OOM 排查：从现象到原因的诊断流程图
---

# 卡顿与 OOM 排查：从现象到原因的诊断流程图

> **本章要回答的问题**：线上突然卡顿或者 OOM，怎么从现象倒推到原因？

线上出问题，最忌讳的是“先重启再说”。重启会清掉现场，下一次还会犯。这一节给一张从现象到原因的诊断流程图：先分四类现象，再按证据一步步缩小范围，最后才改配置。

## 先分四类现象

| 现象 | 典型表现 | 第一现场 |
|---|---|---|
| 启动 OOM | 服务起不来，日志里有 `OutOfMemoryError` / `No available memory` | 启动日志、`nvidia-smi` |
| 运行时 OOM | 服务跑着跑着崩溃/500，显存突然打满 | 崩溃日志、`dmesg`、`nvidia-smi` |
| 首字慢 | TTFT 飙升，生成速度正常 | 队列、prefill、prompt 长度 |
| 生成慢/吞吐掉 | TPOT 变大，聚合吞吐下降 | KV 压力、batch、带宽、降频 |

四类现象的原因和解法完全不同。先确定是哪一类，再往下查。

## 一张诊断流程图

```text
服务起来了吗？
├─ 没起来
│  ├─ 启动日志 OOM
│  │  ├─ 权重 + KV 超过显存 → 降 max-model-len / 换量化 / 换卡
│  │  ├─ gpu-memory-utilization 太高 → 降到 0.85–0.90（经验值）
│  │  └─ 别的进程占着显存 → nvidia-smi 看进程，清掉
│  └─ 不是 OOM
│     ├─ 驱动/CUDA/容器问题 → 7.2
│     └─ 模型路径/权限/端口问题 → 看日志第一条 ERROR
├─ 起来了，但请求失败
│  ├─ 400：prompt + max_tokens 超过 max-model-len → 降 prompt/输出或调大上限
│  ├─ 500：运行时 OOM → 看下面的运行时 OOM
│  └─ 超时：队列太长/上游慢 → 看 TTFT 和队列
├─ 起来了，首字慢
│  ├─ 队列在涨 → 并发超过服务能力；加副本/降并发
│  ├─ 队列不长，prompt 很长 → prefill 慢；chunked prefill / 降长度 / 换算力
│  └─ 前缀重复但没命中 → Prefix Caching 没开/前缀不一致
└─ 起来了，生成慢
   ├─ KV 快满 → 降并发/上下文，或加卡
   ├─ 功耗/温度撞墙 → 看 Clocks Throttle Reasons
   ├─ 其他进程抢卡 → nvidia-smi 看进程
   └─ 带宽瓶颈 → 7.1 的公式，别调算力参数
```

这张图不是穷举，但它强迫你先回答“哪一类”，而不是一上来就改 `max-num-seqs`。vLLM 官方的 Troubleshooting 清单可以和图一起用（引用 [\[1\]](#ref-1)）。

## 每条现象的证据

**启动 OOM**：

```bash
docker logs --tail 200 <容器名>
nvidia-smi
```

看日志里是权重加载阶段还是 KV Cache 分配阶段。`No available memory for the cache blocks` 通常是 KV 池不够；`CUDA out of memory` 可能是权重/激活/其他进程。

**运行时 OOM**：

```bash
docker logs --tail 200 <容器名>
dmesg -T | tail -50
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

`dmesg` 里的 `Xid` 是 NVIDIA 驱动的错误码；Xid 13/31/43 等含义不同，按官方文档查（引用 [\[2\]](#ref-2)）。如果是 Xid，先怀疑硬件/驱动/供电，不要先改模型。

**首字慢**：

```bash
# vLLM 的 metrics（如果开了 /metrics）
curl -s http://127.0.0.1:8000/metrics | \
  grep -E "time_to_first_token|num_requests_waiting|kv_cache_usage|gpu_cache_usage"
```

看队列长度、TTFT、KV Cache 使用率；再看 prompt 长度分布。队列长是容量问题，队列不长是 prefill/前缀问题。

**生成慢/吞吐掉**：

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.current.sm \
  --format=csv,noheader,nounits -l 1
nvidia-smi -q -d PERFORMANCE
```

看功耗、温度、时钟和 `Clocks Throttle Reasons`；如果 GPU 利用率高但 token/s 低，回到 7.1 的带宽公式。

## OOM 的两种

**启动 OOM**：模型加载或 KV 池分配时失败。常见原因：

- `max-model-len` 设得比模型支持的长度还大；
- `gpu-memory-utilization` 太贴近 1.0，没给激活/上下文留余量；
- 权重 + KV + 开销超过显存；
- 别的进程（Jupyter、另一个 vLLM、训练任务）占着显存。

**运行时 OOM**：服务跑起来后，某些请求/并发把显存推到上限。常见原因：

- 并发超过 KV 池能承受的数量；
- 长 prompt 突然变多，KV 按长度吃显存；
- 激活峰值（大 batch、长 prefill）超过预留；
- 内存碎片/缓存分配器持有显存（PyTorch 的缓存分配器行为见引用 [\[3\]](#ref-3)）。

区分方法：看 OOM 发生在启动阶段还是请求阶段；看日志里是权重加载、KV 分配还是 forward。两者的解法不同：启动 OOM 调配置/换卡，运行时 OOM 降并发/限长/加卡。

## 卡顿的三种

**排队**：请求数超过服务能力，TTFT 线性上升，TPOT 可能正常。看队列长度和并发；解法是加副本、降并发、或缩短单请求时间。

**Prefill 堵车**：长 prompt 的 prefill 占住 GPU，短请求排队；TTFT p95 飙升，平均 TTFT 可能还好。看 prompt 长度分布和 chunked prefill 配置；解法是 chunked prefill、限制最大 prompt、或把长请求分流。

**降频**：功耗/温度撞墙，时钟下降，TPOT 变大；GPU 利用率可能还是很高。看 `nvidia-smi -q -d PERFORMANCE` 的 throttle reasons；解法是散热/供电/功耗上限，不是模型参数。

## 常见错误信息

| 错误 | 含义 | 先查 |
|---|---|---|
| `ValueError: The model's max seq len ... is larger than ...` | `max-model-len` 超过模型支持 | 模型 config、`--max-model-len` |
| `torch.OutOfMemoryError: CUDA out of memory` | 显存不够 | 权重/KV/并发/其他进程 |
| `No available memory for the cache blocks` | KV 池不够 | `max-model-len`、并发、`gpu-memory-utilization` |
| `CUDA error: an illegal memory access` | kernel/驱动/硬件 | 驱动版本、Xid、复现步骤 |
| `Xid ...` in `dmesg` | 驱动/硬件错误 | NVIDIA Xid 文档、供电/散热 |
| `Connection refused` / 超时 | 服务没起/端口/防火墙 | 容器日志、端口、SSH 隧道 |

参数的准确含义和默认值见 vLLM 引擎参数文档（引用 [\[4\]](#ref-4)）。

## 动手实验

::: tip 动手实验：人为复现三种故障
**假设**：启动 OOM、运行时 OOM、prefill 堵车都能在 4090 上人为复现；日志里的第一条 ERROR 能区分它们。

**命令**（在单卡 session 末尾做，避免把正常实验搞崩）：

```bash
# 1) 启动 OOM：先起一个占住显存的实例，再起第二个
bash labs/07-gpu-inference/serve_vllm.sh --gpu-mem-util 0.90 --name mem-holder
bash labs/07-gpu-inference/serve_vllm.sh --gpu-mem-util 0.90 --port 8001 --name oom-start
# 第二个会在启动时报 No available memory / CUDA out of memory；
# 如果第二个也起来了，说明第一张卡没占满：把 holder 的 --gpu-mem-util 调到 0.95。
# 记录 docker logs 的第一条 ERROR，然后停掉两个实例：
bash labs/07-gpu-inference/serve_vllm.sh --stop --name oom-start
bash labs/07-gpu-inference/serve_vllm.sh --stop --name mem-holder

# 2) 运行时 OOM：高 max-num-seqs + 高并发长 prompt
# 如果只排队不 OOM，说明调度器在保护显存；记录这个边界，不要硬造崩溃。
bash labs/07-gpu-inference/serve_vllm.sh \
  --max-num-seqs 128 --max-model-len 16384 --gpu-mem-util 0.95 --name oom-runtime
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 128 --requests 256 --prompt-chars 12000 --max-tokens 64 --label oom-runtime

# 3) prefill 堵车：长 prompt + 短请求混排，看 TTFT p95
# 先把服务开到 max-model-len ≥ 12288，或把 --prompt-chars 降到 9000。
uv run labs/07-gpu-inference/bench_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 8 --requests 32 --prompt-chars 12000 --max-tokens 32 --label prefill-jam
```

**记录**：每种的完整报错、`nvidia-smi` 现场、`dmesg` 里的 Xid、TTFT/TPOT、显存曲线。数据存 `benchmarks/gpu/oom/`。**先记录，再重启**；重启会清掉现场。
:::

## 工程含义

**先分类，再动手。** 启动 OOM、运行时 OOM、首字慢、生成慢，四类现象的解法不同；不分类就调参，等于蒙。

**保留现场。** 日志、`nvidia-smi` 时序、`dmesg`、请求分布，至少留一份；重启前先抓。

**告警挂在 SLO 上。** TTFT/TPOT/错误率/队列长度比“GPU 利用率”更能反映用户体感；利用率只用来解释原因。

**把 runbook 写下来。** 每次故障后补一条“现象 → 证据 → 原因 → 解法”；下次遇到同类问题，按图走，不靠记忆。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>vLLM Project. *Troubleshooting*. https://docs.vllm.ai/en/latest/usage/troubleshooting.html
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>NVIDIA. *Xid Errors*. https://docs.nvidia.com/deploy/xid-errors/index.html
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>PyTorch. *CUDA Memory Management*（OOM 与缓存分配器）. https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>vLLM Project. *Engine Arguments*（`max_model_len`、`gpu_memory_utilization`、`max_num_seqs`）. https://docs.vllm.ai/en/latest/configuration/engine_args.html
