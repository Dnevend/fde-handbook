---
title: 性能测量方法：怎么测才可信
---

# 性能测量方法：怎么测才可信

> **本章要回答的问题**：我自己测的速度为什么和网上差很远？怎么测才能拿来比较？

## 先分清三层测量

同一个模型，"快慢"至少有三个不同的量，混在一起就没法讨论：

| 层 | 工具 | 测到什么 | 典型用途 |
|---|---|---|---|
| 引擎层 | `llama-bench`、`/metrics` | 裸推理能力：固定 prompt、无网络、无模板、无排队 | 选硬件、比量化档 |
| 接口层 | 客户端脚本（本节） | 整条链路：HTTP、模板、tokenize、排队、流式回传 | 定 SLO、做容量规划 |
| 用户层 | 端到端脚本 | 加上检索、前端渲染、重试 | 验收 |

三层数字必然不同，通常是引擎层最快、用户层最慢。**比较只在同一层内有效。**

## llama-bench 在测什么

```bash
llama-bench -m "$MODEL" -p 512 -n 128 -r 5 -d 0,4096,16384 -o md
```

四个参数的含义：

- `-p 512`：喂 512 个 token 的 prompt，测 prefill 速度（输出记作 `pp512`）；
- `-n 128`：生成 128 个 token，测 decode 速度（记作 `tg128`）；
- `-r 5`：每组配置重复 5 次，输出均值 ± 标准差；
- `-d 0,4096,16384`：先在指定深度上填上下文，再测 decode，用来看 KV 变长后的衰减。

它有意去掉了服务化的所有开销，所以**适合回答"这个硬件/量化档能跑多快"，不适合回答"用户等多久"**。这一点很多人搞错：拿 `tg128` 当接口延迟报到产品那边，上线必然对不上。

本机参考值（实测，见 `benchmarks/prefill-decode/`）：`pp512` 约 209 tok/s，`tg128` 约 17.2 tok/s，5 次重复的标准差都在 1% 以内。把上下文加深到 4096 和 16384 再测，`tg` 降到 16.4 和 14.5——**KV 变长会拖慢 decode**，这也是"上下文不是免费额度"的又一处证据。

## 接口层怎么测

三个指标，定义要写死：

- **首字延迟（TTFT）**：请求发出 → 第一个内容分片到达。包含排队、模板渲染、tokenize、prefill。
- **每 token 时间（TPOT / ITL）**：decode 阶段平均每个 token 的间隔。用 `(总时长 − TTFT) ÷ 生成 token 数` 估算。
- **吞吐**：单位时间内系统产出的 token 数。单请求吞吐和聚合吞吐是两个数。

本机实测（见 `benchmarks/client-bench/api-bench.csv`，M4/24GB、llama.cpp 0.7.0、8192 上下文、单槽位）：

| 场景 | prompt token | 首字延迟 | 总时长 | decode 速度 |
|---|---|---|---|---|
| 短 prompt，冷 | 26 | 0.26 s | 1.73 s | 17.0 tok/s |
| 短 prompt，热 | 20 | 0.10 s | 1.07 s | 17.3 tok/s |
| 中 prompt，冷 | 167 | 0.98 s | 2.56 s | 17.1 tok/s |
| 中 prompt，热 | 161 | 0.09 s | 1.67 s | 17.2 tok/s |
| 长 prompt，冷 | 711 | 3.56 s | 5.21 s | 17.1 tok/s |
| 长 prompt，热 | 705 | 0.10 s | 1.74 s | 17.1 tok/s |

这张表最重要的一列是**冷热对比**。冷请求的首字延迟随 prompt 长度线性上涨（26→711 token，0.26→3.56 秒，约 200 tok/s 的 prefill 速度，与 `llama-bench` 的 `pp512` 处同一量级）；热请求几乎不变（0.1 秒），因为前缀缓存把整段 prefill 省掉了。

**这意味着：只报热请求的首字延迟，等于把缓存当成了引擎性能。** 长文档 RAG 场景尤其明显：同一份文档反复提问时延迟很低，换一份文档立刻慢一个数量级。两种都要报，并说明命中率假设。

## 噪声与显著性

- **重复至少 5 次，报中位数和极差**。单次测量在 Mac 上很容易被后台任务、热降频、swap 干扰。
- **先测噪声底**。同一配置连续测的极差就是这台机器的噪声；小于它的差异不能当结论。上表里热请求的极差在 0.1–3.3 秒之间，说明缓存命中本身不稳定，这种情况要多跑几轮再看中位数。
- **固定变量**。上下文、槽位、模板、思考开关、`max_tokens`、温度，任何一个变了都不是同一个实验。
- **记录机器状态**。有没有开浏览器、有没有别的服务在跑、是否接了电源。这些看似无关，在 24GB 统一内存上影响很大。

尾巴也要看。平均延迟好看、p99 很难看，是排队型系统的典型特征（引用 [\[3\]](#ref-3)）。上表只给了中位数和极差，正式验收要补 p95/p99。

## 一份可复现的记录长什么样

每条测量至少要写清：

```text
硬件：Apple M4，24GB 统一内存（Apple M4，10 核）
系统：macOS 26.2 (25C56)
引擎：llama.cpp 0.7.0 (build 11669, commit 620d2225c)
模型：Qwen3.5-9B-Q4_K_M.gguf，5.29 GiB，sha256 03b74727…
启动：llama-server -c 8192 -np 1 -ngl 99
采样：temperature 0，seed 42，max_tokens 128，思考关闭
测试：冷/热各 3 次，取中位数
命令：uv run python labs/03-deployment/bench_client.py
```

前七行由 `bash benchmarks/record-env.sh` 帮你打出来，后面的要自己写。缺哪一行，别人就没法复现，包括三个月后的你。

## 五个常见错误

1. **拿热身请求当结果。** 第一次请求包含模型加载、缓存冷启动，必然慢。要么预热，要么明确标成"冷启动"。
2. **只测一次就下结论。** 单次差异很可能小于噪声。
3. **不同上下文长度放在一张表里比。** 8192 和 2048 的 KV 相差 4 倍，速度当然不同。
4. **忘记 `max_tokens` 的影响。** 输出越长，总时长越大；思考模式还会先在预算里烧掉几百 token。
5. **跨硬件搬运数字。** M4 的 tok/s 不能给 4090 做承诺，反过来也不行。能搬的是公式和趋势。

## 工程含义

**先定义指标，再开始测。** 产品说"要快"，要追问：首字多快、整段多长、多少人同时用。三个答案对应三个不同的优化方向。

**把测量脚本当交付物。** 客户环境和你不同，唯一能对齐的是同一份脚本、同一套口径。脚本进仓库，参数写死，别靠记忆。

**对比实验一次只改一个变量。** 换量化档时不要顺手改槽位，换引擎时不要顺手改模板。否则结果没法归因。

**接受"测不准"。** 统一内存机器上的噪声比 GPU 服务器大，这本身就是一条结论：在这类机器上做容量承诺，要留更大的余量。

## 动手实验

::: tip 动手实验：冷热缓存与噪声底
**假设**：冷请求的首字延迟随 prompt 长度线性增长；热请求接近固定开销；同配置重复测量的极差就是噪声底。

**命令**（先起服务）：

```bash
# 终端 1
labs/03-deployment/serve.sh

# 终端 2
cd labs
BASE_URL=http://127.0.0.1:8080/v1 uv run python 03-deployment/bench_client.py

# 引擎层基线，用来和接口层对照
source ../labs/env.sh
llama-bench -m "$MODEL" -p 512 -n 128 -r 5 -o md
```

**记录**：六种场景（短/中/长 × 冷/热）的首字延迟中位数、总时长、decode 速度、极差；`llama-bench` 的 `pp512` 与 `tg128`。数据存 `benchmarks/client-bench/`。

**要算的三个数**：
1. **固定开销**：热请求的首字延迟（约 0.1 秒），是链路本身的成本；
2. **prefill 速度**：`(冷首字 − 热首字) ÷ prompt token 数`，和 `llama-bench` 的 `pp512` 应该同量级；
3. **噪声底**：同配置多次测量的极差，写进下一次实验的判读标准。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp server 文档*（`/metrics` 指标、`/slots`）与 `llama-bench` 工具说明. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>vLLM. *Metrics*（TTFT、TPOT/ITL 的定义与采集）. https://docs.vllm.ai/en/latest/design/metrics.html
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Dean, J., Barroso, L. A. *The Tail at Scale*. Communications of the ACM, 2013. https://doi.org/10.1145/2408776.2408794
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>MLCommons. *MLPerf Inference: Datacenter benchmark rules*（延迟指标的统一定义与报告要求）. https://mlcommons.org/benchmarks/inference-datacenter/
