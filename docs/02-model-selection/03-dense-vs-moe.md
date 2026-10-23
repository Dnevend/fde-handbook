---
title: Dense 与 MoE：总参数和激活参数
---

# Dense 与 MoE：总参数和激活参数

> **本章要回答的问题**：MoE 模型标着几十 B 参数，为什么 24GB 内存也能跑？总参数和激活参数差在哪？

## 一个模型里有很多个"专家"

稠密模型（dense）的每一层前馈网络都只有一份权重，每个 token 都要经过它，没有选择。

混合专家模型（Mixture of Experts，MoE）把这一层拆成很多份"专家"，再加一个路由器（router）：对每个 token，路由器算出各个专家的分数，只选分数最高的几个参与计算，其余不激活。这个设计最早由 Shazeer 等人系统提出（引用 [\[1\]](#ref-1)），Switch Transformer 把它简化成"每个 token 只选一个专家"，从而把参数量推到万亿级（引用 [\[2\]](#ref-2)）；Mixtral 用"8 个专家选 2 个"证明了这条路在开源模型上同样可行（引用 [\[3\]](#ref-3)）。DeepSeek-V3 则用更细的专家粒度和负载均衡策略，把这条路线推到数百 B 规模（引用 [\[5\]](#ref-5)）。

于是就出现了两个不同的参数量：

- **总参数（total parameters）**：所有专家加起来的规模，决定**模型文件多大、要占多少内存**；
- **激活参数（active parameters）**：每个 token 实际参与计算的那部分，决定**算力和带宽开销**。

`Gemma-4-26B-A4B` 就是这个结构：总参数 26B，每个 token 激活约 4B；`config.json` 里有 128 个专家，每次选 8 个（引用 [\[4\]](#ref-4)）。

## 为什么"省算力不省内存"

这是 MoE 最容易被误解的地方。

**所有专家都必须常驻内存。** 路由器事先不知道下一个 token 会用哪 8 个专家，128 个专家里任何一个都可能被选中，所以整个模型文件都得加载进来。16.95 GB 的 `Q4_K_M`，一分不少。

**但每个 token 只读激活的那部分。** 按 1.5 节的带宽公式，每 token 需要读取的字节数大致是 `总字节 × 激活比例`：

```
16.95 GB × (4 / 26) ≈ 2.6 GB
```

在 M4 的 120 GB/s 带宽下，理论上限约 46 token/s（估算）——比 9B dense 的 21 token/s 还快一倍多。

这个估算有一个重要前提：**访存是连续的、高效的**。真实情况里，128 个专家分散在内存各处，每个 token 选中的 8 个不同，读取会碎片化；路由本身也有开销；不同 token 选的专家不同，批处理时可能需要读入更多专家的权重。所以实测往往达不到理论上限，差距有多大取决于引擎实现和专家分布。这是 2.3 节最值得亲手测的一件事。

## 一张对照表

| 维度 | Dense（`Qwen3.5-9B`） | MoE（`Gemma-4-26B-A4B`） |
|---|---|---|
| 总参数 | 9B | 26B |
| 激活参数 | 9B | 约 4B |
| 文件大小 | 5.68 GB | 16.95 GB |
| 每 token 读取（估算） | 5.68 GB | 约 2.6 GB（按激活比例） |
| 显存门槛 | 低 | 高 |
| 单 token 速度（估算上限） | 约 21 token/s | 约 46 token/s |
| 速度稳定性 | 高，几乎无波动 | 受路由和专家负载影响 |
| 量化难度 | 低，各向同性 | 较高，专家之间分布不同 |
| 小显存适配 | 好 | 差（装不下就是装不下） |

一句话总结：**MoE 用内存换速度，dense 用速度换内存。**

## 什么时候 MoE 划算

MoE 的前提是"你的内存足够装下整个模型"。装得下，它能在同样的延迟预算里给你更多参数、更好的质量；装不下，它连启动都做不到，没有中间状态。

对 24GB 的 Mac，`Gemma-4-26B-A4B` 的 16.95 GB 权重加上 KV 和系统开销，已经接近上限（1.6 节估算约 19.5–20 GB）。它"能跑"，但不适合作为日常主力：开着浏览器和编辑器时，swap 会把速度优势全部吃掉。

MoE 真正舒服的场景是显存更大的机器：48GB 的 Mac、单卡 48GB 的 GPU，或者多卡服务器。这也符合它的设计初衷——用稀疏性把"大模型"塞进"单机可接受"的成本里。

## 量化 MoE 要更小心

第四部分会讲量化的通用方法，这里先提一个 MoE 特有的问题：**不同专家的权重分布不一样**。

某些专家专门处理代码，某些专门处理数字，它们的激活值范围差异很大。用一套统一的量化参数（scale）去压所有专家，可能让某些专家损失过大，而另一些几乎无损。结果是整体困惑度看起来没变，但特定能力突然下降——这种退化在通用评测里很难发现，只有在你的业务任务上才暴露。

缓解办法有两个：按专家分别统计量化参数，或者用重要性矩阵（imatrix）加权校准。第四部分会给出具体命令和评估方法。

## 待实测

::: tip 待实测：MoE 的速度与内存
**假设**：`Gemma-4-26B-A4B` 的峰值内存接近 20 GB，但 `tg128` 速度高于 `Qwen3.5-9B`；在 24GB Mac 上同时开其他程序时，速度会因 swap 明显下降。

**命令**：

```bash
# 速度
llama-bench -m Gemma-4-26B-A4B-Q4_K_M.gguf -p 512 -n 128 -r 5 -o csv
llama-bench -m Qwen3.5-9B-Q4_K_M.gguf      -p 512 -n 128 -r 5 -o csv

# 内存（观察是否触发 swap）
/usr/bin/time -l llama-bench -m Gemma-4-26B-A4B-Q4_K_M.gguf -p 512 -n 32 -r 1 -o md 2>&1 | \
  grep "maximum resident set size"
vm_stat | grep -E "Pageouts|Swapouts"
```

**记录**：两个模型的 `tg128`、峰值内存、Pageouts 增量，以及"空闲机器"与"开 10 个浏览器标签"两种条件下的速度差异。数据存 `benchmarks/moe/`。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Shazeer, N. et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv:1701.06538. https://arxiv.org/abs/1701.06538
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Fedus, W., Zoph, B., Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv:2101.03961. https://arxiv.org/abs/2101.03961
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Jiang, A. Q. et al. *Mixtral of Experts*. arXiv:2401.04088. https://arxiv.org/abs/2401.04088
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Google. *Gemma-4-26B-A4B-it `config.json`*（`num_experts: 128`, `top_k_experts: 8`）. https://huggingface.co/google/gemma-4-26B-A4B-it
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437. https://arxiv.org/abs/2412.19437
