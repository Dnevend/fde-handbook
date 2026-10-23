---
title: KV Cache：上下文为什么会吃掉内存
---

# KV Cache：上下文为什么会吃掉内存

> **本章要回答的问题**：对话越聊越慢、越长越占内存，多出来的内存存了什么？为什么叫 Cache，它缓存的是什么？

## 缓存的不是文本，是中间结果

自回归生成每走一步，都要让当前 token "看过"前面所有 token。在注意力计算里，每个 token 会产生三样东西：Query（Q）、Key（K）、Value（V）。

对已经处理过的 token 来说，它们的 K 和 V **不会因为后面新增内容而改变**。既然不变，就没必要每一步重算一遍，把它们存下来即可。这就是 KV Cache：缓存每个历史 token 在每一层的 K 和 V。

它省下的是重复计算，付出的是内存。而且这份内存随对话长度线性增长（图 1.4-1）——这是"聊得越久越吃内存"的直接原因。

<Figure
  id="1.4-1"
  src="/figures/1-4-kv-append.svg"
  alt="KV Cache 的追加结构：每一行是一个 KV 头，每一列是一个 token 位置，新 token 只追加一列，历史列永不改写"
  caption="追加，而不是重算——这就是为什么 decode 每一步的开销是「读一遍权重」而不是「把整段再算一遍」。"
  source="144 KiB/token 与 5.63 GiB 为估算，推导见本节；参数取自 Qwen3-8B 的 config.json（引用 [1]）。"
/>

## 算一笔账

KV Cache 的大小由一个很干净的公式决定：

```
每 token 字节数 = 2 × 层数 × KV 头数 × 每头维度 × 每元素字节数
总字节数        = 每 token 字节数 × 上下文 token 数
```

公式里的 `2` 是 K 和 V 各一份。以 `Qwen3-8B` 为例（参数均取自 `config.json`，引用 [\[1\]](#ref-1)）：36 层、8 个 KV 头、每头 128 维、半精度（2 字节）：

```
2 × 36 × 8 × 128 × 2 = 147,456 字节 ≈ 144 KiB / token
```

代入不同长度：

| 上下文长度 | KV Cache 大小 |
|---|---|
| 1K | 144 MiB |
| 8K | 1.13 GiB |
| 32K | 4.5 GiB |
| 40,960（模型上限） | 5.63 GiB |

（以上为估算，按 f16 计算，不含内存碎片和引擎开销。）

对比一下：这个模型的 `Q4_K_M` 权重约 5.03 GB（引用 [\[2\]](#ref-2)）。**上下文拉满时，KV Cache 比模型权重还大。** 很多人算显存只算权重，然后疑惑为什么 24GB 的卡在 32K 上下文下就 OOM，原因就在这里。

## KV 头数是关键变量

注意公式里乘的是 **KV 头数**，不是注意力头数。`Qwen3-8B` 有 32 个注意力头，但只有 8 个 KV 头——这就是分组查询注意力（Grouped-Query Attention，GQA）：多个 Query 头共享同一组 K/V（引用 [\[3\]](#ref-3)）。

GQA 是介于多头注意力（MHA）和多查询注意力（MQA）之间的折中：

| 方案 | KV 头数 | KV Cache 相对大小 | 质量 |
|---|---|---|---|
| MHA | 等于 Q 头数（32） | 1×（本例 576 KiB/token） | 最好 |
| GQA | 少于 Q 头数（8） | 1/4（144 KiB/token） | 接近 MHA |
| MQA | 1 | 1/32（18 KiB/token） | 略降 |

（表中的相对大小按 `Qwen3-8B` 的 32 个 Q 头估算。）

这层共享关系值得单独画一次（图 1.4-2）：一组 K/V 被 4 个 Q 头共用，省下来的正是三份 KV Cache。

<Figure
  id="1.4-2"
  src="/figures/1-4-gqa-sharing.svg"
  alt="GQA 的头共享关系：8 个 Query 头分成两组，每组 4 个头共享一组 KV，KV 头数因此是 Q 头数的四分之一"
  caption="公式里乘的是 KV 头数而不是注意力头数，这是最容易被漏掉的一处。"
  source="头数取自 Qwen3-8B 的 config.json（32 个 Q 头、8 个 KV 头，引用 [1]）；相对大小为估算。"
/>

MQA 的思路最早由 Shazeer 提出（引用 [\[4\]](#ref-4)），GQA 则是用少量微调把已有 MHA 模型转成 GQA 的方法（引用 [\[3\]](#ref-3)）。今天绝大多数开源模型默认用 GQA，原因很实际：**同样的显存，能装下 4 倍的上下文。**

## 换一代架构，账本会变

`Qwen3.5-9B` 的配置里有一个容易踩坑的地方：它不是纯 Transformer。32 层里有 24 层是线性注意力（linear attention），只有每第 4 层是全注意力（`full_attention_interval: 4`，引用 [\[5\]](#ref-5)）。

这意味着：

- 只有 8 层产生随上下文增长的 KV Cache；
- 线性注意力层维护的是一个**固定大小的状态**，不随对话变长而增长。

按公式重算：2 × 8（全注意力层）× 4（KV 头）× 256（头维度）× 2 字节 = 32 KiB/token。同样是 32K 上下文，约 1 GiB（估算）——只有 `Qwen3-8B` 的四分之一多一点。

这个例子说明一个方法论问题：**公式要对着 `config.json` 用，不要对着"Transformer 通常长这样"用。** 每换一个模型，先看层数、KV 头数、头维度，再看有没有混合结构或滑动窗口（sliding window）。

## Cache 的三种用法

**前缀缓存（prefix cache）**：多个请求共享相同前缀时，复用已算好的 KV。RAG 场景里 system prompt 固定，命中率很高。llama.cpp 默认开启 prompt cache，但要注意：换模型、换上下文长度、改模板都会让缓存失效。

**分页管理（paged attention）**：传统实现要求每个请求的 KV 连续存放，按最大长度预分配，浪费严重。PagedAttention 把 KV 切成固定大小的块，按需分配，像操作系统的虚拟内存一样管理（引用 [\[6\]](#ref-6)）。这是 vLLM 高吞吐的基础。

**KV 量化**：把 KV Cache 从 f16 降到 q8_0 或 q4，直接减半甚至降到四分之一。代价是精度损失，第四部分会专门评估。

## 工程含义

**上下文长度是成本，不是免费额度。** 模型标称支持 262,144 token，不代表你应该用满。每多 1K token，就多一份 KV 内存和读取时间。

**并发请求的 KV 要叠加。** 单个请求 32K 上下文占 1 GiB，10 个并发就是 10 GiB。容量规划时必须按"并发数 × 单请求 KV"算，而不是按一个请求算。

**先量再调。** 遇到 OOM，第一件事是算公式，第二件事是看实际占用，两者对不上再找原因（碎片、引擎预留、多模态投影层等）。

## 待实测

::: tip 待实测：公式与实际的差距
**假设**：实际峰值内存 = 权重 + 公式算出的 KV + 固定开销；固定开销在 0.5–2 GB 之间。

**命令**：

```bash
# 固定同一段 prompt，只改变上下文上限，观察峰值内存
printf '用一句话说明什么是 KV Cache。' > /tmp/p.txt
for c in 4096 8192 16384 32768; do
  echo "== -c $c"
  /usr/bin/time -l llama-completion -m "$MODEL" -c $c -n 16 \
    -f /tmp/p.txt --no-display-prompt 2>&1 | \
    grep -E "maximum resident set size"
done
```

**记录**：每个 `-c` 下的峰值常驻内存（字节），减去权重文件大小，再减去同一 `-c` 下 `-n 0` 的基线，得到 KV 的实测增量。与公式值比较，算出偏差比例。数据存 `benchmarks/kv-cache/`。

**一个已经踩到过的坑**：在 Mac 上 llama.cpp 用内存映射（mmap）加载权重，`maximum resident set size` 不一定把全部权重页算进去，减出来的数可能是负的。遇到负值不要硬解释，改用 `vm_stat` 的 `Pages active` + `Pages wired` 或活动监视器交叉核对，并在记录里写明这次用的是哪种口径——不同口径的数字不能混在一张表里比。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3-8B `config.json`*（`num_hidden_layers: 36`, `num_key_value_heads: 8`, `head_dim: 128`）. https://huggingface.co/Qwen/Qwen3-8B
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3-8B-GGUF*（`Qwen3-8B-Q4_K_M.gguf`，5.03 GB）. https://huggingface.co/Qwen/Qwen3-8B-GGUF
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv:2305.13245. https://arxiv.org/abs/2305.13245
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv:1911.02150. https://arxiv.org/abs/1911.02150
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B `config.json`*（`layer_types` 为 3 层线性 + 1 层全注意力交错）. https://huggingface.co/Qwen/Qwen3.5-9B
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
