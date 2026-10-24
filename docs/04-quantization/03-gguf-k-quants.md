---
title: GGUF K-quants：各个等级在做什么
---

# GGUF K-quants：各个等级在做什么

> **本章要回答的问题**：Q4_K_M 里的 K 和 M 分别指什么？Q4_K_M 和 Q5_K_S 差在哪？

4.1 的表里，int4 逐张量不能用、32 个一组勉强能用。真实的 Q4_K 比"32 个一组"更复杂：它一层套一层，还混着好几种位宽。这一节把这套命名拆开，并给出本地文件里每个张量到底用了什么。

## 名字怎么读

`Q4_K_M` 拆成三段：

- **Q4**：基础位宽，整数部分 4 位。
- **K**：K-quant 家族，用 256 个权重的"超级块"（super-block）组织，块内再分 32 个权重的子块。相对的是老格式 `Q4_0`/`Q4_1`，32 个一组、结构简单。
- **M**：混合档位（mix），介于 `_S`（small）和 `_L`（large）之间。不带后缀的 `Q4_K` 就是 `Q4_K_M` 的别名。

`IQ` 开头的是另一族：I-quant 用格点码本（lattice codebook）代替线性量化，同样位宽下误差更小，但编码更慢，而且**基本依赖重要性矩阵**（4.4 节）。`Q4_0` / `Q4_1` 是更老的线性格式，现在只在兼容场景里见。

## 一个超级块长什么样

以 q4_K 为例，256 个权重组成一个超级块，内部切 8 个子块、每块 32 个权重（图 4.3-1）。

<Figure
  id="4.3-1"
  src="/figures/4-3-kquant-blocks.svg"
  alt="q4_K 的超级块：256 个权重切成 8 个子块，每块 32 个 4 位权重，各配一套 6 位 scale 与 6 位 min，超级块共用两个 fp16 基准"
  caption="两层缩放：子块用 6 位 scale/min 做细调，超级块用两个 fp16 值把它们拉回真实量级。"
/>

按字节算：8 个子块 × 16 字节（32 个 4 位权重）= 128 字节，子块 scale/min 8 × 12 位 = 12 字节，超级块的 `d`、`dmin` 两个 fp16 = 4 字节，合计 144 字节。`144 × 8 ÷ 256 = 4.5` 位/权重。

这就是 4.2 节算出来的 4.5 位的来历：不是"4 位 + 一点零头"，而是"4 位权重 + 6 位子块 scale + 6 位子块 min + fp16 基准"。**min 的存在让 q4_K 是非对称量化**，这也是它比对称的 Q4_0 更准的原因之一。

各档的实际位宽（从本地文件按"字节 × 8 ÷ 参数"反算）：

| 类型 | 位/权重 | 本地张量数 | 参数量 |
|---|---|---|---|
| Q4_K | 4.5000 | 132 | 5.90 B |
| Q5_K | 5.5000 | 48 | 1.21 B |
| Q6_K | 6.5625 | 22 | 1.84 B |
| Q8_0 | 8.5000 | 48 | 6.3 M |
| F32 | 32.0000 | 177 | 1.1 M |

（实测，见 `benchmarks/quantization/tensor-types.csv`）

这些数字和 `ggml-common.h` 里的块定义逐一对得上：q5_K 是 176 字节/256、q6_K 是 210 字节/256、q8_0 是 34 字节/32。**位宽是结构决定的常数，不是估算。**

## M 到底混了什么

本地这个 Q4_K_M 文件的分配是这样的：

| 张量 | 类型 | 说明 |
|---|---|---|
| `token_embd`、`attn_gate`、`ffn_gate`、`ffn_up` | Q4_K | 大头，保持 4 位 |
| `attn_qkv`、`ssm_out` | Q5_K | 24 个线性注意力层 |
| `output`、一半 `ffn_down`、5/8 的 `attn_v` | Q6_K | 敏感张量升档 |
| `ssm_alpha`、`ssm_beta` | Q8_0 | 数量极小，直接给 8 位 |
| 所有 norm、`ssm_a`、`ssm_conv1d`、`ssm_dt` | F32 | 一共 1.1 M 参数，占 4.2 MB |

（实测，见 `benchmarks/quantization/tensor-types.csv` 与 `labs/04-quantization/inspect_tensors.py` 的输出）

两个观察：

**升档是挑过的，不是随机的。** 同为 ffn 的三个投影，`gate`/`up` 留在 4 位，`down` 有一半升到 6 位；8 个全注意力层里，3 个的 `attn_v` 留 4 位、5 个升 6 位。`output` 承担最后的词表投影，直接 6 位。小而关键的 SSM 门控（每层才 4096×32）给 8 位，因为它们总共才 630 万参数，压到 4 位省不下 3 MB，却可能把整层门控搞坏。

**"平均 5.08 位"是混合出来的。** 如果全用 Q4_K，文件会是 5.03 GB；实际 5.68 GB，多出来的 0.65 GB 买了那些升档张量。值不值，取决于任务——这就是 `_S`/`_M`/`_L` 存在的意义。

## 档位差多少：拿官方自己的表

llama.cpp 在 `quantize` 的类型表里直接写了每档的体积和困惑度增量，基准是 Llama-3-8B（引用 [\[1\]](#ref-1)）：

| 类型 | 体积 | 困惑度增量 |
|---|---|---|
| Q2_K | 2.96 GB | +3.5199 |
| Q3_K_S | 3.41 GB | +1.6321 |
| Q3_K_M | 3.74 GB | +0.6569 |
| Q4_K_S | 4.37 GB | +0.2689 |
| **Q4_K_M** | **4.58 GB** | **+0.1754** |
| Q5_K_S | 5.21 GB | +0.1049 |
| Q5_K_M | 5.33 GB | +0.0569 |
| Q6_K | 6.14 GB | +0.0217 |
| Q8_0 | 7.96 GB | +0.0026 |

（引用 [\[1\]](#ref-1)，基准模型 Llama-3-8B；体积是该基准模型下的值，不是 Qwen3.5-9B 的）

这张表的形状比数字更重要：

- **Q2 到 Q3 是一道悬崖**：+3.52 掉到 +0.66，多花 0.8 GB 买回 5 倍的质量。
- **Q3 到 Q4 是第二道坡**：+0.66 到 +0.18，再花 0.8 GB。
- **Q4 往上收益递减**：Q4_K_M → Q5_K_M 只从 +0.18 降到 +0.06，体积涨 16%。
- **Q8_0 几乎无感**：+0.0026，但体积是 Q4 的 1.7 倍。

这就是"4 位是甜点"的量化版本。换成 Qwen3.5-9B，体积和增量都会不同，但拐点通常仍在 Q4 附近；这条判断可以从 4.6 节引用的 k-bit 缩放律得到支持，具体到某个模型仍要自己测。

同一模型各档的实际体积可以从模型仓库的文件列表直接读到（引用 [\[2\]](#ref-2)）：

| 文件 | 体积 | 折合位/权重（估算） |
|---|---|---|
| Q3_K_M | 4.67 GB | 4.17 |
| Q4_K_S | 5.39 GB | 4.82 |
| **Q4_K_M** | **5.68 GB** | **5.08** |
| Q5_K_M | 6.58 GB | 5.88 |
| Q6_K | 7.46 GB | 6.67 |
| Q8_0 | 9.53 GB | 8.52 |

（引用 [\[2\]](#ref-2)；位/权重 = 字节 × 8 ÷ 8.95e9 参数，估算）

注意 Q3_K_M 折合 4.17 位，比它的"基础 3 位"高不少——因为 M 档把一部分张量升到了 5 位。**看文件名猜位宽会猜错，看文件大小才算得准。**

## I-quant 是另一条支线

`IQ2_XXS` 到 `IQ4_XS` 用格点码本：预先定义一组向量，量化时找最近的码字，而不是独立地四舍五入每个权重。同样 2.06 位，IQ2_XXS 的误差明显小于线性 2 位。

代价有两个：编码慢（要找最近邻），以及**基本依赖重要性矩阵**——llama.cpp 的 IQ 量化流程把 imatrix 当成默认输入，没有它就不建议用 IQ 档。所以看到 `IQ` 前缀，先确认它是怎么校准的，再看体积。

本书主线不用 IQ：Q4_K_M 在 24GB Mac 上已经够用，IQ 的收益要到 2–3 位才明显，而那个位宽区间在中文任务上风险太大。

## 工程含义

**体积可以预估，质量不能。** 位/权重 × 参数量 + 少量开销 ≈ 文件大小，误差在 2% 以内。但"Q4_K_M 在我的任务上掉几分"必须自己测（4.6 节）。

**别拿 Q4_0 和 Q4_K_M 当同一档。** 前者是 32 个一组、对称、fp16 scale；后者是 256 超级块、非对称、6 位 scale 与 min。两者都是"4 位"，但结构不在一个层次上，4.2 的分组实验只对应前者的结构。

**升档优先看 output 和 down。** 这是 llama.cpp 在 M 档里给出的经验判断：`output.weight` 和一半 `ffn_down` 被升到 6 位，`attn_v` 升了 5/8。自己要调混合比例时，可以从这三个位置开始，再逐步加。

## 动手实验

::: tip 动手实验：同一份 BF16 源切出整条阶梯
**假设**：从 BF16 出发本地量化，得到的体积与官方文件相差在 1% 内，困惑度趋势与官方表一致。

**命令**：

```bash
# 1. 取 BF16 源（17.9 GB，只需一次）
llama-cli -hf unsloth/Qwen3.5-9B-GGUF:BF16 -st -p "你好" -n 1
BF16=$(find ~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF -name '*BF16.gguf' | head -1)

# 2. 切阶梯（每档几分钟）
for t in Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0; do
  llama-quantize "$BF16" "/tmp/qwen-$t.gguf" "$t"
  ls -l "/tmp/qwen-$t.gguf" | awk -v t=$t '{printf "%-8s %6.2f GB  %.2f 位/权重\n", t, $5/1e9, $5*8/8.95e9}'
done

# 3. 每档跑同一段文本的困惑度（文本用 benchmarks/quantization/build-calib.sh 生成）
for t in Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0; do
  llama-perplexity -m "/tmp/qwen-$t.gguf" -f /tmp/fde-ppl-calib.txt -c 512 --chunks 8 -fa on 2>&1 | grep "Final estimate"
done
```

**记录**：六档的体积、位/权重、困惑度，以及每档相对 Q8_0 的增量。**踩过的坑**：`llama-quantize` 的输入必须是未量化的 16 位模型；拿 Q4 文件当输入会要求加 `--allow-requantize`，那是二次量化，数字不能用来做阶梯对比。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ggml 项目. *`tools/quantize/quantize.cpp`*（各量化类型的体积与困惑度增量表，基准 Llama-3-8B）. https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>unsloth. *Qwen3.5-9B-GGUF 模型仓库文件列表*（各档 GGUF 的体积）. https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>ggml 项目. *`ggml-common.h`*（`block_q4_K`、`block_q5_K`、`block_q6_K`、`block_q8_0` 的字节布局）. https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡*（32 层、混合线性注意力布局、8.95B 参数口径）. https://huggingface.co/Qwen/Qwen3.5-9B
