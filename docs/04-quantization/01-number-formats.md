---
title: 数值表示：FP32 / FP16 / BF16 / INT8 / INT4
---

# 数值表示：FP32 / FP16 / BF16 / INT8 / INT4

> **本章要回答的问题**：这些位宽到底是什么？为什么从 16 位压到 4 位，模型还能正常说话？

3.1 节拆开过一个 GGUF 文件：427 个张量，按字节数除以参数量算，平均 5.08 位/参数。这个"位"不是文件压缩率，而是每个权重实际占的存储宽度。要理解 5 位为什么还能用，先得看清一个数在内存里是怎么摆的。

## 浮点数：位宽是两条轴，不是一条

一个浮点数分三段：符号位、指数、尾数（也叫有效数字）。指数管**能表示多大范围**，尾数管**能表示多精细**。同一句话"16 位浮点"，分成 fp16 和 bf16 两种截然不同的取舍（图 4.1-1）。

<Figure
  id="4.1-1"
  src="/figures/4-1-number-formats.svg"
  alt="五种数值格式的位分配：fp32 是 1+8+23，fp16 是 1+5+10，bf16 是 1+8+7，int8 和 int4 是纯整数外加一个 scale"
  caption="fp16 与 bf16 都占 16 位，区别只在怎么切：一个换精度，一个换范围。"
/>

| 格式 | 符号 | 指数 | 尾数 | 能表示的范围 | 有效精度 |
|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | ±3.4×10³⁸ | 约 7 位十进制 |
| fp16 | 1 | 5 | 10 | ±65504 | 约 3 位十进制 |
| bf16 | 1 | 8 | 7 | ±3.4×10³⁸ | 约 2 位十进制 |
| int8 | — | — | — | -128 ~ 127（配 scale） | 256 个格点 |
| int4 | — | — | — | -8 ~ 7（配 scale） | 16 个格点 |

fp16 的尾数有 10 位，是 bf16 的 7 位的 8 倍精细；代价是指数只有 5 位，最大只能到 65504，再大就溢出成 `inf`。bf16 反过来：它把 fp32 的指数原样保留，只砍尾数，所以范围和 fp32 一样，精度只剩 8 位二进制有效数字（约 2–3 位十进制）。

这个差别在训练里是生死问题。梯度、损失、注意力 logits 都可能瞬间冲到几千几万，fp16 会溢出，得靠损失缩放（loss scaling）兜着；bf16 不会溢出，代价是每次更新都更"粗"。所以训练侧普遍用 bf16，不是因为它更准，而是因为它**不用管范围**。

推理侧没有这个问题：权重和激活都在一个很窄的量级里，两种 16 位都够。选哪个主要看硬件和工具链——GGUF 的量化块里，scale 和 min 就存成 fp16（4.3 节会看到块结构），Apple 的 GPU 也按 fp16 做矩阵乘法，所以这条链路上 fp16 是默认值。

## 实测：同样一批权重，换成不同格式差多少

光看位分配不够，得量误差。取本地 Q4_K_M 文件里的四个代表性张量（中间层的 `attn_qkv`、`ffn_down`、`ffn_up`，以及词嵌入 `token_embd`），各取前 64 行，按每种格式舍入再还原，算归一化均方根误差（RMSE ÷ 权重标准差）。归一化是为了让不同张量能放在一起比。

| 张量 | fp16 | bf16 | int8（逐张量） | int4（逐张量） | int4（32 一组） |
|---|---|---|---|---|---|
| attn_qkv | 0.021% | 0.17% | 1.9% | 33.5% | 10.1% |
| ffn_down | 0.021% | 0.17% | 2.3% | 42.4% | 10.0% |
| ffn_up | 0.021% | 0.17% | 1.3% | 22.8% | 9.9% |
| token_embd | 0.021% | 0.16% | 8.9% | 72.1% | 14.6% |

（实测，见 `benchmarks/quantization/format-error.csv`；脚本 `labs/04-quantization/inspect_tensors.py`）

三件事值得记住：

**第一，bf16 的误差正好是 fp16 的 8 倍。** 尾数 7 位对 10 位，差 3 位就是 2³ = 8。这个倍率是结构决定的，换模型、换张量都不变。0.17% 听起来不小，但和下一行的整数格式比，它仍然小两个数量级——所以 16 位换成 bf16 对推理质量基本无感。

**第二，int8 的误差取决于张量，差 7 倍。** `ffn_up` 只有 1.3%，词嵌入到 8.9%。原因不在位宽，在分布：词嵌入的每一列（对应一个输出维度）量级差得多，一个全局 scale 迁就了最大的那列，小列就被压扁了。这正是 4.2 节要讲的分组量化动机。

**第三，int4 逐张量基本不能用。** 22%–72% 的误差，模型会开始重复、串话。把 256 个权重分成一组、每组一个 scale（最后一列），误差立刻回到 10% 附近——这就是 Q4_K 的结构，4.3 节展开。

一个必须写明的口径：这里的"参考值"是**反量化后的 Q4 权重**，不是训练时的 fp32。fp16/bf16 的误差比 Q4 本身小两个数量级，拿它当参照不影响格式之间的比较；但 int4 那一列是"把已经是 4 位的权重再压成 4 位"，绝对值偏大。想量真实的 fp32→int4 误差，得从 BF16 原始权重出发，命令在文末「待实测」。

## 整数：位宽之外还要一个 scale

整数没有指数，能表示的绝对范围是固定的：int8 是 -128 到 127，int4 是 -8 到 7。要拿它表示浮点权重，必须配一个缩放系数（scale）：

```
x ≈ q × s        q 是整数，s 是浮点 scale
```

于是"4 位量化"这个词其实不严谨：真正占的位数是 4 位整数 + 分摊到每个权重的 scale 开销。每 32 个权重共用一个 fp16 scale，就是 4 + 16/32 = 4.5 位；Q4_K 用更省的 6 位 scale，摊下来正好 4.5 位。**文件名里的 Q4 只是整数部分的位数，不是文件的实际位宽。**

## 工程含义

**看到"Q4"先问两件事：整数几位、scale 怎么分。** 两个都答不上来，这个数字就没法比较。Q4_0、Q4_K_M、IQ4_XS 都是"4 位"，实际位宽从 4.25 到 5.0 不等，质量也不同。

**fp16 溢出不是理论问题。** 长上下文下注意力分数、MoE 路由的 logits 都可能瞬间冲过 65504。这不是量化引入的，是 fp16 本身的边界：一旦溢出成 `inf`，后面整层都会变成 `nan`。训练侧用 bf16 正是为了绕开它，推理侧遇到数值不稳的模型也该优先考虑 bf16 而不是 fp16。

**norm 层不量化是对的。** 本地这个文件里 177 个张量保持 F32，都是 norm、bias 与 SSM 的小参数（`ssm_a`、`ssm_conv1d`），加起来 106 万参数、4.2 MB，占文件 0.07%。把它们也压成 4 位省不下空间，却最容易把模型压坏。

**KV Cache 用的是同一套整数格式。** 4.5 节的 `-ctk q8_0` 就是"8 位整数 + 每 32 个值一个 fp16 scale"，和权重侧的 q8_0 是同一个东西。理解了这一节的位分配，KV 量化就没有新概念。

## 待实测

::: tip 待实测：从 BF16 原始权重复现这张表
**假设**：从 16 位源出发，fp32→int4 的真实误差比本节表格里的值小（本节参照物本身已是 4 位）。

**命令**：

```bash
# 1. 取同一模型的 BF16 源（17.9 GB，只需一次）
llama-cli -hf unsloth/Qwen3.5-9B-GGUF:BF16 -st -p "你好" -n 1

# 2. 指向 BF16 文件重跑本节脚本，对比 format-error.csv
MODEL=$(find ~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF -name '*BF16.gguf' | head -1)
uv run labs/04-quantization/inspect_tensors.py --model "$MODEL" --out benchmarks/quantization/fp32-ref
```

**记录**：两种参照物下 int4 逐张量与 32 一组的 RMSE 差值，以及 fp16/bf16 的误差是否仍是 8 倍关系（应当不变，因为那只由尾数位数决定）。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>IEEE. *IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019)*（二进制浮点的位分配与舍入规则）. https://standards.ieee.org/ieee/754/6210/
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Intel. *BFLOAT16 — Hardware Numerics Definition*（bf16 的 8 位指数 / 7 位尾数与舍入行为）. https://www.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>ggml 项目. *`ggml-common.h`*（`block_q4_K`、`block_q8_0` 等块结构与位宽）. https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>ggml 项目. *GGUF 格式规范*（张量类型枚举与元数据）. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Dettmers, T., Zettlemoyer, L. *The case for 4-bit precision: k-bit Inference Scaling Laws*（ICML 2023；4 位在总位数与零样本精度上的经验最优点）. https://arxiv.org/abs/2212.09720
