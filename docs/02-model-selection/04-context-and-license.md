---
title: 上下文长度、开源协议、中文能力
---

# 上下文长度、开源协议、中文能力

> **本章要回答的问题**：上下文标着 128K，实际能用多少？开源协议会不会卡住商用？中文能力该怎么看？

## 标称上下文不等于有效上下文

模型卡上的"262,144 tokens"，指的是**训练时见过的最大长度**，不是"在这个长度内都能答对"。

RULER 基准专门测这件事：它用检索、多跳追踪、聚合等任务，在不同长度上评估模型。结论是多数模型的有效上下文远小于标称值，而且不同能力衰减的速度不一样——"找到一句话"能撑很长，"把多处信息聚合成一个结论"往往在中段就崩了（引用 [\[1\]](#ref-1)）。

另一个被反复验证的现象是"中间迷失"（lost in the middle）：把关键信息放在长文档的开头或结尾，模型更容易用上；放在中间，准确率明显下降（引用 [\[2\]](#ref-2)）。这不是某个模型的缺陷，是当前架构的普遍特征。

所以"支持 262K"应该翻译成："在 262K 以内不会因为位置编码报错，但你的任务在哪个长度开始退化，需要自己测。"

## 长上下文的真实代价

上下文不是免费的，它至少消耗三样东西：

- **KV Cache 内存**：线性增长。标准结构每 token 160 KiB 的话，128K 就是 20 GB（估算），单这一项就超过多数消费级显卡。
- **prefill 时间**：近似线性增长。128K 的 prompt，首字延迟可能是 8K 的十几倍。
- **注意力计算**：随长度增长，在 prefill 阶段尤其明显。

这就是为什么"把整本手册塞进去"通常不是好方案。RAG 的价值不只是省钱：**它把"在 200 页里找一段"变成"在 5 段里找一句"，让模型在它最擅长的长度区间工作。**

工程上的经验区间：大多数中文问答任务，检索后 4K–16K 的上下文足够；超过 32K 时，先问自己"是不是检索没做好，才需要塞这么多"。

## 协议：三种常见的坑

"开源模型"这个说法很模糊。准确的说法是**开放权重（open weights）**：你能下载权重，但不一定拥有 OSI 定义的开源软件那四项自由。协议决定了你能做什么，这是商用前必须逐条核对的东西。

| 协议 | 商用 | 修改/再分发 | 主要义务 | 典型模型 |
|---|---|---|---|---|
| Apache-2.0（引用 [\[6\]](#ref-6)） | 允许 | 允许 | 保留版权与许可声明；附专利授权条款 | 本书三个模型 |
| MIT | 允许 | 允许 | 保留版权与许可声明 | 部分小模型 |
| Llama Community License（引用 [\[8\]](#ref-8)） | 允许（有门槛） | 允许 | 月活超 7 亿需单独授权；命名与署名要求 | Llama 系列 |
| Gemma Terms（引用 [\[7\]](#ref-7)） | 允许（有使用限制） | 允许 | 遵守禁止用途清单；再分发需附条款 | Gemma 系列 |

三类常见的坑：

**第一，把"权重可下载"当成"随便用"。** 有的协议限制特定用途（比如禁止用于某些监控场景），有的要求下游模型改名，有的对规模设门槛。这些条款不影响技术选型，但会影响交付。

**第二，忽略再分发义务。** 把模型打进客户的离线镜像，属于再分发。Apache-2.0 要求附带许可文本和声明；自定义协议往往要求把条款一并交给客户。

**第三，忽略协议变更。** 同一个家族的不同版本可能换协议。本书的三个模型都标 Apache-2.0（引用 [\[3\]](#ref-3)[\[4\]](#ref-4)[\[5\]](#ref-5)），但 Gemma 历史上使用过自定义条款，旧的条款页仍然在线。**上线前请让法务按你实际下载的那个仓库、那个 commit 核对，不要只看模型卡上的一个字段。**

## 中文能力怎么看

不要看单一分数，看四件事。

**第一，分词效率。** 同样一段中文，token 数越少，上下文越经用、生成越快。这个可以直接用 `llama-tokenize` 数出来（1.1 节），不需要信任何评测。

**第二，中文基准。** C-Eval、CMMLU 这类基准能提供横向参考，但要注意它们的选择题形式与真实问答差距很大，而且存在训练数据污染的可能（2.5 节详述）。看基准时至少确认：评测版本、few-shot 设置、是否用了思考模式。

**第三，模型卡上的语言声明。** 有的模型明确列出支持的语言与中文评测结果，有的只写"multilingual"。声明不等于能力，但缺失声明通常意味着没有专门优化。

**第四，你自己的任务。** 这是唯一有决定权的。企业文档里充满缩写、型号、内部术语，通用基准测不出来。第六部分会给一套构建测试集的方法。

一个务实的判断：**本书三个模型里，Qwen 系列对中文的投入最明确（词表、语料、评测都围绕中英双语设计），Gemma 系列更偏多语言通用。** 如果你的场景以中文为主，这个差异值得在选型时加权。

## 合规清单

交付前逐项确认：

- [ ] 记录模型仓库、commit、文件名、SHA256；
- [ ] 保存协议原文（不是模型卡上的一个字段）；
- [ ] 确认商用、再分发、修改、命名四项要求；
- [ ] 确认没有落在禁止用途清单里；
- [ ] 离线镜像里附上许可文本；
- [ ] 如果客户要求"完全自主可控"，确认协议不会因为你的修改而失效。

## 待实测

::: tip 待实测：有效上下文在哪里开始退化
**假设**：在"单点检索"任务上，模型到 32K 仍能稳定找到目标句；在"跨段聚合"任务上，16K 之后开始明显下降。

**做法**：构造三组文档，每组把一条关键信息放在开头、中间、结尾，长度分别为 4K、16K、64K，问同一个问题，记录是否答对、是否引用正确位置。

**命令**（`-c` 要开到目标长度，KV 会占掉数 GB，见 1.6 节）：

```bash
llama-completion -m "$MODEL" -c 65536 --temp 0 -n 128 \
  -f eval/longctx/needle_64k_mid.txt --no-display-prompt
```

64K 上下文在 24GB 机器上不是免费额度：按 1.6 节的估算，9B 的 KV 每 token 32 KiB，64K 就是约 2 GB，加上权重和运行时开销已经接近可用内存的上限，跑之前先关掉浏览器。

**记录**：长度 × 位置 × 是否正确 的三维表，以及每次的峰值内存（`/usr/bin/time -l` 包住上面的命令）。注意 `-c` 必须真的开到对应长度，否则测的是截断后的效果。数据存 `benchmarks/long-context/`。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Hsieh, C.-P. et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654. https://arxiv.org/abs/2404.06654
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Liu, N. F. et al. *Lost in the Middle: How Language Models Use Long Contexts*. arXiv:2307.03172. https://arxiv.org/abs/2307.03172
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡*（Apache-2.0）. https://huggingface.co/Qwen/Qwen3.5-9B
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3-14B 模型卡*（Apache-2.0）. https://huggingface.co/Qwen/Qwen3-14B
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Google. *Gemma-4-26B-A4B-it 模型卡*（Apache-2.0）. https://huggingface.co/google/gemma-4-26B-A4B-it
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>Apache Software Foundation. *Apache License, Version 2.0*. https://www.apache.org/licenses/LICENSE-2.0
7. <a id="ref-7" style="scroll-margin-top: 96px"></a>Google. *Gemma Terms of Use*. https://ai.google.dev/gemma/terms
8. <a id="ref-8" style="scroll-margin-top: 96px"></a>Meta. *Llama 3.1 Community License Agreement*. https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
