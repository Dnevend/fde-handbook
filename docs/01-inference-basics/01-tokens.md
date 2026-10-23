---
title: Token 与分词：中文为什么更"贵"
---

# Token 与分词：中文为什么更"贵"

> **本章要回答的问题**：同样一句话，中文为什么比英文消耗更多 token？分词器到底是怎么切的，"贵"贵在哪一步？

## 模型看到的不是字，是整数

把一句话喂给模型之前，有一个必经步骤：把文本切成 token，再把每个 token 映射成一个整数 id。模型内部处理的始终是这串整数，输出时再反向映射回文本。

这个"切分 + 编号"的组件叫分词器（tokenizer）。它有三个部分：切分规则、词表（vocabulary）、以及把词表项映射到 id 的查找表。词表大小是模型的固定属性，写在配置里——`Qwen3-8B` 是 151,936（引用 [\[1\]](#ref-1)），`Qwen3.5-9B` 是 248,320（引用 [\[2\]](#ref-2)）。

分词器不是模型"学会"的，是训练前用统计方法在语料上构造出来的。主流做法是字节对编码（Byte Pair Encoding，BPE）：从单个字节开始，反复把语料中最高频的相邻片段合并成一个新 token，合并次数达到预设的词表大小为止（引用 [\[3\]](#ref-3)）。SentencePiece 把这一套做得更通用，可以脱离"空格分词"的假设，直接处理原始文本（引用 [\[4\]](#ref-4)）。

所以词表里有什么，取决于训练语料里什么最常见。这一点直接决定了中文的成本。

## 中文为什么更贵

有三层原因，逐层叠加。

**第一层：字节数。** 一个汉字在 UTF-8 里占 3 个字节，一个英文字母占 1 个。BPE 的起点是字节，同样长度的文本，中文的初始序列天然更长。

**第二层：合并频率。** BPE 只合并"高频相邻字节对"。英语语料里，"ing""tion""the"这类片段出现次数极高，很快就被合并成单个 token；中文没有空格做词边界，常用词（"系统""设备""维护"）能被合并，但组合极其分散，很多两字词、三字词各自只占很小比例。结果是：常见汉字可能是一个 token，生僻字和罕见组合则要拆成两个甚至三个 token。

**第三层：词表配额。** 词表大小是固定的。一个主要用英文训练的模型，词表里大量位置被英文词缀和代码符号占据，留给中文的空间有限。`Qwen3.5` 把词表从 15 万扩到 24.8 万，一部分原因就是为了给多语言（包括中文）更多整词位置（引用 [\[2\]](#ref-2)）。

三层叠加的后果是：**同样一段内容，中文的 token 数往往明显高于英文**。具体高多少，取决于分词器、文本类型和标点习惯，没有一个通用倍数。网上流传的"中文是英文的 2 倍"只能当量级参考，不能当公式用。

这里有一个反直觉的结论：**"贵"不代表中文能力差。** 分词效率影响的是成本和速度，不是理解能力。一个中文 token 数偏多的模型，只要训练充分，照样能答对；只是同样的上下文窗口里装的内容更少，同样的输出要花更长时间。

## 词表大小是有代价的

词表不是越大越好。词表大小 `V` 直接决定两处参数量：

- **输入 Embedding**：`V × hidden_size`。`Qwen3-8B` 的 `hidden_size` 是 4096，仅输入 Embedding 就有 151,936 × 4096 ≈ 6.2 亿参数（估算），占 8.2B 总参数的 7.6% 左右。
- **输出投影**：如果输入输出不共享权重（`tie_word_embeddings: false`），还要再加一份同样大小的矩阵。

词表扩大能降低 token 数，但会把参数预算和显存占用推高。这是模型设计上的一个真实取舍，也是为什么不同厂商的词表大小差异很大：`Gemma-4-26B-A4B` 用了 262,144 的词表（引用 [\[5\]](#ref-5)），比 `Qwen3-8B` 大七成。

## 工程上要记住的三件事

**第一，凡是跟长度有关的地方，都用 token 数，不要用字符数。** 上下文窗口、`max_tokens`、截断策略、计费，单位全是 token。用 `len(text)` 估算，中文场景会系统性低估。

**第二，中文 prompt 要留更多余量。** 如果你按"一句 20 字"去算能塞多少轮对话，实际可能只塞得下一半。RAG 场景里，检索回来的片段同样是中文，拼接后很容易超预算。

**第三，输出长度是延迟的主要来源。** 生成是逐 token 的（1.2 节），输出 500 个 token 和输出 100 个 token，时间差好几倍。让模型"简短回答"不只是风格问题，是性能优化。

```bash
# 用分词器数 token，而不是靠猜
llama-tokenize -m "$MODEL" -p "3 号线液压站多久换一次油？" --show-count
```

## 待实测

::: tip 待实测：中英文 token 成本
**假设**：同一段含义等价的中英文文本，中文的 token 数高于英文；不同分词器的差距不一样。

**命令**（在仓库根目录执行）：

```bash
# 9B（本书主力模型，必做）
llama-tokenize -m "$MODEL" --no-bos -p "3 号线液压站每 500 小时更换一次液压油。" --show-count
llama-tokenize -m "$MODEL" --no-bos -p "Replace the hydraulic oil in Line 3 every 500 hours." --show-count

# 换一个分词器对比（可选，需另行下载 14B，见 2.6 节）
llama-tokenize -m Qwen3-14B-Q4_K_M.gguf --no-bos -p "3 号线液压站每 500 小时更换一次液压油。" --show-count
```

**记录**：两段文本各自的 token 数、比值，以及两个分词器的差异。每行命令连同 `bash benchmarks/record-env.sh` 的输出一起存进 `benchmarks/tokenizer/`，正文再引用。两段都加 `--no-bos`，统一排除开头特殊 token 的干扰。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3-8B 模型卡与 `config.json`*（`vocab_size: 151936`, `hidden_size: 4096`）. https://huggingface.co/Qwen/Qwen3-8B
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡与 `config.json`*（`vocab_size: 248320`）. https://huggingface.co/Qwen/Qwen3.5-9B
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Sennrich, R., Haddow, B., Birch, A. *Neural Machine Translation of Rare Words with Subword Units*. arXiv:1508.07909. https://arxiv.org/abs/1508.07909
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Kudo, T., Richardson, J. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. arXiv:1808.06226. https://arxiv.org/abs/1808.06226
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Google. *Gemma-4-26B-A4B-it 模型卡与 `config.json`*（`vocab_size: 262144`）. https://huggingface.co/google/gemma-4-26B-A4B-it
