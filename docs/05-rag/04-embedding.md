---
title: Embedding：向量如何表示语义
---

# Embedding：向量如何表示语义

> **本章要回答的问题**：一段文字怎么变成一个向量？为什么“意思相近”在向量里就是“方向相近”？维度和精度能省到什么程度？

检索的本质是排序，排序需要一个可计算的“相似”。Embedding 模型把文本压成一个固定长度的向量，让“意思相近”变成“余弦大”。

这一节只讲工程上必须知道的几件事：模型怎么选、查询和文档为什么要分开处理、1024 维能不能砍到 256、int8 存储会不会掉点。

## 一个向量代表什么

Embedding 模型是一个编码器：输入一段文本（最多几千 token），输出一个 1024 维的浮点向量。训练目标让“语义相近的文本”在向量空间里靠近，“不相关”的远离。

相似度用余弦：

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

工程上通常先把向量归一化（除以模长），之后余弦就等于点积，省一次除法。本书的 `raglib.l2_normalize` 就是这么做的。

余弦的绝对值没有意义，只有相对排序有意义。不同模型、不同语料训出来的向量空间不可比；**换模型必须重建整个索引**，不能只换查询侧。

## 选模型：先看语言，再看维度，最后看许可

本书用 `Qwen3-Embedding-0.6B`（Apache-2.0，1024 维，支持 MRL 截断）。选它的原因很实际：

- 中文检索能力强，且 0.6B 在 24GB Mac 上跑得动；
- 支持自定义维度（Matryoshka），可以做“降维换存储”的实验；
- 协议允许商用与再分发，书里可以放心写。

同类可选：BGE-M3（多语言、1024 维，引用 [\[4\]](#ref-4)）、bge-small-zh（更小更快，质量略低）。
没有“最好”的模型，只有“在你的语料上 Recall@k 够不够”的模型；选型方法在 2.5、6.2 还会再讲。

## 查询与文档不是同一种输入

很多模型对查询和文档使用不同的前缀或指令。Qwen3-Embedding 的推荐做法是：查询侧加任务指令，文档侧不加：

```
Instruct: 给定一个中文问题，检索能回答该问题的段落
Query: 有限空间作业为什么必须先通风再检测？
```

本书把这个前缀写在 `config.yaml` 的 `query_instruction`，由 `raglib.embed_texts(kind="query")` 统一加。
忘记加，或者文档侧也加了，召回通常会掉几个点；这类错误很难从日志里看出来，只能靠对照实验。

另一个常见错误是 **pooling 方式不对**。Qwen3-Embedding 用最后一层 token 的向量（last-token pooling），启动服务时要显式指定：

```bash
llama-server -m Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embeddings --pooling last -c 4096 -b 2048 -ub 2048 -ngl 99
```

用错 pooling，向量仍然能算，余弦仍然在 0–1 之间，但检索质量会莫名其妙地差——这是最值得先排查的一项。

## 维度和精度：能省多少

把 734 个块向量化用了 42.6 秒（实测，约 17 块/秒，M4 24GB，Qwen3-Embedding-0.6B Q8）。
在这个规模上，60 道非拒答题的检索结果：

| 维度 | 精度 | Recall@10 | MRR@10 | 索引体积 | 单次暴力查询 |
|---|---|---|---|---|---|
| 1024 | f16 | 1.000 | 0.841 | 1.50 MB | 0.24 ms |
| 1024 | q8 | 1.000 | 0.841 | 0.75 MB | 0.20 ms |
| 512 | f16 | 0.983 | 0.824 | 0.75 MB | 0.17 ms |
| 512 | q8 | 0.983 | 0.824 | 0.38 MB | 0.16 ms |
| 256 | f16 | 0.983 | 0.833 | 0.38 MB | 0.16 ms |
| 256 | q8 | 0.983 | 0.832 | 0.19 MB | 0.17 ms |

（实测，见 `benchmarks/rag/embedding.csv`；q8 是逐行对称量化模拟，不是 llama.cpp 的存储格式。）

两个结论要一起看：

**第一，维度和精度在这个规模上没有明显损失。** 1024 → 256 只掉了 1.7 个百分点，int8 几乎不动。索引从 1.50 MB 降到 0.19 MB，约 8 倍。

**第二，这个实验分辨不出更小的差异。** 734 个块、60 道题，Recall@10 已经贴着 1.0；1.7 个点的差别很可能来自题目与切分，而不是维度。**不要拿这张表外推到十万级向量**——规模上去以后，降维的损失会更明显，必须重测。

为什么降维还能保住召回？因为 MRL 训练时就让前 256 维独立可用的（引用 [\[2\]](#ref-2)）；截断后再归一化，相当于换了一个更低维的度量。不是所有模型都支持这种截断；不支持时，截断等于扔掉信息。

## 存储怎么算

向量库的体积很好估：

```
字节数 = 向量数 × 维度 × 每维字节数
```

f16 每维 2 字节，int8 每维 1 字节。2 万个 1024 维向量：f16 约 41 MB，int8 约 20 MB；加到 100 万条，就是 2 GB / 1 GB。
真正占空间的往往不是向量，而是原文、元数据和索引结构；5.5 会看到 HNSW 的图本身也要内存。

## 工程含义

**文档和查询必须用同一个模型、同一套预处理。** 混用模型、漏加指令、pooling 不一致，都会让余弦“看起来正常但排序不对”。

**归一化后再存。** 省掉查询时的除法，也让不同长度的向量可比；大多数向量库默认按余弦或内积检索，归一化后两者等价。

**维度是存储与质量的旋钮，不是越大越好。** 先用全维度建立基线，再用 5–10 道题扫 512/256；只有召回不掉、存储又紧张时才降。

**把模型版本写进索引元数据。** 换模型、换量化、换 pooling 都要重建索引；没有版本号，半年后没人说得清这个索引是怎么来的。

## 动手实验

::: tip 动手实验：用困难负样本看降维的真实代价
**假设**：在 2 万级向量与困难负样本下，256 维相对 1024 维的 Recall@10 会掉 5 个百分点以上。

**做法**：把语料扩到 2 万块（`corpus/synthetic/generate.py` 的 `PLAN` 扩到约 2000 份），对每道题用 BM25 取前 20 个非 gold 块作为困难负样本，再比较 1024/512/256 与 f16/int8。

**记录**：Recall@10、MRR@10、索引体积、向量化耗时。**注意**：合成语料只能测规模与噪声，不能用来下“业务检索质量”的结论。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Reimers, N., Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*（双塔句向量与余弦检索，EMNLP 2019）. https://arxiv.org/abs/1908.10084
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Kusupati, A. et al. *Matryoshka Representation Learning*（可变维度表示，NeurIPS 2022）. https://arxiv.org/abs/2205.13147
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Zhang, Y. et al. *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models*（模型卡与训练方法）. https://arxiv.org/abs/2506.05176
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Chen, J. et al. *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*（BGE-M3）. https://arxiv.org/abs/2402.03216
