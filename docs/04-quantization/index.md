---
title: 第四部分 量化
---

# 第四部分 量化

> **这一部分要回答的问题**：承接第三部分：跑起来时用的是 Q4 模型。Q4/Q8 是什么？压掉了 3/4 的精度，为什么还能用？

七节从"一个数怎么存"讲到"选哪条路线"（图 4-1）：先看数值表示，再看压缩原理与具体格式，然后是保住质量的两种手段，最后才谈评估与选路。

<Figure
  id="4-1"
  src="/figures/4-part4-map.svg"
  alt="第四部分七节的顺序：数值表示、量化原理、GGUF K-quants、imatrix、KV Cache 量化、评估损失、三条路线"
  caption="先原理后格式，先度量后选择；中间两节（imatrix、KV 量化）是可选的加固手段。"
/>

- [4.1 数值表示](./01-number-formats)
- [4.2 量化原理](./02-quantization-principles)
- [4.3 GGUF K-quants](./03-gguf-k-quants)
- [4.4 imatrix](./04-imatrix)
- [4.5 KV Cache 量化](./05-kv-cache-quantization)
- [4.6 评估量化损失](./06-measuring-loss)
- [4.7 三条量化路线](./07-route-selection)
