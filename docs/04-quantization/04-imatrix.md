---
title: imatrix：低比特量化如何尽量保住质量
---

# imatrix：低比特量化如何尽量保住质量

> **本章要回答的问题**：同样是 4 位，为什么有些量化版本明显更笨？imatrix 补的是什么信息？

4.2 算误差时，我们默认每个权重的重要性一样——RMSE 是全部权重的平均。这个假设不成立。同一个 4 位误差，落在"每个 token 都会经过"的通道上，和落在"偶尔才激活"的通道上，后果差很远。重要性矩阵（importance matrix，imatrix）就是把这件事补进量化过程。

## 误差要按"影响"加权

考虑一个权重行 `w` 和它乘的激活列 `a`，输出是 `Σ w_j a_j`。量化后变成 `Σ q_j a_j`，这个输出上的误差是：

```
F = [ Σ_j (q_j − w_j) · a_j ]²
```

我们要最小化的不是 `Σ (q_j − w_j)²`，而是 `F`。两者的区别就在 `a_j`：激活大的通道，权重的误差被放大得更多。

直接对 `F` 求导会得到一个带非对角项的二次型，需要在训练数据上统计完整的协方差矩阵——那和模型一样大。llama.cpp 的做法是只保留对角项：统计每个输入通道的激活平方均值 `⟨a_j²⟩`，把它当作权重，去做加权最小二乘（引用 [\[1\]](#ref-1)）。

这个近似很粗暴，但存储代价极小：每个张量只需要"输入维度"个浮点数。本地这个模型的 imatrix 有 248 个条目；文件里 177 个一维参数（norm、bias、SSM 标量）不参与，二维矩阵有 250 个，少的 2 个是按惯例跳过的张量——`llama-imatrix` 的 `--process-output` 默认关闭，`output.weight` 不参与统计。

## 本地文件里的证据

Q4_K_M 这个文件是带 imatrix 量化出来的，元数据里留了四个键：

| 键 | 值 |
|---|---|
| `quantize.imatrix.file` | `Qwen3.5-9B-GGUF/imatrix_unsloth.gguf` |
| `quantize.imatrix.dataset` | `unsloth_calibration_Qwen3.5-9B.txt` |
| `quantize.imatrix.entries_count` | 248 |
| `quantize.imatrix.chunks_count` | 80 |

（实测，`llama-gguf "$MODEL" r` 的元数据段）

80 个 chunk、按默认 512 token 一段算，校准语料约 4 万 token（估算）。这个量级和 llama.cpp 作者在 PR 里提到的"7B 模型 5 万 token 约 10 分钟（32 核 Ryzen）"是同一档（引用 [\[1\]](#ref-1)）。

这四个键的实际用途是**审计**：拿到一个别人量化好的文件，先看有没有这四项、校准集是什么。有，说明对方做了这一步；没有，说明是裸量化——不是不能用，但同样标着 Q4_K_M，质量可能差一截。

## imatrix 在量化流程里改了什么

两件事，作用点不同：

**第一，改 scale 的搜索目标。** `llama-quantize` 在给每个子块挑 scale 时，不再最小化未加权的平方误差，而是按 `⟨a_j²⟩` 加权。4.2 的公式没变，变的只是"哪个误差更值得降"。

**第二，决定某些档位能不能用。** 源码里有一张"必须提供 imatrix"的表（引用 [\[2\]](#ref-2)）：

- IQ1_S、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S：必需；
- Q2_K_S：必需；
- 其余 K-quant（Q3_K 以上）：不要求，但给了更好。

所以"下载了一个 IQ2 文件，它一定是校准过的"这句话基本成立——没有 imatrix，`llama-quantize` 会直接拒绝生成这些类型。

## 它救不了什么

**救不了校准集不对口。** 重要性是按校准语料统计的。拿英文维基校准的 imatrix 去压一个中文工业问答模型，被重点保护的通道可能根本不是业务里常用的那些。这不一定会让困惑度变差（困惑度也是在校准集同类文本上测的），但会让你的任务变差。

**救不了太低比特。** 2 位以下，权重本身的信息量已经不够，加权只能缓解、不能逆转。官方表里 Q2_K 的困惑度增量是 +3.52，比 Q3_K_M 的 +0.66 高 5 倍，这个差距不是校准能抹平的。

**也救不了你自己的判断。** 有 imatrix 元数据只说明"流程走了"，不说明"对你的任务有效"。真正的验证只有一条：在你的测试集上跑一遍（4.6 节）。

## 自己做一个 imatrix

流程三步，都在 llama.cpp 里：

```bash
# 1. 用 16 位模型和校准语料统计重要性
llama-imatrix -m "$BF16" -f calib.txt -o imatrix.gguf --chunks 80

# 2. 带着它量化
llama-quantize --imatrix imatrix.gguf "$BF16" qwen-q4km-im.gguf Q4_K_M

# 3. 对比：不带 imatrix 的同一档
llama-quantize "$BF16" qwen-q4km.gguf Q4_K_M
```

三个容易踩的坑：

- **校准语料要是你业务里的文本**，不是维基百科。`calib.txt` 用 `benchmarks/quantization/build-calib.sh` 那种拼法，把法规、制度、工单混在一起，比单一来源稳。
- **`--process-output` 默认关闭**，也就是 `output.weight` 不参与统计。这是有意的：输出层对误差特别敏感，加权反而容易过拟合校准集。
- **imatrix 只对 16 位源有意义**。对已经量化的文件再套 imatrix，是在二次量化的误差上再做优化，结果不可比。

## 工程含义

**看到 Q4_K_M，先问它有没有校准。** 同名不同质的现象在社区量化文件里很常见；元数据是唯一可靠的区分手段。

**自己交付时把校准集写进元数据。** `quantize.imatrix.dataset` 是自由文本，写上文件名和规模（比如 `factory-docs-2026Q1 / 80 chunks`），三个月后还能解释这个文件是怎么来的。

**别把 imatrix 当质量保证。** 它只是把"重要性"这个维度补进误差函数；模型在你的任务上到底行不行，仍然要靠 4.6 的测法。

## 动手实验

::: tip 动手实验：同一份权重，带与不带 imatrix 各量化一次
**假设**：在业务语料上，带 imatrix 的 Q4_K_M 困惑度比不带低 1%–5%，差距在 2–3 位档更明显。

**命令**：

```bash
# 校准语料：法规 + 制度 + 合成工单
benchmarks/quantization/build-calib.sh /tmp/fde-ppl-calib.txt
BF16=$(find ~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF -name '*BF16.gguf' | head -1)

llama-imatrix -m "$BF16" -f /tmp/fde-ppl-calib.txt -o /tmp/imatrix.gguf --chunks 80
llama-quantize --imatrix /tmp/imatrix.gguf "$BF16" /tmp/q4km-im.gguf Q4_K_M
llama-quantize "$BF16" /tmp/q4km.gguf Q4_K_M

for m in /tmp/q4km-im.gguf /tmp/q4km.gguf; do
  llama-perplexity -m "$m" -f /tmp/fde-ppl-calib.txt -c 512 --chunks 8 -fa on 2>&1 | grep "Final estimate"
done
```

**记录**：两版困惑度、两版在 60 题探针上的正确率（`labs/04-quantization/task_probe.py`），以及校准语料换成本领域后差值是否变大。**踩过的坑**：`--chunks` 太少（比如 10）时 imatrix 方差很大，同一份语料重跑两次结果就不一样；至少 50 个 chunk 再比较。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ikawrakow. *Importance Matrix calculation*（llama.cpp PR #4861：用 `⟨a_j²⟩` 做加权最小二乘的原理与耗时）. https://github.com/ggml-org/llama.cpp/pull/4861
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>ggml 项目. *`src/llama-quant.cpp`*（`tensor_requires_imatrix` 的档位要求、加权量化的调用点）. https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>ggml 项目. *`tools/quantize/quantize.cpp`*（`quantize.imatrix.*` 元数据键与 `--imatrix` 参数）. https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Dettmers, T., Zettlemoyer, L. *The case for 4-bit precision: k-bit Inference Scaling Laws*（低比特下权重重要性的非均匀性）. https://arxiv.org/abs/2212.09720
