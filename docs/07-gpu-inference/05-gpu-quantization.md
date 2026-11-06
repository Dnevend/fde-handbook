---
title: GPU 端量化实操：AWQ / GPTQ / FP8
---

# GPU 端量化实操：AWQ / GPTQ / FP8

> **本章要回答的问题**：AWQ 和 GPTQ 怎么把模型压进显存？和 GGUF 的量化有什么本质不同？

第四部分在 Mac 上用 GGUF 做过量化：Q4_K_M、Q5_K_M、Q8_0，看的是文件大小、困惑度和任务正确率。到了 GPU 服务器上，同样是“把权重压小”，但格式、内核和工具链都换了：vLLM 吃的是 AWQ/GPTQ/FP8 的 safetensors，不是 GGUF。**量化不是把文件压小就完了，它必须和部署引擎的 kernel 对上。**

## 三种路线

| 路线 | 典型位宽 | 怎么压 | 典型硬件 | vLLM 支持 |
|---|---|---|---|---|
| AWQ（引用 [\[1\]](#ref-1)） | 4-bit 权重 | 按激活分布挑重要通道，保护少量显著权重 | NVIDIA GPU | 支持（引用 [\[4\]](#ref-4)） |
| GPTQ（引用 [\[2\]](#ref-2)） | 3/4/8-bit 权重 | 逐层用 Hessian 近似做误差补偿 | NVIDIA GPU | 支持（引用 [\[4\]](#ref-4)） |
| FP8（引用 [\[3\]](#ref-3)） | 8-bit 浮点 | 直接把权重/激活转成 E4M3/E5M2 | Ada/Hopper 及更新架构 | 支持（引用 [\[4\]](#ref-4)） |

三者的取舍不一样：

- **AWQ**：4-bit 下质量通常比朴素 round-to-nearest 好，推理内核成熟；需要校准数据，校准集和业务分布差太远会掉点。
- **GPTQ**：常见 4-bit 路线，压缩率高、生态老；不同实现的 kernel 支持差异较大，遇到不支持的模型/层会回退，速度可能不如预期。
- **FP8**：8-bit 浮点，压缩率不如 4-bit，但质量损失通常更小，Ada/Hopper 有硬件支持；对显存特别紧的场景，4-bit 更划算。

## 和 GGUF 的本质不同

GGUF 是**文件格式 + llama.cpp 的 kernel**：量化方案（Q4_K_M 等）和推理引擎绑在一起，换到 vLLM 不一定能高效跑。

AWQ/GPTQ/FP8 是**权重格式 + GPU kernel**：模型以 safetensors 保存，vLLM 用 Marlin/CUTLASS 等 kernel 加载。它们的“快”来自 kernel 对 4-bit/8-bit 矩阵乘的专门优化，不只是权重变小。

所以：

- 在 Mac 上测出的 GGUF 质量，不能直接代表 GPU 上 AWQ 的质量；
- 同一个模型，GGUF 的 Q4_K_M 和 AWQ 的 4-bit 不是同一种量化，别混着比；
- 交付时要写清楚“哪个格式、哪个量化方案、哪个引擎版本”，否则数字不可复现。

## 怎么选

按顺序问：

1. **显存放得下吗？** 先算权重 + KV + 开销（1.6、7.9）。放不下，优先 4-bit；放得下但余量小，FP8 更稳。
2. **质量能接受吗？** 用业务数据跑同一批题，不要只看困惑度；量化对长尾、数字、专有名词更敏感。
3. **硬件支持吗？** FP8 需要 Ada/Hopper 及更新架构；老卡只能走 AWQ/GPTQ。
4. **有现成权重吗？** 官方/社区的 AWQ/FP8 版本能省掉校准时间；没有就自己量化，校准集要来自业务分布。
5. **kernel 支持吗？** 某些模型结构（MoE、混合注意力）在特定量化方案下可能回退；先跑一个 smoke test。

## 质量怎么测

量化对比最容易犯的错，是只报“困惑度涨了一点”。困惑度是平均指标，会掩盖长尾退化。建议三层：

- **通用探针**：一组固定问题，测基本抽取、算术、中文表达；用来发现“明显坏了”。
- **业务测试集**：第六部分的 v2 测试集或第四部分的 60 题抽取探针；同一批题、同一套判分，比较 BF16/FP8/AWQ。
- **人工抽检**：自动判分只能看子串/规则；量化后的语气、格式、拒答行为要人看。

判分口径要固定：同样的 system prompt、同样的 chat template、同样的 `temperature=0`、同样的 `max_tokens`。**只改量化，不改别的。**

## 速度怎么测

量化对速度的影响不是线性的：

- **decode 单请求**：权重变小，带宽压力下降，通常更快；但 4-bit 的 dequant 开销可能吃掉一部分收益。
- **prefill**：算力密集，4-bit/FP8 的 kernel 吞吐可能更高；但效率取决于形状和实现。
- **高并发**：KV Cache 没量化时，KV 读取会逐渐成为瓶颈；权重变小带来的收益会被 KV 摊薄。
- **显存**：量化最确定的收益是“放得下更多 KV/更大模型”，速度收益要看 kernel。

所以报告要分开：权重显存、KV 显存、单请求 decode、prefill TTFT、高并发吞吐。只报一个“快了多少倍”没有意义。

## 常见坑

**校准集和业务分布不一致。** AWQ/GPTQ 的校准数据决定了哪些权重被保护；用英文通用语料校准，中文业务可能掉点。校准集要覆盖你的真实输入。

**chat template 变了。** 量化权重通常保留原模型的 tokenizer/template；换模板会改变输出分布，质量对比就不可比。

**把 KV Cache 量化混进来。** 权重 4-bit 和 KV 8-bit 是两件事；同时改，无法归因。先固定 KV 精度，再比权重格式。

**FP8 在 Ada 和 Hopper 上不是一回事。** 两者都支持 FP8，但 kernel 优化和吞吐不同；4090 上的 FP8 数字不能直接搬到 H100。

**GPTQ 的 kernel 回退。** 某些层/模型结构可能回退到慢路径；启动日志里会有提示，别只看“能跑”。

**量化后忘了重测拒答。** 量化可能让模型更“敢说”；拒答题的正确行为（说资料里没有）要单独检查。

## 动手实验

::: tip 动手实验：BF16 / FP8 / AWQ 同题对比
**假设**：AWQ 的权重大小约为 BF16 的 1/4，FP8 约为 1/2；质量损失在业务题上小于通用探针；高并发下 KV 读取会摊薄权重变小的收益。

**命令**（单卡 24GB 放不下三个模型同时跑；按 BF16 → FP8 → AWQ 顺序，一次起一个。有 48GB 卡可以同时起三个，用 `--endpoints` 一次跑完）：

```bash
# BF16
bash labs/07-gpu-inference/serve_vllm.sh --model-dir models/qwen3.5-9b-bf16 --model-name qwen3.5-9b --port 8001 --name vllm-bf16
uv run labs/07-gpu-inference/quant_eval.py \
  --endpoints bf16=http://127.0.0.1:8001/v1 --model qwen3.5-9b --limit 60 \
  --output benchmarks/gpu/quant/quality-bf16.csv
bash labs/07-gpu-inference/serve_vllm.sh --stop --name vllm-bf16

# FP8
bash labs/07-gpu-inference/serve_vllm.sh --model-dir models/qwen3.5-9b-fp8 --model-name qwen3.5-9b --port 8002 --name vllm-fp8
uv run labs/07-gpu-inference/quant_eval.py \
  --endpoints fp8=http://127.0.0.1:8002/v1 --model qwen3.5-9b --limit 60 \
  --output benchmarks/gpu/quant/quality-fp8.csv
bash labs/07-gpu-inference/serve_vllm.sh --stop --name vllm-fp8

# AWQ
bash labs/07-gpu-inference/serve_vllm.sh --model-dir models/qwen3.5-9b-awq --model-name qwen3.5-9b --port 8003 --name vllm-awq
uv run labs/07-gpu-inference/quant_eval.py \
  --endpoints awq=http://127.0.0.1:8003/v1 --model qwen3.5-9b --limit 60 \
  --output benchmarks/gpu/quant/quality-awq.csv
bash labs/07-gpu-inference/serve_vllm.sh --stop --name vllm-awq
```

**记录**：每个模型分别的显存、单请求 decode、prefill TTFT、高并发吞吐、抽取正确率、拒答行为；启动日志里的量化方式和 kernel。数据存 `benchmarks/gpu/quant/`。**质量判分是字符串口径，只做同题对比，不要当成 RAG 准确率。**
:::

## 工程含义

**为部署卡量化，不为笔记本量化。** 在 Mac 上调好的 GGUF 不能直接搬到 4090；GPU 交付要走 AWQ/GPTQ/FP8，并在目标卡上重测。

**保留 BF16 基线。** 量化版本的质量/速度都要和 BF16 同题对比；没有基线，就不知道“掉了多少”。

**量化权重也是模型资产。** 记录来源、校准集、量化工具版本、协议；不要把一个来历不明的 4-bit 权重写进交付文档。

**先过质量，再谈速度。** 量化最确定的是省显存；速度取决于 kernel 和负载。质量不过关，快也没有意义。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Lin, J. et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv:2306.00978. https://arxiv.org/abs/2306.00978
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Frantar, E. et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323. https://arxiv.org/abs/2210.17323
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Micikevicius, P. et al. *FP8 Formats for Deep Learning*. arXiv:2209.05433. https://arxiv.org/abs/2209.05433
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>vLLM Project. *Quantization*（AWQ/GPTQ/FP8 的支持与 kernel）. https://docs.vllm.ai/en/latest/features/quantization/index.html
