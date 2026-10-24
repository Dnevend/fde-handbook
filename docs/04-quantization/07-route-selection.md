---
title: 三条量化路线怎么选：GGUF / MLX / AWQ-GPTQ（原理在 4.2，AWQ/GPTQ 实操见 7.5）
---

# 三条量化路线怎么选：GGUF / MLX / AWQ-GPTQ

> **本章要回答的问题**：GGUF、MLX、AWQ/GPTQ 三条路各自适合什么场景，我该按什么标准选？

前六节讲的都是"怎么压"。但同样的 4 位，在三个生态里有三套完全不同的实现，文件不能互换，工具不能混用。选错路线的代价不是质量差几个点，而是**在目标机器上根本跑不起来**。

## 三条路线的分界

| | GGUF | MLX | AWQ / GPTQ |
|---|---|---|---|
| 出身 | llama.cpp 生态 | Apple 的 MLX 框架 | 学术界的训练后量化方法 |
| 目标硬件 | CPU、Metal、CUDA 都行 | 只认 Apple Silicon | NVIDIA GPU |
| 文件形态 | 单文件，自带分词器与模板 | safetensors + `scales`/`biases` | safetensors + 量化配置 |
| 量化方式 | K-quant / IQ，块量化 | 分组仿射，组大小可配（常见 64） | 激活感知 / Hessian 近似 |
| 需要校准数据 | K-quant 不强制，IQ 基本必需 | 不需要 | 需要（一小批代表性文本） |
| 服务引擎 | llama.cpp、Ollama、LM Studio | mlx_lm.server | vLLM、SGLang、TensorRT-LLM |
| 离线交付 | 一个文件拷走 | 一个目录拷走 | 一个目录 + 引擎版本 |

（引用 [1][2][3][4]）

三条路线的量化精度没有本质高下，区别在**误差往哪放**：

- GGUF 的 K-quant 把误差控制在块内，靠 scale/min 的两层结构，工程上最省事。
- MLX 直接对权重做分组仿射，组大小 64、4 位是默认，实现简单、在 Apple GPU 上快。
- AWQ 认为"重要的不是权重大小，而是激活大小"，按激活幅度保护少量关键通道；GPTQ 用二阶信息（Hessian 近似）逐列补偿量化误差。两者都需要校准集，质量通常略好，但只在 GPU 上有意义。

## 按约束选，不按精度选

**第一问：跑在哪？**

- 只有 Mac → GGUF 或 MLX。两者都值得试，MLX 在 Apple Silicon 上往往更快（少一层拷贝、直接吃统一内存），但工具链和生态窄。
- 有 NVIDIA 卡 → AWQ/GPTQ + vLLM。GGUF 也能在 CUDA 上跑（llama.cpp 自带 CUDA 后端），但 vLLM 对 GGUF 仍是实验性支持，生产上通常选 safetensors 系权重配 AWQ/GPTQ。
- 只有 CPU、或客户现场是一台不知道什么配置的机器 → GGUF。这是它最大的优势：一个文件、一套二进制，x86 和 ARM 都能跑。

**第二问：怎么交付？**

离线交付、U 盘拷走、客户不许装 Python 环境——这些约束会把选择直接压到 GGUF。AWQ 的目录里除了权重还有 `quantize_config.json`，引擎版本不对就加载失败；MLX 的目录离了 Apple 就是一堆用不上的 safetensors。

**第三问：谁维护？**

GGUF 的量化文件是自包含的：分词器、对话模板、量化参数都在同一个文件里（3.1 节拆过），换台机器不会出现"模型对了、模板是旧的"。AWQ/GPTQ 的模板在 `tokenizer_config.json` 里，和权重分开放，交付时容易漏。

## 三条路线各自的坑

**GGUF 的坑：质量档位太多。** Q4_0、Q4_K_S、Q4_K_M、IQ4_XS 都是"4 位"，误差能差一倍（4.2、4.3 节）。同一个模型在社区里有十几个文件，不看清类型和校准信息，等于随机抽奖。

**MLX 的坑：只能在 Mac 上验证。** 你在 M4 上调好的 4 位模型，拿到 4090 服务器上没有任何对应实现。如果项目最终要上 GPU，MLX 只能当本地试验田，不能当交付形态。

**AWQ/GPTQ 的坑：校准集决定质量。** 和 imatrix 一样，校准文本不对口，保护的就是错误的通道。另外 AWQ 的 4 位不等于 GGUF 的 4 位：AWQ 通过缩放关键通道来降误差，文件大小和普通 4 位 safetensors 相当，但在 vLLM 上能吃满连续批处理与 PagedAttention 的吞吐。

## 跨路线比较质量是无效的

网上常见的对比表是"Q4_K_M vs AWQ 4bit vs MLX 4bit，谁的分高"。这类比较有三个混杂因素：

1. **量化方案不同**（块量化 vs 激活感知 vs 分组仿射），不是同一个"4 位"。
2. **推理引擎不同**（llama.cpp vs vLLM vs MLX），kernel 实现、KV 策略、batch 大小都不一样。
3. **评测条件不同**（上下文长度、采样参数、提示模板）。

要比，只能在**同一引擎、同一提示、同一测试集**下比同一路线的两个档位。跨路线比出来的差异，大部分不是量化带来的。

## 本书的选择

- **第一至第六部分、第八部分**：Mac 上用 GGUF 的 Q4_K_M。理由不是它质量最高，而是它让整条链路（部署、量化、RAG、评估）在同一台机器上闭环，零成本、可复现。
- **第七部分**：租云 GPU 时走 AWQ/GPTQ + vLLM，实操在 7.5。那时比较的基准是"同一个模型、同一批题，GPU 路线比 Mac 路线快多少、贵多少"，不是"谁更聪明"。

如果客户现场是 Mac 且追求极限速度，MLX 值得单独评估；但本书不把它作为主线，因为它学不到第七部分的内容。

## 工程含义

**先定硬件，再定量化。** 反过来做（先选一个量化文件，再找能跑它的机器）是新手最常见的返工原因。

**交付清单里写清路线与版本。** `GGUF / Q4_K_M / llama.cpp 0.7.0 / sha256` 比 `4 位量化版` 有用得多。AWQ 还要加引擎版本，因为 vLLM 的量化算子在不同版本间变过。

**别为了省 0.2 GB 换路线。** 三条路线的体积差异通常在 10% 以内，但迁移成本（重新校准、重新评测、重新打包）远大于这点空间。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ggml 项目. *GGUF 格式规范与 `llama.cpp` 量化类型*（K-quant、IQ 与单文件交付）. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Apple. *MLX 与 mlx-lm*（统一内存上的数组框架、`mlx_lm.convert` 的量化参数）. https://github.com/ml-explore/mlx-lm
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Lin, J. et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*（按激活幅度保护关键通道）. https://arxiv.org/abs/2306.00978
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Frantar, E. et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*（Hessian 近似的逐列补偿）. https://arxiv.org/abs/2210.17323
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>vLLM. *Quantization*（支持的 AWQ/GPTQ/FP8 等格式与引擎侧限制）. https://docs.vllm.ai/en/latest/features/quantization/index.html
