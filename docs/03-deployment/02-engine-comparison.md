---
title: 推理引擎对比：llama.cpp / Ollama / MLX / vLLM
---

# 推理引擎对比：llama.cpp / Ollama / MLX / vLLM

> **本章要回答的问题**：这几个引擎都在做推理，差别在哪？我该用哪个？

## 一句话定位

| 引擎 | 一句话 | 主要硬件 | 权重要求 | 并发能力 |
|---|---|---|---|---|
| llama.cpp | 手动挡的推理工具集 | CPU / Metal / CUDA / Vulkan | GGUF | 有槽位与连续批处理，规模有限 |
| Ollama | 包了 llama.cpp 的模型管家 | 同上 | 自己的模型仓库 | 同上，参数被封装 |
| MLX | Apple 自家的高性能框架 | Apple Silicon | MLX 量化格式 | 弱，偏单用户 |
| vLLM | 为吞吐设计的服务引擎 | NVIDIA / AMD GPU | safetensors / AWQ / GPTQ / FP8 | 强，连续批处理 + PagedAttention |

这不是"谁更强"的排名。它们优化的是不同目标：llama.cpp 优化"在什么机器上都能跑"，Ollama 优化"让人少配参数"，MLX 优化"把 Apple 芯片压干"，vLLM 优化"一台 GPU 服务器扛多少并发"。

## llama.cpp：把推理做成可移植的 C++

核心是一个叫 ggml 的张量计算库，加上针对各后端（Metal、CUDA、Vulkan、ROCm、SYCL）的算子实现。它的价值不在单项最快，而在于**覆盖面**：同一份 GGUF 能在 Mac、Windows、Linux、树莓派、手机上跑，不依赖 Python 环境。

服务端是 `llama-server`，提供 OpenAI 兼容接口。默认就开着连续批处理（`-cb`，引用 [\[1\]](#ref-1)），用 `-np` 控制槽位数。它是个合格的小规模服务端，够 1–10 人用；但没有 PagedAttention 那种精细的显存管理，并发一高就会排队。

代价是**参数多**。上下文、槽位、KV 类型、卸载层数、采样默认值全要自己定，定错了就出现第三部分后面几节讲的那些现象。

## Ollama：把 llama.cpp 藏起来

Ollama 用 Go 写服务层，底层还是 llama.cpp 的推理内核，外面加了三样东西：模型仓库与拉取（`ollama pull`）、模型定义文件（Modelfile，把模板、参数、停止词打包）、常驻加载（`keep_alive`，默认 5 分钟，引用 [\[2\]](#ref-2)）。

它的优点是上手快：`ollama run qwen3.5:9b` 就有交互界面，`/api/chat` 和 OpenAI 兼容接口都能用，适合个人试用和不关心参数的场景。

它的代价是**不透明**：模板、上下文长度、采样参数都被默认值接管。最典型的坑是上下文长度——Ollama 不会自动用满模型标称的窗口，`num_ctx` 要显式设，默认值随版本变化，要用 `ollama show <model>` 核对。你以为在测 32K 长文档，实际可能只喂进去 4K。做正式测量和交付时，这种"看不见的默认值"比性能低更危险。

## MLX：只在 Apple 上快的路

MLX 是 Apple 的数组框架，直接吃统一内存，省掉 CPU↔GPU 拷贝。配套的 `mlx-lm` 负责转换、量化、推理和起服务，量化格式是 4 位（组大小可配，引用 [\[3\]](#ref-3)）。

在 M 系列芯片上，MLX 的 decode 速度通常明显高于 llama.cpp 的 Metal 后端，因为它对 Apple GPU 的调度更贴。但它**完全不跨平台**：同一份权重到了 4090 上没法用，第七部分的多卡、连续批处理、PagedAttention 那一套它也没有。把它当"Mac 上的加速选项"，而不是"跨平台的部署方案"。

## vLLM：为吞吐而生的服务引擎

vLLM 的三个核心机制，都是为了"一台机器扛更多并发"：

- **PagedAttention**：把 KV Cache 切成固定大小的块，按需分配，消除按最大长度预分配造成的浪费（引用 [\[4\]](#ref-4)）；
- **连续批处理**：请求不必等整批结束，谁生成完谁退出、新请求随时插入，GPU 不空转；
- **前缀缓存**：多个请求共享同一段前缀（系统提示、文档）时直接复用 KV。需要显式开启 `enable_prefix_caching`，且只省 prefill、不省 decode（引用 [\[5\]](#ref-5)）。

它只吃 GPU（NVIDIA/AMD），模型要用 safetensors 或 AWQ/GPTQ/FP8 量化版，环境依赖 CUDA 生态。对本书的 24GB Mac 来说直接排除；第七部分会用它做对照。

注意一个和本书模型相关的细节：`Qwen3.5` 是混合结构（线性注意力 + 全注意力），vLLM 对这类模型的 KV 管理与纯 Transformer 不同，前缀缓存的命中粒度受内部块大小限制（引用 [\[5\]](#ref-5)）。这类"引擎对新架构的支持程度"，往往比跑分更能决定你能不能上线。

## 为什么"同一个模型，不同引擎差很多"

看到别人报 40 tok/s、你测出 17 tok/s，先别怀疑自己。差异可能来自这些地方，每一条都不属于模型本身：

| 差异来源 | 说明 |
|---|---|
| 量化档位 | q4_K_M 和 q8_0 的字节数差一倍，速度差接近一倍 |
| 前缀缓存 | 命中缓存的请求几乎不花 prefill，长 prompt 场景差距巨大 |
| 模板与思考模式 | 思考模式多生成几百 token，总时长直接翻倍 |
| 采样参数 | 默认 temperature 0.8 与 0 的差异，在可复现性和输出长度上都能看出来 |
| 批处理与排队 | 单请求测速与并发测速是两种数字，不能混用 |
| 硬件与后端 | Metal、CUDA、CPU 三套算子实现，性能曲线不同 |

第三部分剩下的章节，本质上就是在逐个拆这些变量。**测量方法（3.7）比任何单次数字都重要**——没有统一口径，引擎之间的比较没有意义。

## 工程含义

**原型期用 llama.cpp 或 Ollama，Mac 上想压速度试 MLX，上线 GPU 用 vLLM。** 这是常见路径，不是教条。真正要守的是：换引擎等于换了一整套默认值，之前调好的参数要重新验证一遍。

**把引擎版本写进交付文档。** 同一个 GGUF、同一个 prompt，llama.cpp 0.6 和 0.7 的输出可能不同（采样实现、KV 类型默认值都在变）。客户问"为什么结果变了"，第一个要查的就是版本。

**不要跨引擎搬运性能数字。** 在 Mac 上测出的 tok/s 不能拿去给 GPU 服务器做容量承诺，反过来也不行。可以搬运的是公式和趋势。

## 待实测

::: tip 待实测：同一份权重，换引擎差多少
**假设**：在 M4/24GB 上，MLX 的 decode 速度高于 llama.cpp 的 Metal 后端；Ollama 与 llama.cpp 在相同上下文和槽位配置下速度接近，但默认参数会明显改变结果。

**准备**：本机只装了 llama.cpp 也能做第一组；第二组需要额外安装，按需选择。

```bash
# 第一组：llama.cpp 基线
llama-bench -m "$MODEL" -p 512 -n 128 -r 5 -o md | tee benchmarks/engines/llamacpp.md

# 第二组：Ollama（brew install ollama）
ollama pull qwen3.5:9b
ollama show qwen3.5:9b            # 先看清默认 num_ctx 和模板
# 显式把上下文设成和 llama.cpp 一致，再比
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen3.5:9b",
  "prompt": "用一句话解释什么是 KV Cache。",
  "options": {"num_ctx": 8192, "temperature": 0},
  "stream": false
}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["eval_count"], "token", round(d["eval_count"]/d["eval_duration"]*1e9,1), "tok/s")'

# 第三组：MLX（pip install mlx-lm）
mlx_lm.convert --model Qwen/Qwen3.5-9B -q --mlx-path models/qwen35-9b-mlx
mlx_lm.generate --model models/qwen35-9b-mlx --prompt "用一句话解释什么是 KV Cache。" --max-tokens 128
```

**记录**：每个引擎的 tok/s、首字延迟、上下文长度、模板来源（模型自带还是引擎默认）、量化档位。**同一行里必须写清这几项**，否则数字不可比。数据存 `benchmarks/engines/`。

**判断标准**：差异小于 15% 视为噪声；差异大于 2 倍，先查上表的六个来源，再怀疑引擎。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp server 文档*（`-np`、`-cb`、`-c` 与 OpenAI 兼容端点）. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Ollama. *API 文档*（`keep_alive`、`num_ctx` 等运行参数）. https://github.com/ollama/ollama/blob/main/docs/api.md
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>Apple. *mlx-lm*（`convert`、量化参数与 `mlx_lm.server`）. https://github.com/ml-explore/mlx-lm
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>vLLM. *Automatic Prefix Caching*（`enable_prefix_caching`、只省 prefill、混合 Mamba 的块粒度限制）. https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>Yu, G.-I. et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
