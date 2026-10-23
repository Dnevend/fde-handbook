---
title: 硬件与环境：Apple Silicon 和 NVIDIA 的差异，本书如何对照
---

# 硬件与环境：Apple Silicon 和 NVIDIA 的差异，本书如何对照

> **本章要回答的问题**：Mac 和 NVIDIA 服务器差在哪？我在 Mac 上学到的东西，有多少能直接搬到 GPU 上？

## 两台"基准机"

本书的默认环境是一台 Apple M4、24GB 统一内存的 Mac，系统 macOS 26.2，推理引擎为 llama.cpp 0.7.0（build 11669，commit `620d2225c`）。所有不涉及 CUDA 的实验都在它上面完成。

第七部分需要 NVIDIA 显卡，基准是单卡 RTX 4090（24GB 显存），必要时租用 H100。按小时计费，具体价格以平台为准。

列出具体型号不是为了比较跑分，而是因为**推理性能的数字离开硬件就没有意义**。同一份权重，在 M4 上大约 20 token/s，在 4090 上可能是它的七八倍，这个差距不是引擎写得好不好，是内存带宽决定的。

## 关键差异只有三条

其他差异（指令集、驱动、生态）都可以从这三条推导出来。

**第一条：带宽差一个数量级。** M4 的统一内存带宽是 120 GB/s（引用 [\[1\]](#ref-1)）；RTX 4090 的显存带宽是 1008 GB/s（引用 [\[2\]](#ref-2)）；H100 SXM 是 3.35 TB/s（引用 [\[3\]](#ref-3)）。

为什么带宽这么关键？因为大模型逐 token 生成时，每生成一个 token 都要把参与计算的权重从内存读一遍。读取时间约等于"权重字节数 ÷ 带宽"。以本书主力模型 `Qwen3.5-9B` 的 `Q4_K_M`（5.68 GB，引用 [\[4\]](#ref-4)）为例：

- M4：5.68 GB ÷ 120 GB/s ≈ 47 ms/token，对应约 21 token/s 的理论上限（估算）；
- RTX 4090：5.68 GB ÷ 1008 GB/s ≈ 5.6 ms/token，对应约 177 token/s 的理论上限（估算）。

这是**上限**，实际会因 KV Cache 读取、算子开销、内存碎片而更低。但数量级是对的，也是你在两台机器上会看到差距的根本原因。1.5 节会把这条推导讲完整。

**第二条：内存的组织方式不同。** Mac 是统一内存（unified memory）：CPU 和 GPU 共用同一块物理内存，模型不需要在两端之间拷贝。好处是"能装下就能用"，24GB 的机器可以加载接近 24GB 的模型（还要给系统和 KV Cache 留空间）。坏处是这块内存同时要养操作系统、浏览器和你的编辑器，实际可用额度通常比标称少 4–6 GB（估算，取决于你开了多少东西）。

NVIDIA 那边是独立的显存：模型必须完整放进显存，放不下就报错，报错很干脆。Mac 上放不下时往往不是报错，而是开始用 swap，表现为"整个系统变卡、生成速度断崖式下跌"。这是排查时最容易误判的一种现象——它看起来像模型慢，其实是内存不够。

**第三条：软件栈不同。** Mac 上 llama.cpp 走 Metal，能用的算子、量化格式和优化都比 CUDA 侧少。NVIDIA 上有 CUDA、cuBLAS、FlashAttention，以及 vLLM、TensorRT-LLM 这类为吞吐优化的引擎。所以第七部分讲的连续批处理、PagedAttention、张量并行，在 Mac 上没有对应实现，不要试图在 Mac 上复现。

## 哪些能迁移，哪些不能

把本书的实验按"能不能在 Mac 上做"分成三类：

| 内容 | Mac 上可做 | 必须 NVIDIA |
|---|---|---|
| 分词、自回归、prefill/decode 的定性观察 | 可以 | — |
| KV Cache 与显存的定量估算 | 可以 | — |
| 量化格式对比、困惑度测量 | 可以（llama.cpp） | — |
| RAG 全链路、评估方法 | 可以 | — |
| 并发与吞吐 | 只能小规模观察 | 需要 |
| vLLM / TensorRT-LLM | 不可 | 需要 |
| 多卡、张量并行 | 不可 | 需要 |
| CUDA 算子、FlashAttention 变体 | 不可 | 需要 |

可以迁移的是**概念和公式**：为什么首字慢、KV Cache 怎么算、量化压掉了什么、并发为什么会排队。这些结论在两种硬件上都成立，只是常数不同。

不能迁移的是**具体数字和调优手感**。在 Mac 上把 `-t`（线程数）调到最优，对 4090 毫无意义；在 Mac 上测出的 token/s，也不能拿去给客户做容量承诺。本书在所有性能数字旁边都会写明硬件，就是为了防止这种搬运。

## 环境准备的最小清单

不需要一开始就装全。跑通第一部分和第二部分，只需要：

```bash
# 推理引擎（macOS，Homebrew）
brew install llama.cpp
llama-cli --version        # 记录版本，本书为 0.7.0 (build 11669)

# 下载主力模型（约 5.7 GB，首次需要联网）
llama-cli -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M -st -p "你好" -n 32

# 查看本机内存与 Exchange/swap 使用
sysctl -n hw.memsize
vm_stat | head -5
```

**命令里的两个参数不是可选的。** `-st`（single turn）让命令行在回答完一轮后退出。从 0.7.0 起 `llama-cli` 默认进入连续对话模式，不加 `-st` 会停在输入提示上等你打字，看起来就像"卡住了"。带 `-hf` 下载时，如果仓库里有视觉投影模型（`mmproj-*.gguf`），llama.cpp 会顺带下载一份（`Qwen3.5-9B` 的约 879 MB）。纯文本实验用不到它，但也不会被加载；想省下载时间可以加 `--no-mmproj`。

**下载之后，把模型路径固定成 `$MODEL`。** `-hf` 把权重放在 Hugging Face 缓存里，路径中间带一段版本哈希，每次手写既容易错、也会随更新失效。本书所有命令统一写成 `-m "$MODEL"`：

```bash
# 在仓库根目录执行。脚本会用通配查找缓存里的 Qwen3.5-9B-Q4_K_M.gguf
source labs/env.sh
echo "$MODEL"      # 例如 ~/.cache/huggingface/hub/models--unsloth--.../Qwen3.5-9B-Q4_K_M.gguf
```

`export` 只在当前终端窗口有效，新开窗口要重新 `source` 一次。想一劳永逸，把 `source labs/env.sh` 写进自己的 shell 启动配置；如果设置过 `HF_HOME` 或 `LLAMA_CACHE`，缓存位置会不同，`env.sh` 已经处理了这两种情况。

**后文不再逐节提醒。** 本书所有命令都假定：已在仓库根目录执行过 `source labs/env.sh`，`$MODEL` 指向主力模型。换终端窗口后忘了这一步，报错通常是"模型加载失败"——先 `echo "$MODEL"` 确认它不为空。

后续章节会按需增加 Python 环境、向量库和评测脚本。凡是需要额外依赖的地方，都会在那一节说明用途，不提前铺开。第三部分的实验脚本用 Python 写成，依赖只有 OpenAI 官方 SDK，用 `uv` 装（3.5 节给命令）。

还有一个不起眼但很重要的习惯：**每次实验都记下三样东西——硬件、引擎版本、完整命令**。一周后你会感谢自己。仓库里的 `benchmarks/record-env.sh` 一次打印前两样，复制到数据文件的头部即可：

```bash
bash benchmarks/record-env.sh > benchmarks/<主题>/env.txt
```

本书的"待实测"提示块里，命令是写全的，你跑完把输出填进去就能得到一条可引用的结论。

## 接下来

硬件决定了你的天花板，但天花板下面的空间怎么用，取决于你对推理过程的理解。第一部分从最小的单位开始：token。

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Apple. *Apple introduces M4 Pro and M4 Max*（2024-10）. https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>NVIDIA. *GeForce RTX 4090 规格页*. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>NVIDIA. *NVIDIA H100 Tensor Core GPU*. https://www.nvidia.com/en-us/data-center/h100/
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Unsloth. *Qwen3.5-9B-GGUF*（`Qwen3.5-9B-Q4_K_M.gguf`，5.68 GB）. https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
