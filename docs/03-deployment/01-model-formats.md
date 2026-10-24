---
title: 模型格式：safetensors / GGUF / MLX
---

# 模型格式：safetensors / GGUF / MLX

> **本章要回答的问题**：同一个模型为什么有 safetensors、GGUF、MLX 好几种文件？它们能互相转换吗？

## 先分清两件事：精度和格式

新手最容易把这两个概念混在一起。它们正交：

- **精度**是每个参数用几位表示：fp16（16 位）、q8_0（8.5 位）、q4_K（4.5 位）……第四部分专门讲。
- **格式**是文件怎么组织：张量按什么顺序摆放、元数据（架构参数、分词器、对话模板）存在哪、能不能随机访问。

同一个模型可以同时有 `safetensors`（fp16）、`GGUF`（q4_K_M）、`MLX`（4-bit）三种文件。体积差好几倍，主要来自精度；能不能被某个引擎加载，取决于格式。选型时先问格式，再问精度。

## safetensors：训练侧的交换格式

结构极简：文件开头 8 个字节是 header 长度 N，接下来 N 个字节是 JSON header，记录每个张量的名字、数据类型、形状和字节偏移，剩下的整块是张量数据（引用 [\[1\]](#ref-1)）。

这个设计解决的是**安全**和**加载速度**两件事：

- 它不含可执行代码，不像 pickle 那样可能在你打开文件时跑任意代码；
- 张量数据连续摆放、偏移量写在 header 里，可以懒加载和零拷贝，不必把整个文件读进内存。

代价是**它只管张量**。分词器、对话模板、架构参数在这些文件外面：`config.json`、`tokenizer.json`、`tokenizer_config.json`、`generation_config.json` 各管一摊。取一个模型常常要下七八个文件，少一个就起不来。这也解释了为什么 HF 页面上总在强调"用 `snapshot_download` 而不是手点文件"。

9B 参数的模型用 fp16 存，光权重就约 18 GB（估算：9e9 × 2 字节）。这个体积在 24GB 的 Mac 上只剩理论可能，所以真正拿来推理时几乎总是换成下面两种格式之一。

## GGUF：推理侧的交付格式

GGUF 是 llama.cpp 生态的原生格式，目标很明确：**一个文件装下推理需要的全部东西**（引用 [\[2\]](#ref-2)）。

文件结构分四段：magic（`GGUF`）、版本号、元数据键值对、张量信息表，最后按对齐要求摆放张量数据。元数据里除了架构参数，还塞进了分词器和对话模板——这是它和 safetensors 最大的区别。

本书主力模型的实际内容如下（实测，见 `benchmarks/gguf-format/`，命令见本节"动手实验"）：

| 项目 | 值 |
|---|---|
| 格式版本 | 3 |
| 元数据条目 | 46 |
| 张量数 | 427 |
| header 数据偏移 | 10,968,288 字节（约 10.5 MB） |
| 量化类型分布 | q4_K 132 个、q5_K 48 个、q6_K 22 个、q8_0 48 个、f32 177 个 |
| 文件大小 | 5.29 GiB（8.95B 参数，约 5.08 位/参数） |

四条对工程有直接影响的观察：

**第一，文件是自包含的。** 元数据里有 `tokenizer.ggml.tokens`、`tokenizer.ggml.merges` 和 `tokenizer.chat_template`。这意味着分发一个 GGUF 就等于分发了分词器和模板，不会出现"模型对了但模板是旧的"这种问题。也意味着**模板错了要换整个文件**，不能只换一个 json。

**第二，量化是块结构，不是简单的"每个数砍几位"。** 以 q4_K 为例：256 个权重组成一个超级块，内部再分 8 组，每组 32 个；每组有自己的 6 位缩放系数和最小值，权重本身是 4 位（引用 [\[3\]](#ref-3)）。算下来每个权重占 4.5 位，而不是 4 位——多出来的 0.5 位就是缩放系数。同理 q6_K 是 6.5625 位，q8_0 是 8.5 位（每 32 个权重配一个 16 位缩放因子）。

**第三，`Q4_K_M` 的 M 是"混合"。** 它不是全用 q4_K，而是大部分张量用 q4_K、少数敏感张量（比如输出层）用 q6_K，另有一批用 q5_K。上表里 132 个 q4_K、48 个 q5_K、22 个 q6_K 就是这件事。混合让平均位数落在 5.08 位，比纯 q4_K 的 4.5 位略高，换来的是质量损失更小。第四部分会展开这个取舍。

**第四，元数据里留着量化的来源。** 这个文件的元数据含 `quantize.imatrix.file`、`quantize.imatrix.dataset` 等四项，说明它是用重要性矩阵（imatrix）校准过的，不是裸量化。拿到一个别人量化好的 GGUF 时，先看这几项在不在，能判断对方是不是认真做过。

GGUF 的加载方式是**内存映射（mmap）**：不需要把 5.7 GB 一次性读进内存，而是按页访问，用到哪读哪。这带来一个实际好处（启动快、多个进程共享同一份权重页）和一个实际麻烦（**峰值内存的测量口径会变**，1.4 节已经踩过这个坑）。

## MLX：Apple 生态的格式

MLX 是 Apple 的数组计算框架，模型文件本身是 safetensors 加一组量化张量：权重存成整数，另存 `scales` 和 `biases`，按组还原（引用 [\[4\]](#ref-4)）。`mlx_lm.convert -q` 生成的就是这种文件，4 位、组大小 64 是常见配置，可以按需改（引用 [\[5\]](#ref-5)）。

它的优势和局限同样明显：在 Apple Silicon 上直接吃统一内存、少一层拷贝，速度往往比 llama.cpp 快；但它**只认 Apple GPU**。同一份权重拿到 4090 服务器上没有对应实现，第七部分讲的东西它一样都不适用。

## 能不能互相转换

能，但方向不对称。

| 方向 | 工具 | 有损吗 |
|---|---|---|
| safetensors → GGUF | `convert_hf_to_gguf.py`，再 `llama-quantize` | 量化那一步有损 |
| safetensors → MLX | `mlx_lm.convert -q` | 量化那一步有损 |
| GGUF → safetensors | 反量化后导出 | 不可逆，量化丢的信息回不来 |
| GGUF → GGUF | `llama-quantize`（改量化档） | 每转一次都可能再掉一点 |

实践中容易踩的坑有三个：

- **架构支持滞后。** 新模型发布后，llama.cpp 要先合入架构支持才能转换。`Qwen3.5` 这种混合结构（线性注意力 + 全注意力）尤其如此，旧版本转换脚本会直接报未知架构。
- **分词器和模板可能丢。** 从 GGUF 反推回 HF 目录时，`tokenizer_config.json` 里的模板往往要手工补。反过来的方向（HF → GGUF）反而稳，因为模板本来就在 HF 文件里。
- **词表 id 不能变。** 转换只改存储方式，不改 id 顺序；一旦有人"顺手修一下词表"，模型就会开始胡言乱语，而且不报错。

## 工程含义

**格式跟着引擎走，不跟着模型走。** 你要用 vLLM，就得准备 safetensors（或 AWQ/GPTQ 量化版）；要用 llama.cpp，就得有 GGUF；要用 MLX，就得转换。选型讨论里"这个模型好不好"和"这个模型能不能在我这儿跑"是两个问题。

**交付清单里要写清四样东西**：仓库名、commit、文件名、SHA256。少一样，一个月后就没法解释"为什么客户那边的输出和测试环境不一样"。这里还有一个容易被忽略的口径问题：整文件哈希和张量内容哈希不是一个值。本机这个文件，`shasum -a 256` 得到 `03b74727…`（和 HF 缓存里的 blob 名一致），而 `llama-gguf-hash --sha256 --no-layer` 得到 `2d9e6160…`（只算张量数据）。跟分发方对数时必须先问清楚对方用的是哪一种。

**单文件和多文件是两种运维。** GGUF 一个文件能刻盘带走，适合离线交付；safetensors 一族文件多、体积大，适合放在带版本管理的对象存储里，由部署脚本拉取。

**磁盘占用不等于内存占用。** 5.7 GB 的文件通过 mmap 跑起来，常驻内存取决于访问到的页，不代表 5.7 GB 全在内存里。做容量规划时要用 1.6 节的公式，而不是看文件大小。

## 动手实验

::: tip 动手实验：读懂一个 GGUF 文件的内部
**假设**：本机的 `Qwen3.5-9B-Q4_K_M.gguf` 由 imatrix 校准、含对话模板与分词器、平均约 5 位/参数。

**命令**：

```bash
mkdir -p benchmarks/gguf-format

# 1. 元数据与张量表：版本、条目数、张量数、每层的量化类型
#    注意：工具会把同一份信息打印两遍（一遍校验），下面的 grep 会得到重复行
llama-gguf "$MODEL" r > benchmarks/gguf-format/dump.txt 2>&1
grep -E "version:|alignment:|data offset|n_kv:|n_tensors:" benchmarks/gguf-format/dump.txt | head -5

# 2. 量化类型分布，用来看 Q4_K_M 到底混了哪几种
grep -oE "\b(q[0-9]_K|q[0-9]_0|f32|f16|bf16)\b" benchmarks/gguf-format/dump.txt | sort | uniq -c | sort -rn

# 3. 两种哈希口径，分别记下来
shasum -a 256 "$MODEL"                                  # 整个文件
llama-gguf-hash --sha256 --no-layer "$MODEL"            # 只算张量内容

# 4. 平均位数 = 文件字节数 × 8 ÷ 参数量。注意 $MODEL 是符号链接，要加 -L
#    没有 -L 时 ls 报的是链接本身的 76 字节，会算出 0.00 位/参数
ls -lL "$MODEL" | awk '{printf "%d 字节，%.2f 位/参数\n", $5, $5*8/8.95e9}'
```

**记录**：元数据条目数、张量数、各量化类型的张量个数、实测平均位数、两种哈希，以及 `tokenizer.chat_template` 是否存在。把 `dump.txt` 与 `bash benchmarks/record-env.sh` 的输出放在一起。**踩过的坑**：不用 `-L` 时算出的位数是 0，看到这个数先检查是不是在量符号链接。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Hugging Face. *safetensors*（文件格式：8 字节 header 长度 + JSON header + 数据区）. https://github.com/huggingface/safetensors
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>ggml 项目. *GGUF 格式规范*（版本、元数据 KV、张量信息、对齐）. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>ggml 项目. *`ggml-common.h`*（`block_q4_K`、`block_q6_K` 的块结构与位数）. https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Apple. *MLX*（统一内存上的数组框架与量化张量表示）. https://github.com/ml-explore/mlx
5. <a id="ref-5" style="scroll-margin-top: 96px"></a>Apple. *mlx-lm*（`mlx_lm.convert`、`--q-bits` 与 `--q-group-size`）. https://github.com/ml-explore/mlx-lm
6. <a id="ref-6" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡*. https://huggingface.co/Qwen/Qwen3.5-9B
