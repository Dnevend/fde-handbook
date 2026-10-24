---
title: 采样参数：temperature / top-p / 重复惩罚
---

# 采样参数：temperature / top-p / 重复惩罚

> **本章要回答的问题**：temperature、top-p 到底在改什么？调到多少才算"对"？

## 下一步 token 不是"选出来"的，是"抽出来"的

模型每生成一个 token，输出的不是一个词，而是一个长度为词表大小（`Qwen3.5` 是 248,320）的分数向量。分数先经过 softmax 变成概率分布，然后按某种规则从这个分布里挑一个——这一步就是采样（sampling）。

所有采样参数改的都是同一件事：**怎么从分布里挑**。理解这一点，参数就不再是玄学：

| 参数 | 作用 | 直觉 |
|---|---|---|
| `temperature` | 把分数除以 T 再归一化 | T<1 让分布更尖（保守），T>1 更平（发散） |
| `top_k` | 只保留概率最高的 k 个候选 | 硬性砍掉长尾 |
| `top_p` | 保留累计概率达到 p 的最小集合 | 动态砍尾，比 top_k 更自适应 |
| `min_p` | 保留概率 ≥ (最大概率 × min_p) 的候选 | 按"相对最大概率"砍，跨分布更稳 |
| `repeat_penalty` / `presence_penalty` | 惩罚最近出现过的 token | 抑制绕圈复读 |
| `seed` | 采样器随机数种子 | 固定它才谈得上复现 |

`temperature = 0` 是特例：llama.cpp 直接走贪心（每次取概率最高的 token），不再随机。这就是 1.2 节可复现实验的基础。

严格说，`top_p` 是核采样（nucleus sampling），由 Holtzman 等人提出，用来解决"纯按概率采样会生成胡话、贪心又会重复"的两难（引用 [\[1\]](#ref-1)）。

## 默认值不等于推荐值

llama-server 启动后会把默认采样参数挂在 `/props` 上。本机实测（见 `benchmarks/sampling/`）：`temperature 0.8`、`top_p 0.95`、`top_k 40`、`min_p 0.05`、`repeat_penalty 1.0`、`presence_penalty 0`。

模型卡给的是另一套，按模式和任务分四组（引用 [\[2\]](#ref-2)）：

| 场景 | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| 思考模式，通用任务 | 1.0 | 0.95 | 20 | 1.5 |
| 思考模式，精代码 | 0.6 | 0.95 | 20 | 0.0 |
| 非思考，通用任务 | 0.7 | 0.8 | 20 | 1.5 |
| 非思考，推理任务 | 1.0 | 0.95 | 40 | 2.0 |

两处差异值得记住：

- **`presence_penalty`**：引擎默认 0，模型卡建议 1.5（除精代码）。思考模式下这条影响很大——不惩罚重复，长思考容易在原地打转，最后把预算烧光。
- **`temperature`**：引擎默认 0.8，而模型卡在思考模式建议 1.0。也就是说，"开箱即用"的参数并不在作者推荐的区间里。

这不是引擎的问题。默认值是给"随便试试"用的，正式部署要按任务重设，并把最终参数写进配置。

## 固定 seed 能复现到什么程度

实测（见 `benchmarks/sampling/sweep.csv`）：在同一台机器、同一服务实例上，把 seed 固定为 42，`temperature=1.0` 连跑三次，输出**逐字相同**；换一组温度，输出立刻不同。

结论要分清两层：

- **同一组参数 + 同一 seed + 同一引擎版本 + 同一 prompt**：可复现。这是评估和回归测试的基础。
- **跨引擎、跨版本、跨批大小**：不保证。浮点归约顺序、批处理切分、算子实现都会让 logits 出现微小差异，温度越高越容易被放大。

所以"`temperature=0` 就一定逐字一样"是不严谨的说法。准确的说法是：同一套环境里可复现，换环境要重跑基线。

## 三类任务怎么配

**从材料里摘取答案（RAG 问答、字段抽取）**：要的是稳定和可核对。用 `temperature=0` 或 0.1，`top_p` 收窄，思考模式大多可以关掉。输出越短越可控，别让它复述原文。

**结构化输出（JSON、表格）**：低温度 + 短字段名。格式错了要重试，重试成本高，所以宁可保守。温度高时 JSON 的括号和引号最容易出错。

**需要多样性的场景（起标题、写备选方案）**：温度 0.7–1.0，必要时只保留 top_p 不设 top_k。同一问题多采几次，人工挑。

**评估和回归**：固定 seed + `temperature=0`，把参数写进脚本。参数漂了，评估结果就不可比。

## 工程含义

**采样参数是接口的一部分。** 服务端默认值一变，所有客户端行为都跟着变。上线前把最终值固定写进启动参数或客户端的 `extra_body`，不要依赖默认值。

**温度和长度共同决定成本。** 温度高 → 输出更长、更容易发散 → 总时长上升。1.2 节说过"输出长度就是时间"，采样参数是控制输出长度的第一道闸。

**不要用采样参数掩盖模板问题。** 输出乱、重复、答非所问，先查 3.3 的模板，再调采样。反过来的顺序会让你在错误的配置上反复调参。

**思考模式要单独配参数。** 模型卡把思考和非思考分成两套推荐值，不是随便写的：思考内容更长、更容易重复，`presence_penalty` 的作用完全不同。

## 动手实验

::: tip 动手实验：参数扫描与可复现边界
**假设**：固定 seed 时同参数组内输出逐字相同；温度升高后输出发散；收窄 top_p 能把高温输出拉回稳定区间。

**命令**（先起服务，见 `serve.sh`）：

```bash
cd labs
BASE_URL=http://127.0.0.1:8080/v1 uv run python 03-deployment/sampling_sweep.py
```

脚本固定 seed=42，扫四组参数（T=0、T=0.7/top_p=0.95、T=1.0/top_p=0.95、T=1.0/top_p=0.5），每组重复 3 次，记录输出长度、decode 速度和前 160 字。

**记录**：每组三次输出是否逐字相同、平均生成长度、速度。数据存 `benchmarks/sampling/`。

**建议追加一组对照**：把 `presence_penalty` 从 0 调到 1.5，用同一个需要多步推理的问题跑，比较生成量与是否出现重复段落。这一组能直接回答"模型卡的推荐值到底值不值得照抄"。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Holtzman, A. et al. *The Curious Case of Neural Text Degeneration*（核采样）. arXiv:1904.09751. https://arxiv.org/abs/1904.09751
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡*（四种场景的推荐采样参数、思考模式与输出长度建议）. https://huggingface.co/Qwen/Qwen3.5-9B
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp*（`common/sampling.cpp` 中各采样器的实现与默认值）. https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp server 文档*（`/props` 返回的默认生成参数）. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
