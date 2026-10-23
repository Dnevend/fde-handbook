---
title: 对话模板（Chat Template）：模板错了模型就会胡言乱语
---

# 对话模板（Chat Template）：模板错了模型就会胡言乱语

> **本章要回答的问题**：模型明明能跑，为什么回答开始胡言乱语？对话模板写错了会发生什么？

## 模板做的事：把对话翻译成 token

模型本身只认 token 序列，不认"角色"。你在界面上看到的 user / assistant / system 三段，必须在送进模型之前被拼成一段它训练时见过的文本。这个拼接规则就是对话模板（chat template）。

`Qwen3.5` 的模板用特殊 token 划分说话人：

```text
<|im_start|>system
你是一个严谨的运维助手。<|im_end|>
<|im_start|>user
三号产线今天停机几次？<|im_end|>
<|im_start|>assistant
```

关键词只有三个：`<|im_start|>` 开始一个回合，角色名紧跟其后，`<|im_end|>` 结束。模型在预训练后经过指令微调（SFT）时，见到的就是这种结构；它靠这些 token 判断"接下来轮到我说话"以及"话说完了该停"。

模板做的事情可以用一句话概括（图 3.3-1）：把上面的消息数组拼成下面那串 token，模型看到的只有后者，它从来没见过"角色"这个概念。

<Figure
  id="3.3-1"
  src="/figures/3-3-chat-template.svg"
  alt="对话模板的翻译过程：system、user、assistant 三条消息经模板拼成一串带特殊 token 的文本，assistant 之后紧跟 think 标记"
  caption="模板不只管角色，还管正文从哪里开始——橙色那几个标记是模板插入的，不是模型生成的。"
  source="模板结构以 Qwen3.5-9B 模型卡为准（引用 [2]）；token 串为示意，不是逐字实测输出。"
/>

`Qwen3.5` 的模板还多一层：默认在 assistant 开头插入 `<think>`，思考结束时输出 `</think>`，两者之间是思考内容，之后才是正文（引用 [\[2\]](#ref-2)）。所以模板不只管角色，还管**思考从哪里开始、正文从哪里开始**。

模板的写法是 Jinja2 模板，和分词器一起存放在模型文件里。GGUF 里是 `tokenizer.chat_template` 这个元数据项（3.1 节实测确认过），HF 的 safetensors 文件里则在 `tokenizer_config.json`。llama.cpp 默认用模型自带的模板，也可以用 `--chat-template` 覆盖（引用 [\[3\]](#ref-3)）。

## 模板错了，不会报错

这是本节最重要的一句话。模板错误的典型表现不是崩溃，而是**静默降质**：

- 角色标记错位，模型把用户的话当成自己的续写，开始自问自答；
- 结束标记不匹配，模型不知道该停，一直生成到 `max_tokens` 用完；
- 思考标记缺失，`<think>` 变成普通文本被念出来；
- 模板参数（比如 `enable_thinking`）不生效，因为你传的参数名在错误模板里根本没人读。

第 1 种和第 2 种肉眼可见，第 3 种和第 4 种最危险——输出看起来"还挺像样"，只是变长、变啰嗦，或者该简洁的地方开始长篇大论。

## 实测：把模板换成 chatml 会怎样

用同一个模型、同一个问题（"三条产线各停 2 小时，共损失多少机时"），只改模板和思考开关（实测，见 `benchmarks/template/template.csv`，命令见本节"待实测"）：

| 配置 | 首字延迟 | 总时长 | 生成分片 | 答案字数 | 思考内容 |
|---|---|---|---|---|---|
| 模型自带模板，思考开 | 0.50 s | 16.73 s | 263 | 9 | 有，807 字符 |
| 模型自带模板，思考关 | 0.39 s | 0.81 s | 7 | 9 | 无 |
| `--chat-template chatml`，思考关 | 0.26 s | 16.58 s | 267 | 834 | 无（`<think>` 混进正文） |

三行数据的含义：

**第一，思考模式的成本是二十倍，不是百分之几十。** 同一道题，开思考生成 263 个分片、16.7 秒；关思考 7 个分片、0.81 秒。两者答案都是 `3 × 2 = 6`，都正确。这就是 1.2 节说的"输出长度就是时间"的极端例子。

**第二，换模板后模型开始自言自语。** chatml 模板没有给 assistant 回合插入 `<think>`，但模型仍然习惯性地先写思考（这是训练里学到的行为），于是 `<think>` 作为**正文**输出，834 个字符里大部分是自言自语。服务端的思考解析器认不出它，所以 `reasoning_content` 为空、正文被污染。

**第三，`enable_thinking=false` 在错误模板下静默失效。** 第三行我们传了关闭思考的参数，模板里没有对应逻辑，参数没人读。程序不报错，你以为关掉了，实际没有。

顺便说明表格里首字延迟的差异：0.50 s 与 0.26 s 的差距主要来自模板渲染和 KV 状态，不是思考模式本身——思考是在首字之后才产生的。判断思考成本要看**总时长和生成量**，不是首字延迟。

## 怎么确认模板是对的

不要靠读输出猜，直接看服务端拼出来的字符串。llama.cpp 提供 `/apply-template`，把消息数组渲染成最终 prompt（引用 [\[3\]](#ref-3)）：

```bash
curl -s -X POST http://127.0.0.1:8080/apply-template \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

本机返回：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>
```

对照模型卡里的模板，逐项检查三件事：角色标记对不对、结束标记在不在、思考标记是否按预期插入。这三项对上，模板基本就对了；剩下的是参数传递问题。

一个容易误会的细节：探测请求没带 `enable_thinking`，所以上面显示的是**默认渲染**（带 `<think>`）。要看关闭思考时的渲染，需要把同样的参数一起传进去。别拿默认渲染的探测结果去解释关闭思考的行为。

## 工程含义

**模板属于部署配置，不属于代码。** 模板一变，历史对话、评估结果、prompt 缓存全部要重来。交付文档里要写清模板来源（模型自带 / 自定义文件 / 引擎内置哪一种），以及模板版本。

**升级模型必须重验模板。** 同一家族的新版本常改标记（加字段、改结束符、换思考格式）。升级后第一件事不是跑评测，而是渲染一遍看字符串。

**不要手拼 prompt。** 用 SDK 的 messages 数组，让服务端渲染。手拼字符串的代码在换模型时必然出错，而且错得很难看。

**给思考留预算。** 思考模式下 `max_tokens` 要先被思考吃掉。本机用 256 试过一次，思考写满、正文为空——看起来像"模型没回答"，其实是预算不够。要么关思考，要么把上限设到实际需要的好几倍。

## 待实测

::: tip 待实测：模板与思考模式的四种组合
**假设**：错误模板不报错但改变行为；思考模式在简单摘取任务上是净成本，在多步推理任务上才回本。

**命令**（脚本自己起两个服务、用完就关）：

```bash
cd labs
uv run python 03-deployment/template_check.py
```

脚本会做三件事：用 `/apply-template` 取回渲染结果、记录三种配置的首字延迟与生成量、把答案与思考内容存进 CSV。

想自己加一组，就把问题换成本书的业务题，比如"根据这段维护记录，下次换油应该在第几小时"，然后对比思考开/关的正确率——**成本用时间量，收益用正确率量**，两个都要记，否则"要不要开思考"永远吵不出结论。

**记录**：配置、首字延迟、总时长、生成分片数、答案字数、思考字符数、答案是否正确。数据存 `benchmarks/template/`。不同问题的最优配置可能不同，至少测三类任务（直接摘取、跨段组合、简单计算）再下结论。
:::

## 参考资料

1. <a id="ref-1" style="scroll-margin-top: 96px"></a>Hugging Face. *Chat templating*（Jinja 模板、`chat_template` 的约定）. https://huggingface.co/docs/transformers/main/en/chat_templating
2. <a id="ref-2" style="scroll-margin-top: 96px"></a>Qwen. *Qwen3.5-9B 模型卡*（模板格式、思考模式与 `enable_thinking`）. https://huggingface.co/Qwen/Qwen3.5-9B
3. <a id="ref-3" style="scroll-margin-top: 96px"></a>ggml 项目. *llama.cpp server 文档*（`--chat-template`、`--chat-template-kwargs`、`/apply-template` 端点）. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
4. <a id="ref-4" style="scroll-margin-top: 96px"></a>Pallets. *Jinja*（模板语法）. https://jinja.palletsprojects.com/
