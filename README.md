# FDE Handbook

> 《FDE Handbook：从本地大模型到 RAG 落地》——书籍正文与配套实验代码。

写给**懂工程、想走进 AI 的工程师**（前端 / 后端 / 全栈 / 工控）。不假设你懂 AI 原理，只假设你懂工程：Linux、Docker、HTTP/API、数据库、缓存、搜索。你已有的经验就是起点。

## 这本书想解决什么

不是"科普大模型"，而是让你能独立把一套**离线、私有、可交付**的本地大模型 RAG 系统做出来、量化地证明它有效、并交给客户长期使用。

全书沿一条问题链推进：跑起来一个模型 → 为什么这么吃内存 → 该选哪个模型 → 为什么有这么多格式 → 量化到底压掉了什么 → 模型不知道公司资料怎么办 → 怎么知道它答得好不好 → 50 人同时用怎么办 → 怎么交付。

## 硬件前提

| 阶段 | 环境 | 成本 |
|---|---|---|
| 第一至第六部分、第八部分 | Apple M4 / 24GB 统一内存 | 0 元 |
| 第七部分实操（云 GPU 实验） | 单卡 4090 / 双卡，按小时计费 | 几十至一百多元，以平台实时价格为准 |

- 主力模型为 `Qwen3.5-9B` 的 `Q4_K_M`（约 5.7 GB）；`Qwen3-14B` 的 `Q4_K_M`（约 9.0 GB）只用于对比与消融实验；`Gemma-4-26B-A4B` 的 `Q4_K_M`（约 17 GB）用作 Dense 与 MoE 的对照。三者均为 Apache-2.0 协议。
- 本书**不覆盖**：CUDA/Triton kernel 开发、分布式训练、K8s/Slurm 集群调度、InfiniBand 组网。Mac 上学不到的部分（CUDA、多卡）明确放在第七部分，并说明如何用云 GPU 补齐。

## 目录导览

| 目录 | 内容 |
|---|---|
| `docs/` | VitePress 正文根目录，侧边栏即书的目录 |
| `docs/00-preface/` … `docs/08-delivery/` | 序 + 八个部分的正文 |
| `docs/appendix/` | 速查表、术语表、排查手册、面试题索引、参考文献 |
| `docs/public/` | 图片与架构图（站点直接引用，**图的源在 `figures/src/`**） |
| `figures/src/` | 结构图的图源（JSON，显式坐标）；`figures/render.sh` 渲染成 `docs/public/figures/*.svg` |
| `labs/` | 与章节一一对应的可运行代码（`07-gpu-inference/` 只放云 GPU 脚本，本地不跑） |
| `benchmarks/` | 原始测量数据（CSV/JSON）与生成图表的脚本 |
| `corpus/` | 公开语料清单与下载脚本；**只提交清单，不提交 PDF 本体** |
| `eval/` | 测试问题集（问题 / 标准答案 / 出处） |

## 本地起站

需要 Node.js 20+（开发环境为 Node 26）。

```bash
npm install
npm run docs:dev      # 本地预览，默认 http://localhost:5173
npm run docs:build    # 构建静态站点到 docs/.vitepress/dist
npm run docs:preview  # 预览构建结果
```

## 部署

待定。当前仓库为私有，暂未配置发布流程。

## 写作约定

- 每个数据都注明硬件、软件版本与完整命令，原始数据放 `benchmarks/`。
- 数据来源用文字括注，不用 emoji：估算写「约 5.7 GB（估算，推导见下）」，引用写「（引用 [1]）」，实测写「（实测，见 `benchmarks/…`）」。引用编号对应当节末尾的「参考资料」，只收一手资料：论文、官方文档、源码、模型卡、协议原文。
- 语料不入库，并注明各份资料的协议。
- 插图不手画：结构图的源是 `figures/src/*.json`，定量图由 `benchmarks/estimates/plot.py` 从 CSV 生成，`docs/public/figures/` 里只有渲染产物。图号写 `图 1.3-1`（部分导言页写 `图 3-1`），正文至少引用一次，图里的每个数字同样要标来源。

## 进度

序、第一至第四部分已有正文；第五至第八部分与附录仍为占位页（只有标题与"本章要回答的问题"）。正文按 `docs/00-preface/` → `docs/08-delivery/` 的顺序推进。

第三部分的实验代码在 `labs/03-deployment/`，已在本机跑通；对应实测数据在 `benchmarks/`（`gguf-format/`、`prefill-decode/`、`template/`、`sampling/`、`concurrency/`、`client-bench/`）。

第四部分的实验代码在 `labs/04-quantization/`，同样已跑通；数据在 `benchmarks/quantization/`（张量格式误差、分组量化、KV 量化、跨域困惑度、60 题探针）。这一部分没有下载额外的量化模型：需要从 16 位源重切阶梯的对照写在各节的「待实测」里。

## 协议

| 内容 | 协议 |
|---|---|
| 正文（`docs/` 下的 Markdown 与图表） | [CC BY-NC-SA 4.0](./LICENSE-docs) |
| 基准数据与测试集（`benchmarks/`、`eval/`） | [CC BY-NC-SA 4.0](./LICENSE-docs) |
| 配套代码（`labs/`、站点配置） | [MIT](./LICENSE) |

- 正文与数据**不得用于商业目的**；转载或改编需署名，衍生作品须以相同协议共享。
- 代码可自由使用、修改、分发，保留版权声明即可。
