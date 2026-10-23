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

- 主力模型为 8B 级 `Q4_K_M`（具体型号写到第二部分时按当时最新版本选定）；14B 只用于对比与消融实验。
- 本书**不覆盖**：CUDA/Triton kernel 开发、分布式训练、K8s/Slurm 集群调度、InfiniBand 组网。Mac 上学不到的部分（CUDA、多卡）明确放在第七部分，并说明如何用云 GPU 补齐。

## 目录导览

| 目录 | 内容 |
|---|---|
| `docs/` | VitePress 正文根目录，侧边栏即书的目录 |
| `docs/00-preface/` … `docs/08-delivery/` | 序 + 八个部分的正文 |
| `docs/appendix/` | 速查表、术语表、排查手册、面试题索引、参考文献 |
| `docs/public/` | 图片与架构图 |
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

推送到 `master` 后由 GitHub Actions 自动构建并发布到 GitHub Pages，配置见 `.github/workflows/deploy.yml`（VitePress 官方模板）。

- 线上地址：<https://dnevend.github.io/fde-handbook/>
- 站点部署在子路径下，构建时通过环境变量传入 `base`：`DOCS_BASE=/fde-handbook/ npm run docs:build`。本地开发不设该变量，仍在根路径预览。
- 首次使用需在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
- 也可在 Actions 页面手动触发（`workflow_dispatch`）。

## 写作约定

- 每个数据都注明硬件、软件版本与完整命令，原始数据放 `benchmarks/`。
- 区分数据来源：🧪 实测 / 📐 估算 / 📚 引用，不把估算写成事实。
- 只引一手资料：论文、官方文档、源码，附链接。
- 语料不入库，并注明各份资料的协议。

## 进度

当前仓库处于骨架阶段：站点可运行，侧边栏与大纲一致，各节为占位页（只有标题与"本章要回答的问题"）。正文按 `docs/00-preface/` → `docs/08-delivery/` 的顺序推进。

## 协议

| 内容 | 协议 |
|---|---|
| 正文（`docs/` 下的 Markdown 与图表） | [CC BY-NC-SA 4.0](./LICENSE-docs) |
| 基准数据与测试集（`benchmarks/`、`eval/`） | [CC BY-NC-SA 4.0](./LICENSE-docs) |
| 配套代码（`labs/`、站点配置） | [MIT](./LICENSE) |

- 正文与数据**不得用于商业目的**；转载或改编需署名，衍生作品须以相同协议共享。
- 代码可自由使用、修改、分发，保留版权声明即可。
