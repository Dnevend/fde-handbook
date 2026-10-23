# AGENTS.md

给在本仓库里工作的 agent。面向读者的说明在 `README.md`，这里只写"改这个仓库时该注意什么"。

**仓库是什么**：一本书的站点与配套实验代码——《FDE Handbook：从本地大模型到 RAG 落地》。VitePress 站点 + 章节正文 + `labs/` 实验脚本。
**当前阶段**：序章、第一至第三部分正文已完成；第四至第八部分与附录仍是占位页（只有标题和"本章要回答的问题"）。

## 硬规则

1. **语料本体、模型权重、构建产物一律不入库。** `.gitignore` 已覆盖 `*.gguf`、`*.safetensors`、`corpus/raw/`、PDF/图片、`.vitepress/cache|dist`、`node_modules/`。不要为了"方便"放宽这些规则，也不要 `git add -f` 绕过。
2. **每个数字都标来源。** 用文字括注区分三类数据，不用 emoji：估算写成「约 5.7 GB（估算，推导见下）」，引用写成「C-Eval 90.5（引用 [1]）」，实测写成「21.4 tok/s（实测，见 `benchmarks/…`）」。引用编号对应当节末尾的 `## 参考资料` 列表，列表只收一手资料（论文、官方文档、源码、模型卡、协议原文）。估算不得写成事实。实测必须写明硬件、软件版本与完整命令；原始数据（CSV/JSON）放 `benchmarks/`，正文只放结论和图表。没有跑过的实验写进「待实测」提示块，只给假设、命令和要记录的指标。
3. **增删章节要三处同步**：`docs/.vitepress/config.ts` 的侧边栏、所在部分的 `index.md` 列表、页面 frontmatter 的 `title`。
4. **不写没验证过的结论。** 占位页保留"本章要回答的问题"句式；正文写到什么程度就只声明到什么程度，未做的实验不能写成"实测"。
5. **中文正文，术语首次出现标注英文**（如 首字延迟（TTFT））。
6. **首页动效零依赖。** 封面背景（`HeroAurora.vue`）保持纯 CSS + 内联 SVG 噪声，不引动画库、不写客户端 JS；动效必须在 `prefers-reduced-motion: reduce` 下静态可用，且不得拦截点击或推迟首屏文字渲染。
7. **图从图源生成，不手改 SVG。** 结构图（架构、流程、时序、甘特）的图源是 `figures/src/*.json`，用 `figures/render.sh` 渲染；定量图由 `benchmarks/estimates/plot.py` 读 CSV 生成。`docs/public/figures/*.svg` 是产物，不直接编辑。图号写 `图 <节号>-<序号>`（如 `图 1.3-1`）；部分导言页没有节号，用 `图 <部分号>-<序号>`（如 `图 3-1`，序章用 `图 0-1`）。正文至少引用一次；只要图里出现数字，`<Figure>` 的 `source` 必填，口径同第 2 条。

## 路径约定

- `docs/00-preface/` … `docs/08-delivery/`、`docs/appendix/`：正文，一个目录对应一个部分。
- `docs/.vitepress/theme/`：主题扩展。`index.ts` 负责挂默认主题槽位，`components/` 放组件，`custom.css` 只放必须作用于默认主题元素的全局选择器；能用页面或组件内样式解决的不要放这里。
- `figures/src/`：图源（JSON，显式坐标），文件名 `<节号>-<slug>.json`，**入库**。
- `figures/out/`：渲染报告与审阅 PNG，**不入库**（已进 `.gitignore`）。
- `docs/public/figures/`：站点引用的 SVG 产物，入库，和 `figures/src/` 同步提交。
- `labs/<部分编号>-<部分名>/`：与章节一一对应的可运行代码。
- `labs/07-gpu-inference/`：云 GPU 实验，**本地（Mac）不要求能跑**。
- `benchmarks/`：原始测量数据与生成图表的脚本。`benchmarks/estimates/` 放估算类的 CSV 与绘图脚本（PEP 723 + uv，不写进 `package.json`）。
- `corpus/`：只放 `sources.yaml`（链接、协议、用途）和下载脚本。公开资料也有版权，正文引用时注明协议。
- `eval/`：测试问题集（问题 / 标准答案 / 出处）。

## 常用命令

```bash
npm run docs:dev      # 本地预览
npm run docs:build    # 构建（改完侧边栏或页面后必须跑一次）
npm run docs:preview  # 预览构建结果

./figures/render.sh            # 渲染全部图 + 几何校验
./figures/render.sh 1-3 --png  # 只渲染 1-3 开头的图，并导出审阅 PNG
uv run benchmarks/estimates/plot.py --preview   # 重画定量图
```

画图的几何校验靠 `fireworks-tech-graph`（装在 `~/.agents/skills/fireworks-tech-graph`，用 `FIREWORKS_GRAPH_SKILL` 覆盖路径）。它要求显式坐标，所以图源是 JSON 而不是 Markdown 指令；`check` 会拦文字截断、标注重叠和连线弯折，但字号是启发式估算，导出 PNG 目视一次仍不可省。

## 写图的约定

新增结构图的最省事路径：复制 `figures/src/` 里拓扑最接近的 JSON，改 `nodes` / `arrows`，再 `./figures/render.sh <前缀> --png`。

- 三个字段固定别改：`"style": 4`（Notion Clean，白底、单一蓝色箭头）、`"quality_profile": "standard"`（允许括号端头这类零间距元素）、`"text_policy": "strict"`（文字放不下直接报错，不静默截断）。
- 画布 960 宽。节点 160×64 时中文标签最多 6 个字，超出就拆成两节或挪进 `sublabel`；`title_size` 默认 20，放不下会自动缩到 12。
- 图里不写 `→`、不用 emoji：方向用连线表达，`→` 在 cairosvg 下还会掉字。
- 白底是刻意的：浅色主题下与页面同色，深色主题由 `Figure.vue` 兜成白卡。不要给单张图做深色版。
- 出图后必须打开 `figures/out/preview/*.png` 看一眼。`check` 的宽度是启发式，全角字符估得偏乐观。
- 正文插入方式：

  ```md
  <Figure
    id="1.3-1"
    src="/figures/1-3-prefill-decode-timeline.svg"
    alt="一句话描述图上有什么"
    caption="这张图在说什么，别复述图里的字"
    source="数字的来源与口径，没有数字就删掉这一行"
  />
  ```

## 不要做

- 不要为了"看起来完整"提前填正文，占位页是刻意的。
- 不要调整章节顺序或增删部分，结构由维护者决定。
- 不要擅自更换开源协议（正文与数据 CC BY-NC-SA 4.0、代码 MIT，见 `LICENSE-docs` 与 `LICENSE`），也不要擅自决定部署方式（待定）。
- 不要引入与当前章节无关的依赖；需要新依赖时先说明用途。
- 不要在仓库里写本机绝对路径（`/Users/...`）。
