# AGENTS.md

给在本仓库里工作的 agent。面向读者的说明在 `README.md`，这里只写"改这个仓库时该注意什么"。

**仓库是什么**：一本书的站点与配套实验代码——《FDE Handbook：从本地大模型到 RAG 落地》。VitePress 站点 + 章节正文 + `labs/` 实验脚本。
**当前阶段**：已完成序与第一至第六部分；第七、第八部分与附录仍是占位页。**进度以 `docs/.vitepress/book.json` 为准**，不要在这里复制状态——它曾经和正文对不上。

## 硬规则

1. **语料本体、模型权重、构建产物一律不入库。** `.gitignore` 已覆盖 `*.gguf`、`*.safetensors`、`corpus/raw/`、PDF/图片、`.vitepress/cache|dist`、`node_modules/`。不要为了"方便"放宽这些规则，也不要 `git add -f` 绕过。
2. **每个数字都标来源。** 用文字括注区分三类数据，不用 emoji：估算写成「约 5.7 GB（估算，推导见下）」，引用写成「C-Eval 90.5（引用 [\[1\]](#ref-1)）」，实测写成「21.4 tok/s（实测，见 `benchmarks/…`）」。引用编号对应当节末尾 `## 参考资料` 的第几条，锚点写成 `<a id="ref-1" style="scroll-margin-top: 96px"></a>`，正文里的 `[n]` 必须链到 `#ref-n`——不要写裸的「（引用 [1]）」，那样点不动。列表只收一手资料（论文、官方文档、源码、模型卡、协议原文）。估算不得写成事实。实测必须写明硬件、软件版本与完整命令；原始数据（CSV/JSON）放 `benchmarks/`，正文只放结论和图表。没有跑过的实验写进「动手实验」提示块，只给假设、命令和要记录的指标。
3. **章节信息只改一处。** 部分、章节、状态与阅读路径都写在 `docs/.vitepress/book.json`：侧边栏、部分导言页的 `<ChapterList>`、README 进度块、0.4 的 `<ReadingPath>` 全部由它生成。改完跑 `npm run docs:sync`，再用 `npm run check` 验证。页面 frontmatter 的 `title` 仍需与 book.json 的 `title` 一致（脚本会比对）。
4. **不写没验证过的结论。** 占位页保留"本章要回答的问题"句式；正文写到什么程度就只声明到什么程度，未做的实验不能写成"实测"。
5. **中文正文，术语首次出现标注英文**（如 首字延迟（TTFT））。
6. **首页动效零依赖。** 封面背景（`HeroAurora.vue`）保持纯 CSS + 内联 SVG 噪声，不引动画库、不写客户端 JS；动效必须在 `prefers-reduced-motion: reduce` 下静态可用，且不得拦截点击或推迟首屏文字渲染。
7. **图从图源生成，不手改 SVG。** 结构图（架构、流程、时序、甘特）的图源是 `figures/src/*.json`，用 `figures/render.sh` 渲染；定量图由 `benchmarks/estimates/plot.py` 读 CSV 生成。`docs/public/figures/*.svg` 是产物，不直接编辑。图号写 `图 <节号>-<序号>`（如 `图 1.3-1`）；部分导言页没有节号，用 `图 <部分号>-<序号>`（如 `图 3-1`，序章用 `图 0-1`）。正文至少引用一次；只要图里出现数字，`<Figure>` 的 `source` 必填，口径同第 2 条。

## 路径约定

- `docs/00-preface/` … `docs/08-delivery/`、`docs/appendix/`：正文，一个目录对应一个部分。
- `docs/.vitepress/theme/`：主题扩展。`index.ts` 负责挂默认主题槽位，`components/` 放组件，`custom.css` 只放必须作用于默认主题元素的全局选择器；能用页面或组件内样式解决的不要放这里。
- `figures/src/`：图源（JSON，显式坐标），**入库**。文件名按两种模式：
  - 节内图 `<部分>-<节>-<slug>.json`，对应图 `1.3-1` → `1-3-<slug>.json`；
  - 部分导言图 `<部分>-part<部分>-<slug>.json`，对应图 `1-1` → `1-part1-<slug>.json`。用 `part` 是为了避开图 `1.1-x` 的文件名前缀。
- `figures/out/`：渲染报告与审阅 PNG，**不入库**（已进 `.gitignore`）。
- `docs/public/figures/`：站点引用的 SVG 产物，入库，和 `figures/src/` 同步提交。
- `labs/<部分编号>-<部分名>/`：与章节一一对应的可运行代码。
- `labs/07-gpu-inference/`：云 GPU 实验，**本地（Mac）不要求能跑**。
- `benchmarks/`：原始测量数据与生成图表的脚本。`benchmarks/estimates/` 放估算类的 CSV 与绘图脚本（PEP 723 + uv，不写进 `package.json`）。
- `corpus/`：语料清单与流水线。`sources.yaml` 记出处、协议与 `redistributable` 档位；`index.json` 是条款索引，入库；`raw/` 是下载物、节选与合成产物，**不入库**。规则见「写语料的约定」。
- `eval/`：测试问题集与校验脚本。`qa.jsonl` 是第五部分的 v1（80 题）；`qa-v2.jsonl` 是第六部分的 v2（300 题，dev/test 各 150，`split` 字段区分）。改题后跑 `python3 eval/validate.py`；口径见 `eval/README.md`，统计产物在 `benchmarks/evaluation/`。

## 常用命令

```bash
npm run docs:dev      # 本地预览
npm run docs:build    # 构建（改完页面后必须跑一次）
npm run docs:preview  # 预览构建结果
npm run docs:sync     # 由 book.json 重写 README 进度块
npm run check         # 校验清单、图、引用、禁用词是否漂移

./figures/render.sh            # 渲染全部图 + 几何校验
./figures/render.sh 1-3 --png  # 只渲染 1-3 开头的图，并导出审阅 PNG
uv run benchmarks/estimates/plot.py --preview   # 重画定量图

./corpus/fetch.sh --list                          # 语料清单，不联网
./corpus/fetch.sh                                 # 只下 full 档
uv run corpus/excerpt.py --all                    # 切 excerpt 档
uv run corpus/index.py                            # 重建条款索引
uv run corpus/synthetic/generate.py --clean --validate   # 合成语料 + 自检
python3 eval/validate.py                          # 校验 v1/v2 测试集
```

画图的几何校验靠 `fireworks-tech-graph`（装在 `~/.agents/skills/fireworks-tech-graph`，用 `FIREWORKS_GRAPH_SKILL` 覆盖路径）。它要求显式坐标，所以图源是 JSON 而不是 Markdown 指令；`check` 会拦文字截断、标注重叠和连线弯折，但字号是启发式估算，导出 PNG 目视一次仍不可省。

## 写语料的约定

`corpus/` 只解决"资料从哪来、能不能用"，正文怎么写不在这里。

- **三档授权，语义不能混**：`full`（法规、政府文件，可全文入库）、`excerpt`（招股书、事故报告、专利、环评，只节选）、`none`（国标、厂商手册，只记链接）。"免费公开"不等于"可再分发"。
- **只提交清单与脚本，语料本体不入库。** `sources.yaml` 记出处、协议、`sha256`；`raw/` 已被 `.gitignore` 覆盖，不要 `git add -f`。
- **节选按形态分档**：长文档 ≤5%，短报告 ≤30% 且 ≤3000 字。每份的上限写在条目的 `excerpt_ratio` 里，`excerpt.py` 超限即报错。
- **正文用化名，出处留真名**：经营类材料 → `先赚一个亿科技有限公司`；事故类 → `某注塑厂 A`，且不写伤亡细节。人名为"姓 + 某"，身份证号、住址、电话、统一社会信用代码直接删。真实名称只出现在 `sources.yaml`——清单也化名，就没人能核对了。
- **合成语料只补四类**：规模、权限、版本冲突、脏数据。字段名必须来自 `synthetic/fields.yaml`，且标明 `verbatim`（逐字抄政府范本）还是 `derived`（推导），不许自创"备注"这类万能列。
- **合成件上跑出的数字不许写成实测。** 标记为"实测"的结论必须跑在真实语料上。
- **改了生成器先跑 `--validate`**：验份数、字段声明、四类「脏」、同 seed 的确定性。
- **新增语料要在 `index.py` 的 `AUTHORITY` 里登记效力层级**，否则脚本报错。5.9 的权限过滤靠这个字段。
- 已知缺口记在 `corpus/GAPS.md`，别在正文里用"以后再补"糊过去。

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
- 不要手改生成物：`config.ts` 的 `sidebar`、部分导言页的 `<ChapterList>` 输出、README 进度块、`docs/public/figures/*.svg`。改源头（`book.json` / `figures/src/`），再跑 `npm run docs:sync` 或 `./figures/render.sh`。
- 不要调整章节顺序或增删部分，结构由维护者决定。
- 不要擅自更换开源协议（正文与数据 CC BY-NC-SA 4.0、代码 MIT，见 `LICENSE-docs` 与 `LICENSE`），也不要擅自更换部署方式（已定为 GitHub Pages + GitHub Actions，见 `.github/workflows/deploy.yml`）。
- 不要引入与当前章节无关的依赖；需要新依赖时先说明用途。
- 不要在仓库里写本机绝对路径（`/Users/...`）。
