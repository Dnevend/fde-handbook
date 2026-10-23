# corpus/

公开语料的**清单与获取方式**。正文、PDF 本体、扫描件都不入库——公开资料也有版权。

主线场景是某注塑厂 3 号线（注塑机 / 模温机 / 冷水机 / 机械手 / 模架），模具车间（CNC / EDM / 抛光 / 试模）作为对照。清单见 `sources.yaml`。

## 为什么是这些资料

收什么由"要验证第五、六部分的哪个能力"决定，不由"厂里有什么"决定。一句判据：

> **内容可以合成，缺陷不能合成。**

RAG 的失败大多来自文档的"脏"和"乱"，不来自文字本身。所以凡是脏在载体上的（扫描退化、跨页表格、双栏图注），收真实件；脏在业务逻辑上的（版本冲突、权限、术语不一致、多跳），合成反而更可控也更合规。

## 三档授权，语义不能混

| `redistributable` | 含义 | 脚本行为 |
|---|---|---|
| `full` | 法规、政府文件，可直接全文入库 | `fetch.sh` 自动下载到 `raw/` |
| `excerpt` | 招股书、事故报告、专利、环评，只能节选引用 | 只打印清单，人工挑章节 |
| `none` | 国标、厂商手册，只记链接与条款号 | 不下载，不提醒 |

`excerpt` 的统一口径：**只取与结论直接相关的最小片段**，按文档形态分档——

- 长文档（招股书、环评）≤ 5%；
- 短报告（事故调查、期刊）≤ 30%，且单份不超过 3000 字。

10% 一刀切在 14 页的事故报告上只能取 700 字，连一段设备描述都放不下，反而逼人取整节。原 PDF 停在 `raw/`，不入库。

`none` 存在的意义是留证据、不留文件：国标要求免费公开，但"免费公开"不等于"可再分发"。

## 命令

```bash
corpus/fetch.sh --list        # 看清单，不联网
corpus/fetch.sh --dry-run     # 看会做什么
corpus/fetch.sh               # 只下 full 档
corpus/fetch.sh --only reg-mem-order-13,gov-checklist-hs

uv run corpus/excerpt.py --all                 # 切所有 excerpt 档
uv run corpus/excerpt.py --id acc-fangsu-0518  # 只切一份

uv run corpus/index.py                         # 建条款索引 → corpus/index.json

uv run corpus/synthetic/generate.py --clean --validate   # 生成 60 份并自检
```

`fetch.sh` 是 `fetch.py` 的壳，后者带 PEP 723 声明，首次运行由 uv 装依赖。下载完会在 `raw/<id>/.meta.json` 里记 url、日期、sha256、字节数，并把 sha256 打印出来供回填 `sources.yaml`。

`excerpt.py` 只处理 `excerpt` 档：章节边界写在 `sources.yaml` 的 `ranges` 里（正则），切完报告占原文比例，超上限就报错，逼你回去收窄。节选不脱敏——`raw/` 不入库，真实名称只在这一层存在，写到正文时再按 `redaction.yaml` 替换。

`index.py` 只处理 `full` 档：抽条款号、所在章、开头 40 字，外加效力层级（法律 / 部门规章 / 规范性文件 / 标准 / 内部制度）。索引进库，条款正文不进。5.9 的权限过滤要的就是这张表。

`generate.py --validate` 自检四件事：份数、字段是否都在 `fields.yaml` 里声明、四类「脏」是否齐全、同 seed 两次生成是否一致。动了生成器先跑它，别等写实验时才发现漏项没了。

## 目录

```
corpus/
  sources.yaml            # 清单，入库
  redaction.yaml          # 真实名 → 正文化名的映射，入库
  fetch.sh / fetch.py     # 全量档抓取，入库
  excerpt.py              # excerpt 档切片，入库
  index.py / index.json   # 条款索引，入库
  synthetic/
    spec.md               # 合成规则
    fields.yaml           # 表壳：字段名与出处
    generate.py           # 确定性生成器（同 seed 同结果）
  raw/                    # 下载物、节选、合成产物，不入库（.gitignore）
```

## 脱敏

正文、图注、附录叙述里的企业名一律替换，真实出处只留在 `sources.yaml`——化名会让清单失去可核查性。两套名字分开用：

- 经营类（招股书、管理制度、合成工单）→ `先赚一个亿科技有限公司`
- 事故类（涉人员死亡）→ `某注塑厂 A`，且正文不写伤亡细节

人名一律"姓 + 某"，身份证号、住址、电话、统一社会信用代码直接删除。设备型号与参数不改写，它们才是检索价值所在。规则见 `redaction.yaml`。

## 合成语料

真实手册通常 30–80 页，撑不起索引规模对比；真实语料也没有 ACL。所以规模的、权限的、冲突的实验跑合成件。规则见 `synthetic/spec.md`，实现在 `synthetic/generate.py`（不调模型，固定 seed，同一条命令得到同一份语料），字段名在 `synthetic/fields.yaml` 里标了逐字来源还是派生。

有一条是硬的：**凡是标"实测"的结论，必须跑在真实语料上。**
